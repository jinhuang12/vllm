#!/usr/bin/env python3
"""ammo: Verify Validation Stage gates are properly met.

This script performs BLOCKING verification for Validation Stage completion.
Optimization MUST NOT be declared complete if this script returns non-zero exit code.

Checks:
1. Benchmark uses vLLM baseline (fused_experts/fused_moe), NOT naive PyTorch
2. Tests include numerical comparison (torch.allclose), not just smoke tests
3. Production parity: torch.compile enabled (not TORCH_COMPILE_DISABLE=1)
4. All kill criteria are evaluated (not marked "TODO" or "optional")
5. state.json gate status is properly recorded

Exit codes:
  0 = PASS (all gates met, safe to proceed to Phase 5)
  1 = BLOCKER (invalid validation, do NOT proceed)
  2 = ERROR (script execution error)

Output: JSON report to stdout with detailed gate status.

Usage:
  python verify_phase4_gates.py /path/to/artifact_dir
  python verify_phase4_gates.py /path/to/artifact_dir --json-output /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class GateResult:
    """Result of a single gate check."""
    name: str
    status: str  # "PASS", "FAIL", "WARN"
    message: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class VerificationReport:
    """Full verification report."""
    artifact_dir: str
    phase: str = "5_validation"
    overall_status: str = "UNKNOWN"
    gates: List[GateResult] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self):
        return {
            "artifact_dir": self.artifact_dir,
            "phase": self.phase,
            "overall_status": self.overall_status,
            "gates": [asdict(g) for g in self.gates],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


def find_python_files(artifact_dir: Path, pattern: str = "*.py") -> List[Path]:
    """Find Python files in artifact directory."""
    files = list(artifact_dir.glob(pattern))
    files.extend(artifact_dir.glob(f"**/{pattern}"))
    return files


def _strip_comments(content: str) -> str:
    """Remove Python # comments and triple-quoted docstrings to reduce false positives.

    This is a best-effort heuristic — it won't handle all edge cases (e.g.,
    # inside strings) but eliminates the most common false positive case of
    commented-out code and docstrings containing pattern keywords.
    """
    # Remove triple-quoted docstrings (both ''' and """)
    content = re.sub(r'"""[\s\S]*?"""', '', content)
    content = re.sub(r"'''[\s\S]*?'''", '', content)
    # Remove single-line # comments (but not #! shebangs)
    content = re.sub(r'(?<!.)#!.*$', lambda m: m.group(), content, flags=re.MULTILINE)
    content = re.sub(r'#[^!].*$', '', content, flags=re.MULTILINE)
    return content


def check_baseline_is_vllm(artifact_dir: Path) -> GateResult:
    """Gate 4.1: Check that benchmark baseline is vLLM (not naive PyTorch)."""
    benchmark_files = find_python_files(artifact_dir, "benchmark*.py")
    benchmark_files.extend(find_python_files(artifact_dir, "*validation*.py"))

    if not benchmark_files:
        return GateResult(
            name="baseline_is_vllm",
            status="FAIL",
            message="No benchmark files found",
            evidence=[f"Searched: {artifact_dir}/*.py, benchmark*.py"],
        )

    vllm_baseline_patterns = [
        r"from\s+vllm.*import",  # Generic vLLM import
        r"fused_moe\s*\(",
        r"fused_experts\s*\(",
        r"ops\.fused_moe",
        r"_moe_C\.",
        r"flash_attn",
        r"paged_attention",
        r"vllm\.\w+",
    ]

    naive_pytorch_patterns = [
        r"for\s+\w+\s+in\s+range\s*\(\s*\w*expert",  # for e in range(num_experts)
        r"torch\.matmul.*for\s+",  # torch.matmul in loop
        r"\.mm\s*\(.*for\s+",  # .mm in loop
        r"naive|baseline_naive|pytorch.*baseline",
    ]

    evidence = []
    has_vllm_baseline = False
    has_naive_baseline = False

    for bf in benchmark_files:
        raw_content = bf.read_text(encoding="utf-8", errors="ignore")
        content = _strip_comments(raw_content)

        for pattern in vllm_baseline_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_vllm_baseline = True
                evidence.append(f"Found vLLM baseline in {bf.name}: {pattern}")
                break

        for pattern in naive_pytorch_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_naive_baseline = True
                evidence.append(f"WARNING: Found naive PyTorch in {bf.name}: {pattern}")

    if has_vllm_baseline and not has_naive_baseline:
        return GateResult(
            name="baseline_is_vllm",
            status="PASS",
            message="Benchmark uses vLLM baseline (production kernel)",
            evidence=evidence,
        )
    elif has_naive_baseline:
        return GateResult(
            name="baseline_is_vllm",
            status="FAIL",
            message="Benchmark uses naive PyTorch baseline instead of vLLM production kernels",
            evidence=evidence + [
                "BLOCKER: Must compare against vLLM's actual production kernels",
                "Replace naive loops with: import from vllm production kernel path",
            ],
        )
    else:
        return GateResult(
            name="baseline_is_vllm",
            status="FAIL",
            message="No vLLM baseline detected in benchmark files",
            evidence=evidence + [
                "Required: Import and call vLLM's fused_moe or fused_experts",
                "Searched patterns: vLLM imports, production kernel calls",
            ],
        )


def check_numerical_comparison(artifact_dir: Path) -> GateResult:
    """Gate 4.2: Check tests have numerical comparison (torch.allclose)."""
    test_files = find_python_files(artifact_dir, "test*.py")

    # Also check tests/kernels/moe/ if we can find it
    vllm_root = artifact_dir.parent.parent
    if (vllm_root / "tests" / "kernels" / "moe").exists():
        test_files.extend((vllm_root / "tests" / "kernels" / "moe").glob("test*.py"))

    if not test_files:
        return GateResult(
            name="numerical_comparison",
            status="FAIL",
            message="No test files found",
            evidence=[f"Searched: {artifact_dir}/test*.py"],
        )

    numerical_patterns = [
        r"torch\.allclose\s*\(",
        r"torch\.testing\.assert_close\s*\(",
        r"np\.allclose\s*\(",
        r"assert.*atol.*rtol",
        r"max_abs_diff|max_absolute_error",
    ]

    smoke_only_patterns = [
        r"assert.*\.shape\s*==",  # Shape check only
        r"assert.*not.*nan",  # NaN check only
        r"assert.*not.*inf",  # Inf check only
        r"assert.*dtype\s*==",  # dtype check only
    ]

    evidence = []
    has_numerical = False
    has_smoke_only = False

    for tf in test_files:
        raw_content = tf.read_text(encoding="utf-8", errors="ignore")
        content = _strip_comments(raw_content)

        for pattern in numerical_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_numerical = True
                evidence.append(f"Found numerical comparison in {tf.name}")
                break

        # Check if file ONLY has smoke tests
        smoke_matches = sum(1 for p in smoke_only_patterns if re.search(p, content, re.IGNORECASE))
        numerical_matches = sum(1 for p in numerical_patterns if re.search(p, content, re.IGNORECASE))

        if smoke_matches > 0 and numerical_matches == 0:
            has_smoke_only = True
            evidence.append(f"WARNING: {tf.name} has only smoke tests (shape/NaN checks)")

    if has_numerical:
        return GateResult(
            name="numerical_comparison",
            status="PASS",
            message="Tests include numerical comparison (torch.allclose or equivalent)",
            evidence=evidence,
        )
    elif has_smoke_only:
        return GateResult(
            name="numerical_comparison",
            status="FAIL",
            message="Tests are smoke tests only (shape/NaN checks) without numerical comparison",
            evidence=evidence + [
                "BLOCKER: Must compare optimized kernel output vs vLLM baseline numerically",
                "Add: torch.allclose(optimized_out, vllm_baseline_out, atol=..., rtol=...)",
            ],
        )
    else:
        return GateResult(
            name="numerical_comparison",
            status="FAIL",
            message="No numerical comparison found in test files",
            evidence=evidence + ["Required: torch.allclose() or torch.testing.assert_close()"],
        )


def check_production_parity(artifact_dir: Path) -> GateResult:
    """Gate 4.3: Check benchmarks run with production parity (torch.compile enabled)."""
    benchmark_files = find_python_files(artifact_dir, "benchmark*.py")
    benchmark_files.extend(find_python_files(artifact_dir, "*validation*.py"))

    if not benchmark_files:
        return GateResult(
            name="production_parity",
            status="FAIL",
            message="No benchmark files found",
            evidence=[],
        )

    # Bad patterns (disabling production features)
    bad_patterns = [
        r"TORCH_COMPILE_DISABLE\s*=\s*[\"']?1",
        r"os\.environ.*TORCH_COMPILE.*=.*[\"']1[\"']",
        r"--enforce-eager",
        r"enforce_eager\s*=\s*True",
    ]

    # Good patterns (explicitly enabling production features)
    good_patterns = [
        r"VLLM_TORCH_COMPILE_LEVEL\s*=\s*[\"']?[23]",
        r"--cuda-graph",
        r"cuda_graph\s*=\s*True",
        r"torch\.compile",
    ]

    evidence = []
    has_bad = False
    has_good = False

    for bf in benchmark_files:
        raw_content = bf.read_text(encoding="utf-8", errors="ignore")
        content = _strip_comments(raw_content)

        for pattern in bad_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_bad = True
                evidence.append(f"BLOCKER: Found production-disabling pattern in {bf.name}: {pattern}")

        for pattern in good_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                has_good = True
                evidence.append(f"Found production parity indicator in {bf.name}: {pattern}")

    if has_bad:
        return GateResult(
            name="production_parity",
            status="FAIL",
            message="Benchmark explicitly disables production features (torch.compile, CUDA graphs)",
            evidence=evidence + [
                "BLOCKER: Benchmarks must use production settings",
                "Remove TORCH_COMPILE_DISABLE=1, enforce_eager=True",
                "Use VLLM_TORCH_COMPILE_LEVEL=3 and CUDA graphs enabled",
            ],
        )
    elif has_good:
        return GateResult(
            name="production_parity",
            status="PASS",
            message="Benchmark uses production parity settings",
            evidence=evidence,
        )
    else:
        return GateResult(
            name="production_parity",
            status="WARN",
            message="Cannot determine production parity status from benchmark files",
            evidence=evidence + [
                "Recommended: Explicitly set VLLM_TORCH_COMPILE_LEVEL=3",
                "Recommended: Enable CUDA graphs (default in vllm bench latency)",
            ],
        )


def check_kill_criteria_complete(artifact_dir: Path) -> GateResult:
    """Gate 4.4: Check all kill criteria are evaluated (not TODO/optional).

    Reads kill criteria directly from tracks/{op_id}/validation_results.md files.
    """
    tracks_dir = artifact_dir / "tracks"
    if not tracks_dir.exists():
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message="No tracks/ directory found",
            evidence=[f"Expected: {tracks_dir}"],
        )

    validation_files = list(tracks_dir.glob("*/validation_results.md"))
    if not validation_files:
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message="No validation_results.md files found in tracks/",
            evidence=[f"Searched: {tracks_dir}/*/validation_results.md"],
        )

    # Parse kill criteria from validation_results.md files
    all_incomplete: List[str] = []
    all_complete: List[str] = []
    tracks_checked: List[str] = []

    for vf in validation_files:
        track_id = vf.parent.name
        tracks_checked.append(track_id)
        content = vf.read_text(encoding="utf-8", errors="ignore")

        # Extract kill criteria section — look for "Kill Criteria" or "kill_criteria"
        # heading and parse status entries beneath it
        in_kill_section = False
        for line in content.split("\n"):
            # Detect start of kill criteria section
            if re.search(r"(?i)(kill\s*criteria|kill_criteria)", line) and (
                line.strip().startswith("#") or line.strip().startswith("**")
            ):
                in_kill_section = True
                continue
            # Detect start of a new section (end of kill criteria)
            if in_kill_section and line.strip().startswith("#"):
                in_kill_section = False
                continue
            if not in_kill_section:
                continue

            # Parse criterion lines (e.g., "- correctness: PASS", "- **latency**: PASS")
            criterion_match = re.match(
                r"\s*[-*]\s*\*{0,2}(\w[\w\s]*?)\*{0,2}\s*:\s*(.+)", line
            )
            if criterion_match:
                criterion = criterion_match.group(1).strip()
                result = criterion_match.group(2).strip()
                result_upper = result.upper()
                label = f"{track_id}/{criterion}: {result}"
                if "TODO" in result_upper or "OPTIONAL" in result_upper or "SKIP" in result_upper:
                    all_incomplete.append(label)
                else:
                    all_complete.append(label)

    if not all_complete and not all_incomplete:
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message="No kill criteria found in validation_results.md files",
            evidence=[
                f"Checked tracks: {', '.join(tracks_checked)}",
                "Expected: a 'Kill Criteria' section with criterion status entries",
            ],
        )

    if all_incomplete:
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message=f"{len(all_incomplete)} kill criteria not evaluated",
            evidence=[
                "INCOMPLETE criteria:",
                *[f"  - {c}" for c in all_incomplete],
                "",
                "COMPLETE criteria:",
                *[f"  - {c}" for c in all_complete],
                "",
                "BLOCKER: All kill criteria must be evaluated before SHIP decision",
            ],
        )

    return GateResult(
        name="kill_criteria_complete",
        status="PASS",
        message=f"All {len(all_complete)} kill criteria evaluated across {len(tracks_checked)} track(s)",
        evidence=[f"  - {c}" for c in all_complete],
    )


def check_state_json_gates(artifact_dir: Path) -> GateResult:
    """Gate 4.5: Check state.json has proper gate status recorded."""
    state_file = artifact_dir / "state.json"

    if not state_file.exists():
        return GateResult(
            name="state_json_gates",
            status="FAIL",
            message="state.json does not exist",
            evidence=[],
        )

    try:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return GateResult(
            name="state_json_gates",
            status="FAIL",
            message=f"state.json is invalid JSON: {e}",
            evidence=[],
        )

    # Check gates in state.json (legacy path, may be absent in newer schemas).
    gates = state.get("phase_4_validation", {}).get("gates", {})

    if not gates:
        return GateResult(
            name="state_json_gates",
            status="WARN",
            message="No gates documented in state.json (check tracks/*/validation_results.md instead)",
            evidence=[
                "Expected at: phase_4_validation.gates",
                "Gate results may be in validation_results.md rather than state.json.",
            ],
        )

    # Check each gate has status
    gate_statuses = []
    for gate_name, gate_data in gates.items():
        if isinstance(gate_data, dict):
            status = gate_data.get("status", "UNKNOWN")
        else:
            status = str(gate_data)
        gate_statuses.append(f"{gate_name}: {status}")

    failed_gates = [g for g in gate_statuses if "FAIL" in g.upper()]
    if failed_gates:
        return GateResult(
            name="state_json_gates",
            status="FAIL",
            message=f"{len(failed_gates)} gates failed",
            evidence=gate_statuses,
        )

    return GateResult(
        name="state_json_gates",
        status="PASS",
        message="All gates recorded in state.json",
        evidence=gate_statuses,
    )


def verify_phase4(artifact_dir: Path) -> VerificationReport:
    """Run all Phase 4 verification gates."""
    report = VerificationReport(artifact_dir=str(artifact_dir))

    # Run all gates
    gates = [
        check_baseline_is_vllm(artifact_dir),
        check_numerical_comparison(artifact_dir),
        check_production_parity(artifact_dir),
        check_kill_criteria_complete(artifact_dir),
        check_state_json_gates(artifact_dir),
    ]

    report.gates = gates

    # Collect blockers and warnings
    for gate in gates:
        if gate.status == "FAIL":
            report.blockers.append(f"{gate.name}: {gate.message}")
        elif gate.status == "WARN":
            report.warnings.append(f"{gate.name}: {gate.message}")

    # Determine overall status
    if report.blockers:
        report.overall_status = "BLOCKED"
        report.recommendation = (
            "Validation INVALID. Do NOT declare optimization complete. "
            "Fix all blockers: ensure vLLM baseline, numerical comparison, "
            "production parity, and complete kill criteria evaluation."
        )
    elif report.warnings:
        report.overall_status = "WARN"
        report.recommendation = (
            "Validation conditionally complete. Review warnings before declaring complete. "
            "Ensure all comparisons are against vLLM production kernels."
        )
    else:
        report.overall_status = "PASS"
        report.recommendation = "Validation COMPLETE. Optimization can be declared shipped."

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "artifact_dir",
        type=str,
        help="Path to the artifact directory",
    )
    parser.add_argument(
        "--json-output",
        type=str,
        default=None,
        help="Path to write JSON report (default: stdout only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only output JSON, no human-readable summary",
    )
    parser.add_argument(
        "--track",
        type=str,
        default=None,
        help="Validate a specific worktree track's artifacts. "
             "When provided, artifact_dir is resolved relative to "
             ".claude/worktrees/ammo-track-{track}/{artifact_dir}/",
    )

    args = parser.parse_args()

    if args.track:
        # Resolve artifact_dir relative to the track's worktree
        artifact_dir = (
            Path(".claude/worktrees")
            / f"ammo-track-{args.track}"
            / args.artifact_dir
        ).expanduser().resolve()
    else:
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    if not artifact_dir.exists():
        print(f"ERROR: Artifact directory does not exist: {artifact_dir}", file=sys.stderr)
        return 2

    # Run verification
    report = verify_phase4(artifact_dir)

    # Output JSON
    json_output = json.dumps(report.to_dict(), indent=2)

    if args.json_output:
        Path(args.json_output).write_text(json_output, encoding="utf-8")

    if not args.quiet:
        print("=" * 60)
        print("Validation Stage Verification Report")
        print("=" * 60)
        print(f"Artifact dir: {artifact_dir}")
        print(f"Overall status: {report.overall_status}")
        print()

        for gate in report.gates:
            status_icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(gate.status, "?")
            print(f"  [{status_icon}] {gate.name}: {gate.status}")
            print(f"      {gate.message}")
            for ev in gate.evidence:
                print(f"        - {ev}")
            print()

        if report.blockers:
            print("BLOCKERS:")
            for b in report.blockers:
                print(f"  ✗ {b}")
            print()

        if report.warnings:
            print("WARNINGS:")
            for w in report.warnings:
                print(f"  ⚠ {w}")
            print()

        print(f"Recommendation: {report.recommendation}")
        print()
        print(json_output)
    else:
        print(json_output)

    # Exit code
    if report.overall_status == "BLOCKED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
