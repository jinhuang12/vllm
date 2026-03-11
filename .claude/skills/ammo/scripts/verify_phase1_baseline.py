#!/usr/bin/env python3
"""ammo: Verify Phase 1 baseline requirements are met.

This script performs BLOCKING verification for Phase 1 completion.
Phase 2 MUST NOT proceed if this script returns non-zero exit code.

Checks:
1. nsys profile files exist in {artifact_dir}/runs/
2. constraints.md contains "Baseline Truth Snapshot" section
3. constraints.md documents baseline kernel timings (not just commands)

Exit codes:
  0 = PASS (all gates met, safe to proceed to Phase 2)
  1 = BLOCKER (missing requirements, do NOT proceed)
  2 = ERROR (script execution error)

Output: JSON report to stdout with detailed gate status.

Usage:
  python verify_phase1_baseline.py /path/to/artifact_dir
  python verify_phase1_baseline.py /path/to/artifact_dir --json-output /path/to/report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


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
    phase: str = "1_constraints"
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


def check_nsys_profiles(artifact_dir: Path) -> GateResult:
    """Gate 1.1: Check that nsys profile files exist.

    Searches multiple directories for backward compatibility:
      - {artifact_dir}/runs/
      - {artifact_dir}/nsys/
      - {artifact_dir}/e2e_latency/nsys/
      - {artifact_dir}/ (root)
    """
    search_dirs = []
    for subdir_name in ("runs", "nsys", "e2e_latency/nsys"):
        subdir = artifact_dir / subdir_name
        if subdir.exists():
            search_dirs.append(subdir)
    # Always include artifact root as fallback
    search_dirs.append(artifact_dir)

    # Collect nsys files from all search directories, deduplicating by resolved path
    seen: set = set()
    nsys_files: List[Path] = []
    for d in search_dirs:
        for f in list(d.glob("*.nsys-rep")) + list(d.glob("*.sqlite")):
            resolved = f.resolve()
            if resolved not in seen:
                seen.add(resolved)
                nsys_files.append(f)

    if not nsys_files:
        searched_list = ", ".join(
            str(d) for d in [artifact_dir / "runs", artifact_dir / "nsys", artifact_dir / "e2e_latency/nsys", artifact_dir]
        )
        return GateResult(
            name="nsys_profiles_exist",
            status="FAIL",
            message="No nsys profile files found",
            evidence=[
                f"Searched directories: {searched_list}",
                "Expected patterns: *.nsys-rep, *.sqlite",
                "Run: python scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir} --nsys-profile",
            ],
        )

    return GateResult(
        name="nsys_profiles_exist",
        status="PASS",
        message=f"Found {len(nsys_files)} nsys profile file(s)",
        evidence=[str(f.name) for f in nsys_files[:5]],  # Show first 5
    )


def check_baseline_snapshot_section(artifact_dir: Path) -> GateResult:
    """Gate 1.2: Check constraints.md has Baseline Truth Snapshot section."""
    constraints_file = artifact_dir / "constraints.md"

    if not constraints_file.exists():
        return GateResult(
            name="baseline_snapshot_section",
            status="FAIL",
            message="constraints.md does not exist",
            evidence=[f"Expected: {constraints_file}"],
        )

    content = constraints_file.read_text(encoding="utf-8")

    # Check for baseline snapshot section (various formats)
    # Allow optional section letter prefix (e.g., "## E: Baseline Truth Snapshot")
    patterns = [
        r"##\s*(?:[A-Z]:\s*)?Baseline\s+Truth\s+Snapshot",
        r"##\s*(?:[A-Z]:\s*)?Baseline\s+Snapshot",
        r"##\s*(?:[A-Z]:\s*)?Production\s+Baseline",
        r"baseline_time|baseline_latency|kernel_time",
    ]

    for pattern in patterns:
        if re.search(pattern, content, re.IGNORECASE):
            return GateResult(
                name="baseline_snapshot_section",
                status="PASS",
                message="Found baseline documentation in constraints.md",
                evidence=[f"Matched pattern: {pattern}"],
            )

    return GateResult(
        name="baseline_snapshot_section",
        status="FAIL",
        message="constraints.md missing 'Baseline Truth Snapshot' section",
        evidence=[
            "Required: Document baseline kernel timings from nsys profiling",
            "Expected section: '## Baseline Truth Snapshot' with timing data",
        ],
    )


def check_baseline_kernel_data(artifact_dir: Path) -> GateResult:
    """Gate 1.3: Check that actual kernel timing data is documented (not just commands)."""
    constraints_file = artifact_dir / "constraints.md"

    if not constraints_file.exists():
        return GateResult(
            name="baseline_kernel_data",
            status="FAIL",
            message="constraints.md does not exist",
            evidence=[],
        )

    content = constraints_file.read_text(encoding="utf-8")

    # Look for actual timing data (numbers with units)
    timing_patterns = [
        r"\d+\.?\d*\s*(µs|us|ms|ns)",  # e.g., "123.45 µs"
        r"\d+\.?\d*\s*(microsec|millisec)",
        r"fused_moe|fused_experts|flash_attn|paged_attention|flashinfer|sampling|topk|softmax|triton|CUTLASS",  # vLLM kernel names
        r"triton|CUTLASS",  # Backend names
    ]

    evidence = []
    has_timing_data = False
    has_kernel_names = False

    for pattern in timing_patterns[:2]:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            has_timing_data = True
            evidence.append(f"Found timing data: {matches[:3]}")

    for pattern in timing_patterns[2:]:
        if re.search(pattern, content, re.IGNORECASE):
            has_kernel_names = True
            evidence.append(f"Found kernel reference: {pattern}")

    if has_timing_data and has_kernel_names:
        return GateResult(
            name="baseline_kernel_data",
            status="PASS",
            message="Found baseline kernel timing data",
            evidence=evidence,
        )
    elif has_timing_data:
        return GateResult(
            name="baseline_kernel_data",
            status="WARN",
            message="Found timing data but no vLLM kernel references (target component kernels)",
            evidence=evidence + ["WARN: Ensure baseline uses vLLM production kernels for the target component"],
        )
    else:
        return GateResult(
            name="baseline_kernel_data",
            status="FAIL",
            message="No baseline kernel timing data found in constraints.md",
            evidence=[
                "Required: Actual timing measurements from nsys profiling",
                "Expected: Numbers with units (e.g., '523.4 µs') + kernel names",
                "Commands alone are NOT sufficient - run them and document results",
            ],
        )


def check_production_parity_env(artifact_dir: Path) -> GateResult:
    """Gate 1.4: Check that production parity environment is documented."""
    constraints_file = artifact_dir / "constraints.md"

    if not constraints_file.exists():
        return GateResult(
            name="production_parity_env",
            status="FAIL",
            message="constraints.md does not exist",
            evidence=[],
        )

    content = constraints_file.read_text(encoding="utf-8")

    # Check for production parity indicators
    parity_indicators = {
        "cuda_graphs": r"cuda.?graph|cudaGraph|capture",
        "torch_compile": r"torch\.compile|TORCH_COMPILE|compile_level",
        "vllm_v1": r"VLLM_USE_V1|v1\s+engine",
    }

    found = {}
    for key, pattern in parity_indicators.items():
        if re.search(pattern, content, re.IGNORECASE):
            found[key] = True

    if len(found) >= 2:
        return GateResult(
            name="production_parity_env",
            status="PASS",
            message="Production parity environment documented",
            evidence=[f"Found: {list(found.keys())}"],
        )
    elif len(found) >= 1:
        return GateResult(
            name="production_parity_env",
            status="WARN",
            message="Partial production parity documentation",
            evidence=[
                f"Found: {list(found.keys())}",
                "Recommended: Document CUDA graphs, torch.compile, VLLM_USE_V1 settings",
            ],
        )
    else:
        return GateResult(
            name="production_parity_env",
            status="FAIL",
            message="No production parity environment documented",
            evidence=[
                "Required: Document environment for baseline profiling",
                "Must include: CUDA graphs status, torch.compile level, V1 engine",
            ],
        )


def verify_phase1(artifact_dir: Path) -> VerificationReport:
    """Run all Phase 1 verification gates."""
    report = VerificationReport(artifact_dir=str(artifact_dir))

    # Run all gates
    gates = [
        check_nsys_profiles(artifact_dir),
        check_baseline_snapshot_section(artifact_dir),
        check_baseline_kernel_data(artifact_dir),
        check_production_parity_env(artifact_dir),
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
            "Phase 1 INCOMPLETE. Do NOT proceed to Phase 2. "
            "Fix all blockers listed above. "
            "Run nsys profiling and document baseline kernel timings."
        )
    elif report.warnings:
        report.overall_status = "WARN"
        report.recommendation = (
            "Phase 1 conditionally complete. Review warnings before Phase 2. "
            "Ensure baseline uses vLLM production kernels for the target component."
        )
    else:
        report.overall_status = "PASS"
        report.recommendation = "Phase 1 COMPLETE. Safe to proceed to Phase 2."

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "artifact_dir",
        type=str,
        help="Path to the artifact directory (e.g., kernel_opt_artifacts/component_model_hw_dtype_tp/)",
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

    args = parser.parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    if not artifact_dir.exists():
        print(f"ERROR: Artifact directory does not exist: {artifact_dir}", file=sys.stderr)
        return 2

    # Run verification
    report = verify_phase1(artifact_dir)

    # Output JSON
    json_output = json.dumps(report.to_dict(), indent=2)

    if args.json_output:
        Path(args.json_output).write_text(json_output, encoding="utf-8")

    if not args.quiet:
        print("=" * 60)
        print("Phase 1 Baseline Verification Report")
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
