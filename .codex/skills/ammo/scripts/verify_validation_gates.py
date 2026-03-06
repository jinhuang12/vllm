#!/usr/bin/env python3
"""Verify AMMO Stage 5 validation gates.

This is a blocking gate. Any FAIL or WARN result prevents advancement to Stage 6.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PLACEHOLDER_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [r"\bTODO\b", r"<FILL_ME>", r"\bTBD\b", r"placeholder"]
]
VLLM_BASELINE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"\bvllm\b",
        r"production kernel",
        r"fused_moe",
        r"fused_experts",
        r"paged_attention",
        r"flash_attn",
    ]
]
NUMERICAL_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"torch\.allclose",
        r"torch\.testing\.assert_close",
        r"np\.allclose",
        r"atol\s*=",
        r"rtol\s*=",
    ]
]
BAD_PARITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"TORCH_COMPILE_DISABLE\s*=\s*[\"']?1",
        r"--enforce-eager",
        r"enforce_eager\s*=\s*True",
        r"VLLM_TORCH_COMPILE_LEVEL\s*=\s*[\"']?0",
    ]
]
GOOD_PARITY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in [
        r"VLLM_TORCH_COMPILE_LEVEL\s*=\s*[\"']?3",
        r"cuda graph",
        r"cudagraph",
        r"make_graphed_callables",
        r"torch\.cuda\.CUDAGraph",
        r"torch\.compile",
    ]
]
SECTION_HINTS = {
    "gate_5_1": ["gate 5.1", "correctness"],
    "gate_5_2": ["gate 5.2", "kernel perf", "kernel performance"],
    "gate_5_3": ["gate 5.3", "e2e latency", "end-to-end"],
}
TERMINAL_TRACK_STATUSES = {"PASS", "PASSED", "FAIL", "FAILED", "REGRESSED", "INVALID"}


@dataclass
class GateResult:
    name: str
    status: str
    message: str
    evidence: List[str] = field(default_factory=list)
    track_id: Optional[str] = None


@dataclass
class TrackContext:
    track_id: str
    metadata: Dict[str, Any]
    validation_path: Path
    validation_text: str


@dataclass
class VerificationReport:
    artifact_dir: str
    phase: str = "5_validation"
    overall_status: str = "UNKNOWN"
    gates: List[GateResult] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "phase": self.phase,
            "overall_status": self.overall_status,
            "gates": [asdict(gate) for gate in self.gates],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_state(artifact_dir: Path) -> Dict[str, Any]:
    state_path = artifact_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json: {state_path}")
    return json.loads(state_path.read_text(encoding="utf-8"))


def _normalize_path(path_value: str, artifact_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (artifact_dir / candidate).resolve()


def _resolve_tracks(artifact_dir: Path, state: Dict[str, Any], track_id: Optional[str]) -> List[TrackContext]:
    tracks = state.get("parallel_tracks") or {}
    resolved: List[TrackContext] = []

    if tracks:
        selected_ids: Iterable[str]
        if track_id:
            if track_id not in tracks:
                raise KeyError(f"Track '{track_id}' not found in state.json")
            selected_ids = [track_id]
        else:
            selected_ids = tracks.keys()

        for selected_id in selected_ids:
            metadata = tracks[selected_id]
            validation_value = metadata.get("validation_results_path") or f"tracks/{selected_id}/validation_results.md"
            validation_path = _normalize_path(str(validation_value), artifact_dir)
            validation_text = _read_text(validation_path) if validation_path.exists() else ""
            resolved.append(
                TrackContext(
                    track_id=selected_id,
                    metadata=metadata,
                    validation_path=validation_path,
                    validation_text=validation_text,
                )
            )
        return resolved

    validation_path = artifact_dir / "validation_results.md"
    validation_text = _read_text(validation_path) if validation_path.exists() else ""
    return [TrackContext(track_id=track_id or "root", metadata={}, validation_path=validation_path, validation_text=validation_text)]


def _contains_any(text: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _find_metric_near_keywords(text: str, keywords: List[str], metric_pattern: str) -> Optional[re.Match[str]]:
    lower_text = text.lower()
    for keyword in keywords:
        index = lower_text.find(keyword)
        if index >= 0:
            window = text[index:index + 1600]
            match = re.search(metric_pattern, window, re.IGNORECASE)
            if match:
                return match
    return None


def _section_present(text: str, hints: List[str]) -> bool:
    lower_text = text.lower()
    return any(hint in lower_text for hint in hints)


def _has_numeric_evidence(text: str, hints: List[str]) -> bool:
    match = _find_metric_near_keywords(text, hints, r"([+-]?\d+(?:\.\d+)?)\s*(?:%|x|us|µs|ms|s)")
    return match is not None


def _parse_component_share(texts: Iterable[str]) -> Optional[float]:
    for text in texts:
        patterns = [
            (r"component share[^\n]*?([0-9]+(?:\.[0-9]+)?)\s*%", True),
            (r"component share[^\n]*?f\s*=\s*([0-9]+(?:\.[0-9]+)?)", False),
            (r"\bf\s*=\s*([0-9]+(?:\.[0-9]+)?)", False),
        ]
        for pattern, is_percent in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                return value / 100.0 if is_percent or value > 1.0 else value
    return None


def _parse_speedup_factor(text: str) -> Optional[float]:
    patterns = [
        r"weighted average[^\n]*?([0-9]+(?:\.[0-9]+)?)x",
        r"kernel speedup[^\n]*?([0-9]+(?:\.[0-9]+)?)x",
        r"speedup[^\n]*?([0-9]+(?:\.[0-9]+)?)x",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _parse_e2e_improvement_pct(text: str) -> Optional[float]:
    percent_patterns = [
        r"overall e2e (?:improvement|gain)[^\n]*?([+-]?\d+(?:\.\d+)?)\s*%",
        r"actual e2e (?:improvement|gain)[^\n]*?([+-]?\d+(?:\.\d+)?)\s*%",
        r"e2e (?:improvement|gain)[^\n]*?([+-]?\d+(?:\.\d+)?)\s*%",
        r"improvement[^\n]*?([+-]?\d+(?:\.\d+)?)\s*%",
    ]
    for pattern in percent_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return float(match.group(1))

    factor_patterns = [
        r"speedup_vs_baseline[^\n]*?([0-9]+(?:\.[0-9]+)?)",
        r"e2e speedup[^\n]*?([0-9]+(?:\.[0-9]+)?)x",
    ]
    for pattern in factor_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            factor = float(match.group(1))
            return (factor - 1.0) * 100.0
    return None


def _branch_matches_worktree(metadata: Dict[str, Any]) -> Optional[str]:
    worktree_path = metadata.get("worktree_path")
    expected_branch = metadata.get("branch")
    if not worktree_path or not expected_branch:
        return None
    path = Path(worktree_path)
    if not path.is_absolute():
        path = path.resolve()
    if not path.exists():
        return None
    try:
        output = subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output if output == expected_branch else f"expected {expected_branch}, found {output}"


def check_track_state(track: TrackContext) -> GateResult:
    metadata = track.metadata
    evidence: List[str] = []
    status = str(metadata.get("status", "")).upper()

    if track.track_id != "root":
        if status not in TERMINAL_TRACK_STATUSES:
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Track status is missing or not terminal in state.json",
                evidence=[f"track={track.track_id}", f"status={metadata.get('status')!r}"],
                track_id=track.track_id,
            )
        evidence.append(f"terminal status: {status}")

        for field_name in ["branch", "worktree_path", "validation_results_path"]:
            if not metadata.get(field_name):
                return GateResult(
                    name="track_state_recorded",
                    status="FAIL",
                    message=f"Track metadata missing required field '{field_name}'",
                    evidence=[f"track={track.track_id}"],
                    track_id=track.track_id,
                )
            evidence.append(f"{field_name}: {metadata.get(field_name)}")

        if "correctness" not in metadata:
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Track metadata missing 'correctness' field",
                evidence=[f"track={track.track_id}"],
                track_id=track.track_id,
            )
        evidence.append(f"correctness: {metadata.get('correctness')}")

        branch_check = _branch_matches_worktree(metadata)
        if branch_check and branch_check.startswith("expected "):
            return GateResult(
                name="track_state_recorded",
                status="FAIL",
                message="Worktree branch does not match recorded branch",
                evidence=[branch_check],
                track_id=track.track_id,
            )
        if branch_check:
            evidence.append(f"worktree branch matches: {branch_check}")

    return GateResult(
        name="track_state_recorded",
        status="PASS",
        message="Track metadata is present and terminal",
        evidence=evidence,
        track_id=track.track_id,
    )


def check_validation_completeness(track: TrackContext) -> GateResult:
    if not track.validation_path.exists():
        return GateResult(
            name="validation_completeness",
            status="FAIL",
            message="validation_results.md is missing",
            evidence=[str(track.validation_path)],
            track_id=track.track_id,
        )

    text = track.validation_text
    missing_sections = [
        section_name
        for section_name, hints in SECTION_HINTS.items()
        if not _section_present(text, hints)
    ]
    if missing_sections:
        return GateResult(
            name="validation_completeness",
            status="FAIL",
            message="validation_results.md is missing one or more required sections",
            evidence=missing_sections,
            track_id=track.track_id,
        )

    placeholder_hits = [pattern.pattern for pattern in PLACEHOLDER_PATTERNS if pattern.search(text)]
    if placeholder_hits:
        return GateResult(
            name="validation_completeness",
            status="FAIL",
            message="validation_results.md still contains placeholders or TODO markers",
            evidence=placeholder_hits,
            track_id=track.track_id,
        )

    sections_without_numbers = [
        section_name
        for section_name, hints in SECTION_HINTS.items()
        if not _has_numeric_evidence(text, hints)
    ]
    if sections_without_numbers:
        return GateResult(
            name="validation_completeness",
            status="FAIL",
            message="Required validation sections lack numeric evidence",
            evidence=sections_without_numbers,
            track_id=track.track_id,
        )

    return GateResult(
        name="validation_completeness",
        status="PASS",
        message="validation_results.md contains Gate 5.1, 5.2, and 5.3 evidence",
        evidence=[str(track.validation_path)],
        track_id=track.track_id,
    )


def check_stage1_baseline(track: TrackContext, artifact_dir: Path) -> GateResult:
    baseline_files = sorted((artifact_dir / "runs").glob("baseline_bs*.json"))
    text = track.validation_text
    evidence = [str(path.relative_to(artifact_dir)) for path in baseline_files]

    if not baseline_files:
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="Stage 1 baseline JSON files are missing from artifact_dir/runs",
            evidence=evidence or ["artifact_dir/runs/baseline_bs*.json not found"],
            track_id=track.track_id,
        )

    if not re.search(r"baseline source\s*:\s*stage\s*1\s*\(not re-run\)", text, re.IGNORECASE):
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="validation_results.md does not cite the Stage 1 baseline reuse requirement",
            evidence=evidence + ["Expected citation: Baseline source: Stage 1 (not re-run)"],
            track_id=track.track_id,
        )

    forbidden_worktree_baseline = re.search(
        r"vllm bench latency[^\n]*(baseline|--output-json[^\n]*baseline)",
        text,
        re.IGNORECASE,
    )
    if forbidden_worktree_baseline and "not re-run" not in forbidden_worktree_baseline.group(0).lower():
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="validation_results.md appears to include a worktree baseline run",
            evidence=[forbidden_worktree_baseline.group(0)],
            track_id=track.track_id,
        )

    return GateResult(
        name="stage1_baseline_reuse",
        status="PASS",
        message="Track cites and reuses the Stage 1 baseline",
        evidence=evidence,
        track_id=track.track_id,
    )


def check_vllm_baseline(track: TrackContext) -> GateResult:
    text = track.validation_text
    if not _contains_any(text, VLLM_BASELINE_PATTERNS):
        return GateResult(
            name="baseline_is_vllm",
            status="FAIL",
            message="validation evidence does not mention a vLLM production kernel baseline",
            evidence=["Expected references such as vLLM, fused_moe, fused_experts, paged_attention, or production kernel"],
            track_id=track.track_id,
        )

    if re.search(r"naive|manual per-expert|pytorch baseline", text, re.IGNORECASE):
        return GateResult(
            name="baseline_is_vllm",
            status="FAIL",
            message="validation evidence suggests a naive baseline rather than the vLLM production path",
            evidence=["Remove naive PyTorch baseline references from correctness and performance sections"],
            track_id=track.track_id,
        )

    return GateResult(
        name="baseline_is_vllm",
        status="PASS",
        message="validation evidence references a vLLM production kernel baseline",
        evidence=["vLLM production baseline documented"],
        track_id=track.track_id,
    )


def check_numerical_comparison(track: TrackContext) -> GateResult:
    text = track.validation_text
    if not _contains_any(text, NUMERICAL_PATTERNS):
        return GateResult(
            name="numerical_comparison",
            status="FAIL",
            message="validation evidence does not mention torch.allclose, assert_close, or explicit tolerances",
            evidence=["Document numerical comparison methodology and tolerances in Gate 5.1"],
            track_id=track.track_id,
        )

    return GateResult(
        name="numerical_comparison",
        status="PASS",
        message="validation evidence includes numerical comparison details",
        evidence=["Numerical comparison markers found in validation_results.md"],
        track_id=track.track_id,
    )


def check_production_parity(track: TrackContext) -> GateResult:
    text = track.validation_text
    bad_hits = [pattern.pattern for pattern in BAD_PARITY_PATTERNS if pattern.search(text)]
    if bad_hits:
        return GateResult(
            name="production_parity",
            status="FAIL",
            message="validation evidence contains production-disabling settings",
            evidence=bad_hits,
            track_id=track.track_id,
        )

    good_hits = [pattern.pattern for pattern in GOOD_PARITY_PATTERNS if pattern.search(text)]
    if not good_hits:
        return GateResult(
            name="production_parity",
            status="FAIL",
            message="validation evidence does not prove production-parity settings",
            evidence=["Document VLLM_TORCH_COMPILE_LEVEL=3 and CUDA graph usage in validation_results.md"],
            track_id=track.track_id,
        )

    return GateResult(
        name="production_parity",
        status="PASS",
        message="validation evidence documents production-parity settings",
        evidence=good_hits,
        track_id=track.track_id,
    )


def check_kill_criteria(track: TrackContext, state: Dict[str, Any]) -> GateResult:
    track_results = track.metadata.get("kill_criteria_results")
    global_results = state.get("route_decision", {}).get("kill_criteria_results") or {}
    results = track_results or global_results

    if results:
        incomplete = []
        evidence = []
        for key, value in results.items():
            text = str(value).upper()
            evidence.append(f"{key}: {value}")
            if any(marker in text for marker in ["TODO", "OPTIONAL", "SKIP", "PENDING"]):
                incomplete.append(f"{key}: {value}")
        if incomplete:
            return GateResult(
                name="kill_criteria_complete",
                status="FAIL",
                message="One or more kill criteria are not finalized",
                evidence=incomplete,
                track_id=track.track_id,
            )
        return GateResult(
            name="kill_criteria_complete",
            status="PASS",
            message="Kill criteria are finalized in state.json",
            evidence=evidence,
            track_id=track.track_id,
        )

    text = track.validation_text
    if "kill criteria" not in text.lower():
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message="Kill criteria are missing from both state.json and validation_results.md",
            evidence=["Document every kill criterion with PASS/FAIL verdicts"],
            track_id=track.track_id,
        )
    if re.search(r"TODO|OPTIONAL|SKIP|PENDING", text, re.IGNORECASE):
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message="validation_results.md contains incomplete kill criteria markers",
            evidence=["Replace placeholder kill criteria verdicts with PASS or FAIL"],
            track_id=track.track_id,
        )

    pass_fail_lines = re.findall(r"^[\-\*]\s+.*?(PASS|FAIL)", text, re.IGNORECASE | re.MULTILINE)
    if not pass_fail_lines:
        return GateResult(
            name="kill_criteria_complete",
            status="FAIL",
            message="validation_results.md names kill criteria but does not record definitive PASS/FAIL outcomes",
            evidence=["Record PASS/FAIL for each criterion"],
            track_id=track.track_id,
        )

    return GateResult(
        name="kill_criteria_complete",
        status="PASS",
        message="Kill criteria are documented with definitive verdicts",
        evidence=[f"criteria_with_verdicts={len(pass_fail_lines)}"],
        track_id=track.track_id,
    )


def check_amdahl_sanity(track: TrackContext, artifact_dir: Path) -> GateResult:
    constraints_text = _read_text(artifact_dir / "constraints.md") if (artifact_dir / "constraints.md").exists() else ""
    share = _parse_component_share([track.validation_text, constraints_text])
    speedup = _parse_speedup_factor(track.validation_text)
    improvement = _parse_e2e_improvement_pct(track.validation_text)

    if share is None or speedup is None or improvement is None:
        missing = []
        if share is None:
            missing.append("component share f")
        if speedup is None:
            missing.append("kernel speedup s")
        if improvement is None:
            missing.append("actual E2E improvement")
        return GateResult(
            name="amdahl_sanity",
            status="FAIL",
            message="Amdahl sanity data is incomplete",
            evidence=missing,
            track_id=track.track_id,
        )

    expected_pct = share * (1.0 - (1.0 / speedup)) * 100.0
    ratio = abs(improvement) / max(abs(expected_pct), 0.1)
    evidence = [
        f"component_share_f={share:.4f}",
        f"kernel_speedup={speedup:.4f}x",
        f"expected_e2e_pct={expected_pct:.2f}",
        f"actual_e2e_pct={improvement:.2f}",
        f"ratio={ratio:.2f}",
    ]

    if improvement > expected_pct * 1.5 + 0.5:
        return GateResult(
            name="amdahl_sanity",
            status="FAIL",
            message="Actual E2E gain is implausibly larger than the Amdahl expectation",
            evidence=evidence,
            track_id=track.track_id,
        )

    if speedup > 1.10 and improvement < max(expected_pct * 0.5, 1.0):
        return GateResult(
            name="amdahl_sanity",
            status="FAIL",
            message="Kernel speedup does not translate into expected E2E improvement",
            evidence=evidence,
            track_id=track.track_id,
        )

    return GateResult(
        name="amdahl_sanity",
        status="PASS",
        message="Amdahl sanity check is within expected bounds",
        evidence=evidence,
        track_id=track.track_id,
    )


def check_kernel_e2e_coherence(track: TrackContext) -> GateResult:
    speedup = _parse_speedup_factor(track.validation_text)
    improvement = _parse_e2e_improvement_pct(track.validation_text)
    if speedup is None or improvement is None:
        return GateResult(
            name="kernel_e2e_coherence",
            status="FAIL",
            message="Cannot evaluate kernel-to-E2E coherence without explicit kernel speedup and E2E improvement metrics",
            evidence=["Document weighted kernel speedup and overall E2E improvement"],
            track_id=track.track_id,
        )

    if speedup > 1.10 and improvement < 1.0:
        return GateResult(
            name="kernel_e2e_coherence",
            status="FAIL",
            message="Kernel speedup is meaningful but E2E improvement is within noise",
            evidence=[f"kernel_speedup={speedup:.4f}x", f"e2e_improvement={improvement:.2f}%"],
            track_id=track.track_id,
        )

    return GateResult(
        name="kernel_e2e_coherence",
        status="PASS",
        message="Kernel and E2E results are directionally coherent",
        evidence=[f"kernel_speedup={speedup:.4f}x", f"e2e_improvement={improvement:.2f}%"],
        track_id=track.track_id,
    )


def check_cross_track_contamination(track: TrackContext, all_tracks: List[TrackContext]) -> GateResult:
    if len(all_tracks) <= 1:
        return GateResult(
            name="cross_track_contamination",
            status="PASS",
            message="Single-track run; cross-track contamination is not applicable",
            evidence=["N/A"],
            track_id=track.track_id,
        )

    text = track.validation_text
    has_note = re.search(r"cross[- ]track contamination", text, re.IGNORECASE) is not None
    has_artifact_note = re.search(r"\.so|shared object|provenance", text, re.IGNORECASE) is not None

    sibling_csrc = False
    current_csrc = False
    current_files = track.metadata.get("files_changed") or []
    current_csrc = any(str(path).startswith("csrc/") for path in current_files)
    for sibling in all_tracks:
        if sibling.track_id == track.track_id:
            continue
        sibling_files = sibling.metadata.get("files_changed") or []
        if any(str(path).startswith("csrc/") for path in sibling_files):
            sibling_csrc = True
            break

    if sibling_csrc and not current_csrc and not (has_note and has_artifact_note):
        return GateResult(
            name="cross_track_contamination",
            status="FAIL",
            message="Track needs an explicit contamination audit because another track changed csrc/",
            evidence=["Add a cross-track contamination note with .so provenance evidence"],
            track_id=track.track_id,
        )

    if not has_note:
        return GateResult(
            name="cross_track_contamination",
            status="FAIL",
            message="validation_results.md is missing the required cross-track contamination note",
            evidence=["Document PASS, N/A, or a red flag in validation_results.md"],
            track_id=track.track_id,
        )

    return GateResult(
        name="cross_track_contamination",
        status="PASS",
        message="Cross-track contamination note is present",
        evidence=["Contamination audit documented"],
        track_id=track.track_id,
    )


def verify_validation(artifact_dir: Path, track_id: Optional[str]) -> VerificationReport:
    state = _load_state(artifact_dir)
    tracks = _resolve_tracks(artifact_dir, state, track_id)
    report = VerificationReport(artifact_dir=str(artifact_dir))

    gates: List[GateResult] = []
    for track in tracks:
        gates.extend(
            [
                check_track_state(track),
                check_validation_completeness(track),
                check_stage1_baseline(track, artifact_dir),
                check_vllm_baseline(track),
                check_numerical_comparison(track),
                check_production_parity(track),
                check_kill_criteria(track, state),
                check_amdahl_sanity(track, artifact_dir),
                check_kernel_e2e_coherence(track),
                check_cross_track_contamination(track, tracks),
            ]
        )

    report.gates = gates
    for gate in gates:
        label = f"{gate.track_id}:{gate.name}" if gate.track_id else gate.name
        if gate.status == "FAIL":
            report.blockers.append(f"{label}: {gate.message}")
        elif gate.status == "WARN":
            report.warnings.append(f"{label}: {gate.message}")

    if report.blockers:
        report.overall_status = "BLOCKED"
        report.recommendation = "Validation is blocked. Fix every failing gate before Stage 6."
    elif report.warnings:
        report.overall_status = "WARN"
        report.recommendation = "Validation warnings are blocking. Resolve them before Stage 6."
    else:
        report.overall_status = "PASS"
        report.recommendation = "Validation gates passed. Stage 6 may proceed."

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_dir", help="Artifact directory path")
    parser.add_argument("--json-output", default=None, help="Optional JSON report output path")
    parser.add_argument("--quiet", action="store_true", help="Only emit JSON")
    parser.add_argument("--track", default=None, help="Validate only one track id")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    if not artifact_dir.exists():
        print(f"ERROR: artifact directory does not exist: {artifact_dir}", file=sys.stderr)
        return 2

    try:
        report = verify_validation(artifact_dir, args.track)
    except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    output = json.dumps(report.to_dict(), indent=2)
    if args.json_output:
        Path(args.json_output).write_text(output + "\n", encoding="utf-8")

    if args.quiet:
        print(output)
    else:
        print("=" * 60)
        print("AMMO Validation Gate Report")
        print("=" * 60)
        print(f"Artifact dir: {artifact_dir}")
        print(f"Overall status: {report.overall_status}")
        print()
        for gate in report.gates:
            prefix = f"[{gate.status}]"
            track_label = f" track={gate.track_id}" if gate.track_id else ""
            print(f"{prefix} {gate.name}{track_label}")
            print(f"  {gate.message}")
            for item in gate.evidence:
                print(f"  - {item}")
            print()
        print(output)

    return 0 if report.overall_status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
