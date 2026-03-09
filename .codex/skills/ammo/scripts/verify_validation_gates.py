#!/usr/bin/env python3
"""Verify AMMO Stage 5 validation gates from structured evidence."""

from __future__ import annotations

import argparse
import json
import math
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
TERMINAL_TRACK_STATUSES = {"PASS", "PASSED", "FAIL", "FAILED", "REGRESSED", "INVALID", "BLOCKED"}
PASS_FAIL = {"PASS", "FAIL"}


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
    evidence_path: Path
    evidence: Optional[Dict[str, Any]]


@dataclass
class VerificationReport:
    artifact_dir: str
    phase: str = "5_validation"
    overall_status: str = "UNKNOWN"
    advance_to_stage6: bool = False
    track_outcomes: Dict[str, str] = field(default_factory=dict)
    gates: List[GateResult] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "artifact_dir": self.artifact_dir,
            "phase": self.phase,
            "overall_status": self.overall_status,
            "advance_to_stage6": self.advance_to_stage6,
            "track_outcomes": self.track_outcomes,
            "gates": [asdict(gate) for gate in self.gates],
            "blockers": self.blockers,
            "warnings": self.warnings,
            "recommendation": self.recommendation,
        }


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state(artifact_dir: Path) -> Dict[str, Any]:
    state_path = artifact_dir / "state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"Missing state.json: {state_path}")
    return _read_json(state_path)


def _normalize_path(path_value: str, artifact_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (artifact_dir / candidate).resolve()


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
            evidence_value = metadata.get("evidence_path") or f"tracks/{selected_id}/evidence.json"
            evidence_path = _normalize_path(str(evidence_value), artifact_dir)
            resolved.append(
                TrackContext(
                    track_id=selected_id,
                    metadata=metadata,
                    validation_path=validation_path,
                    validation_text=_read_text(validation_path) if validation_path.exists() else "",
                    evidence_path=evidence_path,
                    evidence=_read_json(evidence_path) if evidence_path.exists() else None,
                )
            )
        return resolved

    validation_path = artifact_dir / "validation_results.md"
    evidence_path = artifact_dir / "evidence.json"
    return [
        TrackContext(
            track_id=track_id or "root",
            metadata={},
            validation_path=validation_path,
            validation_text=_read_text(validation_path) if validation_path.exists() else "",
            evidence_path=evidence_path,
            evidence=_read_json(evidence_path) if evidence_path.exists() else None,
        )
    ]


def _expect_dict(obj: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    value = obj.get(key)
    return value if isinstance(value, dict) else None


def _expect_number(obj: Dict[str, Any], key: str) -> Optional[float]:
    value = obj.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _approx_equal(a: float, b: float, tol: float = 0.25) -> bool:
    return math.isclose(a, b, abs_tol=tol, rel_tol=0.01)


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


def check_structured_evidence(track: TrackContext) -> GateResult:
    if track.evidence is None:
        return GateResult(
            name="structured_evidence",
            status="FAIL",
            message="Structured evidence JSON is missing",
            evidence=[str(track.evidence_path)],
            track_id=track.track_id,
        )

    version = track.evidence.get("schema_version")
    if version != 2:
        return GateResult(
            name="structured_evidence",
            status="FAIL",
            message="Structured evidence must declare schema_version=2",
            evidence=[f"schema_version={version!r}"],
            track_id=track.track_id,
        )

    evidence_track = track.evidence.get("track_id")
    if evidence_track and evidence_track != track.track_id:
        return GateResult(
            name="structured_evidence",
            status="FAIL",
            message="Structured evidence track_id does not match the track being validated",
            evidence=[f"evidence.track_id={evidence_track!r}", f"track_id={track.track_id!r}"],
            track_id=track.track_id,
        )

    return GateResult(
        name="structured_evidence",
        status="PASS",
        message="Structured evidence JSON is present",
        evidence=[str(track.evidence_path)],
        track_id=track.track_id,
    )


def check_validation_summary(track: TrackContext) -> GateResult:
    if not track.validation_path.exists():
        return GateResult(
            name="validation_summary",
            status="FAIL",
            message="validation_results.md summary is missing",
            evidence=[str(track.validation_path)],
            track_id=track.track_id,
        )

    placeholder_hits = [pattern.pattern for pattern in PLACEHOLDER_PATTERNS if pattern.search(track.validation_text)]
    if placeholder_hits:
        return GateResult(
            name="validation_summary",
            status="FAIL",
            message="validation_results.md still contains placeholders or TODO markers",
            evidence=placeholder_hits,
            track_id=track.track_id,
        )

    return GateResult(
        name="validation_summary",
        status="PASS",
        message="validation_results.md summary is present",
        evidence=[str(track.validation_path)],
        track_id=track.track_id,
    )


def check_stage1_baseline(track: TrackContext, artifact_dir: Path) -> GateResult:
    baseline_files = sorted((artifact_dir / "runs").glob("baseline_bs*.json"))
    evidence = track.evidence or {}
    baseline = _expect_dict(evidence, "baseline_source")
    if baseline is None:
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="Structured evidence is missing baseline_source",
            evidence=[str(track.evidence_path)],
            track_id=track.track_id,
        )

    kind = str(baseline.get("kind", "")).lower()
    citation = str(baseline.get("citation", ""))
    if kind != "stage1" or "not re-run" not in citation.lower():
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="baseline_source must cite Stage 1 reuse explicitly",
            evidence=[f"kind={kind!r}", f"citation={citation!r}"],
            track_id=track.track_id,
        )

    if not baseline_files:
        return GateResult(
            name="stage1_baseline_reuse",
            status="FAIL",
            message="Stage 1 baseline JSON files are missing from artifact_dir/runs",
            evidence=["artifact_dir/runs/baseline_bs*.json not found"],
            track_id=track.track_id,
        )

    return GateResult(
        name="stage1_baseline_reuse",
        status="PASS",
        message="Structured evidence cites Stage 1 baseline reuse",
        evidence=[str(path.relative_to(artifact_dir)) for path in baseline_files],
        track_id=track.track_id,
    )


def check_correctness(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    correctness = _expect_dict(evidence, "correctness")
    if correctness is None:
        return GateResult(
            name="correctness_evidence",
            status="FAIL",
            message="Structured evidence is missing correctness",
            evidence=[],
            track_id=track.track_id,
        )

    status = str(correctness.get("status", "")).upper()
    method = str(correctness.get("method", ""))
    atol = _expect_number(correctness, "atol")
    rtol = _expect_number(correctness, "rtol")
    max_abs_diff = _expect_number(correctness, "max_abs_diff")
    nan_inf_check = correctness.get("nan_inf_check")
    graph_replay_check = correctness.get("graph_replay_check")

    if status not in PASS_FAIL:
        return GateResult("correctness_evidence", "FAIL", "correctness.status must be PASS or FAIL", [f"status={status!r}"], track.track_id)
    if "allclose" not in method and "assert_close" not in method:
        return GateResult("correctness_evidence", "FAIL", "correctness.method must name torch.allclose or assert_close", [method], track.track_id)
    if atol is None or rtol is None or max_abs_diff is None:
        return GateResult("correctness_evidence", "FAIL", "correctness must include numeric tolerances and max_abs_diff", [], track.track_id)
    if nan_inf_check is not True or graph_replay_check is not True:
        return GateResult("correctness_evidence", "FAIL", "correctness must explicitly pass NaN/INF and graph replay checks", [f"nan_inf_check={nan_inf_check!r}", f"graph_replay_check={graph_replay_check!r}"], track.track_id)

    return GateResult(
        name="correctness_evidence",
        status="PASS",
        message="Structured correctness evidence is complete",
        evidence=[f"method={method}", f"atol={atol}", f"rtol={rtol}", f"max_abs_diff={max_abs_diff}"],
        track_id=track.track_id,
    )


def check_kernel_bench(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    kernel = _expect_dict(evidence, "kernel_bench")
    if kernel is None:
        return GateResult("kernel_bench", "FAIL", "Structured evidence is missing kernel_bench", [], track.track_id)

    status = str(kernel.get("status", "")).upper()
    weighted_speedup = _expect_number(kernel, "weighted_speedup")
    measured_under_cuda_graphs = kernel.get("measured_under_cuda_graphs")
    buckets = kernel.get("buckets")

    if status not in PASS_FAIL:
        return GateResult("kernel_bench", "FAIL", "kernel_bench.status must be PASS or FAIL", [f"status={status!r}"], track.track_id)
    if weighted_speedup is None:
        return GateResult("kernel_bench", "FAIL", "kernel_bench must include weighted_speedup", [], track.track_id)
    if measured_under_cuda_graphs is not True:
        return GateResult("kernel_bench", "FAIL", "kernel_bench must prove CUDA graph measurement", [f"measured_under_cuda_graphs={measured_under_cuda_graphs!r}"], track.track_id)
    if not isinstance(buckets, list) or not buckets:
        return GateResult("kernel_bench", "FAIL", "kernel_bench must include at least one validated bucket", [], track.track_id)

    return GateResult(
        name="kernel_bench",
        status="PASS",
        message="Structured kernel benchmark evidence is complete",
        evidence=[f"weighted_speedup={weighted_speedup:.4f}x", f"bucket_count={len(buckets)}"],
        track_id=track.track_id,
    )


def check_e2e_validation(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    e2e = _expect_dict(evidence, "e2e")
    if e2e is None:
        return GateResult("e2e_validation", "FAIL", "Structured evidence is missing e2e", [], track.track_id)

    status = str(e2e.get("status", "")).upper()
    run_purpose = str(e2e.get("run_purpose", "")).lower()
    baseline_avg_s = _expect_number(e2e, "baseline_avg_s")
    optimized_avg_s = _expect_number(e2e, "optimized_avg_s")
    speedup = _expect_number(e2e, "speedup")
    improvement_pct = _expect_number(e2e, "improvement_pct")
    admissibility = _expect_dict(e2e, "admissibility") or {}
    fastpath_proof = _expect_dict(e2e, "fastpath_proof") or {}

    if status not in PASS_FAIL:
        return GateResult("e2e_validation", "FAIL", "e2e.status must be PASS or FAIL", [f"status={status!r}"], track.track_id)
    if run_purpose != "official":
        return GateResult("e2e_validation", "FAIL", "Only official runs may determine Stage 5 E2E verdicts", [f"run_purpose={run_purpose!r}"], track.track_id)
    if None in (baseline_avg_s, optimized_avg_s, speedup, improvement_pct):
        return GateResult("e2e_validation", "FAIL", "e2e must include numeric baseline, optimized, speedup, and improvement metrics", [], track.track_id)
    if str(admissibility.get("status", "")).upper() != "PASS":
        return GateResult("e2e_validation", "FAIL", "Official E2E run is not admissible", [json.dumps(admissibility, sort_keys=True)], track.track_id)

    proof_status = str(fastpath_proof.get("status", "")).upper()
    hits = fastpath_proof.get("hits")
    if proof_status != "PASS":
        return GateResult("e2e_validation", "FAIL", "Official optimized run is missing explicit fast-path proof", [json.dumps(fastpath_proof, sort_keys=True)], track.track_id)
    if not isinstance(hits, int) or hits < 1:
        return GateResult("e2e_validation", "FAIL", "Explicit fast-path proof must include hits >= 1", [json.dumps(fastpath_proof, sort_keys=True)], track.track_id)

    source_json = fastpath_proof.get("source_json")
    if isinstance(source_json, str) and source_json:
        source_path = Path(source_json)
        if not source_path.is_absolute():
            source_path = (track.evidence_path.parent / source_path).resolve()
        if not source_path.exists():
            return GateResult("e2e_validation", "FAIL", "fastpath_proof.source_json does not exist", [str(source_path)], track.track_id)

    return GateResult(
        name="e2e_validation",
        status="PASS",
        message="Structured E2E evidence is complete and admissible",
        evidence=[f"speedup={speedup:.4f}x", f"improvement_pct={improvement_pct:.3f}", f"fastpath_hits={hits}"],
        track_id=track.track_id,
    )


def check_kill_criteria(track: TrackContext, state: Dict[str, Any]) -> GateResult:
    evidence = track.evidence or {}
    kill_criteria = evidence.get("kill_criteria")
    if not isinstance(kill_criteria, dict) or not kill_criteria:
        return GateResult("kill_criteria", "FAIL", "Structured evidence is missing kill_criteria", [], track.track_id)

    structured_statuses: Dict[str, str] = {}
    evidence_lines: List[str] = []
    for name, result in kill_criteria.items():
        if not isinstance(result, dict):
            return GateResult("kill_criteria", "FAIL", "Each kill criterion must be an object", [f"{name}={result!r}"], track.track_id)
        status = str(result.get("status", "")).upper()
        source_run_purpose = str(result.get("source_run_purpose", "")).lower()
        promoted = bool(result.get("promoted", False))
        if status not in PASS_FAIL:
            return GateResult("kill_criteria", "FAIL", "Each kill criterion must have PASS or FAIL status", [f"{name}.status={status!r}"], track.track_id)
        if source_run_purpose != "official" and not promoted:
            return GateResult("kill_criteria", "FAIL", "Kill criteria must come from an official run unless explicitly promoted", [f"{name}.source_run_purpose={source_run_purpose!r}", f"{name}.promoted={promoted!r}"], track.track_id)
        structured_statuses[name] = status
        evidence_lines.append(f"{name}: {status}")

    state_results = track.metadata.get("kill_criteria_results") or state.get("route_decision", {}).get("kill_criteria_results") or {}
    if isinstance(state_results, dict) and state_results:
        mismatches = []
        for name, status in structured_statuses.items():
            state_status = str(state_results.get(name, "")).upper()
            if state_status and state_status != status:
                mismatches.append(f"{name}: evidence={status}, state={state_status}")
        if mismatches:
            return GateResult("kill_criteria", "FAIL", "Structured evidence and state.json disagree on kill criteria", mismatches, track.track_id)

    return GateResult("kill_criteria", "PASS", "Kill criteria are structured and consistent", evidence_lines, track.track_id)


def check_amdahl_sanity(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    amdahl = _expect_dict(evidence, "amdahl")
    e2e = _expect_dict(evidence, "e2e") or {}
    if amdahl is None:
        return GateResult("amdahl_sanity", "FAIL", "Structured evidence is missing amdahl", [], track.track_id)

    share = _expect_number(amdahl, "component_share_f")
    kernel_speedup = _expect_number(amdahl, "kernel_speedup")
    expected_pct = _expect_number(amdahl, "expected_e2e_pct")
    actual_pct = _expect_number(amdahl, "actual_e2e_pct")
    e2e_improvement = _expect_number(e2e, "improvement_pct")
    if None in (share, kernel_speedup, expected_pct, actual_pct, e2e_improvement):
        return GateResult("amdahl_sanity", "FAIL", "amdahl must include numeric component share, kernel speedup, expected, and actual values", [], track.track_id)

    expected_calc = share * (1.0 - (1.0 / kernel_speedup)) * 100.0
    if not _approx_equal(expected_pct, expected_calc):
        return GateResult("amdahl_sanity", "FAIL", "amdahl.expected_e2e_pct does not match the structured inputs", [f"expected_e2e_pct={expected_pct:.3f}", f"recomputed={expected_calc:.3f}"], track.track_id)
    if not _approx_equal(actual_pct, e2e_improvement):
        return GateResult("amdahl_sanity", "FAIL", "amdahl.actual_e2e_pct must match e2e.improvement_pct", [f"amdahl.actual_e2e_pct={actual_pct:.3f}", f"e2e.improvement_pct={e2e_improvement:.3f}"], track.track_id)

    return GateResult(
        name="amdahl_sanity",
        status="PASS",
        message="Amdahl sanity inputs are structured and internally consistent",
        evidence=[f"component_share_f={share:.4f}", f"kernel_speedup={kernel_speedup:.4f}x", f"expected_e2e_pct={expected_pct:.3f}", f"actual_e2e_pct={actual_pct:.3f}"],
        track_id=track.track_id,
    )


def check_cross_track_contamination(track: TrackContext, all_tracks: List[TrackContext]) -> GateResult:
    evidence = track.evidence or {}
    contamination = _expect_dict(evidence, "cross_track_contamination")
    if contamination is None:
        return GateResult("cross_track_contamination", "FAIL", "Structured evidence is missing cross_track_contamination", [], track.track_id)

    status = str(contamination.get("status", "")).upper()
    note = str(contamination.get("note", "")).strip()
    if status not in {"PASS", "FAIL", "N/A"}:
        return GateResult("cross_track_contamination", "FAIL", "cross_track_contamination.status must be PASS, FAIL, or N/A", [f"status={status!r}"], track.track_id)
    if len(all_tracks) > 1 and status == "N/A":
        return GateResult("cross_track_contamination", "FAIL", "cross_track_contamination cannot be N/A in multi-track validation", [], track.track_id)
    if not note:
        return GateResult("cross_track_contamination", "FAIL", "cross_track_contamination.note is required", [], track.track_id)
    return GateResult("cross_track_contamination", "PASS", "Cross-track contamination audit is structured", [f"status={status}", note], track.track_id)


def check_candidate_outcome(track: TrackContext) -> GateResult:
    evidence = track.evidence or {}
    e2e = _expect_dict(evidence, "e2e") or {}
    kill_criteria = evidence.get("kill_criteria") or {}
    track_status = str(track.metadata.get("status", "")).upper()
    e2e_status = str(e2e.get("status", "")).upper()

    if track_status in {"PASS", "PASSED"} and e2e_status != "PASS":
        return GateResult("candidate_outcome", "FAIL", "Track is marked passed in state.json but structured E2E verdict is not PASS", [f"track_status={track_status}", f"e2e.status={e2e_status}"], track.track_id)

    failed_criteria = [
        name
        for name, result in kill_criteria.items()
        if isinstance(result, dict) and str(result.get("status", "")).upper() == "FAIL"
    ]
    if track_status in {"PASS", "PASSED"} and failed_criteria:
        return GateResult("candidate_outcome", "FAIL", "Track is marked passed but one or more kill criteria failed", failed_criteria, track.track_id)

    return GateResult("candidate_outcome", "PASS", "Track outcome is consistent with structured evidence", [f"track_status={track_status or 'UNKNOWN'}"], track.track_id)


def verify_validation(artifact_dir: Path, track_id: Optional[str]) -> VerificationReport:
    state = _load_state(artifact_dir)
    tracks = _resolve_tracks(artifact_dir, state, track_id)
    report = VerificationReport(artifact_dir=str(artifact_dir))

    gates: List[GateResult] = []
    for track in tracks:
        report.track_outcomes[track.track_id] = str(track.metadata.get("status", "UNKNOWN")).upper() or "UNKNOWN"
        gates.extend(
            [
                check_track_state(track),
                check_structured_evidence(track),
                check_validation_summary(track),
                check_stage1_baseline(track, artifact_dir),
                check_correctness(track),
                check_kernel_bench(track),
                check_e2e_validation(track),
                check_kill_criteria(track, state),
                check_amdahl_sanity(track),
                check_cross_track_contamination(track, tracks),
                check_candidate_outcome(track),
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
        report.advance_to_stage6 = False
        report.recommendation = "Validation evidence is incomplete or inconsistent. Fix every failing gate before Stage 6."
    elif report.warnings:
        report.overall_status = "WARN"
        report.advance_to_stage6 = False
        report.recommendation = "Validation warnings are blocking. Resolve them before Stage 6."
    else:
        report.overall_status = "PASS"
        report.advance_to_stage6 = True
        if any(outcome in {"PASS", "PASSED"} for outcome in report.track_outcomes.values()):
            report.recommendation = "Validation evidence is complete. Stage 6 may proceed with the passing track set."
        else:
            report.recommendation = "Validation evidence is complete. Stage 6 may proceed, but no track passed Stage 5."

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
        print(f"Advance to Stage 6: {report.advance_to_stage6}")
        print("Track outcomes:")
        for track_name, outcome in sorted(report.track_outcomes.items()):
            print(f"  - {track_name}: {outcome}")
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
