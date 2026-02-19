#!/usr/bin/env python3
"""Validate Stage 5 validation_results.md against bundle + manifests (AMMO v4)."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple


ALLOWED_GATE = {"pass", "fail", "blocked", "not_run"}
SUMMARY_MARKER = "<!-- AMMO_VALIDATION_SUMMARY_V1 -->"


def _load_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _triplet_ok(owner: str, obj: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for gate in ("correctness", "kernel", "e2e"):
        v = str(obj.get(gate, ""))
        if v not in ALLOWED_GATE:
            errors.append(f"{owner}: invalid gate state for {gate}: {v!r}")
    return errors


def _triplet_all_pass(obj: Dict[str, Any]) -> bool:
    return all(str(obj.get(k, "")) == "pass" for k in ("correctness", "kernel", "e2e"))


def _extract_summary(md: str) -> Dict[str, Any]:
    if SUMMARY_MARKER not in md:
        raise SystemExit("validation_results.md missing AMMO_VALIDATION_SUMMARY_V1 marker")
    pattern = re.compile(r"<!--\s*AMMO_VALIDATION_SUMMARY_V1\s*-->\s*```json\s*(\{.*?\})\s*```", re.DOTALL)
    m = pattern.search(md)
    if not m:
        raise SystemExit("validation_results.md missing JSON summary block after marker")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"validation summary JSON parse error: {exc}")


def _check_candidate_summary(entry: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    cid = str(entry.get("candidate_id", "")).strip()
    if not cid:
        errors.append("summary candidate missing candidate_id")
        return errors

    local = entry.get("local")
    acceptance = entry.get("acceptance")
    if not isinstance(local, dict):
        errors.append(f"{cid}: summary.local must be object")
        return errors
    if not isinstance(acceptance, dict):
        errors.append(f"{cid}: summary.acceptance must be object")
        return errors

    errors.extend(_triplet_ok(f"{cid}:summary.local", local))
    errors.extend(_triplet_ok(f"{cid}:summary.acceptance", acceptance))

    if not isinstance(entry.get("a_b_valid"), bool):
        errors.append(f"{cid}: summary.a_b_valid must be boolean")
    if not isinstance(entry.get("activation_proof"), bool):
        errors.append(f"{cid}: summary.activation_proof must be boolean")
    if not isinstance(entry.get("acceptance_run_context"), str) or not entry.get("acceptance_run_context", "").strip():
        errors.append(f"{cid}: summary.acceptance_run_context must be non-empty string")

    evidence_paths = entry.get("evidence_paths")
    if not isinstance(evidence_paths, dict):
        errors.append(f"{cid}: summary.evidence_paths must be object")
    else:
        for gate in ("correctness", "kernel", "e2e"):
            val = evidence_paths.get(gate)
            if str(acceptance.get(gate, "")) == "pass":
                if not isinstance(val, list) or len(val) == 0:
                    errors.append(f"{cid}: evidence_paths.{gate} must be non-empty when acceptance gate is pass")

    if _triplet_all_pass(acceptance):
        if entry.get("a_b_valid") is not True:
            errors.append(f"{cid}: acceptance pass requires a_b_valid=true")
        if entry.get("activation_proof") is not True:
            errors.append(f"{cid}: acceptance pass requires activation_proof=true")

    return errors


def _check(bundle: Dict[str, Any], md: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if "Follow gate order: correctness -> kernel-time -> E2E." in md and len(md.strip().splitlines()) <= 10:
        errors.append("validation_results.md is still scaffold-only")
        return (False, errors)

    if str(bundle.get("version")) != "4":
        return (True, ["bundle version is not 4; validation-results check skipped"])

    summary = _extract_summary(md)
    candidates = summary.get("candidates")
    if not isinstance(candidates, list):
        errors.append("summary.candidates must be an array")
        return (False, errors)

    by_id: Dict[str, Dict[str, Any]] = {}
    for item in candidates:
        if not isinstance(item, dict):
            errors.append("summary candidate entry must be object")
            continue
        cid = str(item.get("candidate_id", "")).strip()
        if not cid:
            errors.append("summary candidate entry missing candidate_id")
            continue
        if cid in by_id:
            errors.append(f"duplicate candidate in summary: {cid}")
            continue
        by_id[cid] = item
        errors.extend(_check_candidate_summary(item))

    parallel = bundle.get("parallel_candidates")
    if not isinstance(parallel, list):
        errors.append("bundle.parallel_candidates must be an array")
        return (False, errors)

    for entry in parallel:
        if not isinstance(entry, dict):
            errors.append("bundle parallel candidate entry must be object")
            continue
        cid = str(entry.get("candidate_id", "")).strip()
        if not cid:
            errors.append("bundle candidate missing candidate_id")
            continue

        if cid not in by_id:
            errors.append(f"{cid}: missing in validation summary")
            continue

        local_bundle = entry.get("local_validation")
        acceptance_bundle = entry.get("acceptance_validation")
        if not isinstance(local_bundle, dict) or not isinstance(acceptance_bundle, dict):
            errors.append(f"{cid}: bundle local/acceptance validation missing")
            continue

        summary_entry = by_id[cid]
        local_summary = summary_entry.get("local")
        acceptance_summary = summary_entry.get("acceptance")

        for gate in ("correctness", "kernel", "e2e"):
            if str(local_summary.get(gate, "")) != str(local_bundle.get(gate, "")):
                errors.append(
                    f"{cid}: summary.local.{gate} mismatch bundle ({local_summary.get(gate)!r} vs {local_bundle.get(gate)!r})"
                )
            if str(acceptance_summary.get(gate, "")) != str(acceptance_bundle.get(gate, "")):
                errors.append(
                    f"{cid}: summary.acceptance.{gate} mismatch bundle ({acceptance_summary.get(gate)!r} vs {acceptance_bundle.get(gate)!r})"
                )

        if _triplet_all_pass(acceptance_summary) and not _triplet_all_pass(local_summary):
            errors.append(f"{cid}: acceptance all-pass cannot happen when local is not all-pass")

    if errors:
        return (False, errors)
    return (True, [])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    bundle = _load_json(artifact_dir / "artifact_bundle.json")
    md = _load_text(artifact_dir / "validation_results.md")

    ok, messages = _check(bundle, md)
    if ok:
        for line in messages:
            print(f"[INFO] {line}")
        print("validation results gate: PASS")
        raise SystemExit(0)

    print("validation results gate: FAIL")
    for msg in messages:
        print(f"- {msg}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
