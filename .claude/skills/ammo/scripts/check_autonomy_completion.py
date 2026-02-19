#!/usr/bin/env python3
"""Validate AMMO autonomous completion for bounded-exhaustion runs (bundle v4)."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


TERMINAL_STATE = {"pass", "fail", "blocked", "aborted"}
TERMINATE_REASON = {
    "pending",
    "no_new_above_threshold",
    "all_remaining_blocked",
    "all_remaining_non_improving",
    "manual_stop",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _run_checker(script: Path, artifact_dir: Path) -> Tuple[bool, str]:
    proc = subprocess.run(
        ["python3", str(script), "--artifact-dir", str(artifact_dir)],
        capture_output=True,
        text=True,
        check=False,
    )
    output = (proc.stdout + proc.stderr).strip()
    return (proc.returncode == 0, output)


def _check(bundle: Dict[str, Any], artifact_dir: Path) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    infos: List[str] = []

    if str(bundle.get("version")) != "4":
        return (True, ["bundle version is not 4; autonomy completion check skipped"])

    if str(bundle.get("autonomy_mode", "")) != "bounded_exhaustion":
        infos.append("autonomy_mode is not bounded_exhaustion; completion check treated as informational")
        return (True, infos)

    exhaustion = bundle.get("exhaustion_state")
    if not isinstance(exhaustion, dict):
        errors.append("exhaustion_state missing")
        return (False, errors)

    terminate_reason = str(exhaustion.get("terminate_reason", ""))
    if terminate_reason not in TERMINATE_REASON:
        errors.append(f"invalid terminate_reason: {terminate_reason!r}")
    if terminate_reason == "pending":
        errors.append("terminate_reason is pending for bounded_exhaustion run")

    eligible_remaining = exhaustion.get("eligible_remaining_count")
    if not isinstance(eligible_remaining, int) or eligible_remaining < 0:
        errors.append("eligible_remaining_count must be non-negative integer")
    else:
        if terminate_reason != "manual_stop" and eligible_remaining != 0:
            errors.append(
                f"eligible_remaining_count must be 0 for terminate_reason={terminate_reason!r}"
            )

    newly_mined = exhaustion.get("newly_mined_count")
    if not isinstance(newly_mined, int) or newly_mined < 0:
        errors.append("newly_mined_count must be non-negative integer")
    elif terminate_reason == "no_new_above_threshold" and newly_mined != 0:
        errors.append("terminate_reason=no_new_above_threshold requires newly_mined_count=0")

    parallel = bundle.get("parallel_candidates")
    if not isinstance(parallel, list) or len(parallel) == 0:
        errors.append("parallel_candidates must be non-empty for bounded_exhaustion runs")
    else:
        for entry in parallel:
            if not isinstance(entry, dict):
                errors.append("parallel candidate entry must be object")
                continue
            cid = str(entry.get("candidate_id", "")).strip() or "<unknown>"
            state = str(entry.get("state", ""))
            if state not in TERMINAL_STATE:
                errors.append(f"{cid}: non-terminal state at completion: {state!r}")

            local = entry.get("local_validation")
            acceptance = entry.get("acceptance_validation")
            if state == "pass":
                if not isinstance(local, dict) or any(str(local.get(k, "")) != "pass" for k in ("correctness", "kernel", "e2e")):
                    errors.append(f"{cid}: state=pass requires local all-pass")
                if not isinstance(acceptance, dict) or any(str(acceptance.get(k, "")) != "pass" for k in ("correctness", "kernel", "e2e")):
                    errors.append(f"{cid}: state=pass requires acceptance all-pass")

    history = bundle.get("promotion_history")
    if not isinstance(history, list):
        errors.append("promotion_history must be an array")
    else:
        expected_step = 1
        for item in history:
            if not isinstance(item, dict):
                errors.append("promotion_history entry must be object")
                continue
            step = item.get("step")
            if step != expected_step:
                errors.append(f"promotion_history step mismatch: expected {expected_step}, got {step!r}")
                expected_step = step if isinstance(step, int) else expected_step
            expected_step += 1
            if not isinstance(item.get("acceptance_gate_hash"), str) or not item.get("acceptance_gate_hash", "").strip():
                errors.append("promotion_history entry missing acceptance_gate_hash")

    script_dir = Path(__file__).resolve().parent
    ok_parallel, out_parallel = _run_checker(script_dir / "check_parallel_evidence.py", artifact_dir)
    if not ok_parallel:
        errors.append("check_parallel_evidence.py failed")
        if out_parallel:
            errors.append(out_parallel)

    ok_validation, out_validation = _run_checker(script_dir / "check_validation_results.py", artifact_dir)
    if not ok_validation:
        errors.append("check_validation_results.py failed")
        if out_validation:
            errors.append(out_validation)

    if errors:
        return (False, errors)
    return (True, infos)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    bundle = _load_json(artifact_dir / "artifact_bundle.json")

    ok, messages = _check(bundle, artifact_dir)
    if ok:
        for line in messages:
            print(f"[INFO] {line}")
        print("autonomy completion gate: PASS")
        raise SystemExit(0)

    print("autonomy completion gate: FAIL")
    for msg in messages:
        print(f"- {msg}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
