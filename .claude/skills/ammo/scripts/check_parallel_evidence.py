#!/usr/bin/env python3
"""Validate AMMO parallel subagent evidence manifests (bundle v4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ALLOWED_GATE = {"pass", "fail", "blocked", "not_run"}
TERMINAL_STATE = {"pass", "fail", "blocked", "aborted"}
REQUIRED_MANIFEST_KEYS = {
    "version",
    "run_id",
    "candidate_id",
    "subagent_id",
    "baseline",
    "candidate",
    "command_fingerprint",
    "gate_run_ids",
    "artifact_hashes",
    "activation_proof",
    "gates",
    "artifacts",
    "notes",
}


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_manifest_path(artifact_dir: Path, raw: str) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (artifact_dir / p).resolve()


def _check_gate_triplet(owner: str, gates: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    for gate in ("correctness", "kernel", "e2e"):
        value = str(gates.get(gate, ""))
        if value not in ALLOWED_GATE:
            errors.append(f"{owner}: invalid gate value for {gate}: {value!r}")
    return errors


def _triplet_is_pass(gates: Dict[str, Any]) -> bool:
    return all(str(gates.get(k, "")) == "pass" for k in ("correctness", "kernel", "e2e"))


def _triplet_is_terminal(gates: Dict[str, Any]) -> bool:
    return all(str(gates.get(k, "")) in ALLOWED_GATE for k in ("correctness", "kernel", "e2e"))


def _non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _check_manifest(candidate_id: str, entry: Dict[str, Any], path: Path) -> Tuple[Dict[str, Any], List[str]]:
    errors: List[str] = []
    manifest = _load_json(path)
    missing = REQUIRED_MANIFEST_KEYS.difference(manifest.keys())
    if missing:
        errors.append(f"{candidate_id}: missing manifest keys {sorted(missing)}")
        return manifest, errors

    if str(manifest.get("candidate_id")) != candidate_id:
        errors.append(f"{candidate_id}: manifest candidate_id mismatch")
    if str(manifest.get("subagent_id", "")).strip() != str(entry.get("subagent_id", "")).strip():
        errors.append(f"{candidate_id}: manifest subagent_id mismatch vs bundle")

    if not _non_empty_string(manifest.get("command_fingerprint")):
        errors.append(f"{candidate_id}: command_fingerprint missing")

    gate_run_ids = manifest.get("gate_run_ids")
    if not isinstance(gate_run_ids, dict):
        errors.append(f"{candidate_id}: gate_run_ids must be object")
    else:
        for gate in ("correctness", "kernel", "e2e"):
            if not _non_empty_string(gate_run_ids.get(gate)):
                errors.append(f"{candidate_id}: gate_run_ids.{gate} missing")

    artifact_hashes = manifest.get("artifact_hashes")
    if not isinstance(artifact_hashes, dict) or len(artifact_hashes) == 0:
        errors.append(f"{candidate_id}: artifact_hashes must be non-empty object")

    gates = manifest.get("gates", {})
    if not isinstance(gates, dict):
        errors.append(f"{candidate_id}: manifest gates must be object")
        return manifest, errors
    errors.extend(_check_gate_triplet(f"{candidate_id}:manifest", gates))

    activation = manifest.get("activation_proof")
    if not isinstance(activation, dict):
        errors.append(f"{candidate_id}: activation_proof must be object")
    else:
        required_signatures = activation.get("required_signatures")
        proof_artifacts = activation.get("proof_artifacts")
        if not isinstance(required_signatures, list) or len(required_signatures) == 0:
            errors.append(f"{candidate_id}: activation_proof.required_signatures must be non-empty list")
        if not isinstance(proof_artifacts, list) or len(proof_artifacts) == 0:
            errors.append(f"{candidate_id}: activation_proof.proof_artifacts must be non-empty list")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append(f"{candidate_id}: artifacts must be object")
    else:
        for gate in ("correctness", "kernel", "e2e"):
            gate_state = str(gates.get(gate, ""))
            value = artifacts.get(gate)
            if gate_state == "pass":
                if not isinstance(value, list) or len(value) == 0:
                    errors.append(f"{candidate_id}: artifacts.{gate} must be non-empty for pass gate")

    return manifest, errors


def _check_bundle(artifact_dir: Path, bundle: Dict[str, Any]) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    infos: List[str] = []

    if str(bundle.get("version")) != "4":
        return (True, ["bundle version is not 4; parallel evidence check skipped"])

    candidates = bundle.get("parallel_candidates")
    stage = str(bundle.get("stage", ""))
    if not isinstance(candidates, list):
        errors.append("parallel_candidates must be an array for bundle v4")
        return (False, errors)
    if not candidates:
        if stage.startswith(("4", "5", "6")):
            errors.append("parallel_candidates is empty for Stage 4+ run")
            return (False, errors)
        return (True, ["parallel_candidates empty before Stage 4; check skipped"])

    for entry in candidates:
        if not isinstance(entry, dict):
            errors.append("parallel_candidates entry must be object")
            continue

        candidate_id = str(entry.get("candidate_id", "")).strip()
        if not candidate_id:
            errors.append("parallel_candidates entry missing candidate_id")
            continue

        local = entry.get("local_validation")
        acceptance = entry.get("acceptance_validation")
        if not isinstance(local, dict):
            errors.append(f"{candidate_id}: local_validation missing")
            continue
        if not isinstance(acceptance, dict):
            errors.append(f"{candidate_id}: acceptance_validation missing")
            continue

        errors.extend(_check_gate_triplet(f"{candidate_id}:bundle.local", local))
        errors.extend(_check_gate_triplet(f"{candidate_id}:bundle.acceptance", acceptance))

        manifest_ref = str(entry.get("evidence_manifest", "")).strip()
        if not manifest_ref:
            errors.append(f"{candidate_id}: evidence_manifest missing")
            continue
        manifest_path = _resolve_manifest_path(artifact_dir, manifest_ref)
        if not manifest_path.exists():
            errors.append(f"{candidate_id}: manifest path not found: {manifest_path}")
            continue

        manifest, manifest_errors = _check_manifest(candidate_id, entry, manifest_path)
        errors.extend(manifest_errors)
        if manifest_errors:
            continue

        manifest_gates = manifest.get("gates", {})
        for gate in ("correctness", "kernel", "e2e"):
            if str(manifest_gates.get(gate, "")) != str(local.get(gate, "")):
                errors.append(
                    f"{candidate_id}: manifest gate {gate} != bundle local_validation ({manifest_gates.get(gate)!r} vs {local.get(gate)!r})"
                )

        state = str(entry.get("state", "")).strip()
        if state == "running":
            errors.append(f"{candidate_id}: state cannot remain 'running' after evidence manifest is recorded")

        if state == "pass":
            if not _triplet_is_pass(local):
                errors.append(f"{candidate_id}: state=pass requires local_validation all pass")
            if not _triplet_is_pass(acceptance):
                errors.append(f"{candidate_id}: state=pass requires acceptance_validation all pass")

        if _triplet_is_pass(acceptance) and not _triplet_is_pass(local):
            errors.append(f"{candidate_id}: acceptance pass cannot occur when local_validation is not all pass")

        if _triplet_is_terminal(local) and state in {"pending", "running"}:
            infos.append(f"{candidate_id}: local gates are terminal but state={state}; verify state progression")

        if state not in TERMINAL_STATE and stage.startswith(("5", "6")):
            errors.append(f"{candidate_id}: non-terminal state {state!r} is invalid in Stage 5/6")

    if errors:
        return (False, errors)
    return (True, infos)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    bundle = _load_json(artifact_dir / "artifact_bundle.json")
    ok, messages = _check_bundle(artifact_dir, bundle)

    if ok:
        for line in messages:
            print(f"[INFO] {line}")
        print("parallel evidence gate: PASS")
        raise SystemExit(0)

    print("parallel evidence gate: FAIL")
    for err in messages:
        print(f"- {err}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
