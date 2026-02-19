#!/usr/bin/env python3
"""Translate legacy optimization state into AMMO artifact_bundle.json.

Inputs:
- required: state.json
- optional: validation_summary.json

This translator is intentionally generic and does not assume runtime-specific
schema beyond common fields.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional


PLACEHOLDER = "<FILL_ME>"
ALLOWED_GATES = {"pass", "fail", "waived", "blocked"}
ALLOWED_ROI = {"S", "A", "B", "C"}
ALLOWED_SHIP = {"ship", "ship_restricted", "reject"}


def _load_json(path: Path, required: bool) -> Optional[Dict[str, Any]]:
    if not path.exists():
        if required:
            raise SystemExit(f"Missing required file: {path}")
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse JSON {path}: {exc}")
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected JSON object at {path}, got {type(obj).__name__}")
    return obj


def _write_json(path: Path, obj: Dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {path} (use --overwrite)")
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _coerce_gate(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "pass" if value else "fail"
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    aliases = {
        "pass": "pass",
        "passed": "pass",
        "ok": "pass",
        "success": "pass",
        "fail": "fail",
        "failed": "fail",
        "error": "fail",
        "waived": "waived",
        "skip": "waived",
        "skipped": "waived",
        "blocked": "blocked",
        "pending": "blocked",
        "unknown": "blocked",
    }
    out = aliases.get(v)
    return out if out in ALLOWED_GATES else None


def _coerce_roi(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip().upper() in ALLOWED_ROI:
        return value.strip().upper()
    return None


def _coerce_ship(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    aliases = {
        "ship": "ship",
        "ship_restricted": "ship_restricted",
        "ship-restricted": "ship_restricted",
        "restricted": "ship_restricted",
        "reject": "reject",
        "do_not_ship": "reject",
    }
    out = aliases.get(value.strip().lower())
    return out if out in ALLOWED_SHIP else None


def _extract_decision(src: Dict[str, Any], validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    decision = {
        "correctness_gate": "blocked",
        "kernel_gate": "blocked",
        "e2e_gate": "blocked",
        "roi_tier": "C",
        "complexity_penalty": "",
        "ship_recommendation": "reject",
        "envelope": "",
        "rollback": "",
    }

    fragments = [src, src.get("decision"), src.get("validation")]
    if validation:
        fragments.extend([validation, validation.get("summary"), validation.get("decision")])

    for frag in fragments:
        if not isinstance(frag, dict):
            continue
        for key in ("correctness_gate", "kernel_gate", "e2e_gate"):
            gate = _coerce_gate(frag.get(key))
            if gate is not None:
                decision[key] = gate
        roi = _coerce_roi(frag.get("roi_tier"))
        if roi:
            decision["roi_tier"] = roi
        ship = _coerce_ship(frag.get("ship_recommendation"))
        if ship:
            decision["ship_recommendation"] = ship
        for k in ("complexity_penalty", "envelope", "rollback"):
            if isinstance(frag.get(k), str):
                decision[k] = frag[k]

    return decision


def _build_bundle(state: Dict[str, Any], validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    target = state.get("target", {}) if isinstance(state.get("target"), dict) else {}
    incumbent = state.get("incumbent", {}) if isinstance(state.get("incumbent"), dict) else {}

    framework_name = target.get("framework") if isinstance(target.get("framework"), str) else "legacy_project"
    adapter_name = target.get("adapter") if isinstance(target.get("adapter"), str) else "legacy_adapter"

    workload_model = target.get("model_id") if isinstance(target.get("model_id"), str) else PLACEHOLDER
    workload_dtype = target.get("dtype") if isinstance(target.get("dtype"), str) else PLACEHOLDER

    bucket_hint = state.get("buckets") if isinstance(state.get("buckets"), list) else ["legacy_scenario"]
    buckets = [str(x) for x in bucket_hint] if bucket_hint else ["legacy_scenario"]

    return {
        "version": "3",
        "system": {
            "gpu": target.get("hardware") if isinstance(target.get("hardware"), str) else PLACEHOLDER,
            "driver": PLACEHOLDER,
            "cuda": PLACEHOLDER,
        },
        "framework": {
            "name": framework_name,
            "adapter": adapter_name,
            "adapter_maturity": "beta",
            "adapter_config": {
                "translated_from": "legacy_state",
            },
        },
        "workload": {
            "model_id": workload_model,
            "dtype": workload_dtype,
            "tp": int(target.get("tp", 1)) if isinstance(target.get("tp", 1), int) else 1,
            "ep": int(target.get("ep", 1)) if isinstance(target.get("ep", 1), int) else 1,
            "buckets": buckets,
        },
        "parity": {
            "cuda_graphs": "unknown",
            "torch_compile": "unknown",
            "notes": "translated from legacy state",
        },
        "constraints": {
            "status": "blocked",
            "snapshot_id": "",
            "evidence_links": [],
            "parity_signature": "",
            "adapter_required_fields_complete": False,
            "notes": "legacy translation: reconstruct constraints.md before Stage 3 planning",
        },
        "baseline": {},
        "incumbent": {
            "variant_id": incumbent.get("variant_id", "baseline") if isinstance(incumbent, dict) else "baseline",
            "reason": incumbent.get("reason", "translated from legacy state") if isinstance(incumbent, dict) else "translated from legacy state",
            "metrics": incumbent.get("metrics", {}) if isinstance(incumbent, dict) else {},
        },
        "candidates": [],
        "validation": {},
        "decision": _extract_decision(state, validation),
        "stage": state.get("phase", "legacy_phase") if isinstance(state.get("phase"), str) else "legacy_phase",
        "status": state.get("status", "pending") if isinstance(state.get("status"), str) else "pending",
        "last_update": state.get("last_update", date.today().isoformat()),
        "migration": {
            "source": "state.json",
            "warnings": [
                "review translated parity knobs and workload buckets before using this bundle for decisioning"
            ],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--state-json", default=None)
    p.add_argument("--validation-summary", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    state_path = Path(args.state_json).expanduser().resolve() if args.state_json else artifact_dir / "state.json"
    validation_path = (
        Path(args.validation_summary).expanduser().resolve()
        if args.validation_summary
        else artifact_dir / "validation_summary.json"
    )
    out_path = Path(args.out).expanduser().resolve() if args.out else artifact_dir / "artifact_bundle.json"

    state = _load_json(state_path, required=True)
    validation = _load_json(validation_path, required=False)

    bundle = _build_bundle(state, validation)
    _write_json(out_path, bundle, overwrite=args.overwrite)

    print(f"[OK] Wrote {out_path}")


if __name__ == "__main__":
    main()
