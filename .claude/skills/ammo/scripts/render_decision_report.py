#!/usr/bin/env python3
"""Render maintenance_decision.md from artifact_bundle or legacy gate outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


_ALLOWED_GATES = {"pass", "fail", "waived", "blocked"}
_ALLOWED_ROI = {"S", "A", "B", "C"}
_ALLOWED_SHIP = {"ship", "ship_restricted", "reject"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    obj = _load_json(path)
    if not isinstance(obj, dict):
        raise SystemExit(f"Expected JSON object at {path}, got {type(obj).__name__}")
    return obj


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


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
    if out in _ALLOWED_GATES:
        return out
    return None


def _coerce_roi(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    out = value.strip().upper()
    if out in _ALLOWED_ROI:
        return out
    return None


def _coerce_ship(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    aliases = {
        "ship": "ship",
        "ship_restricted": "ship_restricted",
        "ship-restricted": "ship_restricted",
        "restricted": "ship_restricted",
        "reject": "reject",
        "do_not_ship": "reject",
    }
    out = aliases.get(v)
    if out in _ALLOWED_SHIP:
        return out
    return None


def _decision_defaults() -> Dict[str, Any]:
    return {
        "correctness_gate": "blocked",
        "kernel_gate": "blocked",
        "e2e_gate": "blocked",
        "roi_tier": "C",
        "complexity_penalty": "",
        "ship_recommendation": "reject",
        "envelope": "",
        "rollback": "",
    }


def _extract_decision_fragment(obj: Any) -> Dict[str, Any]:
    if not isinstance(obj, dict):
        return {}

    out: Dict[str, Any] = {}

    for key in ("correctness_gate", "kernel_gate", "e2e_gate"):
        gate = _coerce_gate(obj.get(key))
        if gate is not None:
            out[key] = gate

    roi = _coerce_roi(obj.get("roi_tier"))
    if roi is not None:
        out["roi_tier"] = roi

    ship = _coerce_ship(obj.get("ship_recommendation"))
    if ship is not None:
        out["ship_recommendation"] = ship

    if isinstance(obj.get("complexity_penalty"), str):
        out["complexity_penalty"] = obj["complexity_penalty"]
    if isinstance(obj.get("envelope"), str):
        out["envelope"] = obj["envelope"]
    if isinstance(obj.get("rollback"), str):
        out["rollback"] = obj["rollback"]

    gates = obj.get("gates")
    if isinstance(gates, dict):
        aliases = {
            "correctness": "correctness_gate",
            "kernel": "kernel_gate",
            "e2e": "e2e_gate",
        }
        for src, dst in aliases.items():
            gate = _coerce_gate(gates.get(src))
            if gate is not None:
                out[dst] = gate

    return out


def _infer_recommendation(decision: Dict[str, Any]) -> str:
    gates = [decision.get("correctness_gate"), decision.get("kernel_gate"), decision.get("e2e_gate")]
    if any(g == "fail" for g in gates):
        return "reject"
    if any(g == "blocked" for g in gates):
        return "ship_restricted"
    if decision.get("roi_tier") in ("S", "A"):
        return "ship"
    if decision.get("roi_tier") == "B":
        return "ship_restricted"
    return "reject"


def _load_decision_from_artifact_bundle(artifact_dir: Path) -> Optional[Tuple[Dict[str, Any], str, List[str]]]:
    bundle_path = artifact_dir / "artifact_bundle.json"
    if not bundle_path.exists():
        return None

    bundle = _load_json(bundle_path)
    if not isinstance(bundle, dict):
        raise SystemExit(f"Expected JSON object at {bundle_path}, got {type(bundle).__name__}")

    decision = _decision_defaults()
    fragment = _extract_decision_fragment(bundle.get("decision"))
    decision.update(fragment)
    defaults_used: List[str] = []
    if "correctness_gate" not in fragment:
        defaults_used.append("correctness_gate=blocked")
    if "kernel_gate" not in fragment:
        defaults_used.append("kernel_gate=blocked")
    if "e2e_gate" not in fragment:
        defaults_used.append("e2e_gate=blocked")
    if "roi_tier" not in fragment:
        defaults_used.append("roi_tier=C")

    return decision, "artifact_bundle.json", defaults_used


def _load_decision_from_legacy(artifact_dir: Path) -> Tuple[Dict[str, Any], str, List[str]]:
    state_path = artifact_dir / "state.json"
    validation_path = artifact_dir / "validation_summary.json"

    state = _load_optional_json(state_path)
    validation_summary = _load_optional_json(validation_path)

    if state is None and validation_summary is None:
        raise SystemExit(
            "No decision inputs found. Expected artifact_bundle.json, or legacy state.json and/or validation_summary.json"
        )

    decision = _decision_defaults()
    provided_keys = set()
    for payload in (state, validation_summary):
        if isinstance(payload, dict):
            for fragment in (
                _extract_decision_fragment(payload),
                _extract_decision_fragment(payload.get("decision")),
                _extract_decision_fragment(payload.get("validation")),
                _extract_decision_fragment(payload.get("summary")),
            ):
                provided_keys.update(fragment.keys())
                decision.update(fragment)

    defaults_used: List[str] = []
    if "correctness_gate" not in provided_keys:
        defaults_used.append("correctness_gate=blocked")
    if "kernel_gate" not in provided_keys:
        defaults_used.append("kernel_gate=blocked")
    if "e2e_gate" not in provided_keys:
        defaults_used.append("e2e_gate=blocked")
    if "roi_tier" not in provided_keys:
        defaults_used.append("roi_tier=C")

    if state is not None and isinstance(state.get("status"), str):
        status = state.get("status", "")
        if status in {"blocked", "needs_investigation"} and decision.get("kernel_gate") == "blocked":
            # Keep conservative default; this line exists to make fallback intent explicit.
            pass

    source = "legacy(state.json/validation_summary.json)"
    return decision, source, defaults_used


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()

    loaded = _load_decision_from_artifact_bundle(artifact_dir)
    if loaded is None:
        decision, source, defaults_used = _load_decision_from_legacy(artifact_dir)
    else:
        decision, source, defaults_used = loaded

    recommendation = decision.get("ship_recommendation")
    if recommendation not in _ALLOWED_SHIP:
        recommendation = _infer_recommendation(decision)

    md = f"""# Maintenance Decision

## Data source

- Source: {source}
- Defaults (fallback-safe): {', '.join(defaults_used) if defaults_used else 'none'}

## Gate outcomes

- Correctness: {decision.get('correctness_gate', 'blocked')}
- Kernel-time: {decision.get('kernel_gate', 'blocked')}
- E2E: {decision.get('e2e_gate', 'blocked')}

## ROI and complexity

- ROI tier: {decision.get('roi_tier', 'C')}
- Complexity penalty: {decision.get('complexity_penalty', '')}

## Recommendation

- {recommendation}
- Envelope: {decision.get('envelope', '')}
- Rollback: {decision.get('rollback', '')}
"""

    _write_text(artifact_dir / "maintenance_decision.md", md)
    print(f"[OK] Wrote {artifact_dir / 'maintenance_decision.md'}")


if __name__ == "__main__":
    main()
