#!/usr/bin/env python3
"""Validate Stage 1 constraints completeness for AMMO runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


PLACEHOLDER_PATTERNS = (
    "",
    "<FILL_ME>",
    "pending",
    "n/a",
    "unknown",
)


def _load_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _load_bundle(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_fields(md: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for raw in md.splitlines():
        line = raw.strip()
        m = re.match(r"^- ([^:]+):\s*(.*)$", line)
        if not m:
            continue
        key = m.group(1).strip()
        value = m.group(2).strip()
        fields[key] = value
    return fields


def _is_filled(value: str) -> bool:
    if not value:
        return False
    v = value.strip().lower()
    if any(v == p for p in PLACEHOLDER_PATTERNS):
        return False
    if v.startswith("`") and v.endswith("`"):
        v = v[1:-1].strip().lower()
        if any(v == p for p in PLACEHOLDER_PATTERNS):
            return False
    return True


def _require_fields(fields: Dict[str, str], keys: List[str]) -> List[str]:
    missing: List[str] = []
    for key in keys:
        if key not in fields or not _is_filled(fields[key]):
            missing.append(key)
    return missing


def _check_sections(md: str) -> List[str]:
    required = [
        "## 0) Snapshot metadata",
        "## 1) Target envelope",
        "## 2) Production parity signature (blocking)",
        "## 3) Baseline evidence manifest",
        "## 4) Baseline truth snapshot",
        "## 5) Incumbent optimization map",
        "## 6) Candidate feasibility constraints",
        "## 7) Adapter-required appendix (blocking)",
        "## 8) Open risks and blockers",
        "## 9) Stage 1 completion record",
    ]
    return [sec for sec in required if sec not in md]


def _check_vllm_appendix(md: str) -> List[str]:
    required_snippets = [
        "vLLM version/commit",
        "CUDA graphs mode",
        "compile/runtime mode",
        "baseline",
        "nsys",
        "correctness",
    ]
    lower = md.lower()
    missing = [s for s in required_snippets if s.lower() not in lower]
    return missing


def _check_bundle(bundle: Dict[str, object]) -> List[str]:
    errs: List[str] = []
    constraints = bundle.get("constraints")
    if not isinstance(constraints, dict):
        return ["artifact_bundle.json missing constraints object"]

    status = str(constraints.get("status", ""))
    snapshot = str(constraints.get("snapshot_id", ""))
    links = constraints.get("evidence_links")
    adapter_ok = constraints.get("adapter_required_fields_complete")

    if status != "complete":
        errs.append("artifact_bundle.constraints.status must be 'complete'")
    if not _is_filled(snapshot):
        errs.append("artifact_bundle.constraints.snapshot_id missing")
    if not isinstance(links, list) or len(links) == 0:
        errs.append("artifact_bundle.constraints.evidence_links must be non-empty")
    if adapter_ok is not True:
        errs.append("artifact_bundle.constraints.adapter_required_fields_complete must be true")
    return errs


def _check(md: str, bundle: Dict[str, object], adapter: str) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if "Use `references/constraints-template.md` from the AMMO skill." in md and len(md.strip()) < 300:
        errors.append("constraints.md is still the scaffold template")

    missing_sections = _check_sections(md)
    if missing_sections:
        errors.append(f"missing sections: {missing_sections}")

    fields = _parse_fields(md)
    required_fields = [
        "Constraints snapshot ID",
        "Stage status",
        "Framework",
        "Adapter",
        "Model/workload target",
        "Hardware (GPU/driver/CUDA)",
        "Dtype / quant format",
        "Topology (TP/EP/other partitioning)",
        "Primary buckets",
        "CUDA graphs mode",
        "Compile/runtime mode",
        "Parity signature string/hash",
        "Baseline command(s)",
        "Profiling command(s)",
        "Per-bucket baseline metrics",
        "Dominant hotspot groups",
        "Launch/API vs kernel split",
    ]
    missing = _require_fields(fields, required_fields)
    if missing:
        errors.append(f"missing/placeholder fields: {missing}")

    if adapter == "vllm":
        vllm_missing = _check_vllm_appendix(md)
        if vllm_missing:
            errors.append(f"vLLM appendix hints missing: {vllm_missing}")

    errors.extend(_check_bundle(bundle))
    return (len(errors) == 0, errors)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--adapter", default="generic")
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    md = _load_text(artifact_dir / "constraints.md")
    bundle = _load_bundle(artifact_dir / "artifact_bundle.json")

    ok, errors = _check(md, bundle, args.adapter)
    if ok:
        print("constraints gate: PASS")
        raise SystemExit(0)

    print("constraints gate: FAIL")
    for err in errors:
        print(f"- {err}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
