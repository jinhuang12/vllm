#!/usr/bin/env python3
"""Validate Stage 2 optimization plan completeness for AMMO runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Tuple


def _load_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def _has_scaffold_only(md: str) -> bool:
    marker = "Use `references/optimization-plan-template.md` from the AMMO skill."
    return marker in md and len(md.strip().splitlines()) <= 5


def _load_bundle(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parallel_required(bundle: dict) -> bool:
    version = str(bundle.get("version", ""))
    if version in {"2", "3"}:
        return False
    mode = str(bundle.get("execution_mode", "parallel_default"))
    return mode != "serial"


def _autonomy_bounded(bundle: dict) -> bool:
    return str(bundle.get("autonomy_mode", "")) == "bounded_exhaustion"


def _check_substantive(md: str, parallel_required: bool, bounded_exhaustion: bool) -> List[str]:
    required = [
        "## 0) Constraints citations",
        "## 1) Context and envelope",
        "## 3) Ranked opportunity backlog",
        "## 4) Selected hypotheses",
        "## 5) Dependency map and nullification risk",
        "## 7) Measurement protocol",
        "## 8) Enablement envelope and rollback",
    ]
    if parallel_required:
        required.extend(
            [
                "### 4.2 Parallel execution slate",
                "### 7.5 Subagent evidence package contract",
            ]
        )
    if bounded_exhaustion:
        required.append("### 4.3 Autonomy continuation policy")
    missing = [h for h in required if h not in md]

    # Require enough hypothesis IDs for selected mode, unless explicitly blocked.
    hypothesis_ids = set(re.findall(r"\bOP-\d{3}\b", md))
    min_ids = 3 if parallel_required else 1
    has_hypothesis = len(hypothesis_ids) >= min_ids
    has_blocked = "Phase 2 blocked decision" in md or "status=blocked" in md
    reduced_allowed = (
        "fewer than 3" in md.lower() or "less than 3" in md.lower()
    ) and has_blocked
    if not has_hypothesis and not has_blocked:
        missing.append(f"at least {min_ids} hypothesis ID(s) (OP-xxx) or explicit blocker decision")
    if parallel_required and not has_hypothesis and has_blocked and not reduced_allowed:
        missing.append("explicit reduced-candidate rationale (e.g. fewer than 3 with blocker evidence)")
    return missing


def _check(md: str, bundle: dict) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    if _has_scaffold_only(md):
        errors.append("optimization_plan.md is still scaffold-only")
        return (False, errors)

    missing = _check_substantive(md, _parallel_required(bundle), _autonomy_bounded(bundle))
    if missing:
        errors.append(f"missing sections/content: {missing}")
    return (len(errors) == 0, errors)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    md = _load_text(artifact_dir / "optimization_plan.md")
    bundle = _load_bundle(artifact_dir / "artifact_bundle.json")
    ok, errors = _check(md, bundle)
    if ok:
        print("optimization plan gate: PASS")
        raise SystemExit(0)

    print("optimization plan gate: FAIL")
    for err in errors:
        print(f"- {err}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
