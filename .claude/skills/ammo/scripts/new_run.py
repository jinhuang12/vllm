#!/usr/bin/env python3
"""Scaffold AMMO artifact directory with generic artifact_bundle state."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, Dict

PLACEHOLDER = "<FILL_ME>"


def _write_text(path: Path, text: str, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; use --force")
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any], force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {path}; use --force")
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_bundle_v3(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "version": "3",
        "system": {
            "gpu": args.gpu or PLACEHOLDER,
            "driver": PLACEHOLDER,
            "cuda": PLACEHOLDER,
        },
        "framework": {
            "name": args.framework,
            "adapter": args.adapter,
            "adapter_maturity": args.adapter_maturity,
            "adapter_config": {},
        },
        "workload": {
            "model_id": args.model_id or PLACEHOLDER,
            "dtype": args.dtype or PLACEHOLDER,
            "tp": args.tp,
            "ep": args.ep,
            "buckets": args.buckets,
        },
        "parity": {
            "cuda_graphs": args.cuda_graphs,
            "torch_compile": args.torch_compile,
            "notes": "",
        },
        "constraints": {
            "status": "pending",
            "snapshot_id": "",
            "evidence_links": [],
            "parity_signature": "",
            "adapter_required_fields_complete": False,
            "notes": "",
        },
        "baseline": {},
        "incumbent": {
            "variant_id": "baseline",
            "reason": "initial baseline",
            "metrics": {},
        },
        "candidates": [],
        "validation": {},
        "decision": {
            "correctness_gate": "blocked",
            "kernel_gate": "blocked",
            "e2e_gate": "blocked",
            "roi_tier": "C",
            "complexity_penalty": "",
            "ship_recommendation": "reject",
            "envelope": "",
            "rollback": "",
        },
        "stage": "1_constraints_and_baseline",
        "status": "pending",
        "last_update": date.today().isoformat(),
    }


def _artifact_bundle_v4(args: argparse.Namespace) -> Dict[str, Any]:
    bundle = _artifact_bundle_v3(args)
    bundle["version"] = "4"
    bundle["execution_mode"] = "parallel_default"
    bundle["autonomy_mode"] = args.autonomy_mode
    bundle["orchestration"] = {
        "strategy": "top3_parallel_worktrees",
        "monitor_policy": "strict_auto_interrupt",
        "last_checkpoint": "",
        "interventions": [],
    }
    bundle["parallel_candidates"] = []
    bundle["acceptance_validation"] = {
        "status": "pending",
        "summary": "",
    }
    bundle["exhaustion_state"] = {
        "cycle": 0,
        "newly_mined_count": 0,
        "eligible_remaining_count": 0,
        "terminate_reason": "pending",
    }
    bundle["validation_convergence"] = {
        "failure_signatures": [],
    }
    bundle["promotion_history"] = []
    return bundle


def _constraints_template() -> str:
    return """# Constraints

Use `references/constraints-template.md` from the AMMO skill.

This artifact is Stage 1 output and is mandatory before `optimization_plan.md`.
"""


def _plan_template() -> str:
    return """# Optimization Plan\n\nUse `references/optimization-plan-template.md` from the AMMO skill.\n"""


def _impl_template() -> str:
    return """# Implementation Notes\n\nRecord code paths changed, enablement envelope, and fallback behavior.\n"""


def _validation_template() -> str:
    return """# Validation Results

Follow gate order: correctness -> kernel-time -> E2E.

<!-- AMMO_VALIDATION_SUMMARY_V1 -->
```json
{
  "candidates": []
}
```
"""


def _decision_template() -> str:
    return """# Maintenance Decision\n\nUse `references/maintenance-decision-template.md` from the AMMO skill.\n"""


def _integration_template() -> str:
    return """# Integration\n\nRecord deployment envelope, feature gate, and rollback procedure.\n"""


def _orchestrator_log_template() -> str:
    return """# Orchestrator Log

Record subagent launches, checkpoint reviews, interventions, and acceptance reruns.
"""


def _validation_records_readme() -> str:
    return """Store per-gate validation records (local and acceptance) as JSON files.
Each record should conform to `schemas/validation_record.v1.json`.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", required=True)
    p.add_argument("--framework", default="generic_project")
    p.add_argument("--adapter", default="project_adapter")
    p.add_argument("--adapter-maturity", choices=["stable", "beta"], default="stable")
    p.add_argument("--model-id", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--dtype", default="")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--buckets", nargs="+", default=["primary_scenario", "stress_scenario"])
    p.add_argument("--cuda-graphs", default="enabled")
    p.add_argument("--torch-compile", default="disabled")
    p.add_argument("--bundle-version", choices=["3", "4"], default="4")
    p.add_argument("--autonomy-mode", choices=["single_wave", "bounded_exhaustion"], default="bounded_exhaustion")
    p.add_argument("--force", action="store_true")

    args = p.parse_args()
    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    bundle = _artifact_bundle_v4(args) if args.bundle_version == "4" else _artifact_bundle_v3(args)
    _write_json(artifact_dir / "artifact_bundle.json", bundle, args.force)
    _write_text(artifact_dir / "constraints.md", _constraints_template(), args.force)
    _write_text(artifact_dir / "optimization_plan.md", _plan_template(), args.force)
    _write_text(artifact_dir / "implementation_notes.md", _impl_template(), args.force)
    _write_text(artifact_dir / "validation_results.md", _validation_template(), args.force)
    _write_text(artifact_dir / "maintenance_decision.md", _decision_template(), args.force)
    _write_text(artifact_dir / "integration.md", _integration_template(), args.force)
    if args.bundle_version == "4":
        parallel_dir = artifact_dir / "parallel"
        evidence_dir = parallel_dir / "evidence"
        validation_records_dir = parallel_dir / "validation_records"
        worktree_dir = parallel_dir / "worktrees"
        evidence_dir.mkdir(parents=True, exist_ok=True)
        validation_records_dir.mkdir(parents=True, exist_ok=True)
        worktree_dir.mkdir(parents=True, exist_ok=True)
        _write_text(parallel_dir / "orchestrator_log.md", _orchestrator_log_template(), args.force)
        _write_text(validation_records_dir / "README.txt", _validation_records_readme(), args.force)

    print(f"[OK] AMMO run scaffolded at {artifact_dir} (bundle v{args.bundle_version})")


if __name__ == "__main__":
    main()
