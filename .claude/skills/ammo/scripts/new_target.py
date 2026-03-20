#!/usr/bin/env python3
"""ammo v2: scaffold a new target artifact directory.

Creates:
  - constraints.md
  - state.json (simplified v2 schema)
  - target.json (input for run_vllm_bench_latency_sweep.py)

Safety: refuses to overwrite existing files unless --force is provided.

Example:
  python .claude/skills/ammo/scripts/new_target.py \\
    --artifact-dir kernel_opt_artifacts/auto_qwen3_l40s_fp8_tp1 \\
    --model-id Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \\
    --hardware L40S --dtype fp8 --tp 1 --ep 1
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


PLACEHOLDER = "<FILL_ME>"


@dataclass(frozen=True)
class TargetFields:
    model_id: str
    hardware: str
    dtype: str
    tp: int
    ep: int
    max_model_len: int
    input_len: int
    output_len: int
    batch_sizes: List[int]
    num_iters: int
    noise_tolerance_pct: float
    catastrophic_regression_pct: float


def _write_text(path: Path, text: str, *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Dict[str, Any], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path} (use --force)")
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _default_target_fields(args: argparse.Namespace) -> TargetFields:
    model_id = args.model_id or PLACEHOLDER
    hardware = args.hardware or PLACEHOLDER
    dtype = args.dtype or PLACEHOLDER

    return TargetFields(
        model_id=model_id,
        hardware=hardware,
        dtype=dtype,
        tp=args.tp,
        ep=args.ep,
        max_model_len=args.max_model_len,
        input_len=args.input_len,
        output_len=args.output_len,
        batch_sizes=args.batch_sizes,
        num_iters=args.num_iters,
        noise_tolerance_pct=args.noise_tolerance_pct,
        catastrophic_regression_pct=args.catastrophic_regression_pct,
    )


def _constraints_md(fields: TargetFields) -> str:
    return f"""# Constraints (Stage 1)

## Target envelope

- Model: {fields.model_id}
- Hardware: {fields.hardware}
- Dtype / quant format: {fields.dtype}
- TP / EP: tp={fields.tp}, ep={fields.ep}
- Max model len: {fields.max_model_len}
- Decode buckets (batch sizes): {fields.batch_sizes}
- E2E workload: input_len={fields.input_len}, output_len={fields.output_len}

## TODOs (required before Stage 2)

- [ ] Read vLLM source for target component, document forward path and correctness invariants
- [ ] Capture baseline truth snapshot under production parity (CUDA graphs / torch.compile)
- [ ] Record baseline kernel timings from nsys profiling

"""

def _state_json(fields: TargetFields, artifact_dir: Path, diminishing_threshold: int = 3,
                enable_delegation: bool = True, delegates_per_champion: int = 1,
                noise_tolerance_pct: float = 0.5,
                catastrophic_regression_pct: float = 5.0) -> Dict[str, Any]:
    return {
        "target": {
            "model_id": fields.model_id,
            "hardware": fields.hardware,
            "dtype": fields.dtype,
            "tp": fields.tp,
            "ep": fields.ep,
            "component": "auto",
        },
        "stage": "1_baseline",
        "summary": "Initialized.",
        "gpu_resources": {
            "gpu_count": 1,
            "gpu_model": PLACEHOLDER,
            "memory_total_gib": 0,
            "cuda_visible_devices": "0",
        },
        "debate": {
            "team_name": None,
            "candidates": [],
            "rounds_completed": 0,
            "max_rounds": 4,
            "selected_winners": [],
            "selection_rationale": None,
            "delegation": {
                "enabled": enable_delegation,
                "delegates_per_champion": delegates_per_champion,
                "champion_delegate_mapping": {},
                "delegate_results": {},
            },
            "next_round_overlap": {
                "active": False,
                "phase": None,
                "selected_winners": [],
                "profiling_basis": None,
                "f_values_at_proposal": {},
            },
        },
        "parallel_tracks": {},
        "integration": {
            "status": "pending",
            "passing_candidates": [],
            "conflict_analysis": None,
            "combined_patch_branch": None,
            "combined_e2e_result": None,
            "final_decision": None,
        },
        "stage_timestamps": {
            "1_baseline": {"started_at": None, "completed_at": None},
            "2_bottleneck_mining": {"started_at": None, "completed_at": None},
            "3_debate": {"started_at": None, "completed_at": None},
            "4_5_parallel_tracks": {"started_at": None, "completed_at": None},
            "6_integration": {"started_at": None, "completed_at": None},
            "7_campaign_eval": {"started_at": None, "completed_at": None},
        },
        "session_id": None,  # Lead records session UUID at campaign start; eval extracts timing/costs from logs
        "agent_costs": [],  # Auto-populated by eval pipeline from session logs
        "campaign": {
            "status": "active",
            "current_round": 1,
            "diminishing_returns_threshold_pct": diminishing_threshold,
            "noise_tolerance_pct": noise_tolerance_pct,
            "catastrophic_regression_pct": catastrophic_regression_pct,
            "cumulative_e2e_speedup": 1.0,
            "rounds": [],
            "shipped_optimizations": [],
        },
    }


def _target_json(fields: TargetFields, artifact_dir: Path) -> Dict[str, Any]:
    # This schema is consumed by run_vllm_bench_latency_sweep.py.
    return {
        "artifact_dir": str(artifact_dir),
        "target": {
            "model_id": fields.model_id,
            "dtype": fields.dtype,
            "tp": fields.tp,
            "ep": fields.ep,
            "max_model_len": fields.max_model_len,
        },
        "workload": {
            "input_len": fields.input_len,
            "output_len": fields.output_len,
            "batch_sizes": fields.batch_sizes,
            "num_iters": fields.num_iters,
        },
        "bench": {
            "runner": "vllm_bench_latency",
            "vllm_cmd": "vllm",
            "extra_args": [],
            "baseline_extra_args": [],
            "opt_extra_args": [],
            "baseline_env": {},
            "opt_env": {
                "<ENABLE_FLAG>": "1"
            },
            "baseline_label": "baseline",
            "opt_label": "opt",
            "fastpath_evidence": {
                "baseline": {
                    "require_patterns": [],
                    "forbid_patterns": [],
                },
                "opt": {
                    "require_patterns": [],
                    "forbid_patterns": [],
                },
                "note": "Fill require_patterns to assert optimized fast-path executed (recommended).",
            },
        },
        "gating": {
            "noise_tolerance_pct": fields.noise_tolerance_pct,
            "catastrophic_regression_pct": fields.catastrophic_regression_pct,
        },
        "notes": {
            "production_parity": "Ensure CUDA graphs / torch.compile settings match production. See references/e2e-latency-guide.md.",
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact-dir", type=str, required=True, help="Directory to create/populate")
    p.add_argument("--force", action="store_true", help="Overwrite existing files")

    # Optional target metadata (safe defaults + placeholders)
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--hardware", type=str, default=None)
    p.add_argument("--dtype", type=str, default=None)

    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--diminishing-returns-threshold", type=float, default=0.5,
                   help="Stop campaign when top bottleneck < this %% of total latency (default: 0.5)")

    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--input-len", type=int, default=64)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--num-iters", type=int, default=5)

    # Delegation options (Stage 3 debate)
    p.add_argument("--enable-delegation", action="store_true", default=True,
                   help="Enable delegate sub-agents for debate champions (default: disabled)")
    p.add_argument("--delegates-per-champion", type=int, default=1,
                   help="Number of Sonnet delegate agents per Opus champion (default: 1)")

    # Gating options (BS-dependent optimization support)
    p.add_argument("--noise-tolerance-pct", type=float, default=0.5,
                   help="Per-BS speedup within this %% of 1.0 is classified NOISE (default: 0.5)")
    p.add_argument("--catastrophic-regression-pct", type=float, default=5.0,
                   help="Per-BS regression beyond this %% is classified CATASTROPHIC (default: 5.0)")

    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fields = _default_target_fields(args)

    # Create standard subdirs.
    (artifact_dir / "investigation").mkdir(exist_ok=True)
    (artifact_dir / "runs").mkdir(exist_ok=True)
    (artifact_dir / "nsys").mkdir(exist_ok=True)
    (artifact_dir / "blockers").mkdir(exist_ok=True)

    _write_text(artifact_dir / "constraints.md", _constraints_md(fields), force=args.force)
    _write_json(artifact_dir / "state.json", _state_json(
        fields, artifact_dir, args.diminishing_returns_threshold,
        enable_delegation=args.enable_delegation,
        delegates_per_champion=args.delegates_per_champion,
        noise_tolerance_pct=args.noise_tolerance_pct,
        catastrophic_regression_pct=args.catastrophic_regression_pct,
    ), force=args.force)
    _write_json(artifact_dir / "target.json", _target_json(fields, artifact_dir), force=args.force)

    print(f"Initialized artifact directory: {artifact_dir}")
    print("Created: constraints.md, state.json, target.json")
    print("Next: fill constraints.md (Phase 1) and run collect_env.py")


if __name__ == "__main__":
    main()
