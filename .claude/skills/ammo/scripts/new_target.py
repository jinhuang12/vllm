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
    data_parallel_size: int = 1
    enable_expert_parallel: bool = False


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
        data_parallel_size=getattr(args, "data_parallel_size", 1),
        enable_expert_parallel=getattr(args, "enable_expert_parallel", False),
    )


def _constraints_md(fields: TargetFields) -> str:
    return f"""# Constraints (Stage 1)

## Target envelope

- Model: {fields.model_id}
- Hardware: {fields.hardware}
- Dtype / quant format: {fields.dtype}
- TP / EP: tp={fields.tp}, ep={fields.ep}
- Data Parallel / Expert Parallel: dp={fields.data_parallel_size}, enable_expert_parallel={fields.enable_expert_parallel}
- Max model len: {fields.max_model_len}
- Decode buckets (batch sizes): {fields.batch_sizes}
- E2E workload: input_len={fields.input_len}, output_len={fields.output_len}

## TODOs (required before Stage 2)

- [ ] Read vLLM source for target component, document forward path and correctness invariants
- [ ] Capture baseline truth snapshot under production parity (CUDA graphs / torch.compile)
- [ ] Record baseline kernel timings from nsys profiling

"""

_SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "state.schema.json"


def _schema_default(path: str, fallback: float) -> float:
    """Read a default value from state.schema.json at a dotted `campaign.config.*` path.

    Keeping the defaults in the schema lets us change them once and have writers
    pick them up automatically — no per-call-site drift.
    """
    try:
        schema = json.loads(_SCHEMA_PATH.read_text())
        props = schema["properties"]["campaign"]["properties"]["config"]["properties"]
        return float(props[path]["default"])
    except (FileNotFoundError, KeyError, json.JSONDecodeError, TypeError, ValueError):
        return fallback


def _state_json(fields: TargetFields, artifact_dir: Path, min_e2e_improvement_pct: float = 0.25,
                noise_tolerance_pct: float | None = None,
                catastrophic_regression_pct: float | None = None) -> Dict[str, Any]:
    # Defaults resolved from state.schema.json — schema is the single source of truth.
    if noise_tolerance_pct is None:
        noise_tolerance_pct = _schema_default("noise_tolerance_pct", 0.5)
    if catastrophic_regression_pct is None:
        catastrophic_regression_pct = _schema_default("catastrophic_regression_pct", 5.0)
    return {
        "target": {
            "model_id": fields.model_id,
            "hardware": fields.hardware,
            "dtype": fields.dtype,
            "tp": fields.tp,
            "dp": fields.data_parallel_size,
            "ep": fields.ep,
            "component": "auto",
        },
        "session_id": None,
        "gpu_resources": {
            "gpu_count": 1,
            "gpu_model": PLACEHOLDER,
            "memory_total_gib": 0,
            "cuda_visible_devices": "0",
        },
        "campaign": {
            "schema_version": "4.1",
            "status": "active",
            "current_round": 1,
            "current_stage": "1_baseline",
            "config": {
                "min_e2e_improvement_pct": min_e2e_improvement_pct,
                "noise_tolerance_pct": noise_tolerance_pct,
                "catastrophic_regression_pct": catastrophic_regression_pct,
            },
            "shipped_optimizations": [],
            "agent_costs": [],
            "rounds": [
                {
                    "round_id": 1,
                    "status": "IN_PROGRESS",
                    "team_name": None,
                    "profiling_baseline_path": None,
                    "baseline": {
                        "started_at": None,
                        "completed_at": None,
                        "e2e_latency": None,
                        "per_bs_verdict": None,
                    },
                    "bottleneck_mining": {
                        "started_at": None,
                        "completed_at": None,
                        "top_bottleneck_share_pct": None,
                    },
                    "debate": {
                        "started_at": None,
                        "completed_at": None,
                        "candidates": [],
                        "rounds_completed": 0,
                        "max_rounds": 4,
                        "selected_winners": [],
                    },
                    "parallel_tracks": {
                        "started_at": None,
                        "completed_at": None,
                        "tracks": {},
                    },
                    "integration": {
                        "started_at": None,
                        "completed_at": None,
                        "status": "pending",
                        "passing_candidates": [],
                        "failed_candidates": [],
                        "selected_candidates": [],
                        "conflict_analysis": None,
                        "combined_patch_branch": None,
                        "combined_e2e_result": None,
                        "e2e_latency_combined": None,
                        "per_bs_verdict": None,
                        "commit_sha": None,
                        "final_decision": None,
                        "resolver_invoked": None,
                        "resolver_outcome": None,
                        "conflicting_tracks": None,
                    },
                    "campaign_eval": {
                        "started_at": None,
                        "completed_at": None,
                    },
                    "audit": {},
                    "shipped": [],
                    "dropped": [],
                    "cumulative_speedup_after": None,
                    "combined_e2e_speedup_x": None,
                    "combined_e2e_delta_pp": None,
                    "note": None,
                    "round_summary": None,
                }
            ],
        },
    }


def _compose_extra_args(fields: TargetFields) -> List[str]:
    """Compose bench.extra_args from DP / EP fields.

    DP > 1 triggers auto-injection of --distributed-executor-backend
    external_launcher, which vLLM requires for data-parallel execution via
    torchrun. The sweep script validates that no conflicting backend value
    was appended post-hoc (e.g., via frontend additionalFlags).
    """
    extra: List[str] = []
    if fields.data_parallel_size > 1:
        extra.extend([
            "--data-parallel-size", str(fields.data_parallel_size),
            "--distributed-executor-backend", "external_launcher",
        ])
    if fields.enable_expert_parallel:
        extra.append("--enable-expert-parallel")
    return extra


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
            "extra_args": _compose_extra_args(fields),
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
    p.add_argument("--ep", type=int, default=1,
                   help="Legacy expert-parallel sizing field — persisted to target.ep "
                        "in state.json. Does NOT enable vLLM expert parallelism. "
                        "Use --enable-expert-parallel to actually enable EP.")
    p.add_argument("--data-parallel-size", type=int, default=1,
                   help="vLLM data-parallel size. When > 1, new_target.py injects "
                        "--distributed-executor-backend external_launcher into "
                        "bench.extra_args (required for torchrun DP).")
    p.add_argument("--enable-expert-parallel", action="store_true",
                   help="Enable vLLM expert parallelism (passes "
                        "--enable-expert-parallel through to bench.extra_args).")
    p.add_argument("--min-e2e-improvement", type=float, default=0.25,
                   help="Stop campaign when no candidate can yield >= this %% E2E improvement (default: 0.25)")

    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--input-len", type=int, default=64)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--num-iters", type=int, default=10)

    # Gating options (BS-dependent optimization support)
    p.add_argument("--noise-tolerance-pct", type=float, default=0.5,
                   help="Per-BS speedup within this %% of 1.0 is classified NOISE (default: 0.5)")
    p.add_argument("--catastrophic-regression-pct", type=float, default=5.0,
                   help="Per-BS regression beyond this %% is classified CATASTROPHIC (default: 5.0)")

    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    fields = _default_target_fields(args)

    # Create v2 round-scoped scaffold for round 1.
    # Spec: docs/superpowers/specs/2026-05-12-ammo-artifact-layout-design.md (§Bootstrap)
    # Reference: ai_cli_session/.claude/skills/ammo/references/artifact-layout.md
    round_dir = artifact_dir / "rounds" / "1"
    for subdir in (
        "profiling/nsys",
        "profiling/ncu",
        "sweeps/baseline/json",
        "sweeps/baseline/logs",
        "sweeps/baseline/status",
        "sweeps/opt",
        "sweeps/integration",
        "sweeps/golden_capture",
        "mining",
        "debate/proposals",
        "debate/micro_experiments",
        "debate/monitor_audits",
        "tracks",
        "audits",
        "_archive",
    ):
        (round_dir / subdir).mkdir(parents=True, exist_ok=True)
    # Cross-round (campaign-level) blockers dir — NOT round-scoped.
    (artifact_dir / "blockers").mkdir(exist_ok=True)

    state = _state_json(
        fields, artifact_dir, args.min_e2e_improvement,
        noise_tolerance_pct=args.noise_tolerance_pct,
        catastrophic_regression_pct=args.catastrophic_regression_pct,
    )
    _write_json(artifact_dir / "state.json", state, force=args.force)
    _write_json(artifact_dir / "target.json", _target_json(fields, artifact_dir), force=args.force)

    print(f"Initialized artifact directory: {artifact_dir}")
    print("Created: state.json, target.json, rounds/1/ scaffold")
    print("Next: write rounds/1/constraints.md (Phase 1) and run collect_env.py")


if __name__ == "__main__":
    main()
