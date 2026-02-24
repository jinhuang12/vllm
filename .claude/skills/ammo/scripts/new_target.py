#!/usr/bin/env python3
"""ammo v2: scaffold a new target artifact directory.

Creates:
  - constraints.md, optimization_plan.md, implementation_notes.md,
    validation_results.md, integration.md
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
from datetime import date
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


def _optimization_plan_md() -> str:
    return """# Optimization Plan (Stage 3)

## Optimization approach

- Approach:
- Rationale (with profiling evidence):

## Feasibility math

- Required savings (per bucket):
- Upper bound savings (bytes/BW):
- Kill criteria:

## Implementation plan

1.
2.
3.

"""


def _implementation_notes_md() -> str:
    return """# Implementation Notes (Stage 4)

Record:
- kernel structure and specialization decisions
- CUDA graphs safety decisions (stream, allocations, stable shapes)
- how correctness is preserved (component semantics, reduction/accumulation invariants)

"""


def _validation_results_md() -> str:
    return """# Validation Results (Stage 5)

> Default gates + reporting checklist: `references/validation-defaults.md`.

## Correctness

- Status:
- Tolerance (atol/rtol):
- Max abs diff:

## Kernel perf (CUDA graphs)

- Bucket set:
- Baseline vs optimized (µs):

## E2E latency (vllm bench latency)

- Workload:
- Baseline vs optimized (s):

## Decision

- Ship / Restrict envelope / Pivot route / Stop

"""


def _integration_md() -> str:
    return """# Integration

Record:
- fast-path enablement envelope (model id, dtype, TP/EP, dims, buckets)
- fallback behavior
- how to reproduce validation

"""


def _state_json(fields: TargetFields, artifact_dir: Path) -> Dict[str, Any]:
    # Derive a short target name for the team
    model_short = fields.model_id.split("/")[-1].lower().replace("-", "")[:12] if fields.model_id != PLACEHOLDER else "target"
    hw_short = fields.hardware.lower() if fields.hardware != PLACEHOLDER else "gpu"
    target_name = f"{model_short}-{hw_short}"

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
        "status": "in_progress",
        "current_opportunity_id": None,
        "max_attempts": 3,
        "opportunity_attempts": [],
        "route_decision": {},
        "verification_run": {
            "stage1": None,
            "validation": None,
        },
        "last_update": date.today().isoformat(),
        "summary": "Initialized.",
        "team": {
            "name": f"ammo-{target_name}",
            "members": ["lead", "researcher", "implementer"],
        },
        "gpu_resources": {
            "gpu_count": 1,
            "gpu_model": PLACEHOLDER,
            "memory_total_gib": 0,
            "cuda_visible_devices": "0",
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

    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--input-len", type=int, default=64)
    p.add_argument("--output-len", type=int, default=512)
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    p.add_argument("--num-iters", type=int, default=20)

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
    _write_text(artifact_dir / "optimization_plan.md", _optimization_plan_md(), force=args.force)
    _write_text(artifact_dir / "implementation_notes.md", _implementation_notes_md(), force=args.force)
    _write_text(artifact_dir / "validation_results.md", _validation_results_md(), force=args.force)
    _write_text(artifact_dir / "integration.md", _integration_md(), force=args.force)

    _write_json(artifact_dir / "state.json", _state_json(fields, artifact_dir), force=args.force)
    _write_json(artifact_dir / "target.json", _target_json(fields, artifact_dir), force=args.force)

    print(f"Initialized artifact directory: {artifact_dir}")
    print("Created: constraints.md, optimization_plan.md, implementation_notes.md, validation_results.md, integration.md, state.json, target.json")
    print("Next: fill constraints.md (Phase 1) and run collect_env.py")


if __name__ == "__main__":
    main()
