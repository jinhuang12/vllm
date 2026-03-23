# Nsys Probe & Profiling Fallbacks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a standalone nsys probe script that estimates profiling cost before expensive nsys captures, and update AMMO guidance docs so agents use it.

**Architecture:** A new `scripts/nsys_probe.py` reads `target.json`, runs a quick nsys capture at OL=2, parses kernel counts from `nsys stats`, and outputs per-bucket risk estimates as JSON + console table. Three guidance files get small edits to reference the probe and add a fallback escalation hierarchy.

**Tech Stack:** Python 3 (stdlib only — json, argparse, subprocess, re, pathlib). No GPU packages imported. External deps: `nsys` CLI, `vllm` CLI (both already on PATH in AMMO sessions).

**Spec:** `docs/specs/2026-03-23-nsys-probe-and-profiling-fallbacks-design.md`

---

### Task 1: Scaffold `nsys_probe.py` — target.json parsing and CLI

**Files:**
- Create: `.claude/skills/ammo/scripts/nsys_probe.py`
- Reference: `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` (reuse `_load_json`, `_require`, `_require_int`, `_require_list_int`, `_maybe_list_str` patterns)

- [ ] **Step 1: Create the script with CLI arg parsing and target.json loading**

```python
#!/usr/bin/env python3
"""Nsys profiling probe: estimate nsys cost before expensive captures.

Runs a quick nsys trace at OL=2 on the target model, counts kernel
instances, and outputs per-bucket risk estimates with suggested
--nsys-output-len values.

Required for TP > 1 or models > 10B params. Optional otherwise.

Usage:
  python scripts/nsys_probe.py --artifact-dir <dir>
  python scripts/nsys_probe.py --artifact-dir <dir> --probe-bs 8
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# --- Overhead model from nsys-profiling-guide.md §3.9 ---
# Maps total kernel events to estimated wall time in minutes.
# Interpolated linearly between breakpoints.
_OVERHEAD_TABLE = [
    # (events, minutes)
    (0, 0.0),
    (30_000, 5.0),
    (200_000, 30.0),
    (3_000_000, 100.0),
]


def _estimate_time_min(total_events: int) -> float:
    """Estimate nsys wall time from total kernel events using overhead model."""
    for i in range(1, len(_OVERHEAD_TABLE)):
        ev_lo, t_lo = _OVERHEAD_TABLE[i - 1]
        ev_hi, t_hi = _OVERHEAD_TABLE[i]
        if total_events <= ev_hi:
            frac = (total_events - ev_lo) / max(1, ev_hi - ev_lo)
            return t_lo + frac * (t_hi - t_lo)
    # Extrapolate beyond last breakpoint.
    ev_lo, t_lo = _OVERHEAD_TABLE[-2]
    ev_hi, t_hi = _OVERHEAD_TABLE[-1]
    frac = (total_events - ev_lo) / max(1, ev_hi - ev_lo)
    return t_lo + frac * (t_hi - t_lo)


def _risk_level(time_min: float) -> str:
    if time_min < 5.0:
        return "GREEN"
    elif time_min < 15.0:
        return "YELLOW"
    else:
        return "RED"


# --- JSON helpers (same pattern as run_vllm_bench_latency_sweep.py) ---

PLACEHOLDER = "<FILL_ME>"


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Target spec not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"Failed to parse JSON {path}: {e}")


def _is_placeholder(v: Any) -> bool:
    return isinstance(v, str) and (
        v.strip() == PLACEHOLDER
        or (v.strip().startswith("<") and v.strip().endswith(">"))
    )


def _require(obj: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise SystemExit(f"Missing required field: {ctx}.{key}")
    val = obj[key]
    if _is_placeholder(val):
        raise SystemExit(f"Field still placeholder: {ctx}.{key}={val!r}")
    return val


def _require_int(obj: Dict[str, Any], key: str, ctx: str) -> int:
    val = _require(obj, key, ctx)
    if not isinstance(val, int):
        raise SystemExit(f"Expected int for {ctx}.{key}, got {type(val).__name__}")
    return val


def _require_list_int(obj: Dict[str, Any], key: str, ctx: str) -> List[int]:
    val = _require(obj, key, ctx)
    if not isinstance(val, list) or not all(isinstance(x, int) for x in val):
        raise SystemExit(f"Expected list[int] for {ctx}.{key}, got {val!r}")
    return val


def _maybe_list_str(obj: Dict[str, Any], key: str) -> List[str]:
    val = obj.get(key, [])
    if val is None:
        return []
    if not isinstance(val, list) or not all(isinstance(x, str) for x in val):
        raise SystemExit(f"Expected list[str] for {key}, got {val!r}")
    return val


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _bench_exe_tokens(vllm_cmd: Any) -> List[str]:
    if isinstance(vllm_cmd, list):
        return vllm_cmd
    if isinstance(vllm_cmd, str):
        return shlex.split(vllm_cmd)
    raise SystemExit(f"bench.vllm_cmd must be str or list[str], got {type(vllm_cmd).__name__}")


def load_target(artifact_dir: Path) -> Dict[str, Any]:
    """Load and validate target.json, returning parsed fields."""
    target_path = artifact_dir / "target.json"
    spec = _load_json(target_path)

    target = _require(spec, "target", "root")
    model_id = _require(target, "model_id", "target")
    tp = _require_int(target, "tp", "target")
    max_model_len = _require_int(target, "max_model_len", "target")

    workload = _require(spec, "workload", "root")
    batch_sizes = _require_list_int(workload, "batch_sizes", "workload")

    bench = _require(spec, "bench", "root")
    vllm_cmd = _require(bench, "vllm_cmd", "bench")
    vllm_exe = _bench_exe_tokens(vllm_cmd)
    # Fallback: if `vllm` isn't on PATH, try python -m
    if vllm_exe and vllm_exe[0] == "vllm" and shutil.which("vllm") is None:
        vllm_exe = [sys.executable, "-m", "vllm.entrypoints.cli.main"]

    extra_args = _maybe_list_str(bench, "extra_args")
    baseline_extra_args = _maybe_list_str(bench, "baseline_extra_args")

    return {
        "model_id": model_id,
        "tp": tp,
        "max_model_len": max_model_len,
        "batch_sizes": batch_sizes,
        "vllm_exe": vllm_exe,
        "extra_args": extra_args + baseline_extra_args,
        "num_layers": target.get("num_layers"),  # optional, for heuristic
        "architecture": target.get("architecture"),  # optional
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description="Nsys profiling probe: estimate cost before expensive captures."
    )
    p.add_argument("--artifact-dir", type=str, required=True)
    p.add_argument(
        "--probe-bs",
        type=int,
        default=None,
        help="Batch size for the probe (default: smallest from workload.batch_sizes)",
    )
    p.add_argument(
        "--prewarm-timeout-s",
        type=int,
        default=1800,
        help="Pre-warm step timeout in seconds (default: 1800, for cold torch.compile)",
    )
    p.add_argument(
        "--probe-timeout-s",
        type=int,
        default=600,
        help="Nsys probe step timeout in seconds (default: 600)",
    )
    args = p.parse_args()

    # Validate nsys is available.
    if not shutil.which("nsys"):
        print("ERROR: nsys not found on PATH. Install Nsight Systems CLI.", file=sys.stderr)
        raise SystemExit(1)

    artifact_dir = Path(args.artifact_dir).expanduser().resolve()
    cfg = load_target(artifact_dir)

    probe_bs = args.probe_bs or min(cfg["batch_sizes"])
    print(f"=== nsys probe ===")
    print(f"Model: {cfg['model_id']} (TP={cfg['tp']})")
    print(f"Probe batch size: {probe_bs}")
    print(f"Workload batch sizes: {cfg['batch_sizes']}")

    # Steps 1-5 will be implemented in subsequent tasks.
    # For now, validate parsing works.
    print(f"Target parsed successfully. artifact_dir={artifact_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script runs and parses a target.json**

Run: `python .claude/skills/ammo/scripts/nsys_probe.py --artifact-dir /tmp/test_probe --help`
Expected: Help text printed, exit 0.

Run: `python .claude/skills/ammo/scripts/nsys_probe.py --artifact-dir /nonexistent`
Expected: "Target spec not found" error, exit 1.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/scripts/nsys_probe.py
git commit -m "feat(ammo): scaffold nsys_probe.py with CLI and target.json parsing"
```

---

### Task 2: Implement pre-warm step (Step 1 of probe flow)

**Files:**
- Modify: `.claude/skills/ammo/scripts/nsys_probe.py`

- [ ] **Step 1: Add the `run_prewarm()` function**

Add this function after `load_target()`:

```python
def run_prewarm(
    *,
    cfg: Dict[str, Any],
    probe_bs: int,
    artifact_dir: Path,
    timeout_s: int,
) -> None:
    """Pre-warm: load model once to populate torch.compile + Triton caches.

    This ensures the nsys probe step (Step 2) doesn't include JIT overhead
    in the trace, making the kernel count estimate more reliable.
    """
    cmd = (
        cfg["vllm_exe"]
        + [
            "bench", "latency",
            "--model", cfg["model_id"],
            "--tensor-parallel-size", str(cfg["tp"]),
            "--max-model-len", str(cfg["max_model_len"]),
            "--cudagraph-capture-sizes", str(probe_bs),
            "--batch-size", str(probe_bs),
            "--input-len", "64",
            "--output-len", "2",
            "--num-iters-warmup", "1",
            "--num-iters", "1",
        ]
        + cfg["extra_args"]
    )

    print(f"\n--- Step 1: Pre-warm (no nsys) ---")
    print(f"Cmd: {shlex.join(cmd)}")
    print(f"Timeout: {timeout_s}s")

    try:
        proc = subprocess.run(
            cmd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"\nERROR: Pre-warm timed out after {timeout_s}s.\n"
            "This usually means torch.compile is running on a cold cache.\n"
            "Try: --prewarm-timeout-s 3600 (or run one iteration manually first).",
            file=sys.stderr,
        )
        raise SystemExit(2)

    if proc.returncode != 0:
        print(f"\nERROR: Pre-warm failed (exit code {proc.returncode}).", file=sys.stderr)
        # Print last 30 lines of stderr for diagnostics.
        stderr_lines = proc.stderr.strip().splitlines()
        for line in stderr_lines[-30:]:
            print(f"  {line}", file=sys.stderr)
        # Suggest common fixes.
        stderr_lower = proc.stderr.lower()
        if "trust_remote_code" in stderr_lower or "trust-remote-code" in stderr_lower:
            print(
                "\nHINT: Model may require --trust-remote-code. "
                "Add it to bench.extra_args in target.json.",
                file=sys.stderr,
            )
        if "tokenizer" in stderr_lower:
            print(
                "\nHINT: Tokenizer error — check tokenizer_config.json for "
                "custom tokenizer_class or unsupported extra_special_tokens.",
                file=sys.stderr,
            )
        raise SystemExit(2)

    print("Pre-warm completed successfully.")
```

- [ ] **Step 2: Wire `run_prewarm()` into `main()`**

In `main()`, after the print statements, add:

```python
    run_prewarm(
        cfg=cfg,
        probe_bs=probe_bs,
        artifact_dir=artifact_dir,
        timeout_s=args.prewarm_timeout_s,
    )
```

- [ ] **Step 3: Test pre-warm with a nonexistent model (fast failure)**

Run: Create a minimal target.json that references a model not on disk, verify exit code 2 with error output.

- [ ] **Step 4: Commit**

```bash
git add .claude/skills/ammo/scripts/nsys_probe.py
git commit -m "feat(ammo): nsys_probe pre-warm step with error hints"
```

---

### Task 3: Implement nsys probe step (Step 2) and kernel count parsing (Step 3)

**Files:**
- Modify: `.claude/skills/ammo/scripts/nsys_probe.py`

- [ ] **Step 1: Add `run_nsys_probe()` and `parse_kernel_count()` functions**

```python
def run_nsys_probe(
    *,
    cfg: Dict[str, Any],
    probe_bs: int,
    artifact_dir: Path,
    timeout_s: int,
) -> Path:
    """Run a short nsys capture at OL=2 to count kernel instances.

    Returns the path to the .nsys-rep file.
    """
    nsys_dir = artifact_dir / "nsys"
    nsys_dir.mkdir(parents=True, exist_ok=True)
    nsys_out = nsys_dir / "probe"

    bench_cmd = (
        cfg["vllm_exe"]
        + [
            "bench", "latency",
            "--model", cfg["model_id"],
            "--tensor-parallel-size", str(cfg["tp"]),
            "--max-model-len", str(cfg["max_model_len"]),
            "--cudagraph-capture-sizes", str(probe_bs),
            "--batch-size", str(probe_bs),
            "--input-len", "64",
            "--output-len", "2",
            "--num-iters-warmup", "3",
            "--num-iters", "1",
        ]
        + cfg["extra_args"]
    )

    nsys_cmd = [
        "nsys", "profile",
        "--trace=cuda",
        "--sample=none",
        "--cuda-graph-trace=node",
        "--trace-fork-before-exec=true",
        "--force-overwrite=true",
        "-o", str(nsys_out),
    ] + bench_cmd

    print(f"\n--- Step 2: Nsys probe (OL=2, BS={probe_bs}) ---")
    print(f"Cmd: {shlex.join(nsys_cmd)}")
    print(f"Timeout: {timeout_s}s")

    try:
        proc = subprocess.run(
            nsys_cmd,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print(
            f"\nERROR: Nsys probe timed out after {timeout_s}s.\n"
            "Model may be too heavy for --cuda-graph-trace=node even at OL=2.\n"
            "Try:\n"
            "  (1) Reduce --cudagraph-capture-sizes further\n"
            "  (2) Use torch.profiler for kernel identification\n"
            "  (3) Use --cuda-graph-trace=graph (loses per-kernel detail, see §3.6)",
            file=sys.stderr,
        )
        raise SystemExit(3)

    if proc.returncode != 0:
        print(f"\nERROR: Nsys probe failed (exit code {proc.returncode}).", file=sys.stderr)
        stderr_lines = proc.stderr.strip().splitlines()
        for line in stderr_lines[-20:]:
            print(f"  {line}", file=sys.stderr)
        raise SystemExit(3)

    nsys_rep = nsys_out.with_suffix(".nsys-rep")
    if not nsys_rep.exists():
        # nsys sometimes adds numbers
        candidates = list(nsys_dir.glob("probe*.nsys-rep"))
        if candidates:
            nsys_rep = candidates[0]
        else:
            print("ERROR: nsys produced no .nsys-rep file.", file=sys.stderr)
            raise SystemExit(3)

    print(f"Probe trace: {nsys_rep} ({nsys_rep.stat().st_size / 1024 / 1024:.1f} MB)")
    return nsys_rep


def parse_kernel_count(nsys_rep: Path) -> int:
    """Parse cuda_gpu_kern_sum from nsys stats and estimate kernels_per_decode_step.

    Uses the §3.9 rough-division method: total_instances / ~12.
    With warmup=3, iters=1, OL=2:
      warmup: 3 iters × ~3 steps = ~9 steps
      benchmark: 1 iter × 3 steps = ~3 steps
      total ≈ 12 steps

    Returns kernels_per_decode_step (before TP multiplication — caller applies TP).
    """
    print(f"\n--- Step 3: Parse kernel count ---")
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(nsys_rep)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(f"ERROR: nsys stats failed: {proc.stderr[:500]}", file=sys.stderr)
        raise SystemExit(3)

    # Parse CSV: header line contains "Instances" column.
    total_instances = 0
    lines = proc.stdout.strip().splitlines()
    header_idx = None
    instances_col = None
    for i, line in enumerate(lines):
        if "Instances" in line:
            header_idx = i
            cols = [c.strip().strip('"') for c in line.split(",")]
            for j, col in enumerate(cols):
                if col == "Instances":
                    instances_col = j
                    break
            break

    if header_idx is None or instances_col is None:
        print("ERROR: Could not find 'Instances' column in nsys stats output.", file=sys.stderr)
        print(f"First 10 lines:\n{chr(10).join(lines[:10])}", file=sys.stderr)
        raise SystemExit(3)

    for line in lines[header_idx + 1 :]:
        cols = [c.strip().strip('"') for c in line.split(",")]
        if len(cols) > instances_col:
            try:
                total_instances += int(cols[instances_col])
            except ValueError:
                continue

    # Rough-division: ~12 total steps in the trace.
    kernels_per_step = max(1, total_instances // 12)
    print(f"Total kernel instances: {total_instances:,}")
    print(f"Estimated kernels/decode step: {kernels_per_step:,} (total / 12)")

    return kernels_per_step
```

- [ ] **Step 2: Wire into `main()` after pre-warm**

```python
    nsys_rep = run_nsys_probe(
        cfg=cfg,
        probe_bs=probe_bs,
        artifact_dir=artifact_dir,
        timeout_s=args.probe_timeout_s,
    )

    kernels_per_step = parse_kernel_count(nsys_rep)
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/scripts/nsys_probe.py
git commit -m "feat(ammo): nsys_probe Steps 2-3 — nsys capture and kernel count parsing"
```

---

### Task 4: Implement per-bucket estimates, heuristic cross-check, and output (Steps 4-5)

**Files:**
- Modify: `.claude/skills/ammo/scripts/nsys_probe.py`

- [ ] **Step 1: Add `compute_estimates()` and `format_output()` functions**

```python
def compute_estimates(
    *,
    kernels_per_step: int,
    tp: int,
    batch_sizes: List[int],
    num_layers: Optional[int],
    architecture: Optional[str],
) -> Dict[str, Any]:
    """Compute per-bucket and total sweep estimates."""
    # Heuristic cross-check.
    heuristic = None
    heuristic_str = None
    if num_layers:
        # MoE/hybrid: 13-20 kernels/layer, standard: 10-15
        is_moe = architecture and any(
            k in (architecture or "").lower() for k in ["moe", "mixture", "expert"]
        )
        factor = 15 if is_moe else 12
        heuristic = num_layers * factor
        heuristic_str = f"{num_layers} layers × {factor} = {heuristic}"

        ratio = max(kernels_per_step, 1) / max(heuristic, 1)
        if ratio > 2.0 or ratio < 0.5:
            print(
                f"\nWARNING: Measured kernels/step ({kernels_per_step:,}) differs from "
                f"heuristic ({heuristic:,}) by {ratio:.1f}x. "
                "Possible transient overhead inflation. Using lower value.",
                file=sys.stderr,
            )
            kernels_per_step = min(kernels_per_step, heuristic)

    # Per-bucket estimates.
    effective_kernels = kernels_per_step * tp
    safe_nsys_ol = min(32, max(1, 20_000 // max(1, effective_kernels)))

    per_bucket = {}
    for bs in batch_sizes:
        events = effective_kernels * safe_nsys_ol
        time_min = _estimate_time_min(events)
        per_bucket[str(bs)] = {
            "safe_nsys_OL": safe_nsys_ol,
            "estimated_events": events,
            "estimated_time_min": round(time_min, 1),
            "risk_level": _risk_level(time_min),
        }

    # Total sweep estimate (additive: each bucket is a separate capture range).
    num_buckets = len(batch_sizes)
    per_bucket_events = effective_kernels * safe_nsys_ol
    per_bucket_time = _estimate_time_min(per_bucket_events)
    total_events = per_bucket_events * num_buckets
    total_time = per_bucket_time * num_buckets

    # Suggested sweep args.
    suggested_timeout = max(600, int(total_time * 60 * 1.5))  # 1.5x safety margin

    return {
        "kernels_per_decode_step": kernels_per_step,
        "estimation_method": "total_instances / 12 (§3.9 rough division)",
        "heuristic_estimate": heuristic_str,
        "per_bucket": per_bucket,
        "total_sweep": {
            "num_buckets": num_buckets,
            "total_events": total_events,
            "estimated_time_min": round(total_time, 1),
            "risk_level": _risk_level(total_time),
        },
        "suggested_sweep_args": {
            "--nsys-output-len": safe_nsys_ol,
            "--nsys-num-iters": 1,
            "--nsys-timeout-s": suggested_timeout,
        },
    }


def format_output(
    *,
    model_id: str,
    tp: int,
    probe_bs: int,
    estimates: Dict[str, Any],
) -> str:
    """Format the probe results as a human-readable table."""
    lines = []
    lines.append(f"\n=== nsys probe results ===")
    lines.append(f"Model: {model_id} (TP={tp})")

    kps = estimates["kernels_per_decode_step"]
    heuristic = estimates.get("heuristic_estimate") or "N/A"
    lines.append(f"Kernels/decode step: {kps:,} (measured) vs {heuristic} (heuristic)")
    lines.append("")
    lines.append("| Batch Size | Safe nsys_OL | Est. Events | Est. Time | Risk |")
    lines.append("|---:|---:|---:|---:|---|")

    for bs_str, info in estimates["per_bucket"].items():
        lines.append(
            f"| {bs_str} | {info['safe_nsys_OL']} | "
            f"{info['estimated_events']:,} | ~{info['estimated_time_min']} min | "
            f"{info['risk_level']} |"
        )

    ts = estimates["total_sweep"]
    lines.append(
        f"| TOTAL SWEEP ({ts['num_buckets']} buckets) | — | "
        f"{ts['total_events']:,} | ~{ts['estimated_time_min']} min | "
        f"{ts['risk_level']} |"
    )

    sa = estimates["suggested_sweep_args"]
    lines.append("")
    lines.append("Suggested sweep args:")
    lines.append(
        f"  --nsys-output-len {sa['--nsys-output-len']} "
        f"--nsys-num-iters {sa['--nsys-num-iters']} "
        f"--nsys-timeout-s {sa['--nsys-timeout-s']}"
    )

    # Warnings.
    if sa["--nsys-output-len"] <= 4:
        lines.append("")
        lines.append(
            f"WARNING: nsys_OL <= {sa['--nsys-output-len']} due to high kernel count × TP."
        )
        lines.append(
            "Decode profiling at low OL captures kernel identity and count accurately"
        )
        lines.append(
            "but NOT duration scaling with KV length. For duration analysis, use"
        )
        lines.append("targeted ncu on specific kernels (see nsys-profiling-guide.md §4).")

    if any(info["risk_level"] == "RED" for info in estimates["per_bucket"].values()):
        lines.append("")
        lines.append(
            "WARNING: Some batch sizes are RED (>15 min). Consider skipping nsys"
        )
        lines.append(
            "for those batch sizes and documenting the gap in bottleneck_analysis.md."
        )

    return "\n".join(lines)
```

- [ ] **Step 2: Complete `main()` with estimates computation and output**

Replace the placeholder at the end of `main()` with:

```python
    estimates = compute_estimates(
        kernels_per_step=kernels_per_step,
        tp=cfg["tp"],
        batch_sizes=cfg["batch_sizes"],
        num_layers=cfg.get("num_layers"),
        architecture=cfg.get("architecture"),
    )

    # Console output.
    output = format_output(
        model_id=cfg["model_id"],
        tp=cfg["tp"],
        probe_bs=probe_bs,
        estimates=estimates,
    )
    print(output)

    # JSON output.
    result = {
        "probe_time": datetime.now(timezone.utc).isoformat(),
        "model_id": cfg["model_id"],
        "tp": cfg["tp"],
        "probe_bs": probe_bs,
        **estimates,
    }
    out_path = artifact_dir / "nsys" / "probe_results.json"
    _write_json(out_path, result)
    print(f"\nWrote: {out_path}")
```

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/scripts/nsys_probe.py
git commit -m "feat(ammo): nsys_probe Steps 4-5 — estimates, heuristic cross-check, output"
```

---

### Task 5: Unit tests for pure functions (overhead model, risk levels, kernel parsing, estimates)

**Files:**
- Create: `.claude/skills/ammo/tests/test_nsys_probe.py`
- Reference: `.claude/skills/ammo/scripts/nsys_probe.py`

- [ ] **Step 1: Write unit tests for pure functions**

```python
"""Unit tests for nsys_probe.py pure functions.

These test the overhead model, risk levels, estimate computation, and
CSV parsing without needing nsys or a GPU.
"""

import json
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the module under test.
import importlib.util
import sys

_spec = importlib.util.spec_from_file_location(
    "nsys_probe",
    str(Path(__file__).resolve().parents[1] / "scripts" / "nsys_probe.py"),
)
nsys_probe = importlib.util.module_from_spec(_spec)
sys.modules["nsys_probe"] = nsys_probe
_spec.loader.exec_module(nsys_probe)


class TestOverheadModel:
    def test_zero_events(self):
        assert nsys_probe._estimate_time_min(0) == 0.0

    def test_low_events_green(self):
        t = nsys_probe._estimate_time_min(10_000)
        assert 0 < t < 5.0

    def test_30k_events_is_5_min(self):
        t = nsys_probe._estimate_time_min(30_000)
        assert abs(t - 5.0) < 0.1

    def test_200k_events_is_30_min(self):
        t = nsys_probe._estimate_time_min(200_000)
        assert abs(t - 30.0) < 0.1

    def test_interpolation(self):
        # Midpoint between 30k (5min) and 200k (30min)
        t = nsys_probe._estimate_time_min(115_000)
        assert 15.0 < t < 20.0

    def test_extrapolation_beyond_table(self):
        t = nsys_probe._estimate_time_min(5_000_000)
        assert t > 100.0


class TestRiskLevel:
    def test_green(self):
        assert nsys_probe._risk_level(3.0) == "GREEN"

    def test_yellow(self):
        assert nsys_probe._risk_level(10.0) == "YELLOW"

    def test_red(self):
        assert nsys_probe._risk_level(20.0) == "RED"

    def test_boundary_green_yellow(self):
        assert nsys_probe._risk_level(4.99) == "GREEN"
        assert nsys_probe._risk_level(5.0) == "YELLOW"

    def test_boundary_yellow_red(self):
        assert nsys_probe._risk_level(14.99) == "YELLOW"
        assert nsys_probe._risk_level(15.0) == "RED"


class TestComputeEstimates:
    def test_small_model_tp1(self):
        """Small TP=1 model: all GREEN, OL=32."""
        est = nsys_probe.compute_estimates(
            kernels_per_step=300,
            tp=1,
            batch_sizes=[1, 8, 32],
            num_layers=None,
            architecture=None,
        )
        assert est["suggested_sweep_args"]["--nsys-output-len"] == 32
        for bs_info in est["per_bucket"].values():
            assert bs_info["risk_level"] == "GREEN"

    def test_large_moe_tp8(self):
        """Large MoE TP=8: low OL, YELLOW/RED."""
        est = nsys_probe.compute_estimates(
            kernels_per_step=1170,
            tp=8,
            batch_sizes=[1, 8, 32],
            num_layers=78,
            architecture="glm_moe_dsa",
        )
        # effective_kernels = 1170 * 8 = 9360
        # safe_OL = floor(20000 / 9360) = 2
        assert est["suggested_sweep_args"]["--nsys-output-len"] == 2

    def test_total_sweep_calculated(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=500,
            tp=2,
            batch_sizes=[1, 8],
            num_layers=None,
            architecture=None,
        )
        assert est["total_sweep"]["num_buckets"] == 2
        # Per-bucket events should multiply by num_buckets for total
        per_bucket_events = list(est["per_bucket"].values())[0]["estimated_events"]
        assert est["total_sweep"]["total_events"] == per_bucket_events * 2

    def test_heuristic_crosscheck_uses_lower(self):
        """When measured >> heuristic, uses lower value."""
        est = nsys_probe.compute_estimates(
            kernels_per_step=5000,  # suspiciously high
            tp=1,
            batch_sizes=[1],
            num_layers=30,
            architecture="standard",
        )
        # Heuristic: 30 * 12 = 360. 5000/360 = 13.9x > 2x threshold.
        # Should use min(5000, 360) = 360.
        assert est["kernels_per_decode_step"] == 360

    def test_nsys_num_iters_always_1(self):
        est = nsys_probe.compute_estimates(
            kernels_per_step=300,
            tp=1,
            batch_sizes=[1],
            num_layers=None,
            architecture=None,
        )
        assert est["suggested_sweep_args"]["--nsys-num-iters"] == 1

    def test_total_sweep_time_is_additive(self):
        """Total sweep time = sum of per-bucket times (not nonlinear lookup on total events)."""
        est = nsys_probe.compute_estimates(
            kernels_per_step=1170,
            tp=8,
            batch_sizes=[1, 8, 32],
            num_layers=78,
            architecture="glm_moe_dsa",
        )
        per_bs_time = list(est["per_bucket"].values())[0]["estimated_time_min"]
        expected_total = round(per_bs_time * 3, 1)
        assert abs(est["total_sweep"]["estimated_time_min"] - expected_total) < 0.5


class TestLoadTarget:
    def test_minimal_target(self, tmp_path):
        target_json = {
            "artifact_dir": str(tmp_path),
            "target": {
                "model_id": "test/model",
                "dtype": "fp8",
                "tp": 2,
                "ep": 1,
                "max_model_len": 4096,
            },
            "workload": {
                "input_len": 64,
                "output_len": 512,
                "batch_sizes": [1, 8],
                "num_iters": 5,
            },
            "bench": {
                "runner": "vllm_bench_latency",
                "vllm_cmd": "vllm",
                "extra_args": [],
                "baseline_extra_args": [],
                "opt_extra_args": [],
                "baseline_env": {},
                "opt_env": {},
                "baseline_label": "baseline",
                "opt_label": "opt",
            },
        }
        (tmp_path / "target.json").write_text(json.dumps(target_json))
        cfg = nsys_probe.load_target(tmp_path)
        assert cfg["model_id"] == "test/model"
        assert cfg["tp"] == 2
        assert cfg["batch_sizes"] == [1, 8]

    def test_missing_target_json(self, tmp_path):
        with pytest.raises(SystemExit):
            nsys_probe.load_target(tmp_path)

    def test_vllm_path_fallback(self, tmp_path):
        """When vllm is not on PATH, falls back to python -m."""
        target_json = {
            "artifact_dir": str(tmp_path),
            "target": {
                "model_id": "test/model",
                "dtype": "fp8",
                "tp": 1,
                "ep": 1,
                "max_model_len": 4096,
            },
            "workload": {
                "input_len": 64,
                "output_len": 512,
                "batch_sizes": [1],
                "num_iters": 5,
            },
            "bench": {
                "runner": "vllm_bench_latency",
                "vllm_cmd": "vllm",
                "extra_args": [],
                "baseline_extra_args": [],
                "opt_extra_args": [],
                "baseline_env": {},
                "opt_env": {},
                "baseline_label": "baseline",
                "opt_label": "opt",
            },
        }
        (tmp_path / "target.json").write_text(json.dumps(target_json))
        with patch("nsys_probe.shutil.which", return_value=None):
            cfg = nsys_probe.load_target(tmp_path)
        assert cfg["vllm_exe"][0] == sys.executable
        assert "vllm.entrypoints.cli.main" in cfg["vllm_exe"][-1]
```

- [ ] **Step 2: Run tests**

Run: `cd /home/jinhun/vllm && python -m pytest .claude/skills/ammo/tests/test_nsys_probe.py -v`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/tests/test_nsys_probe.py
git commit -m "test(ammo): unit tests for nsys_probe pure functions"
```

---

### Task 6: Update `ammo-researcher.md` — probe step and fallback hierarchy

**Files:**
- Modify: `/home/jinhun/vllm/.claude/agents/ammo-researcher.md` (E2E Baseline section + new fallback section)

- [ ] **Step 1: Add pre-profiling probe block and update sweep example**

Insert the probe block BEFORE the line `**Combined E2E baseline + nsys profiling (default for Stage 1)**:`. Then REPLACE the existing sweep command block (the 3-line `python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` example ending with `--nsys-profile`) with the updated version that includes `--nsys-output-len`:

Before line 53, insert:
```markdown
**Pre-profiling probe (REQUIRED for TP > 1 or models > 10B params)**:

Before running `--nsys-profile`, estimate profiling cost:

```bash
python .claude/skills/ammo/scripts/nsys_probe.py --artifact-dir {artifact_dir}
```

This takes ~5-15 minutes and outputs per-BS risk estimates with suggested
`--nsys-output-len`, `--nsys-num-iters`, and `--nsys-timeout-s` values.
See `references/nsys-profiling-guide.md` §3.9-3.10 for the theory.

For small TP=1 models (< 10B params), the probe is optional — nsys
profiling at default settings rarely has issues.
```

Then update the existing sweep example (lines 54-58) to:
```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --nsys-profile \
  --nsys-output-len {probe_suggested_OL}
```

- [ ] **Step 2: Add "When nsys Profiling Fails" section**

Insert after the paragraph ending with "Do not rank them as optimization candidates." (the last substantive line of the "Steady-State vs Transient Classification" section, before "## Key Constraints"):

```markdown
## When nsys Profiling Fails

If nsys `--cuda-graph-trace=node` fails or hangs for a batch size, follow this escalation hierarchy:

1. **Reduce `--nsys-output-len`** to the probe's suggested value (or lower)
2. **Restrict `--cudagraph-capture-sizes`** to `[target_bs]` only
3. **Skip nsys for that BS** — document "BS=N profiling unavailable" in bottleneck_analysis.md
4. **NEVER fall back to `--enforce-eager`** for profiling

If a batch size has no profiling data, flag it explicitly:

> WARNING: No nsys profiling data for BS={N}. Debate proposals targeting this batch size lack empirical grounding for kernel-level claims.

If the probe itself times out at OL=2, the model may be too heavy for `--cuda-graph-trace=node` entirely. In that case:
- Use `torch.profiler` for lightweight kernel identification
- Or use `--cuda-graph-trace=graph` (loses per-kernel detail inside CUDA graphs — see nsys-profiling-guide.md §3.6 for caveats)
- Document all methodology caveats prominently in bottleneck_analysis.md
```

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/ammo-researcher.md
git commit -m "docs(ammo): add nsys probe step and fallback hierarchy to researcher agent"
```

---

### Task 7: Update `nsys-profiling-guide.md` — add §3.10 decision tree

**Files:**
- Modify: `/home/jinhun/vllm/.claude/skills/ammo/references/nsys-profiling-guide.md:392` (after the §3.9 escape hatches table, before §4)

- [ ] **Step 1: Insert §3.10 after line 392 (end of §3.9), before `## 4) Nsight Compute`**

```markdown
### 3.10 Profiling Decision Tree

Before attempting nsys profiling on models with TP > 1 or > 10B params, run the probe script to estimate cost:

```bash
python scripts/nsys_probe.py --artifact-dir {artifact_dir}
```

Decision tree based on probe results:

```
Probe succeeds?
├── YES: All BS green/yellow?
│   ├── YES → Run --nsys-profile with suggested --nsys-output-len
│   └── NO (some BS red) →
│       ├── Use suggested --nsys-output-len (auto-reduces OL for expensive BS)
│       └── OR skip nsys for red BS, document the gap
└── NO: Probe timed out at OL=2?
    ├── Restrict --cudagraph-capture-sizes to [1] only, retry probe
    ├── If still fails → use torch.profiler for kernel identification
    └── Document all caveats in bottleneck_analysis.md
```

For small TP=1 models (< 10B params), the probe is optional — proceed directly to `--nsys-profile` with default settings.

If nsys profiling fails AFTER the probe passed (unexpected), run the probe as a diagnostic to compare expected vs actual behavior.
```

- [ ] **Step 2: Verify the section numbering is consistent**

Read the file to confirm §3.10 sits between §3.9 and §4.

- [ ] **Step 3: Commit**

```bash
git add .claude/skills/ammo/references/nsys-profiling-guide.md
git commit -m "docs(ammo): add §3.10 profiling decision tree to nsys guide"
```

---

### Task 8: Update `SKILL.md` — orchestrator probe instruction

**Files:**
- Modify: `/home/jinhun/vllm/.claude/skills/ammo/SKILL.md:160` (the "Profiling strategy selection" paragraph)

- [ ] **Step 1: Insert probe instruction after the existing paragraph**

At line 162 (after "This is NOT a parity violation — the profiled sizes are exact matches in vLLM's default capture list, so the graphs are identical to production."), insert:

```markdown
The lead should also instruct the researcher to run `scripts/nsys_probe.py` first to estimate profiling cost and determine safe `--nsys-output-len` values. See `references/nsys-profiling-guide.md` §3.10.
```

- [ ] **Step 2: Commit**

```bash
git add .claude/skills/ammo/SKILL.md
git commit -m "docs(ammo): add nsys probe instruction to orchestrator SKILL.md"
```

---

### Task 9: Final integration test and cleanup

**Files:**
- Reference: `.claude/skills/ammo/scripts/nsys_probe.py`
- Reference: `.claude/skills/ammo/tests/test_nsys_probe.py`

- [ ] **Step 1: Run the full test suite**

```bash
cd /home/jinhun/vllm && python -m pytest .claude/skills/ammo/tests/test_nsys_probe.py -v
```

Expected: All tests pass.

- [ ] **Step 2: Verify script help text and --help flag**

```bash
python .claude/skills/ammo/scripts/nsys_probe.py --help
```

Expected: Clean help output with all documented flags.

- [ ] **Step 3: Verify all guidance files are consistent**

```bash
grep -n "nsys_probe" /home/jinhun/vllm/.claude/agents/ammo-researcher.md
grep -n "nsys_probe\|§3.10" /home/jinhun/vllm/.claude/skills/ammo/SKILL.md
grep -n "nsys_probe\|3.10" /home/jinhun/vllm/.claude/skills/ammo/references/nsys-profiling-guide.md
```

Expected: Each grep finds at least one match, confirming cross-references.

- [ ] **Step 4: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore(ammo): nsys probe integration test and cleanup"
```
