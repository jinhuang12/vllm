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


def run_prewarm(
    *,
    cfg: Dict[str, Any],
    probe_bs: int,
    artifact_dir: Path,
    timeout_s: int,
) -> None:
    """Pre-warm: load model once to populate torch.compile + Triton caches."""
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
        stderr_lines = proc.stderr.strip().splitlines()
        for line in stderr_lines[-30:]:
            print(f"  {line}", file=sys.stderr)
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
    Returns kernels_per_decode_step (before TP multiplication — caller applies TP).
    """
    print(f"\n--- Step 3: Parse kernel count ---")
    cmd = ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv", str(nsys_rep)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        print(f"ERROR: nsys stats failed: {proc.stderr[:500]}", file=sys.stderr)
        raise SystemExit(3)

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

    for line in lines[header_idx + 1:]:
        cols = [c.strip().strip('"') for c in line.split(",")]
        if len(cols) > instances_col:
            try:
                total_instances += int(cols[instances_col])
            except ValueError:
                continue

    kernels_per_step = max(1, total_instances // 12)
    print(f"Total kernel instances: {total_instances:,}")
    print(f"Estimated kernels/decode step: {kernels_per_step:,} (total / 12)")

    return kernels_per_step


def compute_estimates(
    *,
    kernels_per_step: int,
    tp: int,
    batch_sizes: List[int],
    num_layers: Optional[int],
    architecture: Optional[str],
) -> Dict[str, Any]:
    """Compute per-bucket and total sweep estimates."""
    heuristic = None
    heuristic_str = None
    if num_layers:
        is_moe = architecture and any(
            k in (architecture or "").lower() for k in ["moe", "mixture", "expert"]
        )
        factor = 15 if is_moe else 12
        heuristic = num_layers * factor
        heuristic_str = f"{num_layers} layers x {factor} = {heuristic}"

        ratio = max(kernels_per_step, 1) / max(heuristic, 1)
        if ratio > 2.0 or ratio < 0.5:
            print(
                f"\nWARNING: Measured kernels/step ({kernels_per_step:,}) differs from "
                f"heuristic ({heuristic:,}) by {ratio:.1f}x. "
                "Possible transient overhead inflation. Using lower value.",
                file=sys.stderr,
            )
            kernels_per_step = min(kernels_per_step, heuristic)

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

    suggested_timeout = max(600, int(total_time * 60 * 1.5))

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
        f"| TOTAL SWEEP ({ts['num_buckets']} buckets) | - | "
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

    if sa["--nsys-output-len"] <= 4:
        lines.append("")
        lines.append(
            f"WARNING: nsys_OL <= {sa['--nsys-output-len']} due to high kernel count x TP."
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

    run_prewarm(
        cfg=cfg,
        probe_bs=probe_bs,
        artifact_dir=artifact_dir,
        timeout_s=args.prewarm_timeout_s,
    )

    nsys_rep = run_nsys_probe(
        cfg=cfg,
        probe_bs=probe_bs,
        artifact_dir=artifact_dir,
        timeout_s=args.probe_timeout_s,
    )

    kernels_per_step = parse_kernel_count(nsys_rep)

    estimates = compute_estimates(
        kernels_per_step=kernels_per_step,
        tp=cfg["tp"],
        batch_sizes=cfg["batch_sizes"],
        num_layers=cfg.get("num_layers"),
        architecture=cfg.get("architecture"),
    )

    output = format_output(
        model_id=cfg["model_id"],
        tp=cfg["tp"],
        probe_bs=probe_bs,
        estimates=estimates,
    )
    print(output)

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


if __name__ == "__main__":
    main()
