#!/usr/bin/env python3
"""
AMMO Kernel Benchmark Template (Gate 5.2)

Independent kernel benchmark for adversarial validation. The kernel validation
sub-agent (or champion's delegate) adapts this template — filling in
imports, tensor shapes, and kernel invocations.

METHODOLOGY (non-negotiable):
- Both baseline and optimized kernels captured in CUDA graphs
- Uses getCurrentCUDAStream (not default stream)
- All batch sizes from target.json tested
- Both warm-cache and cold-cache timings reported
- Output is structured JSON with raw microsecond timings (no speedup computation)

USAGE:
  python benchmark_kernel.py \
    --artifact-dir <path> \
    --op-id <op_id> \
    --output <path/to/gate_5_2_results.json>

Fill in the sections marked FILL below.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.cuda

# ============================================================
# FILL: Import baseline and optimized kernels
# ============================================================
# Example for MoE:
#   from vllm.model_executor.layers.fused_moe import fused_experts as baseline_fn
#   from vllm.model_executor.layers.fused_moe.fused_moe_custom import fused_experts_opt as optimized_fn
#
# The baseline MUST be vLLM's production kernel (not naive PyTorch).
# See references/validation-defaults.md for valid baseline imports.
# ============================================================


def load_target_config(artifact_dir: str) -> dict:
    """Load target.json for batch sizes and GPU config."""
    target_path = Path(artifact_dir) / "target.json"
    with open(target_path) as f:
        return json.load(f)


def create_input_tensors(batch_size: int, config: dict, device: str = "cuda"):
    """
    FILL: Create input tensors for the target kernel.

    Use the kernel's expected input shapes based on the model config.
    Example for MoE:
        hidden_size = config.get("hidden_size", 4096)
        x = torch.randn(batch_size, hidden_size, device=device, dtype=torch.bfloat16)
        w1 = torch.randn(num_experts, intermediate_size, hidden_size, ...)
        ...
        return {"x": x, "w1": w1, "w2": w2, ...}
    """
    raise NotImplementedError("Champion must fill in create_input_tensors()")


def run_baseline(inputs: dict):
    """
    FILL: Run the vLLM production baseline kernel.

    Example:
        return baseline_fn(inputs["x"], inputs["w1"], inputs["w2"], ...)
    """
    raise NotImplementedError("Champion must fill in run_baseline()")


def run_optimized(inputs: dict):
    """
    FILL: Run the optimized kernel.

    Example:
        return optimized_fn(inputs["x"], inputs["w1"], inputs["w2"], ...)
    """
    raise NotImplementedError("Champion must fill in run_optimized()")


def benchmark_under_cuda_graph(
    fn,
    inputs: dict,
    warmup_iters: int = 10,
    measure_iters: int = 100,
    cold_cache: bool = False,
    l2_cache_size_bytes: int = 0,
) -> float:
    """
    Benchmark a kernel under CUDA graph capture and replay.

    Returns average time in microseconds.

    For cold-cache mode, allocates L2-busting tensors between measurements
    to flush the cache hierarchy.
    """
    stream = torch.cuda.current_stream()

    # Capture the kernel in a CUDA graph
    # Warmup run (outside graph) to trigger JIT compilation
    for _ in range(3):
        fn(inputs)
    torch.cuda.synchronize()

    # Capture graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fn(inputs)

    # Warmup graph replays
    for _ in range(warmup_iters):
        graph.replay()
    torch.cuda.synchronize()

    if cold_cache and l2_cache_size_bytes > 0:
        # Cold-cache: flush L2 between each measurement
        # Allocate a tensor > 2.5x L2 cache to bust the cache
        flush_size = int(l2_cache_size_bytes * 2.5)
        flush_elements = flush_size // 4  # float32
        flush_tensor = torch.empty(flush_elements, device="cuda", dtype=torch.float32)

        times = []
        for _ in range(measure_iters):
            # Bust L2 cache
            flush_tensor.fill_(1.0)
            torch.cuda.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graph.replay()
            end.record(stream)
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)  # ms -> us

        del flush_tensor
        return sum(times) / len(times)
    else:
        # Warm-cache: tight replay loop
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream)
        for _ in range(measure_iters):
            graph.replay()
        end.record(stream)
        torch.cuda.synchronize()

        total_us = start.elapsed_time(end) * 1000  # ms -> us
        return total_us / measure_iters


def get_l2_cache_size() -> int:
    """Get L2 cache size in bytes for the current GPU."""
    props = torch.cuda.get_device_properties(0)
    return props.l2_cache_size


def main():
    parser = argparse.ArgumentParser(description="AMMO Gate 5.2 Kernel Benchmark")
    parser.add_argument("--artifact-dir", required=True, help="Campaign artifact directory")
    parser.add_argument("--op-id", required=True, help="Optimization track ID")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=100)
    args = parser.parse_args()

    config = load_target_config(args.artifact_dir)
    batch_sizes = config.get("workload", {}).get("batch_sizes", [1, 8, 32])
    l2_cache_bytes = get_l2_cache_size()

    device = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}"
    torch.cuda.set_device(device)

    results = {
        "op_id": args.op_id,
        "gpu_model": torch.cuda.get_device_name(0),
        "l2_cache_bytes": l2_cache_bytes,
        "cuda_graphs_used": True,
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "batch_sizes": {},
    }

    for bs in batch_sizes:
        print(f"\n--- Batch size {bs} ---")
        inputs = create_input_tensors(bs, config, device=device)

        # Warm-cache baseline
        baseline_warm_us = benchmark_under_cuda_graph(
            run_baseline, inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            cold_cache=False,
        )

        # Cold-cache baseline
        baseline_cold_us = benchmark_under_cuda_graph(
            run_baseline, inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            cold_cache=True,
            l2_cache_size_bytes=l2_cache_bytes,
        )

        # Warm-cache optimized
        opt_warm_us = benchmark_under_cuda_graph(
            run_optimized, inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            cold_cache=False,
        )

        # Cold-cache optimized
        opt_cold_us = benchmark_under_cuda_graph(
            run_optimized, inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
            cold_cache=True,
            l2_cache_size_bytes=l2_cache_bytes,
        )

        results["batch_sizes"][str(bs)] = {
            "baseline_warm_us": round(baseline_warm_us, 2),
            "baseline_cold_us": round(baseline_cold_us, 2),
            "optimized_warm_us": round(opt_warm_us, 2),
            "optimized_cold_us": round(opt_cold_us, 2),
        }

        print(f"  Baseline: warm={baseline_warm_us:.2f}us, cold={baseline_cold_us:.2f}us")
        print(f"  Optimized: warm={opt_warm_us:.2f}us, cold={opt_cold_us:.2f}us")

    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
