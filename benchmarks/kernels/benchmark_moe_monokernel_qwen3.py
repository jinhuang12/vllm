"""
Benchmark for MoE Monokernel vs FusedMoE on Qwen3-Coder-30B-A3B-Instruct-FP8

Compares the optimized monokernel implementation against the baseline FusedMoE.
Target: Qwen3-Coder-30B-A3B on g6e.24xlarge (4x L40S, TP=4)
"""

import argparse
import json
import os
import torch

from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

# Try importing custom ops for monokernel
try:
    from vllm import _custom_ops as ops
    HAS_MONOKERNEL = ops.moe_monokernel_supported()
except (ImportError, AttributeError):
    HAS_MONOKERNEL = False

FP8_DTYPE = current_platform.fp8_dtype()

# Qwen3-Coder-30B-A3B dimensions
HIDDEN_SIZE = 2048  # K
INTERMEDIATE_SIZE = 768  # N
NUM_EXPERTS = 128  # E
TOP_K = 8


def create_test_tensors(batch_size: int, device: str = "cuda"):
    """Create test tensors matching Qwen3-30B-A3B MoE dimensions."""

    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    ) / 10.0

    router_logits = torch.randn(
        batch_size, NUM_EXPERTS, dtype=torch.bfloat16, device=device
    )

    w13 = torch.randn(
        NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
        dtype=torch.float32, device=device
    ).to(FP8_DTYPE)

    w2 = torch.randn(
        NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        dtype=torch.float32, device=device
    ).to(FP8_DTYPE)

    # Block quantization scales (128x128 blocks)
    # w13: [E, 2*N, K] -> scale shape [E, ceil(2*N/128), ceil(K/128)] = [E, 12, 16]
    # w2: [E, K, N] -> scale shape [E, ceil(K/128), ceil(N/128)] = [E, 16, 6]
    BLOCK_SIZE = 128
    w13_scale_rows = (2 * INTERMEDIATE_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 12
    w13_scale_cols = (HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 16
    w2_scale_rows = (HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 16
    w2_scale_cols = (INTERMEDIATE_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 6

    w13_scale = torch.ones(NUM_EXPERTS, w13_scale_rows, w13_scale_cols,
                           dtype=torch.float32, device=device)
    w2_scale = torch.ones(NUM_EXPERTS, w2_scale_rows, w2_scale_cols,
                          dtype=torch.float32, device=device)
    a1_scale = torch.ones(1, dtype=torch.float32, device=device)
    a2_scale = torch.ones(1, dtype=torch.float32, device=device)

    return (hidden_states, router_logits, w13, w2,
            w13_scale, w2_scale, a1_scale, a2_scale)


def benchmark_fused_moe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict:
    """Benchmark baseline FusedMoE."""

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, router_logits, TOP_K, renormalize=True
    )

    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=FP8_DTYPE,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=[128, 128],
    )

    # Warmup
    for _ in range(num_warmup):
        out = fused_experts(
            hidden_states, w13, w2, topk_weights, topk_ids,
            inplace=False, quant_config=quant_config,
        )

    torch.cuda.synchronize()

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(num_iters):
        start_event.record()
        out = fused_experts(
            hidden_states, w13, w2, topk_weights, topk_ids,
            inplace=False, quant_config=quant_config,
        )
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    latencies = sorted(latencies)
    return {
        "mean_ms": sum(latencies) / len(latencies),
        "median_ms": latencies[len(latencies) // 2],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p90_ms": latencies[int(0.9 * len(latencies))],
        "p99_ms": latencies[int(0.99 * len(latencies))],
    }


def benchmark_monokernel(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict:
    """Benchmark MoE Monokernel with block quantization."""

    if not HAS_MONOKERNEL:
        return None

    batch_size = hidden_states.size(0)

    # Allocate scratchpad for block quantization
    scratchpad_size = ops.moe_monokernel_qwen3_block_quant_scratchpad_size(batch_size)
    scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

    # Allocate output
    output = torch.empty_like(hidden_states)

    # Warmup
    for _ in range(num_warmup):
        ops.moe_monokernel_qwen3_block_quant(
            hidden_states, router_logits, w13, w13_scale,
            w2, w2_scale, output, scratchpad
        )

    torch.cuda.synchronize()

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(num_iters):
        start_event.record()
        ops.moe_monokernel_qwen3_block_quant(
            hidden_states, router_logits, w13, w13_scale,
            w2, w2_scale, output, scratchpad
        )
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    latencies = sorted(latencies)
    return {
        "mean_ms": sum(latencies) / len(latencies),
        "median_ms": latencies[len(latencies) // 2],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p90_ms": latencies[int(0.9 * len(latencies))],
        "p99_ms": latencies[int(0.99 * len(latencies))],
    }


def main():
    parser = argparse.ArgumentParser(
        description="MoE Monokernel vs FusedMoE Benchmark for Qwen3-Coder-30B-A3B"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=10,
        help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=100,
        help="Number of timed iterations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/tmp/qwen3_moe_monokernel_benchmark",
        help="Output directory for results"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("Qwen3-Coder-30B-A3B MoE Monokernel Benchmark")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Hidden Size (K): {HIDDEN_SIZE}")
    print(f"Intermediate Size (N): {INTERMEDIATE_SIZE}")
    print(f"Num Experts (E): {NUM_EXPERTS}")
    print(f"Top-K: {TOP_K}")
    print(f"Monokernel Available: {HAS_MONOKERNEL}")
    print(f"Batch Sizes: {args.batch_sizes}")
    print("=" * 80 + "\n")

    results = []

    for bs in args.batch_sizes:
        print(f"\nBenchmarking batch_size={bs}...")

        tensors = create_test_tensors(bs)
        (hidden_states, router_logits, w13, w2,
         w13_scale, w2_scale, a1_scale, a2_scale) = tensors

        # Benchmark FusedMoE (baseline)
        baseline_stats = benchmark_fused_moe(
            hidden_states, router_logits, w13, w2,
            w13_scale, w2_scale, a1_scale, a2_scale,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )

        print(f"  FusedMoE (baseline):")
        print(f"    Mean: {baseline_stats['mean_ms']:.3f} ms")
        print(f"    Median: {baseline_stats['median_ms']:.3f} ms")

        # Benchmark Monokernel if available
        monokernel_stats = None
        speedup = None
        if HAS_MONOKERNEL:
            monokernel_stats = benchmark_monokernel(
                hidden_states, router_logits, w13, w2,
                w13_scale, w2_scale,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )

            if monokernel_stats:
                speedup = baseline_stats['mean_ms'] / monokernel_stats['mean_ms']
                print(f"  Monokernel:")
                print(f"    Mean: {monokernel_stats['mean_ms']:.3f} ms")
                print(f"    Median: {monokernel_stats['median_ms']:.3f} ms")
                print(f"    Speedup: {speedup:.2f}x")

        results.append({
            "batch_size": bs,
            "baseline": baseline_stats,
            "monokernel": monokernel_stats,
            "speedup": speedup,
        })

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Latency Comparison (ms)")
    print("=" * 80)
    print(f"{'BS':>8} {'FusedMoE':>12} {'Monokernel':>12} {'Speedup':>10}")
    print("-" * 44)
    for r in results:
        baseline_mean = r['baseline']['mean_ms']
        mono_mean = r['monokernel']['mean_ms'] if r['monokernel'] else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r['speedup'] else "N/A"

        if isinstance(mono_mean, float):
            print(f"{r['batch_size']:>8} {baseline_mean:>12.3f} {mono_mean:>12.3f} {speedup_str:>10}")
        else:
            print(f"{r['batch_size']:>8} {baseline_mean:>12.3f} {mono_mean:>12} {speedup_str:>10}")

    # Save results
    os.makedirs(args.output, exist_ok=True)
    output_file = f"{args.output}/monokernel_benchmark.json"
    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "hidden_size": HIDDEN_SIZE,
                "intermediate_size": INTERMEDIATE_SIZE,
                "num_experts": NUM_EXPERTS,
                "top_k": TOP_K,
                "monokernel_available": HAS_MONOKERNEL,
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
