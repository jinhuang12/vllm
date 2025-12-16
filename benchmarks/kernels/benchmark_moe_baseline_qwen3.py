"""
Baseline benchmark for FusedMoE on Qwen3-Coder-30B-A3B-Instruct-FP8

This script profiles the existing FusedMoE implementation to establish
baseline metrics before implementing the monokernel optimization.

Target: Qwen3-Coder-30B-A3B-Instruct-FP8 on g6e.24xlarge (4x L40S, TP=4)
"""

import argparse
import torch
import torch.distributed as dist
import time
from typing import Tuple

# vLLM imports
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts, fused_topk
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()


def create_test_tensors(
    batch_size: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Tuple[torch.Tensor, ...]:
    """Create test tensors matching Qwen3-30B-A3B MoE dimensions."""

    # Input hidden states: [batch_size, hidden_size]
    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=dtype, device=device
    ) / 10.0  # Scale to avoid overflow in FP8

    # Router logits: [batch_size, num_experts]
    router_logits = torch.randn(
        batch_size, num_experts, dtype=dtype, device=device
    )

    # Expert weights (FP8 for quantized model)
    # w1 (gate): [num_experts, intermediate_size, hidden_size]
    # w2 (down): [num_experts, hidden_size, intermediate_size]
    # w3 (up):   [num_experts, intermediate_size, hidden_size]
    # Combined w13: [num_experts, 2*intermediate_size, hidden_size]

    # Create FP8 tensors by converting from float
    w13 = torch.randn(
        num_experts, 2 * intermediate_size, hidden_size,
        dtype=torch.float32, device=device
    ).to(FP8_DTYPE)
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size,
        dtype=torch.float32, device=device
    ).to(FP8_DTYPE)

    # Scales for FP8 - per-tensor scales
    w13_scale = torch.ones(num_experts, 1, 1,
                           dtype=torch.float32, device=device)
    w2_scale = torch.ones(num_experts, 1, 1,
                          dtype=torch.float32, device=device)
    a1_scale = torch.ones(1, dtype=torch.float32, device=device)
    a2_scale = torch.ones(1, dtype=torch.float32, device=device)

    return (hidden_states, router_logits, w13, w2,
            w13_scale, w2_scale, a1_scale, a2_scale)


def benchmark_fused_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    top_k: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> dict:
    """Benchmark FusedMoE kernel and return timing statistics."""

    # Get topk weights and indices (returns 3 values: weights, ids, expert_map)
    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, router_logits, top_k, renormalize=True
    )

    # Create FP8 quant config using new API
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=FP8_DTYPE,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=None,  # per-tensor quantization
    )

    # Warmup
    for _ in range(num_warmup):
        out = fused_experts(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            quant_config=quant_config,
        )

    torch.cuda.synchronize()

    # Timed iterations
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    for _ in range(num_iters):
        start_event.record()
        out = fused_experts(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            quant_config=quant_config,
        )
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    # Statistics
    latencies = sorted(latencies)
    return {
        "mean_ms": sum(latencies) / len(latencies),
        "median_ms": latencies[len(latencies) // 2],
        "min_ms": min(latencies),
        "max_ms": max(latencies),
        "p90_ms": latencies[int(0.9 * len(latencies))],
        "p99_ms": latencies[int(0.99 * len(latencies))],
    }


def benchmark_with_profiler(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    top_k: int,
    output_dir: str = "/tmp/qwen3_moe_baseline_profile",
) -> None:
    """Run profiler to capture detailed kernel metrics."""

    topk_weights, topk_ids, _ = fused_topk(
        hidden_states, router_logits, top_k, renormalize=True
    )

    # Create FP8 quant config using new API
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=FP8_DTYPE,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=None,  # per-tensor quantization
    )

    # Warmup
    for _ in range(5):
        out = fused_experts(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            quant_config=quant_config,
        )
    torch.cuda.synchronize()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
        profile_memory=True,
    ) as prof:
        for _ in range(20):
            out = fused_experts(
                hidden_states,
                w13,
                w2,
                topk_weights,
                topk_ids,
                inplace=False,
                quant_config=quant_config,
            )
        torch.cuda.synchronize()

    # Export trace
    import os
    os.makedirs(output_dir, exist_ok=True)
    prof.export_chrome_trace(f"{output_dir}/moe_baseline_trace.json")

    # Print summary
    print("\n" + "="*80)
    print("CUDA Kernel Summary (sorted by total time)")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    ))

    print("\n" + "="*80)
    print("Memory Summary")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Baseline MoE benchmark for Qwen3-Coder-30B-A3B"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch sizes to test"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=2048,
        help="Hidden size (K) - Qwen3-30B uses 2048"
    )
    parser.add_argument(
        "--intermediate-size",
        type=int,
        default=768,  # moe_intermediate_size for Qwen3-30B-A3B
        help="MoE intermediate size (N) - Qwen3-30B-A3B uses 768"
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=128,
        help="Number of experts (E) - Qwen3-30B-A3B uses 128"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Top-K experts per token - Qwen3-30B-A3B uses 8"
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
        "--profile",
        action="store_true",
        help="Run with PyTorch profiler"
    )
    parser.add_argument(
        "--profile-output",
        type=str,
        default="/tmp/qwen3_moe_baseline_profile",
        help="Output directory for profiler traces"
    )
    args = parser.parse_args()

    print("="*80)
    print("Qwen3-Coder-30B-A3B MoE Baseline Benchmark")
    print("="*80)
    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Hidden Size (K): {args.hidden_size}")
    print(f"Intermediate Size (N): {args.intermediate_size}")
    print(f"Num Experts (E): {args.num_experts}")
    print(f"Top-K: {args.top_k}")
    print(f"Batch Sizes: {args.batch_sizes}")
    print("="*80 + "\n")

    results = []

    for bs in args.batch_sizes:
        print(f"\nBenchmarking batch_size={bs}...")

        tensors = create_test_tensors(
            batch_size=bs,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            num_experts=args.num_experts,
            top_k=args.top_k,
        )

        if args.profile and bs == args.batch_sizes[-1]:
            # Only profile the largest batch size
            benchmark_with_profiler(
                *tensors,
                top_k=args.top_k,
                output_dir=args.profile_output,
            )

        stats = benchmark_fused_experts(
            *tensors,
            top_k=args.top_k,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )

        results.append({
            "batch_size": bs,
            **stats
        })

        print(f"  Mean: {stats['mean_ms']:.3f} ms")
        print(f"  Median: {stats['median_ms']:.3f} ms")
        print(f"  Min/Max: {stats['min_ms']:.3f} / {stats['max_ms']:.3f} ms")
        print(f"  P90/P99: {stats['p90_ms']:.3f} / {stats['p99_ms']:.3f} ms")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: FusedMoE Baseline Latency (ms)")
    print("="*80)
    print(f"{'Batch Size':>12} {'Mean':>10} {'Median':>10} {'P90':>10} {'P99':>10}")
    print("-"*54)
    for r in results:
        print(f"{r['batch_size']:>12} {r['mean_ms']:>10.3f} {r['median_ms']:>10.3f} "
              f"{r['p90_ms']:>10.3f} {r['p99_ms']:>10.3f}")

    # Save results
    import json
    output_file = f"{args.profile_output}/baseline_results.json"
    import os
    os.makedirs(args.profile_output, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump({
            "config": {
                "hidden_size": args.hidden_size,
                "intermediate_size": args.intermediate_size,
                "num_experts": args.num_experts,
                "top_k": args.top_k,
            },
            "results": results
        }, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
