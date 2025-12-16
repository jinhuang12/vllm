#!/usr/bin/env python3
"""
Nsight Compute profiling wrapper for MoE monokernel.

This script provides a minimal wrapper for profiling the MoE monokernel
with NVIDIA Nsight Compute (ncu). It creates inputs matching the Qwen3-Coder-30B-A3B
model dimensions and invokes the monokernel for profiling.

Usage:
    # Profile with full metrics (recommended for cooperative kernels):
    ncu --replay-mode application \
        --set full \
        --clock-control base \
        --kernel-name regex:moe_kernel \
        --launch-skip 3 --launch-count 1 \
        -o moe_monokernel_bs64 \
        python benchmarks/kernels/ncu_profile_moe_monokernel.py --batch-size 64

    # Profile specific GEMM metrics:
    ncu --replay-mode application \
        --metrics \
        sm__inst_executed_pipe_tensor.avg.pct_of_peak_sustained_active,\
        sm__throughput.avg.pct_of_peak_sustained_elapsed,\
        dram__throughput.avg.pct_of_peak_sustained_elapsed,\
        sm__warps_active.avg.pct_of_peak_sustained_active,\
        smsp__warp_issue_stall_barrier.pct,\
        smsp__warp_issue_stall_membar.pct,\
        smsp__warp_issue_stall_wait.pct,\
        lts__average_hit_rate.pct \
        --kernel-name regex:moe_kernel \
        --launch-skip 3 --launch-count 1 \
        python benchmarks/kernels/ncu_profile_moe_monokernel.py --batch-size 64

    # Compare BS=1 vs BS=64:
    ncu --replay-mode application --set full --clock-control base \
        --kernel-name regex:moe_kernel --launch-skip 3 --launch-count 1 \
        -o moe_monokernel_bs1 \
        python benchmarks/kernels/ncu_profile_moe_monokernel.py --batch-size 1

Target metrics for GEMM optimality assessment:
    - Tensor Core utilization (expect ~1-3% based on roofline analysis)
    - Memory bandwidth utilization
    - Warp stall reasons (likely memory or synchronization)
    - L2 cache hit rate (for weight/activation locality)

Key Findings from Roofline Analysis:
    - FP8 Tensor Core Peak: ~362 TFLOPS (L40S)
    - Achieved: ~1.5% efficiency at BS=64
    - Root causes: Small M=8 tiles, sequential expert processing

LLM Council Approved: 2025-12-09
"""

import argparse
import torch

# Check for CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. NCU profiling requires a GPU.")


def profile_monokernel(batch_size: int, warmup_iters: int = 3, profile_iters: int = 1):
    """
    Run the MoE monokernel for NCU profiling.

    Args:
        batch_size: Number of tokens to process
        warmup_iters: Number of warmup iterations before profiling
        profile_iters: Number of iterations to profile
    """
    # Import here to allow script to be run without vllm installed (for help text)
    from vllm import _custom_ops as ops
    from vllm.platforms import current_platform
    fp8_dtype = current_platform.fp8_dtype()

    # Qwen3-Coder-30B-A3B dimensions
    HIDDEN = 2048  # Hidden size
    INTERMEDIATE = 768  # Intermediate size (after TP split)
    TOP_K = 8  # Top-k experts per token
    NUM_EXPERTS = 128  # Total number of experts

    print(f"NCU Profiling MoE Monokernel")
    print(f"=" * 60)
    print(f"Batch Size: {batch_size}")
    print(f"Hidden: {HIDDEN}")
    print(f"Intermediate: {INTERMEDIATE}")
    print(f"Top-K: {TOP_K}")
    print(f"Experts: {NUM_EXPERTS}")
    print(f"Total pairs: {batch_size * TOP_K}")
    print(f"Warmup iters: {warmup_iters}")
    print(f"Profile iters: {profile_iters}")
    print(f"=" * 60)

    # Create inputs
    # Hidden states (BF16 input)
    x = torch.randn(batch_size, HIDDEN, device="cuda", dtype=torch.bfloat16) / 10

    # Router logits (BF16)
    router_logits = torch.randn(
        batch_size, NUM_EXPERTS, device="cuda", dtype=torch.bfloat16
    )

    # Up-projection weights (FP8): [E, 2*N, K] - gate and x combined
    w13 = torch.randn(
        NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", dtype=torch.float32
    ).to(fp8_dtype)
    # Block quantization scales (128x128 blocks): [E, ceil(2N/128)=12, ceil(K/128)=16]
    w13_scale = torch.ones(NUM_EXPERTS, 12, 16, device="cuda", dtype=torch.float32)

    # Down-projection weights (FP8): [E, K, N]
    w2 = torch.randn(
        NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", dtype=torch.float32
    ).to(fp8_dtype)
    # Block quantization scales (128x128 blocks): [E, ceil(K/128)=16, ceil(N/128)=6]
    w2_scale = torch.ones(NUM_EXPERTS, 16, 6, device="cuda", dtype=torch.float32)

    # Output tensor (BF16)
    output = torch.empty(batch_size, HIDDEN, device="cuda", dtype=torch.bfloat16)

    # Scratchpad for intermediate data
    scratchpad_size = ops.moe_monokernel_qwen3_block_quant_scratchpad_size(batch_size)
    scratchpad = torch.empty(scratchpad_size, device="cuda", dtype=torch.uint8)

    print(f"Input shapes:")
    print(f"  x: {x.shape} ({x.dtype})")
    print(f"  router_logits: {router_logits.shape} ({router_logits.dtype})")
    print(f"  w13: {w13.shape} ({w13.dtype})")
    print(f"  w2: {w2.shape} ({w2.dtype})")
    print(f"  output: {output.shape} ({output.dtype})")
    print(f"  scratchpad: {scratchpad.numel() // 1024 // 1024} MB")
    print()

    # Warmup iterations (NCU will skip these with --launch-skip)
    print(f"Running {warmup_iters} warmup iterations...")
    for i in range(warmup_iters):
        ops.moe_monokernel_qwen3_block_quant(
            x, router_logits, w13, w13_scale, w2, w2_scale, output, scratchpad
        )
        torch.cuda.synchronize()

    # Profile iterations (NCU will capture these with --launch-count)
    print(f"Running {profile_iters} profile iterations...")
    for i in range(profile_iters):
        ops.moe_monokernel_qwen3_block_quant(
            x, router_logits, w13, w13_scale, w2, w2_scale, output, scratchpad
        )
        torch.cuda.synchronize()

    print("Profiling complete.")
    print()
    print("To analyze results:")
    print(
        "  ncu --import moe_monokernel_bs{}.ncu-rep --print-summary per-kernel".format(
            batch_size
        )
    )
    print(
        "  ncu --import moe_monokernel_bs{}.ncu-rep --csv > moe_bs{}.csv".format(
            batch_size, batch_size
        )
    )


def main():
    parser = argparse.ArgumentParser(
        description="NCU profiling wrapper for MoE monokernel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example NCU commands:

  # Full profile (recommended for first analysis):
  ncu --replay-mode application --set full --clock-control base \\
      --kernel-name regex:moe_kernel --launch-skip 3 --launch-count 1 \\
      -o moe_monokernel_bs64 \\
      python %(prog)s --batch-size 64

  # Quick profile (specific metrics only):
  ncu --replay-mode application --clock-control base \\
      --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed \\
      --kernel-name regex:moe_kernel --launch-skip 3 --launch-count 1 \\
      python %(prog)s --batch-size 64

Key metrics to look for:
  - sm__inst_executed_pipe_tensor: Tensor Core utilization
  - sm__throughput: Overall SM compute efficiency
  - dram__throughput: Memory bandwidth utilization
  - smsp__warp_issue_stall_*: Warp stall reasons
  - lts__average_hit_rate: L2 cache efficiency
""",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Number of tokens (batch size). Default: 64",
    )
    parser.add_argument(
        "--warmup-iters",
        "-w",
        type=int,
        default=3,
        help="Number of warmup iterations. Default: 3",
    )
    parser.add_argument(
        "--profile-iters",
        "-p",
        type=int,
        default=1,
        help="Number of iterations to profile. Default: 1",
    )

    args = parser.parse_args()

    profile_monokernel(
        batch_size=args.batch_size,
        warmup_iters=args.warmup_iters,
        profile_iters=args.profile_iters,
    )


if __name__ == "__main__":
    main()
