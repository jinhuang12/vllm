"""
Per-stage MoE profiling: Reference vs Monokernel

Measures individual stage latencies to identify bottlenecks in each implementation.
Target: Qwen3-Coder-30B-A3B on L40S

Stages measured:
- Reference: routing, sort, quant1, up_proj, silu, quant2, down_proj, sum
- Monokernel: total (internal stages not separable without code modification)
"""

import sys
import os

# Debug: print which vllm we're importing from
# print("sys.path before imports:")
# for p in sys.path[:5]:
#     print(f"  {p}")

import argparse
import torch
from typing import Dict, List, Tuple

import triton.language as tl

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, invoke_fused_moe_kernel,
    moe_kernel_quantize_input
)
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm._custom_ops import moe_sum
from vllm.platforms import current_platform

# Try importing custom ops for monokernel
try:
    from vllm import _custom_ops as ops
    HAS_MONOKERNEL = ops.moe_monokernel_supported()
except (ImportError, AttributeError) as e:
    print(f"Warning: Monokernel import failed: {e}")
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

    # W13 shape: [E, 2*N, K] for up-projection (produces x and gate)
    w13 = torch.randn(
        NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, HIDDEN_SIZE,
        dtype=torch.float32, device=device
    ).to(FP8_DTYPE)

    # W2 shape: [E, K, N] for down-projection
    w2 = torch.randn(
        NUM_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE,
        dtype=torch.float32, device=device
    ).to(FP8_DTYPE)

    # Scales
    w13_scale = torch.ones(NUM_EXPERTS, 2 * INTERMEDIATE_SIZE, dtype=torch.float32, device=device)
    w2_scale = torch.ones(NUM_EXPERTS, HIDDEN_SIZE, dtype=torch.float32, device=device)
    a1_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    a2_scale = torch.tensor(1.0, dtype=torch.float32, device=device)

    return (hidden_states, router_logits, w13, w2,
            w13_scale, w2_scale, a1_scale, a2_scale)


def profile_reference_stages(
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
) -> Dict[str, Tuple[float, float, float]]:
    """
    Profile each stage of reference MoE implementation.

    Returns dict mapping stage name -> (avg_ms, min_ms, max_ms)
    """
    M, K = hidden_states.shape
    N = INTERMEDIATE_SIZE
    E = NUM_EXPERTS

    # Pre-allocate intermediate buffers
    intermediate_cache1 = torch.empty(M, TOP_K, 2 * N, dtype=torch.float32, device="cuda")
    intermediate_cache2 = torch.empty(M * TOP_K, N, dtype=torch.float32, device="cuda")
    intermediate_cache3 = torch.empty(M, TOP_K, K, dtype=torch.float32, device="cuda")
    output = torch.empty(M, K, dtype=torch.bfloat16, device="cuda")

    # Collect timings for each stage
    stages = ['routing', 'sort', 'quant1', 'up_proj', 'silu', 'quant2', 'down_proj', 'sum']
    timings = {s: [] for s in stages}

    # Get compute_type for invoke_fused_moe_kernel (Triton type based on hidden_states dtype)
    compute_type = tl.bfloat16  # hidden_states is BF16

    # Warmup
    for _ in range(num_warmup):
        topk_weights, topk_ids, _ = fused_topk(hidden_states, router_logits, TOP_K, renormalize=True)
        sorted_token_ids, expert_ids, num_tokens_post = moe_align_block_size(
            topk_ids, 64, E, None
        )
    torch.cuda.synchronize()

    for _ in range(num_iters):
        events = {s: (torch.cuda.Event(enable_timing=True),
                      torch.cuda.Event(enable_timing=True))
                  for s in stages}

        # Stage 1: Routing (topk selection + softmax normalization)
        events['routing'][0].record()
        topk_weights, topk_ids, _ = fused_topk(hidden_states, router_logits, TOP_K, renormalize=True)
        events['routing'][1].record()

        # Stage 2: Sort (align block size for efficient expert processing)
        events['sort'][0].record()
        sorted_token_ids, expert_ids, num_tokens_post = moe_align_block_size(
            topk_ids, 64, E, None
        )
        events['sort'][1].record()

        # Stage 3: Quantize input activations (BF16 -> FP8)
        events['quant1'][0].record()
        qhidden, a1q_scale = moe_kernel_quantize_input(
            A=hidden_states,
            A_scale=a1_scale,
            quant_dtype=FP8_DTYPE,
            per_act_token_quant=False,
            block_shape=None,
        )
        events['quant1'][1].record()

        # Stage 4: Up-projection GEMM (W13 @ activations)
        events['up_proj'][0].record()
        invoke_fused_moe_kernel(
            qhidden,
            w13,
            intermediate_cache1,
            a1q_scale,
            w13_scale,
            None,  # w1_zp
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post,
            False,  # apply_router_weight_on_input
            TOP_K,
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        events['up_proj'][1].record()

        # Stage 5: SiLU activation with gate multiplication
        events['silu'][0].record()
        torch.ops._C.silu_and_mul(intermediate_cache2, intermediate_cache1.view(-1, 2 * N))
        events['silu'][1].record()

        # Stage 6: Quantize intermediate activations
        events['quant2'][0].record()
        qintermediate, a2q_scale = moe_kernel_quantize_input(
            A=intermediate_cache2,
            A_scale=a2_scale,
            quant_dtype=FP8_DTYPE,
            per_act_token_quant=False,
            block_shape=None,
        )
        events['quant2'][1].record()

        # Stage 7: Down-projection GEMM (W2 @ intermediate)
        events['down_proj'][0].record()
        invoke_fused_moe_kernel(
            qintermediate,
            w2,
            intermediate_cache3,
            a2q_scale,
            w2_scale,
            None,  # w2_zp
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post,
            True,  # apply_router_weight_on_input (apply during down-proj)
            1,
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
             "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=None,
        )
        events['down_proj'][1].record()

        # Stage 8: Sum across top-k experts
        events['sum'][0].record()
        moe_sum(intermediate_cache3, output)
        events['sum'][1].record()

        torch.cuda.synchronize()

        for s in stages:
            timings[s].append(events[s][0].elapsed_time(events[s][1]))

    # Compute statistics
    results = {}
    for s in stages:
        t = timings[s]
        results[s] = (sum(t) / len(t), min(t), max(t))

    return results


def profile_monokernel(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> Tuple[Tuple[float, float, float], Dict[str, Tuple[float, float, float]]]:
    """
    Profile monokernel with per-stage breakdown using internal clock64() timing.

    Returns:
        (total_timing, stage_timings)
        - total_timing: (avg_ms, min_ms, max_ms) from CUDA events
        - stage_timings: dict of stage -> (avg_ms, min_ms, max_ms) from clock64()
    """
    if not HAS_MONOKERNEL:
        return None, None

    batch_size = hidden_states.size(0)

    # Allocate scratchpad
    scratchpad_size = ops.moe_monokernel_qwen3_scratchpad_size(batch_size)
    scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")

    # Allocate output
    output = torch.empty_like(hidden_states)

    # L40S GPU frequency for converting clock64() cycles to milliseconds
    GPU_FREQ_GHZ = 2.505  # L40S SM clock frequency
    CYCLE_TO_MS = 1.0 / (GPU_FREQ_GHZ * 1e6)

    # Stage names for timing indices
    stage_names = ['routing', 'prepare', 'quantize', 'grid_sync_1',
                   'up_proj', 'grid_sync_2', 'down_proj', 'grid_sync_3', 'convert']

    # Warmup
    for _ in range(num_warmup):
        ops.moe_monokernel_qwen3(
            hidden_states, router_logits, w13, w13_scale,
            w2, w2_scale, output, scratchpad
        )
    torch.cuda.synchronize()

    # Timed iterations
    total_timings = []
    stage_timings = {s: [] for s in stage_names}

    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        ops.moe_monokernel_qwen3(
            hidden_states, router_logits, w13, w13_scale,
            w2, w2_scale, output, scratchpad
        )
        end.record()

        torch.cuda.synchronize()
        total_timings.append(start.elapsed_time(end))

        # Read internal clock64() timing from scratchpad
        timing = ops.moe_monokernel_get_timing(scratchpad, batch_size)

        # Convert cycle differences to milliseconds
        # timing[0]=kernel_start, timing[1]=routing_end, ..., timing[9]=kernel_end
        stage_timings['routing'].append((timing[1].item() - timing[0].item()) * CYCLE_TO_MS)
        stage_timings['prepare'].append((timing[2].item() - timing[1].item()) * CYCLE_TO_MS)
        stage_timings['quantize'].append((timing[3].item() - timing[2].item()) * CYCLE_TO_MS)
        stage_timings['grid_sync_1'].append((timing[4].item() - timing[3].item()) * CYCLE_TO_MS)
        stage_timings['up_proj'].append((timing[5].item() - timing[4].item()) * CYCLE_TO_MS)
        stage_timings['grid_sync_2'].append((timing[6].item() - timing[5].item()) * CYCLE_TO_MS)
        stage_timings['down_proj'].append((timing[7].item() - timing[6].item()) * CYCLE_TO_MS)
        stage_timings['grid_sync_3'].append((timing[8].item() - timing[7].item()) * CYCLE_TO_MS)
        stage_timings['convert'].append((timing[9].item() - timing[8].item()) * CYCLE_TO_MS)

    # Compute statistics
    total_result = (sum(total_timings) / len(total_timings),
                    min(total_timings), max(total_timings))

    stage_results = {}
    for s in stage_names:
        t = stage_timings[s]
        stage_results[s] = (sum(t) / len(t), min(t), max(t))

    return total_result, stage_results


def print_results(batch_size: int, ref_results: Dict,
                  mono_total: Tuple, mono_stages: Dict):
    """Print formatted comparison table with per-stage breakdown."""
    print(f"\n{'='*70}")
    print(f"MoE Per-Stage Profiling (BS={batch_size}, E={NUM_EXPERTS}, TOP_K={TOP_K})")
    print(f"{'='*70}")

    print("\nReference Implementation (fused_experts):")
    print(f"  {'Stage':<18} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | {'% Total':>8}")
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

    total_ref = sum(r[0] for r in ref_results.values())

    for stage, (avg, min_t, max_t) in ref_results.items():
        pct = 100 * avg / total_ref if total_ref > 0 else 0
        print(f"  {stage:<18} | {avg:>10.4f} | {min_t:>10.4f} | {max_t:>10.4f} | {pct:>7.1f}%")

    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
    print(f"  {'TOTAL':<18} | {total_ref:>10.4f} | {'':<10} | {'':<10} | {'100.0%':>8}")

    if mono_total:
        avg, min_t, max_t = mono_total
        print(f"\nMonokernel (clock64() per-stage timing):")
        print(f"  {'Stage':<18} | {'Avg (ms)':>10} | {'Min (ms)':>10} | {'Max (ms)':>10} | {'% Total':>8}")
        print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")

        total_mono_stages = sum(r[0] for r in mono_stages.values())

        for stage, (s_avg, s_min, s_max) in mono_stages.items():
            pct = 100 * s_avg / total_mono_stages if total_mono_stages > 0 else 0
            # Mark the bottleneck
            marker = " <-- BOTTLENECK" if pct > 40 else ""
            print(f"  {stage:<18} | {s_avg:>10.4f} | {s_min:>10.4f} | {s_max:>10.4f} | {pct:>7.1f}%{marker}")

        print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}")
        print(f"  {'TOTAL (clock64)':<18} | {total_mono_stages:>10.4f} | {'':<10} | {'':<10} | {'100.0%':>8}")
        print(f"  {'TOTAL (CUDA evt)':<18} | {avg:>10.4f} | {min_t:>10.4f} | {max_t:>10.4f} |")

        speedup = total_ref / avg if avg > 0 else 0
        print(f"\nSpeedup: {speedup:.2f}x ", end="")
        if speedup >= 1:
            print(f"(monokernel is {speedup:.1f}x faster)")
        else:
            print(f"(monokernel is {1/speedup:.1f}x slower)")

        # Bottleneck analysis
        print("\nBottleneck Analysis:")
        up_pct_ref = 100 * ref_results['up_proj'][0] / total_ref
        down_pct_ref = 100 * ref_results['down_proj'][0] / total_ref
        print(f"  Reference: up_proj={up_pct_ref:.1f}%, down_proj={down_pct_ref:.1f}%")

        up_pct_mono = 100 * mono_stages['up_proj'][0] / total_mono_stages if total_mono_stages > 0 else 0
        down_pct_mono = 100 * mono_stages['down_proj'][0] / total_mono_stages if total_mono_stages > 0 else 0
        print(f"  Monokernel: up_proj={up_pct_mono:.1f}%, down_proj={down_pct_mono:.1f}%")

        # Grid sync overhead
        sync_pct = 100 * (mono_stages['grid_sync_1'][0] + mono_stages['grid_sync_2'][0] +
                         mono_stages['grid_sync_3'][0]) / total_mono_stages if total_mono_stages > 0 else 0
        print(f"  Grid sync overhead: {sync_pct:.1f}%")

        if speedup < 1:
            print("\n  Key bottleneck: ", end="")
            if down_pct_mono > 40:
                print("Scalar down-projection loop (N=768 iterations per thread)")
            elif sync_pct > 20:
                print("Grid synchronization barriers")
            else:
                print("Unknown - consider profiling with Nsight Compute")
    else:
        print("\nMonokernel: NOT AVAILABLE")


def main():
    parser = argparse.ArgumentParser(
        description="Per-stage MoE profiling: Reference vs Monokernel"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8, 64],
        help="Batch sizes to profile"
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
    args = parser.parse_args()

    print("="*70)
    print("MoE Per-Stage Profiling: Reference vs Monokernel")
    print("="*70)
    print(f"Model: Qwen3-Coder-30B-A3B")
    print(f"Dimensions: K={HIDDEN_SIZE}, N={INTERMEDIATE_SIZE}, E={NUM_EXPERTS}, TOP_K={TOP_K}")
    print(f"Warmup: {args.num_warmup}, Iterations: {args.num_iters}")
    print(f"Monokernel available: {HAS_MONOKERNEL}")

    for bs in args.batch_sizes:
        print(f"\n--- Profiling BS={bs} ---")

        # Create tensors
        (hidden_states, router_logits, w13, w2,
         w13_scale, w2_scale, a1_scale, a2_scale) = create_test_tensors(bs)

        # Profile reference
        ref_results = profile_reference_stages(
            hidden_states, router_logits, w13, w2,
            w13_scale, w2_scale, a1_scale, a2_scale,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
        )

        # Profile monokernel
        mono_total = None
        mono_stages = None
        if HAS_MONOKERNEL:
            mono_total, mono_stages = profile_monokernel(
                hidden_states, router_logits, w13, w2,
                w13_scale, w2_scale,
                num_warmup=args.num_warmup,
                num_iters=args.num_iters,
            )

        # Print results
        print_results(bs, ref_results, mono_total, mono_stages)

        # Clear GPU memory
        del hidden_states, router_logits, w13, w2
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
