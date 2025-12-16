"""
Per-stage MoE profiling: FusedMoE vs Monokernel (Qwen3, FP8 block quant).

This script is a bottleneck finder for the small-batch (BS=1/2/4) gap under
CUDA graphs. It reports:
- Baseline: `topk_softmax`, `fused_experts`, and total (CUDA graph, per-op)
- Monokernel: total + internal per-stage timing (clock64 deltas, scaled)

Run with the repo venv:
  `./.venv/bin/python benchmarks/kernels/profile_moe_stages.py --batch-sizes 1 2 --use-cuda-graph`
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.platforms import current_platform

try:
    from vllm import _custom_ops as ops
    HAS_MONOKERNEL = ops.moe_monokernel_supported()
except (ImportError, AttributeError):
    HAS_MONOKERNEL = False

FP8_DTYPE = current_platform.fp8_dtype()

# Qwen3-Coder-30B-A3B dimensions (TP=1 shapes for kernel benchmarks)
HIDDEN_SIZE = 2048  # K
INTERMEDIATE_SIZE = 768  # N
NUM_EXPERTS = 128  # E
TOP_K = 8


def create_test_tensors(batch_size: int, device: str = "cuda"):
    hidden_states = torch.randn(
        batch_size, HIDDEN_SIZE, dtype=torch.bfloat16, device=device
    ) / 10.0
    router_logits = torch.randn(
        batch_size, NUM_EXPERTS, dtype=torch.bfloat16, device=device
    )

    w13 = torch.randn(
        NUM_EXPERTS,
        2 * INTERMEDIATE_SIZE,
        HIDDEN_SIZE,
        dtype=torch.float32,
        device=device,
    ).to(FP8_DTYPE)

    w2 = torch.randn(
        NUM_EXPERTS,
        HIDDEN_SIZE,
        INTERMEDIATE_SIZE,
        dtype=torch.float32,
        device=device,
    ).to(FP8_DTYPE)

    # Block quantization scales (128x128 blocks)
    BLOCK_SIZE = 128
    w13_scale_rows = (2 * INTERMEDIATE_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 12
    w13_scale_cols = (HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 16
    w2_scale_rows = (HIDDEN_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 16
    w2_scale_cols = (INTERMEDIATE_SIZE + BLOCK_SIZE - 1) // BLOCK_SIZE  # 6

    w13_scale = torch.ones(
        NUM_EXPERTS, w13_scale_rows, w13_scale_cols, dtype=torch.float32, device=device
    )
    w2_scale = torch.ones(
        NUM_EXPERTS, w2_scale_rows, w2_scale_cols, dtype=torch.float32, device=device
    )

    # Activation scales (static, matches current benchmark defaults)
    a1_scale = torch.ones(1, dtype=torch.float32, device=device)
    a2_scale = torch.ones(1, dtype=torch.float32, device=device)

    return (
        hidden_states,
        router_logits,
        w13,
        w2,
        w13_scale,
        w2_scale,
        a1_scale,
        a2_scale,
    )


def _time_cuda_graph(
    fn: Callable[[], None],
    *,
    num_warmup: int,
    num_iters: int,
    num_ops_per_graph: int,
) -> tuple[float, float]:
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(num_ops_per_graph):
            fn()
    torch.cuda.synchronize()

    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    latencies = []
    for _ in range(num_iters):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        latencies.append(start.elapsed_time(end) / num_ops_per_graph)

    graph.reset()
    latencies.sort()
    return (sum(latencies) / len(latencies), latencies[len(latencies) // 2])


def profile_fusedmoe(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor,
    a2_scale: torch.Tensor,
    *,
    num_warmup: int,
    num_iters: int,
    use_cuda_graph: bool,
    num_ops_per_graph: int,
) -> dict[str, tuple[float, float]]:
    quant_config = FusedMoEQuantConfig.make(
        quant_dtype=FP8_DTYPE,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=[128, 128],
    )

    topk_weights = torch.empty(
        hidden_states.size(0), TOP_K, dtype=torch.float32, device="cuda"
    )
    topk_ids = torch.empty(
        hidden_states.size(0), TOP_K, dtype=torch.int32, device="cuda"
    )
    token_expert_indices = torch.empty(
        hidden_states.size(0), TOP_K, dtype=torch.int32, device="cuda"
    )

    def topk_only():
        ops.topk_softmax(
            topk_weights,
            topk_ids,
            token_expert_indices,
            router_logits,
            True,  # renormalize
        )

    # Precompute topk for experts-only.
    topk_only()
    torch.cuda.synchronize()

    def experts_only():
        fused_experts(
            hidden_states,
            w13,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            quant_config=quant_config,
        )

    def full():
        topk_only()
        experts_only()

    if not use_cuda_graph:
        raise ValueError("Only CUDA graph mode is supported for this script.")

    return {
        "topk_softmax": _time_cuda_graph(
            topk_only,
            num_warmup=num_warmup,
            num_iters=num_iters,
            num_ops_per_graph=num_ops_per_graph,
        ),
        "fused_experts": _time_cuda_graph(
            experts_only,
            num_warmup=num_warmup,
            num_iters=num_iters,
            num_ops_per_graph=num_ops_per_graph,
        ),
        "total": _time_cuda_graph(
            full,
            num_warmup=num_warmup,
            num_iters=num_iters,
            num_ops_per_graph=num_ops_per_graph,
        ),
    }


def profile_monokernel(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    num_warmup: int,
    num_iters: int,
    use_cuda_graph: bool,
    num_ops_per_graph: int,
) -> dict | None:
    if not HAS_MONOKERNEL:
        return None
    if not use_cuda_graph:
        raise ValueError("Only CUDA graph mode is supported for this script.")

    batch_size = hidden_states.size(0)
    scratchpad_size = ops.moe_monokernel_qwen3_block_quant_scratchpad_size(batch_size)
    scratchpad = torch.empty(scratchpad_size, dtype=torch.uint8, device="cuda")
    output = torch.empty_like(hidden_states)

    def run_once():
        ops.moe_monokernel_qwen3_block_quant(
            hidden_states,
            router_logits,
            w13,
            w13_scale,
            w2,
            w2_scale,
            output,
            scratchpad,
        )

    total_mean_ms, total_median_ms = _time_cuda_graph(
        run_once,
        num_warmup=num_warmup,
        num_iters=num_iters,
        num_ops_per_graph=num_ops_per_graph,
    )

    # Timing array layout:
    # [kernel_start, routing_end, prepare_end, quantize_end, grid_sync_1,
    #  up_proj_end, grid_sync_2, down_proj_end, grid_sync_3, kernel_end]
    timing = ops.moe_monokernel_block_quant_get_timing(scratchpad, batch_size).cpu().tolist()
    deltas = [
        timing[1] - timing[0],
        timing[2] - timing[1],
        timing[3] - timing[2],
        timing[4] - timing[3],
        timing[5] - timing[4],
        timing[6] - timing[5],
        timing[7] - timing[6],
        timing[8] - timing[7],
        timing[9] - timing[8],
    ]
    total_cycles = sum(deltas)

    stage_names = [
        "routing",
        "prepare",
        "quantize",
        "grid_sync_1",
        "up_proj",
        "grid_sync_2",
        "down_proj",
        "grid_sync_3",
        "convert",
    ]

    stage_ms = {}
    if total_cycles > 0:
        cycle_to_ms = total_mean_ms / total_cycles
        stage_ms = {k: v * cycle_to_ms for k, v in zip(stage_names, deltas)}

    return {
        "total": (total_mean_ms, total_median_ms),
        "stage_ms": stage_ms,
    }


def _fmt_us(ms: float) -> str:
    return f"{ms * 1000.0:.2f} us"


def print_results(bs: int, fused: dict[str, tuple[float, float]], mono: dict | None) -> None:
    print(f"\n{'='*80}")
    print(f"MoE Profiling (BS={bs}, E={NUM_EXPERTS}, TOP_K={TOP_K})")
    print(f"{'='*80}")

    topk_mean, topk_median = fused["topk_softmax"]
    experts_mean, experts_median = fused["fused_experts"]
    total_mean, total_median = fused["total"]

    print("Baseline (FusedMoE, CUDA graph):")
    print(f"  topk_softmax:  mean={_fmt_us(topk_mean)}  median={_fmt_us(topk_median)}")
    print(f"  fused_experts: mean={_fmt_us(experts_mean)}  median={_fmt_us(experts_median)}")
    print(f"  total:        mean={_fmt_us(total_mean)}  median={_fmt_us(total_median)}")

    if not mono:
        print("Monokernel: unavailable")
        return

    mono_mean, mono_median = mono["total"]
    speedup = total_mean / mono_mean if mono_mean > 0 else float("nan")
    print("Monokernel (CUDA graph):")
    print(f"  total:        mean={_fmt_us(mono_mean)}  median={_fmt_us(mono_median)}  speedup={speedup:.2f}x")

    stage_ms = mono.get("stage_ms", {})
    if stage_ms:
        print("  internal stages (scaled):")
        for name in [
            "routing",
            "prepare",
            "quantize",
            "grid_sync_1",
            "up_proj",
            "grid_sync_2",
            "down_proj",
            "grid_sync_3",
            "convert",
        ]:
            val = stage_ms.get(name)
            if val is not None:
                print(f"    {name:>12}: {_fmt_us(val)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-stage MoE profiling: FusedMoE vs Monokernel (Qwen3 FP8 block quant)"
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64],
        help="Batch sizes to profile.",
    )
    parser.add_argument("--num-warmup", type=int, default=20)
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--use-cuda-graph", action="store_true")
    parser.add_argument("--num-ops-per-graph", type=int, default=10)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if not args.use_cuda_graph:
        raise ValueError("Pass `--use-cuda-graph` for accurate small-batch timing.")

    print(f"Device: {torch.cuda.get_device_name()}")
    print(f"Monokernel available: {HAS_MONOKERNEL}")

    for bs in args.batch_sizes:
        (
            hidden_states,
            router_logits,
            w13,
            w2,
            w13_scale,
            w2_scale,
            a1_scale,
            a2_scale,
        ) = create_test_tensors(bs)

        fused = profile_fusedmoe(
            hidden_states,
            router_logits,
            w13,
            w2,
            w13_scale,
            w2_scale,
            a1_scale,
            a2_scale,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            use_cuda_graph=args.use_cuda_graph,
            num_ops_per_graph=args.num_ops_per_graph,
        )

        mono = profile_monokernel(
            hidden_states,
            router_logits,
            w13,
            w2,
            w13_scale,
            w2_scale,
            num_warmup=args.num_warmup,
            num_iters=args.num_iters,
            use_cuda_graph=args.use_cuda_graph,
            num_ops_per_graph=args.num_ops_per_graph,
        )

        print_results(bs, fused, mono)


if __name__ == "__main__":
    main()

