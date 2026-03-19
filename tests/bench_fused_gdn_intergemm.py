# SPDX-License-Identifier: Apache-2.0
"""
Kernel benchmark for fused GDN inter-GEMM kernels.
Measures both baseline and optimized under CUDA graph capture.
"""

import torch
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn
from vllm.model_executor.layers.fla.ops.fused_gdn_intergemm import (
    fused_split_rearrange, fused_rmsnorm_gated,
)
from einops import rearrange

DEVICE = "cuda"
DTYPE = torch.bfloat16
NUM_K_HEADS, NUM_V_HEADS = 16, 32
HEAD_K_DIM, HEAD_V_DIM = 128, 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM
CONV_DIM = KEY_DIM * 2 + VALUE_DIM
WARMUP, ITERS, BS = 50, 200, 8


def benchmark_cuda_graph(run_fn, inputs, name):
    """Benchmark under CUDA graph capture."""
    for _ in range(3):
        run_fn(*inputs)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            run_fn(*inputs)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g, stream=s):
        run_fn(*inputs)
    torch.cuda.synchronize()

    for _ in range(WARMUP):
        g.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000 / ITERS


def main():
    torch.manual_seed(42)
    mqkv = torch.randn(BS, CONV_DIM, dtype=DTYPE, device=DEVICE)
    x = torch.randn(BS * NUM_V_HEADS, HEAD_V_DIM, dtype=DTYPE, device=DEVICE)
    z = torch.randn(BS * NUM_V_HEADS, HEAD_V_DIM, dtype=DTYPE, device=DEVICE)
    w = torch.ones(HEAD_V_DIM, dtype=DTYPE, device=DEVICE) + \
        torch.randn(HEAD_V_DIM, dtype=DTYPE, device=DEVICE) * 0.1

    # Baseline split: split + rearrange + 3x contiguous
    def bl_split(m):
        q, k, v = torch.split(m, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
        q = rearrange(q, "l (h d) -> 1 l h d", d=HEAD_K_DIM).contiguous()
        k = rearrange(k, "l (h d) -> 1 l h d", d=HEAD_K_DIM).contiguous()
        v = rearrange(v, "l (h d) -> 1 l h d", d=HEAD_V_DIM).contiguous()
        return q, k, v

    # Fused split
    def opt_split(m):
        return fused_split_rearrange(m, KEY_DIM, VALUE_DIM,
                                     HEAD_K_DIM, HEAD_V_DIM,
                                     NUM_K_HEADS, NUM_V_HEADS)

    # Baseline rmsnorm
    def bl_norm(x, z, w):
        return rmsnorm_fn(x, w, None, z=z, eps=1e-6, group_size=None,
                          norm_before_gate=True)

    # Fused rmsnorm
    def opt_norm(x, z, w):
        return fused_rmsnorm_gated(x, z, w, eps=1e-6)

    print(f"Kernel Benchmark: Fused GDN Inter-GEMM (BS={BS})")
    print(f"  CUDA graph: capture + {ITERS} replay iterations")
    print()

    # Split/rearrange benchmark
    bl_s = benchmark_cuda_graph(bl_split, (mqkv,), "bl_split")
    opt_s = benchmark_cuda_graph(opt_split, (mqkv,), "opt_split")
    sp_s = bl_s / opt_s
    print(f"=== Split/Rearrange ===")
    print(f"  Baseline (3 copies):    {bl_s:.2f} us")
    print(f"  Fused (1 kernel):       {opt_s:.2f} us")
    print(f"  Speedup:                {sp_s:.2f}x")
    print()

    # RMSNormGated benchmark
    bl_r = benchmark_cuda_graph(bl_norm, (x, z, w), "bl_norm")
    opt_r = benchmark_cuda_graph(opt_norm, (x, z, w), "opt_norm")
    sp_r = bl_r / opt_r
    print(f"=== RMSNormGated ===")
    print(f"  Baseline (LayerNormFn): {bl_r:.2f} us")
    print(f"  Fused (direct Triton):  {opt_r:.2f} us")
    print(f"  Speedup:                {sp_r:.2f}x")
    print()

    saved_s = (bl_s - opt_s) * 24
    saved_r = (bl_r - opt_r) * 24
    total = saved_s + saved_r
    pct = total / 14953 * 100
    print(f"=== Per-Step (24 layers) ===")
    print(f"  Split saved:    {saved_s:.1f} us")
    print(f"  RMSNorm saved:  {saved_r:.1f} us")
    print(f"  Total saved:    {total:.1f} us")
    print(f"  Projected E2E:  {pct:.2f}%")


if __name__ == "__main__":
    main()
