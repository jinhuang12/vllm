# SPDX-License-Identifier: Apache-2.0
"""
Correctness tests for fused GDN inter-GEMM kernels.

Tests compare fused kernels against the vLLM production baseline:
1. fused_split_rearrange vs split + rearrange + contiguous (3 copy kernels)
2. fused_rmsnorm_gated vs RMSNormGated (LayerNormFn.apply path)

Tolerances: atol=0.04, rtol=0.02 (bfloat16, different accumulation order)
"""

import torch
import pytest

# Production baseline imports
from vllm.model_executor.layers.fla.ops.layernorm_guard import rmsnorm_fn

# Fused kernel imports
from vllm.model_executor.layers.fla.ops.fused_gdn_intergemm import (
    fused_split_rearrange,
    fused_rmsnorm_gated,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16

# Qwen3.5-4B GDN dimensions (TP=1)
NUM_K_HEADS = 16
NUM_V_HEADS = 32
HEAD_K_DIM = 128
HEAD_V_DIM = 128
KEY_DIM = NUM_K_HEADS * HEAD_K_DIM   # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V_DIM  # 4096
CONV_DIM = KEY_DIM + KEY_DIM + VALUE_DIM  # 8192

# bf16 tolerances: fused kernel reads the same data but writes directly;
# the values should be bit-identical since no computation is done.
ATOL = 1e-6
RTOL = 1e-6


def rearrange_mixed_qkv_baseline(mixed_qkv):
    """Production baseline: split + rearrange + contiguous."""
    from einops import rearrange
    if mixed_qkv is None:
        return None, None, None
    query, key, value = torch.split(
        mixed_qkv, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1,
    )
    query, key = map(
        lambda x: rearrange(x, "l (h d) -> 1 l h d", d=HEAD_K_DIM),
        (query, key),
    )
    value = rearrange(value, "l (h d) -> 1 l h d", d=HEAD_V_DIM)
    return query.contiguous(), key.contiguous(), value.contiguous()


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_fused_split_rearrange_correctness(batch_size):
    """Test fused_split_rearrange vs production split+rearrange+contiguous."""
    torch.manual_seed(42)

    mixed_qkv = torch.randn(batch_size, CONV_DIM, dtype=DTYPE, device=DEVICE)

    # Baseline
    q_bl, k_bl, v_bl = rearrange_mixed_qkv_baseline(mixed_qkv.clone())

    # Fused
    q_f, k_f, v_f = fused_split_rearrange(
        mixed_qkv.clone(), KEY_DIM, VALUE_DIM,
        HEAD_K_DIM, HEAD_V_DIM, NUM_K_HEADS, NUM_V_HEADS,
    )

    # Shape check
    assert q_bl.shape == q_f.shape, f"q shape: {q_bl.shape} vs {q_f.shape}"
    assert k_bl.shape == k_f.shape, f"k shape: {k_bl.shape} vs {k_f.shape}"
    assert v_bl.shape == v_f.shape, f"v shape: {v_bl.shape} vs {v_f.shape}"

    # Value check -- should be bit-identical since it's just a data copy
    assert torch.equal(q_bl, q_f), (
        f"q mismatch: max_diff={torch.max(torch.abs(q_bl - q_f)).item()}"
    )
    assert torch.equal(k_bl, k_f), (
        f"k mismatch: max_diff={torch.max(torch.abs(k_bl - k_f)).item()}"
    )
    assert torch.equal(v_bl, v_f), (
        f"v mismatch: max_diff={torch.max(torch.abs(v_bl - v_f)).item()}"
    )

    # No NaN/Inf
    for name, t in [("q", q_f), ("k", k_f), ("v", v_f)]:
        assert not torch.isnan(t).any(), f"{name} has NaN"
        assert not torch.isinf(t).any(), f"{name} has Inf"

    print(f"  BS={batch_size}: PASS (bit-identical)")


@pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
def test_fused_rmsnorm_gated_correctness(batch_size):
    """Test fused_rmsnorm_gated vs production RMSNormGated (rmsnorm_fn)."""
    torch.manual_seed(42)

    M = batch_size * NUM_V_HEADS
    D = HEAD_V_DIM
    eps = 1e-6

    x = torch.randn(M, D, dtype=DTYPE, device=DEVICE)
    z = torch.randn(M, D, dtype=DTYPE, device=DEVICE)
    weight = torch.ones(D, dtype=DTYPE, device=DEVICE) + \
             torch.randn(D, dtype=DTYPE, device=DEVICE) * 0.1

    # Baseline: production rmsnorm_fn
    out_bl = rmsnorm_fn(
        x.clone(), weight, None, z=z.clone(), eps=eps,
        group_size=None, norm_before_gate=True,
    )

    # Fused kernel
    out_f = fused_rmsnorm_gated(x.clone(), z.clone(), weight, eps=eps)

    assert out_bl.shape == out_f.shape
    max_diff = torch.max(torch.abs(out_bl - out_f)).item()
    # Same computation order, should be very close or identical
    assert torch.allclose(out_bl, out_f, atol=1e-3, rtol=1e-3), (
        f"RMSNormGated mismatch: max_diff={max_diff:.6f}"
    )
    assert not torch.isnan(out_f).any(), "Output has NaN"
    assert not torch.isinf(out_f).any(), "Output has Inf"

    print(f"  BS={batch_size} (M={M}): PASS (max_diff={max_diff:.6f})")


def test_fused_rmsnorm_gated_edge_values():
    """Test RMSNormGated with edge values."""
    torch.manual_seed(42)
    M, D, eps = 8, 128, 1e-6
    weight = torch.ones(D, dtype=DTYPE, device=DEVICE)

    # Near-zero
    x_s = torch.randn(M, D, dtype=DTYPE, device=DEVICE) * 1e-3
    z_s = torch.randn(M, D, dtype=DTYPE, device=DEVICE) * 1e-3
    out_bl = rmsnorm_fn(x_s.clone(), weight, None, z=z_s.clone(), eps=eps,
                        group_size=None, norm_before_gate=True)
    out_f = fused_rmsnorm_gated(x_s.clone(), z_s.clone(), weight, eps=eps)
    assert not torch.isnan(out_f).any()
    assert torch.allclose(out_bl, out_f, atol=1e-3, rtol=1e-3)

    # Large values
    x_l = torch.randn(M, D, dtype=DTYPE, device=DEVICE) * 100
    z_l = torch.randn(M, D, dtype=DTYPE, device=DEVICE) * 100
    out_bl = rmsnorm_fn(x_l.clone(), weight, None, z=z_l.clone(), eps=eps,
                        group_size=None, norm_before_gate=True)
    out_f = fused_rmsnorm_gated(x_l.clone(), z_l.clone(), weight, eps=eps)
    assert not torch.isnan(out_f).any()
    assert torch.allclose(out_bl, out_f, atol=0.5, rtol=0.05)
    print("  Edge values: PASS")


def test_cuda_graph_split():
    """Test fused_split_rearrange under CUDA graph capture + replay."""
    torch.manual_seed(42)
    BS = 8
    mixed_qkv = torch.randn(BS, CONV_DIM, dtype=DTYPE, device=DEVICE)

    # Warmup
    for _ in range(3):
        fused_split_rearrange(mixed_qkv.clone(), KEY_DIM, VALUE_DIM,
                              HEAD_K_DIM, HEAD_V_DIM, NUM_K_HEADS, NUM_V_HEADS)

    mqkv_g = mixed_qkv.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        q, k, v = fused_split_rearrange(
            mqkv_g, KEY_DIM, VALUE_DIM, HEAD_K_DIM, HEAD_V_DIM,
            NUM_K_HEADS, NUM_V_HEADS,
        )

    for _ in range(5):
        mqkv_g.copy_(mixed_qkv)
        g.replay()

    assert not torch.isnan(q).any()
    q_ref, _, _ = rearrange_mixed_qkv_baseline(mixed_qkv)
    assert torch.equal(q, q_ref)
    print("  CUDA graph split: PASS")


def test_cuda_graph_rmsnorm():
    """Test fused_rmsnorm_gated under CUDA graph capture + replay."""
    torch.manual_seed(42)
    M, D = 256, 128
    x = torch.randn(M, D, dtype=DTYPE, device=DEVICE)
    z = torch.randn(M, D, dtype=DTYPE, device=DEVICE)
    w = torch.ones(D, dtype=DTYPE, device=DEVICE)

    for _ in range(3):
        fused_rmsnorm_gated(x.clone(), z.clone(), w, eps=1e-6)

    x_g, z_g = x.clone(), z.clone()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fused_rmsnorm_gated(x_g, z_g, w, eps=1e-6)

    for _ in range(5):
        x_g.copy_(x); z_g.copy_(z)
        g.replay()

    assert not torch.isnan(out).any()
    out_eager = fused_rmsnorm_gated(x.clone(), z.clone(), w, eps=1e-6)
    assert torch.allclose(out, out_eager, atol=1e-6, rtol=1e-6)
    print("  CUDA graph rmsnorm: PASS")


if __name__ == "__main__":
    print("\n=== Test 1: fused_split_rearrange correctness ===")
    for bs in [1, 4, 8, 16, 32]:
        test_fused_split_rearrange_correctness(bs)

    print("\n=== Test 2: fused_rmsnorm_gated correctness ===")
    for bs in [1, 4, 8, 16, 32]:
        test_fused_rmsnorm_gated_correctness(bs)

    print("\n=== Test 3: Edge values ===")
    test_fused_rmsnorm_gated_edge_values()

    print("\n=== Test 4: CUDA graph capture (split) ===")
    test_cuda_graph_split()

    print("\n=== Test 5: CUDA graph capture (rmsnorm) ===")
    test_cuda_graph_rmsnorm()

    print("\n=== ALL TESTS PASSED ===")
