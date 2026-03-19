# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused GDN inter-GEMM Triton kernels for decode path optimization.

Two fused operations replacing 4+ kernel launches per GDN layer during decode:

1. fused_split_rearrange_kernel: Replaces split + 3x rearrange(..).contiguous()
   with a single kernel that reads conv1d output and writes q, k, v in the
   target [1, BS, H, D] layout. Eliminates 3 contiguous-copy kernel launches.

2. fused_rmsnorm_gated_kernel: Fuses RMSNorm(x) * silu(z) into a single
   pass, replacing the LayerNormFn.apply -> layer_norm_fwd_kernel path.

Enable with: VLLM_GDN_FUSED_INTERGEMM=1
"""

import os

import torch
import triton
import triton.language as tl

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

# Enable flag -- uses registered vLLM env var
VLLM_GDN_FUSED_INTERGEMM: bool = envs.VLLM_GDN_FUSED_INTERGEMM

# Call counters for fastpath verification
_fused_split_call_count: int = 0
_fused_rmsnorm_call_count: int = 0

if VLLM_GDN_FUSED_INTERGEMM:
    logger.info(
        "VLLM_GDN_FUSED_INTERGEMM=1: fused GDN inter-GEMM kernels ENABLED"
    )


# =============================================================================
# Kernel 1: Fused Split + Rearrange (eliminates 3 contiguous copies)
# =============================================================================
# Replaces: torch.split() + 3x rearrange("l (h d) -> 1 l h d").contiguous()
#
# Input:  mixed_qkv [BS, conv_dim] (already processed by conv1d)
# Output: q [1, BS, Hk, Dk], k [1, BS, Hk, Dk], v [1, BS, Hv, Dv]
#
# The input is contiguous but split creates non-contiguous views for k, v.
# The rearrange is a view-only reshape, but .contiguous() triggers a copy.
# This kernel reads the flat input and writes directly to the target layout.
# =============================================================================


@triton.jit
def fused_split_rearrange_kernel(
    X_ptr,       # [BS, conv_dim] input
    Q_ptr,       # [1, BS, Hk, Dk] output
    K_ptr,       # [1, BS, Hk, Dk] output
    V_ptr,       # [1, BS, Hv, Dv] output
    conv_dim,
    key_dim,     # Hk * Dk
    value_dim,   # Hv * Dv
    BLOCK_D: tl.constexpr,
):
    """
    Grid: (BS, cdiv(conv_dim, BLOCK_D))
    Each program copies one (batch, channel_block) tile to the correct output.
    """
    i_b = tl.program_id(0)
    i_d = tl.program_id(1)

    offs = i_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask = offs < conv_dim

    # Load from input
    val = tl.load(X_ptr + i_b * conv_dim + offs, mask=mask)

    # Write to Q: channels [0, key_dim)
    q_mask = mask & (offs < key_dim)
    tl.store(Q_ptr + i_b * key_dim + offs, val, mask=q_mask)

    # Write to K: channels [key_dim, 2*key_dim)
    k_mask = mask & (offs >= key_dim) & (offs < 2 * key_dim)
    tl.store(K_ptr + i_b * key_dim + (offs - key_dim), val, mask=k_mask)

    # Write to V: channels [2*key_dim, conv_dim)
    v_mask = mask & (offs >= 2 * key_dim)
    tl.store(V_ptr + i_b * value_dim + (offs - 2 * key_dim), val, mask=v_mask)


# =============================================================================
# Kernel 2: Fused RMSNorm + SiLU Gating (post-recurrent)
# =============================================================================


@triton.jit
def fused_rmsnorm_gated_kernel(
    X_ptr,
    Z_ptr,
    W_ptr,
    Y_ptr,
    stride_x,
    stride_z,
    stride_y,
    M,
    eps,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    One row per program. Grid: (M,)
    Computes: y = RMSNorm(x, weight, eps) * silu(z)
    """
    row = tl.program_id(0)
    if row >= M:
        return

    cols = tl.arange(0, BLOCK_D)
    col_mask = cols < D

    # Load x
    x = tl.load(X_ptr + row * stride_x + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )

    # RMS norm: rstd = rsqrt(mean(x^2) + eps)
    var = tl.sum(x * x, axis=0) / D
    rstd = tl.rsqrt(var + eps)

    # Load weight and normalize
    w = tl.load(W_ptr + cols, mask=col_mask, other=0.0).to(tl.float32)
    x_hat = x * rstd * w

    # Load z and compute silu(z)
    z = tl.load(Z_ptr + row * stride_z + cols, mask=col_mask, other=0.0).to(
        tl.float32
    )
    z_act = z * tl.sigmoid(z)

    # norm_before_gate: out = norm(x) * silu(z)
    y = x_hat * z_act

    tl.store(
        Y_ptr + row * stride_y + cols,
        y.to(Y_ptr.dtype.element_ty),
        mask=col_mask,
    )


# =============================================================================
# Python wrappers
# =============================================================================


def fused_split_rearrange(
    mixed_qkv: torch.Tensor,    # [BS, conv_dim] -- already conv1d-processed
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
    num_k_heads: int,
    num_v_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused split + rearrange + contiguous for GDN decode.

    Replaces:
        q, k, v = torch.split(mixed_qkv, [key_dim, key_dim, value_dim], dim=-1)
        q = rearrange(q, "l (h d) -> 1 l h d", d=head_k_dim).contiguous()
        k = rearrange(k, "l (h d) -> 1 l h d", d=head_k_dim).contiguous()
        v = rearrange(v, "l (h d) -> 1 l h d", d=head_v_dim).contiguous()

    Returns:
        q: [1, BS, num_k_heads, head_k_dim] contiguous
        k: [1, BS, num_k_heads, head_k_dim] contiguous
        v: [1, BS, num_v_heads, head_v_dim] contiguous
    """
    global _fused_split_call_count
    _fused_split_call_count += 1

    BS = mixed_qkv.shape[0]
    conv_dim = mixed_qkv.shape[1]

    # Allocate outputs in target layout (contiguous 4D)
    q = torch.empty((1, BS, num_k_heads, head_k_dim), dtype=mixed_qkv.dtype,
                    device=mixed_qkv.device)
    k = torch.empty((1, BS, num_k_heads, head_k_dim), dtype=mixed_qkv.dtype,
                    device=mixed_qkv.device)
    v = torch.empty((1, BS, num_v_heads, head_v_dim), dtype=mixed_qkv.dtype,
                    device=mixed_qkv.device)

    BLOCK_D = 256
    grid = (BS, triton.cdiv(conv_dim, BLOCK_D))

    fused_split_rearrange_kernel[grid](
        mixed_qkv, q, k, v,
        conv_dim, key_dim, value_dim,
        BLOCK_D,
    )

    return q, k, v


def fused_rmsnorm_gated(
    x: torch.Tensor,      # [M, D] already reshaped
    z: torch.Tensor,      # [M, D] already reshaped
    weight: torch.Tensor, # [D]
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm(x) * silu(z).

    Input x and z should already be reshaped to [M, D] where D = head_v_dim.
    Returns output of same shape as x.
    """
    global _fused_rmsnorm_call_count
    _fused_rmsnorm_call_count += 1

    assert x.dim() == 2 and z.dim() == 2
    M, D = x.shape

    if x.stride(-1) != 1:
        x = x.contiguous()
    if z.stride(-1) != 1:
        z = z.contiguous()

    y = torch.empty_like(x)
    BLOCK_D = triton.next_power_of_2(D)

    fused_rmsnorm_gated_kernel[(M,)](
        x, z, weight, y,
        x.stride(0), z.stride(0), y.stride(0),
        M, eps,
        D, BLOCK_D,
    )

    return y
