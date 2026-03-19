# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Selective Triton GEMM dispatch with fused gate_up SiLU kernel.

Fused SiLU kernel computes SiLU(x @ W_gate.T) * (x @ W_up.T) in one pass,
where W_gate and W_up are the two halves of a merged [2*N_half, K] weight.
Standard Triton GEMM replaces cuBLAS for shapes that win in warm cache.

Enabled by VLLM_TRITON_GEMM_SELECTIVE=1.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

# ============================================================
# Fused Gate-Up SiLU GEMM Kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
            num_stages=4, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=4, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=8,
        ),
    ],
    key=["N_half", "K"],
)
@triton.jit
def _fused_gate_up_silu_kernel(
    # Pointers
    a_ptr, w_ptr, c_ptr,
    # Dimensions
    M, N_half, K,
    # Strides (A is [M, K] row-major, W is [N_full, K] row-major)
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Fused gate_up GEMM + SiLU.

    Computes: C[M, N_half] = SiLU(A @ W[:N_half].T) * (A @ W[N_half:].T)

    Weight W is [2*N_half, K] with gate in rows [0, N_half) and up in
    rows [N_half, 2*N_half). Each CTA computes a [BLOCK_M, BLOCK_N] tile
    of the output, maintaining two FP32 accumulators for gate and up.
    SiLU epilogue: silu(gate) * up = gate * sigmoid(gate) * up in FP32.
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N_half, BLOCK_N)

    # L2 swizzle grouping
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # A tile pointers: A[m, k]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak

    # Weight tile pointers for gate half: W[n, k] where n in [0, N_half)
    # Loaded as [BK, BN] for A[BM,BK] @ W_tile[BK,BN] = C[BM,BN]
    wg_ptrs = (w_ptr
               + offs_k[:, None] * stride_wk
               + offs_n[None, :] * stride_wn)

    # Weight tile pointers for up half: W[n + N_half, k]
    wu_ptrs = (w_ptr
               + offs_k[:, None] * stride_wk
               + (offs_n + N_half)[None, :] * stride_wn)

    # FP32 accumulators
    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K-loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_half)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        wg_tile = tl.load(wg_ptrs, mask=w_mask, other=0.0)
        wu_tile = tl.load(wu_ptrs, mask=w_mask, other=0.0)

        acc_gate += tl.dot(a_tile, wg_tile)
        acc_up += tl.dot(a_tile, wu_tile)

        # Advance K pointers
        a_ptrs += BLOCK_K * stride_ak
        wg_ptrs += BLOCK_K * stride_wk
        wu_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    # SiLU epilogue in FP32: silu(x) = x * sigmoid(x)
    result = acc_gate * tl.sigmoid(acc_gate) * acc_up

    # Store as BF16
    c_ptrs = (c_ptr
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_half)
    tl.store(c_ptrs, result.to(tl.bfloat16), mask=c_mask)


def fused_gate_up_silu_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Fused gate_up GEMM + SiLU activation.

    Args:
        x: [M, K] input tensor (BF16)
        weight: [2*N_half, K] merged gate+up weight (BF16)

    Returns:
        [M, N_half] = SiLU(x @ W_gate.T) * (x @ W_up.T)
    """
    M, K = x.shape
    N_full = weight.shape[0]
    N_half = N_full // 2

    out = torch.empty(M, N_half, dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N_half, META["BLOCK_N"]),
    )

    _fused_gate_up_silu_kernel[grid](
        x, weight, out,
        M, N_half, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32,
    )
    return out


def fused_gate_up_silu_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    N_half = weight.shape[0] // 2
    return x.new_empty(x.shape[0], N_half)


direct_register_custom_op(
    op_name="fused_gate_up_silu",
    op_func=fused_gate_up_silu_impl,
    fake_impl=fused_gate_up_silu_fake,
)


# ============================================================
# Standard Triton GEMM Kernel (for out_proj, qkv_proj)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=4, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 128, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=4,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=3, num_warps=8,
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 64, "GROUP_SIZE_M": 1},
            num_stages=4, num_warps=8,
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _gemm_m32_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Triton GEMM for small M (decode).

    Computes C = A @ B.T where A is [M, K] and B is [N, K].
    """
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = (b_ptr
              + offs_k[:, None] * stride_bk
              + offs_n[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a_tile, b_tile)

        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    c_ptrs = (c_ptr
              + offs_m[:, None] * stride_cm
              + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)


def gemm_m32_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Triton GEMM for small M (decode)."""
    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty(M, N, dtype=x.dtype, device=x.device)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )

    _gemm_m32_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=32,
    )
    return out


def gemm_m32_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    return x.new_empty(x.shape[0], weight.shape[0])


direct_register_custom_op(
    op_name="gemm_m32",
    op_func=gemm_m32_impl,
    fake_impl=gemm_m32_fake,
)


# ============================================================
# Selective Dispatch
# ============================================================

def selective_triton_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shape-aware GEMM dispatch: Triton for warm-winning shapes, cuBLAS otherwise.

    Uses Triton GEMM when:
    - M in [2, 32] (decode batch sizes)
    - BF16 dtype
    - No bias (Triton kernel doesn't support bias)
    """
    M = x.numel() // x.shape[-1]
    N, K = weight.shape

    if (2 <= M <= 32
            and x.dtype == torch.bfloat16
            and bias is None):
        x_2d = x.reshape(-1, K) if x.ndim != 2 else x
        out = torch.ops.vllm.gemm_m32(x_2d, weight)
        if x.ndim != 2:
            out = out.reshape(*x.shape[:-1], N)
        return out

    return torch.nn.functional.linear(x, weight, bias)


def should_use_fused_silu(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Check if fused SiLU GEMM should be used for gate_up_proj."""
    M = x.numel() // x.shape[-1]
    return (2 <= M <= 32
            and x.dtype == torch.bfloat16
            and weight.shape[0] % 2 == 0)
