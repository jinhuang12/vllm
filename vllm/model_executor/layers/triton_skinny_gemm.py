# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Triton GEMM kernel for small-M BF16 decode on NVIDIA GPUs.

Replaces cuBLAS CUTLASS WMMA 16x16x128 dispatch for M <= 16 with a Triton
kernel using larger CTA tiles (up to BLOCK_N=256, 8 warps, 5-stage pipeline)
for better DRAM bandwidth utilization on bandwidth-bound small-M GEMMs.

Activated via VLLM_TRITON_SKINNY_GEMM=1.
"""

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

BLOCK_M: int = 16  # Fixed — WMMA minimum tile, M is padded up to this


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 32, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=5, num_warps=8
        ),
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 128}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=4
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 64}, num_stages=4, num_warps=8
        ),
        triton.Config(
            {"BLOCK_N": 128, "BLOCK_K": 128}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 64}, num_stages=2, num_warps=8
        ),
        triton.Config(
            {"BLOCK_N": 256, "BLOCK_K": 128}, num_stages=2, num_warps=8
        ),
    ],
    key=["N", "K"],
)
@triton.jit
def _skinny_gemm_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    out_ptr,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """BF16 GEMM: out[BLOCK_M, N] = x[BLOCK_M, K] @ w[N, K]^T + bias, FP32 accum."""
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = k_offs[None, :] < K
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        # Load w tile [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        # Accumulate [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # Fuse bias addition in FP32 (matches cuBLAS epilogue behavior)
    if HAS_BIAS:
        n_mask = n_offs < N
        bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store [BLOCK_M, BLOCK_N]
    out_ptrs = out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    out_mask = n_offs[None, :] < N
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def triton_skinny_gemm_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton GEMM for M <= 16 BF16, cuBLAS fallback otherwise."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    if M > BLOCK_M or x.dtype != torch.bfloat16:
        return F.linear(x, weight, bias)

    # Pad input to BLOCK_M
    if M < BLOCK_M:
        x_padded = x_2d.new_empty(BLOCK_M, K)
        x_padded[:M] = x_2d
    else:
        x_padded = x_2d

    out = x_padded.new_empty(BLOCK_M, N)

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
    _skinny_gemm_kernel[grid](
        x_padded,
        weight,
        bias if bias is not None else x_padded,  # dummy ptr when no bias
        out,
        N,
        K,
        x_padded.stride(0),
        x_padded.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        HAS_BIAS=bias is not None,
        BLOCK_M=BLOCK_M,
    )

    return out[:M].reshape(*orig_shape[:-1], N)


def triton_skinny_gemm_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight.shape[0])


direct_register_custom_op(
    op_name="triton_skinny_gemm",
    op_func=triton_skinny_gemm_impl,
    fake_impl=triton_skinny_gemm_fake,
)


def cuda_skinny_gemm(
    layer: torch.nn.Module,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Entry point for dispatch_unquantized_gemm on CUDA with skinny GEMM."""
    return torch.ops.vllm.triton_skinny_gemm(x, weight, bias)
