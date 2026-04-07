# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Triton GEMM kernels for small-M BF16 decode on NVIDIA GPUs.

Replaces cuBLAS CUTLASS WMMA 16x16x128 dispatch for M <= 16 with a Triton
kernel using larger CTA tiles (up to BLOCK_N=256, 8 warps, 5-stage pipeline)
for better DRAM bandwidth utilization on bandwidth-bound small-M GEMMs.

Also provides an FP8 weight-only variant that loads FP8 E4M3 weights,
dequantizes to BF16 in-register, and uses BF16 WMMA MMA. This halves DRAM
weight traffic for ~1.5x speedup on large-N shapes. SM89 (Ada/L40S) has no
native FP8 MMA -- the speedup is purely from reduced memory bandwidth.

Activated via VLLM_TRITON_SKINNY_GEMM=1 (BF16) or VLLM_FP8_WEIGHT_GEMM=1 (FP8).
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


# ---- FP8 weight-only dequant-fused kernel --------------------------------

FP8_MAX = 448.0  # torch.finfo(torch.float8_e4m3fn).max


def quantize_weights_per_channel(
    weight_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize BF16 weight [N, K] → FP8 E4M3 with per-channel (per-N) scale.

    Returns (weight_fp8 [N, K], scale [N] float32).
    """
    abs_max = weight_bf16.abs().amax(dim=1)  # [N]
    scale = (abs_max / FP8_MAX).clamp(min=1e-12)
    weight_fp8 = (weight_bf16 / scale[:, None]).to(torch.float8_e4m3fn)
    return weight_fp8, scale.float()


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_N": 64, "BLOCK_K": 64}, num_stages=4, num_warps=4
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
def _fp8_skinny_gemm_kernel(
    x_ptr,
    w_fp8_ptr,
    scale_ptr,
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
    """FP8 weight-only GEMM: load FP8 weights, dequant to BF16, BF16 MMA.

    out[M, N] = x[M, K] @ (w_fp8[N, K] * scale[N])^T + bias[N]
    Scale applied in FP32 epilogue for full precision.
    """
    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # Load BF16 input tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = k_offs[None, :] < K
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load FP8 weight tile [BLOCK_N, BLOCK_K] and cast to BF16
        w_ptrs = (
            w_fp8_ptr
            + n_offs[:, None] * stride_wn
            + k_offs[None, :] * stride_wk
        )
        w_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        w_bf16 = w_fp8.to(tl.bfloat16)

        # BF16 MMA with FP32 accumulation
        acc += tl.dot(x, tl.trans(w_bf16))

    # Apply per-channel scale in FP32 (mathematically equivalent to
    # per-element dequant before MMA, but avoids intermediate BF16 truncation)
    n_mask = n_offs < N
    scales = tl.load(scale_ptr + n_offs, mask=n_mask, other=1.0)
    acc = acc * scales[None, :]

    # Fuse bias addition in FP32 (matches cuBLAS epilogue behavior)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store [BLOCK_M, BLOCK_N] as BF16
    out_ptrs = out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=n_mask[None, :])


# ---- FP8 occupancy-aware kernel (BN=32 / BK=256 / 2-stage) ---------------
# Doubles grid blocks for small-N projections (e.g., o_proj N=2560: 40→80
# blocks on 142 SMs), improving DRAM BW utilization from 64→75% to ~80%.
# Activated via VLLM_FP8_OCC_GEMM=1 (default on when FP8 weight GEMM is on).

# N threshold: shapes with N <= this use the occupancy-tuned kernel.
# 6144 covers o_proj (2560), qkv_proj (6144), down_proj (2560).
# gate_up (18432) stays on the autotuned baseline.
_FP8_OCC_N_THRESHOLD = 6144


@triton.jit
def _fp8_occaware_skinny_gemm_kernel(
    x_ptr,
    w_fp8_ptr,
    scale_ptr,
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
):
    """Occupancy-aware FP8 GEMM with BN=32/BK=256/2-stage pipeline.

    Hardcoded tile dimensions for higher SM utilization on small-N shapes.
    Same FP8→BF16 dequant, BF16 MMA, FP32 accumulation as baseline.
    """
    BLOCK_N: tl.constexpr = 32
    BLOCK_K: tl.constexpr = 256

    pid = tl.program_id(0)
    n_start = pid * BLOCK_N
    m_offs = tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # Load BF16 input tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = k_offs[None, :] < K
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load FP8 weight tile [BLOCK_N, BLOCK_K] and cast to BF16
        w_ptrs = (
            w_fp8_ptr
            + n_offs[:, None] * stride_wn
            + k_offs[None, :] * stride_wk
        )
        w_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        w_fp8 = tl.load(w_ptrs, mask=w_mask, other=0.0)
        w_bf16 = w_fp8.to(tl.bfloat16)

        # BF16 MMA with FP32 accumulation
        acc += tl.dot(x, tl.trans(w_bf16))

    # Apply per-channel scale in FP32
    n_mask = n_offs < N
    scales = tl.load(scale_ptr + n_offs, mask=n_mask, other=1.0)
    acc = acc * scales[None, :]

    # Fuse bias addition in FP32
    if HAS_BIAS:
        bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]

    # Store [BLOCK_M, BLOCK_N] as BF16
    out_ptrs = out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=n_mask[None, :])


def fp8_skinny_gemm_impl(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """FP8 weight-only Triton GEMM for M <= 16, cuBLAS fallback otherwise."""
    from vllm import envs

    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight_fp8.shape[0]

    if M > BLOCK_M or x.dtype != torch.bfloat16:
        # Fallback: dequant in FP32, cast to BF16 for F.linear
        w_bf16 = (weight_fp8.to(torch.float32) * weight_scale[:, None]).to(
            torch.bfloat16
        )
        return F.linear(x, w_bf16, bias)

    # Pad input to BLOCK_M
    if M < BLOCK_M:
        x_padded = x_2d.new_empty(BLOCK_M, K)
        x_padded[:M] = x_2d
    else:
        x_padded = x_2d

    out = x_padded.new_empty(BLOCK_M, N)

    # Shape-adaptive dispatch: small-N uses occupancy-tuned kernel
    use_occ = envs.VLLM_FP8_OCC_GEMM and N <= _FP8_OCC_N_THRESHOLD
    if use_occ:
        grid = (triton.cdiv(N, 32),)  # BN=32 hardcoded
        _fp8_occaware_skinny_gemm_kernel[grid](
            x_padded,
            weight_fp8,
            weight_scale,
            bias if bias is not None else x_padded,
            out,
            N,
            K,
            x_padded.stride(0),
            x_padded.stride(1),
            weight_fp8.stride(0),
            weight_fp8.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_M=BLOCK_M,
            num_stages=2,
            num_warps=4,
        )
    else:
        grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]),)
        _fp8_skinny_gemm_kernel[grid](
            x_padded,
            weight_fp8,
            weight_scale,
            bias if bias is not None else x_padded,
            out,
            N,
            K,
            x_padded.stride(0),
            x_padded.stride(1),
            weight_fp8.stride(0),
            weight_fp8.stride(1),
            out.stride(0),
            out.stride(1),
            HAS_BIAS=bias is not None,
            BLOCK_M=BLOCK_M,
        )

    return out[:M].reshape(*orig_shape[:-1], N)


def fp8_skinny_gemm_fake(
    x: torch.Tensor,
    weight_fp8: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return x.new_empty(*x.shape[:-1], weight_fp8.shape[0])


direct_register_custom_op(
    op_name="triton_fp8_skinny_gemm",
    op_func=fp8_skinny_gemm_impl,
    fake_impl=fp8_skinny_gemm_fake,
)

# ---- BF16 kernel wrapper -------------------------------------------------


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
    if hasattr(layer, "weight_fp8"):
        return torch.ops.vllm.triton_fp8_skinny_gemm(
            x, layer.weight_fp8, layer.weight_scale, bias
        )
    return torch.ops.vllm.triton_skinny_gemm(x, weight, bias)
