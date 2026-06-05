# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Fused Mamba2 GatedRMSNorm Triton kernel (single-launch).

Replaces the Inductor-emitted 2-Triton-kernel chain that lowers
``Mixer2RMSNormGated.forward_native()`` (silu + mul + grouped variance + rsqrt
+ weight broadcast). The native implementation lowers under torch.compile to:

  1. ``triton_red_fused__to_copy_mean_mul_pow_silu_view_0``  (reduction)
  2. ``triton_poi_fused__to_copy_add_..._mean_mul_pow_rsqrt_silu_view_1``
     (pointwise)

Inductor cannot fuse those two kernels because of three structural barriers:
(a) cross-dtype boundary at ``gate.to(float32)``, (b) grouped reduction with
``view -> mean(-1) -> view``, (c) no IR-level op for grouped GatedRMSNorm. See
the OP-006-R7 proposal for the full argument.

This module fuses the chain into a single Triton kernel:
    one program per (token, group), 1024-element reduction in registers.

Inputs/outputs match ``forward_native`` exactly: BF16 in/out with FP32
intermediate (silu, variance, rsqrt). The kernel is functional (no in-place
mutation) and is wrapped in an opaque ``direct_register_custom_op`` so that
torch.compile/Inductor treats it as an atomic IR node.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _gated_rms_norm_fused_kernel(
    X_ptr,
    GATE_ptr,
    W_ptr,
    OUT_ptr,
    M,
    GS,
    eps,
    stride_x_m,
    stride_gate_m,
    stride_out_m,
    BLOCK_GS: tl.constexpr,
):
    """One program per (token, group).

    Layout: 2D ``[M, H]`` with arbitrary per-row stride; the inner (hidden)
    dimension is assumed contiguous (unit stride). Group ``g`` of token ``m``
    starts at byte offset ``m * stride_*_m + g * GS`` (in element units).

    Per-row strides are parameterized so the kernel works zero-copy when one
    of the input tensors is a non-contiguous slice of a wider buffer (e.g.
    ``gate = projected_states[..., :hidden]`` in ``mamba_mixer2.py``).
    """
    pid_m = tl.program_id(0)
    pid_g = tl.program_id(1)

    offs = tl.arange(0, BLOCK_GS)
    mask = offs < GS

    g_off = pid_g * GS + offs
    x = tl.load(
        X_ptr + pid_m * stride_x_m + g_off, mask=mask, other=0.0
    ).to(tl.float32)
    g = tl.load(
        GATE_ptr + pid_m * stride_gate_m + g_off, mask=mask, other=0.0
    ).to(tl.float32)
    w = tl.load(W_ptr + g_off, mask=mask, other=0.0).to(tl.float32)

    # silu(gate) * x — gate promoted to fp32 (matches forward_native:109)
    g_silu = g * tl.sigmoid(g)
    y = x * g_silu

    # Grouped variance: mean(y^2) over the group lane
    var = tl.sum(y * y, axis=0) / GS
    rstd = 1.0 / tl.sqrt(var + eps)
    y = y * rstd * w

    tl.store(
        OUT_ptr + pid_m * stride_out_m + g_off,
        y.to(OUT_ptr.dtype.element_ty),
        mask=mask,
    )


def _gated_rms_norm_fused_impl(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    n_groups: int,
) -> torch.Tensor:
    """Fused gated-RMS-norm CUDA implementation.

    Equivalent to::

        z = x.to(input_dtype) * F.silu(gate.to(float32))
        zg = z.view(..., n_groups, group_size)
        var = zg.pow(2).mean(-1, keepdim=True)
        zg = zg * torch.rsqrt(var + eps)
        out = weight * zg.view(..., hidden).to(input_dtype)

    where ``input_dtype`` matches ``x.dtype`` (BF16 in production).

    Args:
        x: [..., hidden] BF16/FP16.
        gate: same shape and dtype as ``x``.
        weight: [hidden] same dtype as ``x``.
        eps: variance epsilon.
        n_groups: number of groups along the hidden dim. ``hidden`` must be
            divisible by ``n_groups``.

    Returns:
        Fresh tensor, same shape and dtype as ``x``.
    """
    # Flatten leading dims so the kernel sees a 2D [M, H] view of each input.
    # We must NOT call ``.reshape(-1, hidden)`` blindly — if the tensor is a
    # non-contiguous slice along the last dim (e.g. gate = projected_states[
    # ..., :hidden]), reshape would silently materialize a contiguous copy and
    # we would lose the zero-copy property; if the slice is along an interior
    # dim, the resulting strides would be incompatible with our 2D kernel.
    # Instead, we collapse the leading dims via ``view`` when contiguous in
    # the leading dims, and otherwise call ``.contiguous()`` to fall back to
    # safe behavior.
    orig_shape = x.shape
    hidden = orig_shape[-1]
    assert hidden % n_groups == 0, (
        f"hidden={hidden} not divisible by n_groups={n_groups}"
    )
    group_size = hidden // n_groups

    def _to_2d(t: torch.Tensor) -> torch.Tensor:
        # If the inner (hidden) dim is unit stride and the leading dims have
        # the canonical contiguous strides for that hidden stride, we can
        # ``view`` without a copy; otherwise call ``.contiguous()`` (rare
        # path; only triggers for unusual layouts the kernel cannot handle).
        if t.dim() == 2:
            if t.stride(-1) == 1:
                return t
            return t.contiguous()
        # >2D: collapse leading dims. The flatten is safe iff the inner dim
        # is unit stride and the second-innermost dim's stride equals the
        # inner extent (i.e. (hidden,) is the contiguous suffix). Otherwise,
        # fall back to ``.contiguous()``.
        if t.stride(-1) == 1 and t.is_contiguous():
            return t.view(-1, hidden)
        # Non-contiguous higher-rank input: contiguous() copies.
        return t.contiguous().view(-1, hidden)

    x_2d = _to_2d(x)
    gate_2d = _to_2d(gate)
    M = x_2d.shape[0]

    # Allocate a fresh contiguous output (mutates_args=[] requires fresh).
    out_2d = torch.empty(M, hidden, device=x.device, dtype=x.dtype)

    if M == 0:
        return out_2d.view(orig_shape)

    # Kernel REQUIRES inner-dim contiguity (unit stride along hidden).
    assert x_2d.stride(-1) == 1, "x must be unit-stride along the hidden dim"
    assert gate_2d.stride(-1) == 1, (
        "gate must be unit-stride along the hidden dim"
    )

    # next_power_of_2 for the BLOCK constexpr; mask handles non-pow2 group_size.
    BLOCK_GS = triton.next_power_of_2(group_size)

    grid = (M, n_groups)
    _gated_rms_norm_fused_kernel[grid](
        x_2d,
        gate_2d,
        weight,
        out_2d,
        M,
        group_size,
        eps,
        x_2d.stride(0),
        gate_2d.stride(0),
        out_2d.stride(0),
        BLOCK_GS=BLOCK_GS,
        num_warps=4,
    )

    return out_2d.view(orig_shape)


def _gated_rms_norm_fused_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    n_groups: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="mamba2_gated_rms_norm_fused",
    op_func=_gated_rms_norm_fused_impl,
    mutates_args=[],
    fake_impl=_gated_rms_norm_fused_fake,
)


def gated_rms_norm_fused(
    x: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    n_groups: int,
) -> torch.Tensor:
    """Public entry point. Dispatches through the registered opaque custom op
    so that torch.compile/Inductor treats this as an atomic IR node.
    """
    return torch.ops.vllm.mamba2_gated_rms_norm_fused(
        x, gate, weight, eps, n_groups
    )
