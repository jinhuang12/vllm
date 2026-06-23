# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Fused Triton MHC Post + Add kernel (OP-001).

Replaces tilelang mhc_post_tilelang_kernel + triton_add with a single
fused Triton kernel that has better SM utilization at small batch sizes.

The kernel computes:
  out[n, h, i] = post_mix[n, h] * (x[n, i] + x_add[n, i])
                 + sum_j(comb_mix[n, j, h] * residual[n, j, i])

When x_add is None (attention block path), the add is skipped:
  out[n, h, i] = post_mix[n, h] * x[n, i]
                 + sum_j(comb_mix[n, j, h] * residual[n, j, i])

Grid: (num_tokens * n_h_tiles,) where n_h_tiles = H / BLOCK_H
  - At BS=8, H=4096, BLOCK_H=1024: grid = 8*4 = 32 CTAs
  - vs tilelang grid=(8,) = 4x more parallelism on 192-SM B200

Dispatch gate: Triton for M <= 12, tilelang for M > 12
  (Triton benefits from better SM fill at small M; tilelang is
   already efficient at larger M where grid is naturally larger)
"""

import torch
import triton
import triton.language as tl

from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _mhc_post_fused_kernel(
    comb_ptr,   # [N, HC, HC] fp32
    res_ptr,    # [N, HC, H] bf16
    pmix_ptr,   # [N, HC] fp32
    x_ptr,      # [N, H] bf16
    x_add_ptr,  # [N, H] bf16 or null (0 pointer)
    out_ptr,    # [N, HC, H] bf16
    N,
    H: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_H: tl.constexpr,
    HAS_ADD: tl.constexpr,
):
    """
    Fused MHC Post kernel with optional input-side add.

    Each program handles one (token, h_tile) pair.
    For each head h in HC, computes:
      out[n, h, h_start:h_start+BLOCK_H] =
          pmix[n, h] * (x[n, h_range] + x_add[n, h_range])
          + sum_j comb[n, j, h] * res[n, j, h_range]
    """
    pid = tl.program_id(0)
    n_tiles = tl.cdiv(H, BLOCK_H)
    token_id = pid // n_tiles
    tile_id = pid % n_tiles

    if token_id >= N:
        return

    h_start = tile_id * BLOCK_H
    h_offs = h_start + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    # Load x[token_id, h_start:h_start+BLOCK_H] -> fp32
    x_vals = tl.load(
        x_ptr + token_id * H + h_offs, mask=h_mask, other=0.0
    ).to(tl.float32)

    # Fused add: x_vals += x_add[token_id, h_range]
    # CRITICAL: Truncate to bf16 after add to match production's intermediate
    # bf16 store (triton_add outputs bf16, then hc_post loads bf16).
    # Without this truncation, fp32 precision would cause divergence.
    if HAS_ADD:
        x_add_vals = tl.load(
            x_add_ptr + token_id * H + h_offs, mask=h_mask, other=0.0
        ).to(tl.float32)
        x_vals = (x_vals + x_add_vals).to(tl.bfloat16).to(tl.float32)

    # Load comb_mix and post_mix bases for this token
    comb_base = comb_ptr + token_id * HC * HC
    pmix_base = pmix_ptr + token_id * HC

    # Process each output head
    for h_idx in tl.static_range(HC):
        # post_mix[n, h] * x_vals
        pmix_val = tl.load(pmix_base + h_idx)
        acc = pmix_val * x_vals

        # + sum_j comb[n, j, h] * res[n, j, h_range]
        for j in tl.static_range(HC):
            comb_val = tl.load(comb_base + j * HC + h_idx)
            res_offs = res_ptr + token_id * HC * H + j * H + h_offs
            res_vals = tl.load(res_offs, mask=h_mask, other=0.0).to(tl.float32)
            acc += comb_val * res_vals

        # CRITICAL: Truncate fp32 accumulator to bf16 before store
        # This matches production's tilelang kernel which stores bf16
        # intermediate results. Without truncation, the higher precision
        # of fp32 accumulation would cause divergence from baseline.
        out_offs = out_ptr + token_id * HC * H + h_idx * H + h_offs
        tl.store(out_offs, acc.to(tl.bfloat16), mask=h_mask)


def mhc_post_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    x_add: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Triton implementation of mhc_post with optional fused input add.

    Args:
        x: [N, H] bf16 — layer output (attention or FFN/MoE)
        residual: [N, HC, H] bf16 — multi-head residual
        post_layer_mix: [N, HC, 1] fp32 — post-mix weights
        comb_res_mix: [N, HC, HC] fp32 — combine mixing matrix
        x_add: [N, H] bf16 or None — optional tensor to add to x
               (shared expert output for MoE block)

    Returns:
        out: [N, HC, H] bf16
    """
    N = residual.shape[0]
    HC = residual.shape[1]
    H = residual.shape[2]

    out = torch.empty_like(residual)

    # Squeeze post_layer_mix from [N, HC, 1] to [N, HC]
    pmix = post_layer_mix.squeeze(-1)

    BLOCK_H = 1024
    n_tiles = (H + BLOCK_H - 1) // BLOCK_H
    grid = (N * n_tiles,)

    has_add = x_add is not None

    _mhc_post_fused_kernel[grid](
        comb_res_mix,
        residual,
        pmix,
        x,
        x_add if has_add else x,  # dummy pointer when no add
        out,
        N,
        H,
        HC,
        BLOCK_H,
        HAS_ADD=has_add,
        num_warps=4,
        num_stages=2,
    )
    return out


def mhc_post_fused_add(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    x_add: torch.Tensor,
) -> torch.Tensor:
    """
    Fused mhc_post + input-side add. Used in MoE block path where
    x = routed_experts_output and x_add = shared_expert_output.

    Computes: out = mhc_post(x + x_add, residual, post_layer_mix, comb_res_mix)
    in a single kernel launch (eliminates separate triton_add).

    Dispatch gate: uses Triton fused kernel for M <= 12, falls back to
    tilelang + separate add for M > 12 (to avoid regression at large M).
    """
    from vllm.model_executor.layers.mhc import _MHC_POST_TRITON_M_THRESHOLD

    num_tokens = residual.shape[0] if residual.dim() == 3 else residual.shape[-3]
    if num_tokens <= _MHC_POST_TRITON_M_THRESHOLD:
        return mhc_post_triton(x, residual, post_layer_mix, comb_res_mix, x_add)

    # Fallback: separate add + tilelang mhc_post for large M
    from vllm.model_executor.layers.mhc import mhc_post_tilelang

    x_combined = x + x_add
    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x_combined,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


def _mhc_post_fused_add_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    x_add: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


# Register the fused op for torch.compile compatibility
direct_register_custom_op(
    op_name="mhc_post_fused_add",
    op_func=mhc_post_fused_add,
    mutates_args=[],
    fake_impl=_mhc_post_fused_add_fake,
)
