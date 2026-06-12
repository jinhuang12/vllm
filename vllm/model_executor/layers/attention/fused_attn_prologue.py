# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMMO OP-016: fused attention-prologue kernel for Qwen3.5 hybrid models.

Fuses the 4-kernel decode prologue of the FULL-ATTENTION layers
(q/k Gemma-RMSNorm + neox RoPE + FP8-e4m3 q-quant + FP8-e4m3 paged KV scatter)
into ONE Triton kernel per layer, entirely downstream of the qkv GEMM and
upstream of the (opaque) FlashInfer FMHA cubin.

Replaces, per full-attn layer:
  triton_red_fused (q/k RMSNorm reduce)
  triton_poi rms_norm pointwise (cat/clamp/mul/reciprocal — also folds q-quant)
  triton_poi RoPE apply
  reshape_and_cache_flash (bf16->fp8 paged KV scatter)

Collapses 4 occupancy-starved launches + 3 intermediate q/k DRAM round-trips
into one pass. The qkv GEMM (nvjet g136), the FMHA cubin, and the GDN path are
NOT touched.

The kernel reads q/k/v directly from the contiguous qkv_proj output (column
offsets + explicit row stride), so there are no intermediate split/reshape
copies and no contiguity assumption on the slices.

Correctness notes (vs the production unfused chain):
  * Qwen3.5 q_norm/k_norm are GemmaRMSNorm: effective weight is (stored + 1.0),
    reduction in fp32 over head_dim, multiply in weight dtype, cast back.
  * RoPE: the model's MRotaryEmbedding (mrope_interleaved, section [11,11,10])
    reduces to plain neox RoPE on rotary_dim=64 for TEXT-ONLY input (all 3 mrope
    position rows carry the same token index — see gpu_model_runner comment and
    mrope.py partial-section sum). This kernel implements the text-only neox
    reduction; non-text (image/video) input is NOT supported and the call site
    must fall back.
  * cos_sin_cache layout is [max_pos, rotary_dim] = [cos(rotary_dim/2) |
    sin(rotary_dim/2)] per row (RotaryEmbedding._compute_cos_sin_cache).
  * FP8 q-quant matches QuantFP8 static per-tensor: clamp(q / q_scale,
    -448, 448).to(fp8_e4m3). KV scatter matches reshape_and_cache_flash /
    CopyWithScaleOp SATFINITE: clamp(kv / kv_scale, -448, 448).to(fp8_e4m3).
  * Paged KV scatter uses the ACTUAL cache strides (block/page/head), so it is
    byte-identical to reshape_and_cache_flash on both NHD and HND (B200) layouts.
  * Padded CUDA-graph tokens (slot_mapping < 0) skip the scatter, mirroring the
    C kernel's `if (slot_idx < 0) return;`.
"""

import torch
import triton
import triton.language as tl

from vllm.model_executor.layers.attention.attention import get_attention_context
from vllm.utils.torch_utils import (
    LayerNameType,
    _resolve_layer_name,
    direct_register_custom_op,
)

# fp8_e4m3 finite max (matches QuantFP8 _FP8_MAX and __NV_SATFINITE).
_FP8_MAX = 448.0


@triton.jit
def _fused_attn_prologue_kernel(
    # base inputs
    qkv_ptr,         # bf16 [T, qkv_width]  (q|gate interleaved, then k, then v)
    q_w_ptr,         # fp32/bf16 [D]  (stored q_norm weight; effective = w + 1)
    k_w_ptr,         # fp32/bf16 [D]  (stored k_norm weight; effective = w + 1)
    cos_sin_ptr,     # bf16 [max_pos, ROT]  ([cos(ROT/2) | sin(ROT/2)])
    pos_ptr,         # int64 [T]  (token sequence position -> cos_sin row index)
    slot_ptr,        # int64 [T]  (paged slot per token; <0 = padded -> skip scatter)
    q_scale_ptr,     # fp32 [1]
    k_scale_ptr,     # fp32 [1]
    v_scale_ptr,     # fp32 [1]
    # outputs
    q_out_ptr,       # fp8_e4m3 [T, Hq * D]  (contiguous)
    k_cache_ptr,     # fp8_e4m3 paged cache (strided)
    v_cache_ptr,     # fp8_e4m3 paged cache (strided)
    # qkv layout
    qkv_row_stride,  # elements per token row in qkv
    k_col_off,       # column offset of k block within a qkv row (= Hq*2*D)
    v_col_off,       # column offset of v block within a qkv row (= Hq*2*D + Hkv*D)
    # paged cache strides (elements)
    block_stride,
    page_stride,
    head_stride,
    # cos_sin row stride
    cos_sin_row_stride,
    # compile-time params
    BLOCK_SIZE: tl.constexpr,    # paged block size (slot % BLOCK_SIZE)
    Hq: tl.constexpr,
    Hkv: tl.constexpr,
    D: tl.constexpr,             # head_dim
    ROT: tl.constexpr,           # rotary_dim
    EPS: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    fp8_max: tl.constexpr = 448.0  # fp8_e4m3 finite max (SATFINITE / QuantFP8)

    half = ROT // 2
    in_rot = d < ROT
    # cos/sin row is indexed by the token's SEQUENCE POSITION, not its index.
    pos = tl.load(pos_ptr + tok)
    cos_sin_row = pos * cos_sin_row_stride
    # neox: freq index = d % half within the rotary block; cos at [freq],
    # sin at [half + freq]; partner lane is d+half (sign -1) for d<half,
    # d-half (sign +1) for half<=d<ROT.
    freq = d % half
    cos = tl.load(
        cos_sin_ptr + cos_sin_row + freq, mask=in_rot, other=1.0
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr + cos_sin_row + half + freq, mask=in_rot, other=0.0
    ).to(tl.float32)
    partner = tl.where(d < half, d + half, d - half)
    rot_sign = tl.where(d < half, -1.0, 1.0)

    # ---- Q path (gated): read ONLY the q half at per-head stride 2*D ----
    if head < Hq:
        inv_q = 1.0 / tl.load(q_scale_ptr)
        # q half lives in the first Hq*2*D columns; per head stride 2*D, q at +0.
        qoff = tok * qkv_row_stride + head * (2 * D)
        q = tl.load(qkv_ptr + qoff + d).to(tl.float32)
        # Gemma RMSNorm: fp32 reduction over head_dim, effective weight (w + 1).
        rms = tl.rsqrt(tl.sum(q * q, axis=0) / D + EPS)
        qw = tl.load(q_w_ptr + d).to(tl.float32) + 1.0
        qn = q * rms * qw
        # neox RoPE on the normed q: normed partner lane (re-load raw, re-norm
        # with the same per-head rms scalar and the partner's norm weight).
        qp_raw = tl.load(qkv_ptr + qoff + partner).to(tl.float32)
        qpw = tl.load(q_w_ptr + partner).to(tl.float32) + 1.0
        qp_n = qp_raw * rms * qpw
        q_roped = tl.where(in_rot, qn * cos + rot_sign * qp_n * sin, qn)
        # FP8 q-quant: clamp(q / q_scale, -448, 448).to(fp8).
        q_scaled = q_roped * inv_q
        q_scaled = tl.minimum(tl.maximum(q_scaled, -fp8_max), fp8_max)
        tl.store(q_out_ptr + tok * (Hq * D) + head * D + d, q_scaled.to(tl.float8e4nv))

    # ---- K + V path: k norm+rope -> fp8 scatter; v cast -> fp8 scatter ----
    if head < Hkv:
        slot = tl.load(slot_ptr + tok)
        if slot >= 0:
            inv_k = 1.0 / tl.load(k_scale_ptr)
            inv_v = 1.0 / tl.load(v_scale_ptr)
            koff = tok * qkv_row_stride + k_col_off + head * D
            voff = tok * qkv_row_stride + v_col_off + head * D
            k = tl.load(qkv_ptr + koff + d).to(tl.float32)
            krms = tl.rsqrt(tl.sum(k * k, axis=0) / D + EPS)
            kw = tl.load(k_w_ptr + d).to(tl.float32) + 1.0
            kn = k * krms * kw
            kp_raw = tl.load(qkv_ptr + koff + partner).to(tl.float32)
            kpw = tl.load(k_w_ptr + partner).to(tl.float32) + 1.0
            kp_n = kp_raw * krms * kpw
            k_roped = tl.where(in_rot, kn * cos + rot_sign * kp_n * sin, kn)
            k_scaled = k_roped * inv_k
            k_scaled = tl.minimum(tl.maximum(k_scaled, -fp8_max), fp8_max)

            v = tl.load(qkv_ptr + voff + d).to(tl.float32)
            v_scaled = v * inv_v
            v_scaled = tl.minimum(tl.maximum(v_scaled, -fp8_max), fp8_max)

            block_idx = slot // BLOCK_SIZE
            block_off = slot % BLOCK_SIZE
            dst = block_idx * block_stride + block_off * page_stride + head * head_stride
            tl.store(k_cache_ptr + dst + d, k_scaled.to(tl.float8e4nv))
            tl.store(v_cache_ptr + dst + d, v_scaled.to(tl.float8e4nv))


def _launch_fused_attn_prologue(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> torch.Tensor:
    """Run the fused prologue kernel. Mutates k_cache/v_cache in place; returns
    the FP8-e4m3 query [num_tokens, num_heads * head_dim]."""
    num_tokens = qkv.shape[0]
    q_out = torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=torch.float8_e4m3fn,
        device=qkv.device,
    )
    # qkv column layout: [q|gate (Hq*2*D)] [k (Hkv*D)] [v (Hkv*D)]
    k_col_off = num_heads * 2 * head_dim
    v_col_off = k_col_off + num_kv_heads * head_dim
    # Paged cache addressing strides (in elements), identical basis as
    # reshape_and_cache_flash: block_stride = stride(0), page_stride = stride(1),
    # head_stride = stride(2); block_size = size(1).
    block_stride = k_cache.stride(0)
    page_stride = k_cache.stride(1)
    head_stride = k_cache.stride(2)
    block_size = k_cache.shape[1]
    grid = (num_tokens, num_heads)
    _fused_attn_prologue_kernel[grid](
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        q_scale,
        k_scale,
        v_scale,
        q_out,
        k_cache,
        v_cache,
        qkv.stride(0),
        k_col_off,
        v_col_off,
        block_stride,
        page_stride,
        head_stride,
        cos_sin_cache.stride(0),
        BLOCK_SIZE=block_size,
        Hq=num_heads,
        Hkv=num_kv_heads,
        D=head_dim,
        ROT=rotary_dim,
        EPS=eps,
    )
    return q_out


def fused_attn_prologue_impl(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: LayerNameType,
) -> torch.Tensor:
    """Opaque custom op: fetches kv_cache + slot_mapping + scales from the
    forward context (mirrors unified_kv_cache_update), runs the fused kernel,
    scatters the FP8 KV cache as a side effect, and RETURNS the FP8 query.

    The returned q_fp8 is consumed by the FMHA as `query`, which creates the
    data dependency that keeps torch.compile from reordering the FMHA before
    this op (so the KV scatter is always visible to the FMHA). No separate
    dummy-dep tensor is required.
    """
    layer_name = _resolve_layer_name(layer_name)
    _, attn_layer, kv_cache, layer_slot_mapping = get_attention_context(layer_name)

    # MRoPE passes positions as [3, num_tokens]; for text-only decode all three
    # rows are identical, so the text row (row 0) is the sequence position.
    if positions.dim() == 2:
        positions = positions[0]

    num_tokens = qkv.shape[0]
    if layer_slot_mapping is None:
        # Profiling / no-cache run: still emit FP8 q so the FMHA contract holds.
        return torch.empty(
            (num_tokens, num_heads * head_dim),
            dtype=torch.float8_e4m3fn,
            device=qkv.device,
        )

    # Split the logical [num_blocks, 2, ...] cache and view as fp8_e4m3 (matches
    # FlashInferImpl.forward's kv_cache.view(fp8) and do_kv_cache_update's
    # kv_cache[:, 0] / [:, 1]). Strides are read from these views in the kernel.
    k_cache = kv_cache[:, 0]
    v_cache = kv_cache[:, 1]
    if k_cache.dtype != torch.float8_e4m3fn:
        k_cache = k_cache.view(torch.float8_e4m3fn)
        v_cache = v_cache.view(torch.float8_e4m3fn)

    return _launch_fused_attn_prologue(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        layer_slot_mapping,
        attn_layer._q_scale,
        attn_layer._k_scale,
        attn_layer._v_scale,
        k_cache,
        v_cache,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
    )


def fused_attn_prologue_fake(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
    layer_name: LayerNameType,
) -> torch.Tensor:
    num_tokens = qkv.shape[0]
    return torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=torch.float8_e4m3fn,
        device=qkv.device,
    )


direct_register_custom_op(
    op_name="fused_attn_prologue",
    op_func=fused_attn_prologue_impl,
    mutates_args=[],
    fake_impl=fused_attn_prologue_fake,
)


def is_fused_attn_prologue_supported(attn: "Attention") -> bool:
    """Eligibility check for the OP-016 fused prologue, evaluated at call time.

    Requires:
      * env flag VLLM_FUSE_ATTN_PROLOGUE enabled,
      * FP8-e4m3 paged KV cache with a working static q-quant (query_quant set),
      * scales loaded (not runtime-calibrated),
      * opaque attention op (the FlashInfer torch.ops.vllm.* path; B200/CUDA),
      * a separate (non-fused) backend KV-write path that we are taking over.

    Falls back to the unfused production chain whenever any condition fails, so
    the optimization is strictly additive and never changes correctness when off.
    """
    import vllm.envs as envs

    if not envs.VLLM_FUSE_ATTN_PROLOGUE:
        return False
    if attn.query_quant is None:
        return False
    if not getattr(attn.impl, "supports_quant_query_input", False):
        return False
    if attn.calculate_kv_scales:
        # Scales not yet calibrated — defer to the baseline chain.
        return False
    if attn.kv_sharing_target_layer_name is not None:
        return False
    if attn.attn_backend.forward_includes_kv_cache_update:
        # Backend writes KV inside its own forward; we cannot take it over.
        return False
    if attn.use_direct_call:
        # Non-opaque path: this integration targets the opaque torch.ops path.
        return False
    return True


def fused_attn_prologue_forward(
    attn: "Attention",
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    eps: float,
) -> torch.Tensor:
    """Fused replacement for the full-attention decode prologue + FMHA call.

    Runs the fused Triton op (q/k norm + RoPE + FP8 q-quant + FP8 KV scatter),
    which returns the FP8 query and writes the paged KV cache as a side effect,
    then dispatches the FlashInfer FMHA directly over the freshly-written cache.

    The internal query-quant and unified_kv_cache_update of Attention.forward are
    intentionally bypassed (the fused op already did both). The FP8 query carries
    the data dependency that orders the FMHA after the KV scatter.

    Returns the attention output [num_tokens, num_heads * head_dim] (pre-gate).
    """
    from vllm.model_executor.layers.attention.attention import _encode_layer_name

    q_fp8 = torch.ops.vllm.fused_attn_prologue(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        eps,
        _encode_layer_name(attn.layer_name),
    )

    num_tokens = q_fp8.shape[0]
    output = torch.empty(
        (num_tokens, num_heads * head_dim),
        dtype=qkv.dtype,
        device=qkv.device,
    )
    q_view = q_fp8.view(-1, num_heads, head_dim)
    out_view = output.view(-1, num_heads, head_dim)
    # FMHA reads K/V from the paged cache; key/value args are sliced but unused on
    # the non-DCP decode and prefill paths. Pass the FP8 q (which the FMHA cubin
    # requires) as both query and the key/value placeholders to satisfy the
    # custom-op schema (non-None tensors) without an extra allocation.
    encoded = _encode_layer_name(attn.layer_name)
    torch.ops.vllm.unified_attention_with_output(
        q_view,
        q_view,
        q_view,
        out_view,
        encoded,
        kv_cache_dummy_dep=None,
    )
    return output.view(-1, num_heads * head_dim)
