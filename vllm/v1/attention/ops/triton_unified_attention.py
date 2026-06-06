# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Authors:
#  - Burkhard Ringlein <ngl@zurich.ibm.com>
#  - Jan van Lunteren <jvl@zurich.ibm.com>
#  - Chih-Chieh Yang <chih.chieh.yang@ibm.com>
#  - Thomas Parnell <tpa@zurich.ibm.com>

import atexit
import os
from typing import Any

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_attention_helpers import (
    apply_alibi_to_score,
    apply_softcap,
    cdiv_fn,
    compute_kv_seq_mask,
    compute_tile_loop_bounds,
    compute_window_segments,
    find_seq_idx,
    init_softmax_M,
    load_qq_bias_tile,
    resolve_seq_and_query_len,
    softmax_step,
    store_segm_reduce_scalars,
)
from vllm.v1.kv_cache_interface import KVQuantMode

logger = init_logger(__name__)
is_batch_invariant = envs.VLLM_BATCH_INVARIANT
float8_info = torch.finfo(current_platform.fp8_dtype())

# AMMO OP-017: dispatch-coverage instrumentation.  Counts the number of
# unified_attention launches by category so we can compute the realized
# coverage fraction of the BLOCK_M=32 / num_warps=8 reroute on the real
# workload (chunked-prefill input-len 27000 + MTP spec-decode).  Both
# fastpath-evidence (proving the opt arm is not silently a baseline run)
# and the deflation-by-coverage requirement from the lead's HARD pre-
# validator gate read this counter.  Only updated when VLLM_OP017_COVERAGE
# is enabled to keep the production hot path overhead-free.
_op017_dispatch_counters: dict[str, int] = {
    "rerouted_global_2d_prefill": 0,
    "passthru_global_3d_decode": 0,
    "passthru_sliding_2d": 0,
    "passthru_sliding_3d": 0,
    "passthru_other_head_size": 0,
    "passthru_other_nqpkv": 0,
    "env_off_total_launches": 0,
}
_OP017_FIRST_FIRE_LOGGED = False
_OP017_FIRST_PASSTHRU_LOGGED = False

# AMMO OP-019: dispatch-coverage instrumentation (mirrors OP-017's pattern).
# Counts the number of unified_attention launches by category so we can
# compute the realized coverage fraction of the BLOCK_M=64 / num_warps=8
# reroute on the real workload (chunked-prefill input-len 27000 + MTP
# spec-decode).  Both fastpath-evidence (proving the opt arm is not silently
# a baseline run) and the deflation-by-coverage requirement from the lead's
# obligation C read this counter.  Coverage =
# rerouted_sliding_2d_prefill / (rerouted_sliding_2d_prefill +
# passthru_sliding_3d_decode + passthru_sliding_other_nqpkv).  The
# denominator excludes global-attention shapes (head_size==512, owned by
# OP-017) and includes only launches where the sliding-attention head_size /
# nqpkv could in principle be rerouted by OP-019.  Per memory
# `[ammo-opt-arm-silently-disabled-masquerades-as-clean]` this is the guard
# against a silently-disabled opt arm.
_op019_dispatch_counters: dict[str, int] = {
    "rerouted_sliding_2d_prefill": 0,
    "passthru_sliding_3d_decode": 0,
    "passthru_sliding_global": 0,        # head_size==256 but window_size<0
    "passthru_sliding_other_nqpkv": 0,    # head_size==256 + sliding but nqpkv != 2
    "passthru_global_head_size": 0,       # head_size==512 (OP-017 territory)
    "passthru_other_head_size": 0,        # head_size not in {256, 512}
    "env_off_total_launches": 0,
}
_OP019_FIRST_FIRE_LOGGED = False
_OP019_FIRST_PASSTHRU_LOGGED = False


# AMMO OP-033: dispatch-coverage instrumentation (mirrors OP-017 / OP-019).
# OP-033 is a strict reslice of OP-019's coverage set: the TILE=64 widening
# fires IFF the OP-019 BLOCK_M=64 reroute has fired AND VLLM_OP033 is on.
# Same fastpath-evidence + deflation-by-coverage rationale per memory
# `[ammo-opt-arm-silently-disabled-masquerades-as-clean]`.  Coverage =
# rerouted_tile64_2d_prefill / (rerouted_tile64_2d_prefill +
# passthru_op019_off_2d_prefill + passthru_sliding_3d_decode +
# passthru_sliding_other_nqpkv).
_op033_dispatch_counters: dict[str, int] = {
    "rerouted_tile64_2d_prefill": 0,
    "passthru_op019_off_2d_prefill": 0,    # head_size==256+nqpkv==2+sliding+2D
                                            # but VLLM_OP019 is OFF -> BLOCK_M=16
    "passthru_sliding_3d_decode": 0,
    "passthru_sliding_global": 0,           # head_size==256, sliding_window<0
    "passthru_sliding_other_nqpkv": 0,
    "passthru_global_head_size": 0,
    "passthru_other_head_size": 0,
    "env_off_total_launches": 0,
}
_OP033_FIRST_FIRE_LOGGED = False


def _op017_dump_counters_atexit() -> None:
    """Dump the dispatch-coverage counters on process exit.

    Engine subprocesses (multiproc executor + spec-decode drafter) each
    own their own copy of the module state; each subprocess runs its own
    atexit hook.  We print to stderr so the line lands in the bench
    supervisor log even if stdout is captured.

    Coverage = rerouted_global_2d_prefill / (rerouted_global_2d_prefill +
    passthru_global_3d_decode + passthru_other_nqpkv + passthru_other_head_size).
    The denominator excludes sliding-window shapes (separately scoped to
    §6.4) and includes only launches where the global-attention head_size /
    nqpkv could in principle be rerouted.
    """
    pid = os.getpid()
    if envs.VLLM_OP017:
        c = _op017_dispatch_counters
        total_global_eligible_shapes = (
            c["rerouted_global_2d_prefill"]
            + c["passthru_global_3d_decode"]
            + c["passthru_other_nqpkv"]
        )
        # head_size==512 launches that did or could have rerouted (excluding
        # sliding hd256 — those have their own §6.4 obligation).
        if total_global_eligible_shapes > 0:
            cov_pct = (
                100.0
                * c["rerouted_global_2d_prefill"]
                / total_global_eligible_shapes
            )
        else:
            cov_pct = 0.0
        # Stamp the message with a unique tag so the harness can grep it
        # out of the supervisor log and the engine subprocess logs.
        msg = (
            f"OP017_COVERAGE_REPORT pid={pid} env=ON "
            f"rerouted={c['rerouted_global_2d_prefill']} "
            f"passthru_global_3d_decode={c['passthru_global_3d_decode']} "
            f"passthru_sliding_2d={c['passthru_sliding_2d']} "
            f"passthru_sliding_3d={c['passthru_sliding_3d']} "
            f"passthru_other_head_size={c['passthru_other_head_size']} "
            f"passthru_other_nqpkv={c['passthru_other_nqpkv']} "
            f"global_eligible_total={total_global_eligible_shapes} "
            f"coverage_global_2d_prefill_pct={cov_pct:.2f}"
        )
    else:
        msg = (
            f"OP017_COVERAGE_REPORT pid={pid} env=OFF "
            f"env_off_total_launches="
            f"{_op017_dispatch_counters['env_off_total_launches']}"
        )
    # Print to stderr so it lands even if stdout is buffered/captured.
    import sys as _sys
    print(msg, flush=True, file=_sys.stderr)


# Register the atexit hook exactly once per process.  The smoke test
# uses importlib.reload() to flip envs.VLLM_OP017 mid-run; without this
# guard each reload would push another copy of the hook onto the atexit
# stack and the report would print N times at process exit.
if not getattr(atexit, "_op017_registered", False):
    atexit.register(_op017_dump_counters_atexit)
    atexit._op017_registered = True  # type: ignore[attr-defined]


def _op019_dump_counters_atexit() -> None:
    """Dump the OP-019 dispatch-coverage counters on process exit (mirrors
    OP-017).

    Coverage = rerouted_sliding_2d_prefill / (rerouted_sliding_2d_prefill +
    passthru_sliding_3d_decode + passthru_sliding_other_nqpkv).  The
    denominator covers launches where the sliding-window head_size==256 path
    was eligible in principle; head_size==512 / global / other heads are
    excluded (different OP-id territories).
    """
    pid = os.getpid()
    if envs.VLLM_OP019:
        c = _op019_dispatch_counters
        total_sliding_eligible_shapes = (
            c["rerouted_sliding_2d_prefill"]
            + c["passthru_sliding_3d_decode"]
            + c["passthru_sliding_other_nqpkv"]
        )
        if total_sliding_eligible_shapes > 0:
            cov_pct = (
                100.0
                * c["rerouted_sliding_2d_prefill"]
                / total_sliding_eligible_shapes
            )
        else:
            cov_pct = 0.0
        msg = (
            f"OP019_COVERAGE_REPORT pid={pid} env=ON "
            f"rerouted={c['rerouted_sliding_2d_prefill']} "
            f"passthru_sliding_3d_decode={c['passthru_sliding_3d_decode']} "
            f"passthru_sliding_global={c['passthru_sliding_global']} "
            f"passthru_sliding_other_nqpkv={c['passthru_sliding_other_nqpkv']} "
            f"passthru_global_head_size={c['passthru_global_head_size']} "
            f"passthru_other_head_size={c['passthru_other_head_size']} "
            f"sliding_eligible_total={total_sliding_eligible_shapes} "
            f"coverage_sliding_2d_prefill_pct={cov_pct:.2f}"
        )
    else:
        msg = (
            f"OP019_COVERAGE_REPORT pid={pid} env=OFF "
            f"env_off_total_launches="
            f"{_op019_dispatch_counters['env_off_total_launches']}"
        )
    import sys as _sys
    print(msg, flush=True, file=_sys.stderr)


if not getattr(atexit, "_op019_registered", False):
    atexit.register(_op019_dump_counters_atexit)
    atexit._op019_registered = True  # type: ignore[attr-defined]


def _op033_dump_counters_atexit() -> None:
    """Dump the OP-033 dispatch-coverage counters on process exit (mirrors
    OP-017 / OP-019).

    Coverage = rerouted_tile64_2d_prefill / (rerouted_tile64_2d_prefill +
    passthru_op019_off_2d_prefill + passthru_sliding_3d_decode +
    passthru_sliding_other_nqpkv).  The denominator covers launches where
    the sliding-window head_size==256 path was eligible in principle for
    the TILE=64 stacked widening (i.e. would have rerouted under OP-019
    BLOCK_M=64).  Other head sizes / shapes are tagged for diagnostics.
    """
    pid = os.getpid()
    if envs.VLLM_OP033:
        c = _op033_dispatch_counters
        total_eligible = (
            c["rerouted_tile64_2d_prefill"]
            + c["passthru_op019_off_2d_prefill"]
            + c["passthru_sliding_3d_decode"]
            + c["passthru_sliding_other_nqpkv"]
        )
        if total_eligible > 0:
            cov_pct = (
                100.0
                * c["rerouted_tile64_2d_prefill"]
                / total_eligible
            )
        else:
            cov_pct = 0.0
        msg = (
            f"OP033_COVERAGE_REPORT pid={pid} env=ON "
            f"rerouted={c['rerouted_tile64_2d_prefill']} "
            f"passthru_op019_off_2d_prefill="
            f"{c['passthru_op019_off_2d_prefill']} "
            f"passthru_sliding_3d_decode={c['passthru_sliding_3d_decode']} "
            f"passthru_sliding_global={c['passthru_sliding_global']} "
            f"passthru_sliding_other_nqpkv="
            f"{c['passthru_sliding_other_nqpkv']} "
            f"passthru_global_head_size={c['passthru_global_head_size']} "
            f"passthru_other_head_size={c['passthru_other_head_size']} "
            f"sliding_eligible_total={total_eligible} "
            f"coverage_tile64_2d_prefill_pct={cov_pct:.2f}"
        )
    else:
        msg = (
            f"OP033_COVERAGE_REPORT pid={pid} env=OFF "
            f"env_off_total_launches="
            f"{_op033_dispatch_counters['env_off_total_launches']}"
        )
    import sys as _sys
    print(msg, flush=True, file=_sys.stderr)


if not getattr(atexit, "_op033_registered", False):
    atexit.register(_op033_dump_counters_atexit)
    atexit._op033_registered = True  # type: ignore[attr-defined]


@triton.jit
def _cast_kv_tile(data, Q, tensor_scale, KV_QUANT_MODE: tl.constexpr):
    """Cast a loaded KV tile to Q's dtype, dequantizing if needed.

    Modes handled inside the core kernel:

    - ``KV_QUANT_MODE == 0`` (NONE) and ``2`` (INT8 per-token-head) and
      ``3`` (FP8 per-token-head): plain cast.  Per-token-head modes apply
      their scales separately on S/P inside the loop.
    - ``KV_QUANT_MODE == 1`` (FP8 per-tensor): dequantize using the
      tensor-wide scale.
    """
    if KV_QUANT_MODE == 1:
        if Q.dtype.is_fp8():
            return data.to(Q.dtype)
        return (data.to(tl.float32) * tl.load(tensor_scale)).to(Q.dtype)
    return data.to(Q.dtype)


@triton.jit
def kernel_unified_attention(
    # Output destinations.  In 2D mode we write the final result into
    # ``output_ptr``; in 3D mode we write per-segment partials into the
    # three ``segm_*`` tensors and ``output_ptr`` is unused (callers may
    # pass any non-null pointer).
    output_ptr,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    # Inputs
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    # Per-(token, head) scale caches (used iff KV_QUANT_MODE in {2, 3}).
    # For other modes callers may pass any non-null pointer.
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    # Scalars
    scale,
    k_scale,
    v_scale,
    out_scale,
    softcap,
    num_query_heads: tl.constexpr,  # int
    num_queries_per_kv: tl.constexpr,  # int
    block_table_stride: tl.int64,  # int
    query_stride_0: tl.int64,  # int
    query_stride_1: tl.int64,  # int, should be equal to head_size
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    qq_bias_stride_0: tl.int64,  # int
    BLOCK_SIZE: tl.constexpr,  # int
    TILE_SIZE: tl.constexpr,  # int must be power of 2
    HEAD_SIZE: tl.constexpr,  # int
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    USE_ALIBI_SLOPES: tl.constexpr,  # bool
    USE_ALIBI_SQRT: tl.constexpr,  # bool
    USE_QQ_BIAS: tl.constexpr,  # bool
    USE_SOFTCAP: tl.constexpr,  # bool
    USE_SINKS: tl.constexpr,  # bool
    SLIDING_WINDOW: tl.constexpr,  # int
    USE_MM_PREFIX: tl.constexpr,  # bool
    MAX_MM_RANGES: tl.constexpr,  # int
    mm_prefix_range_ptr,
    stride_k_cache_0: tl.int64,  # int
    stride_k_cache_1: tl.int64,  # int
    stride_k_cache_2: tl.int64,  # int
    stride_k_cache_3: tl.constexpr,  # int
    stride_v_cache_0: tl.int64,  # int
    stride_v_cache_1: tl.int64,  # int
    stride_v_cache_2: tl.int64,  # int
    stride_v_cache_3: tl.constexpr,  # int
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    query_start_len_ptr,
    BLOCK_Q: tl.constexpr,
    num_seqs: tl.int32,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
    USE_FP8: tl.constexpr,
    # Toggles 2D vs 3D layout.  The 2D path runs the full sequence in one
    # tile loop and writes to ``output_ptr``.  The 3D path scopes the loop
    # to ``[segm_idx, segm_idx+1) × tiles_per_segment`` and writes
    # per-segment partials, finalized by ``reduce_segments``.
    IS_3D: tl.constexpr,
    # KV cache quantization mode handled inside this kernel via constexpr
    # branches: NONE (0), FP8_PER_TENSOR (1), INT8_PER_TOKEN_HEAD (2),
    # FP8_PER_TOKEN_HEAD (3).
    KV_QUANT_MODE: tl.constexpr = 0,
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # Chunked / block-local attention.  ``CHUNK_LOOKBACK >= 0`` enables
    # chunked masking (used by Gemma3 block-local layers); takes precedence
    # over ``SLIDING_WINDOW`` inside the helpers.  ``-1`` disables.
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    # OP-003: when True (3D + sliding-window + spec-decode regime), the
    # ``NUM_SEGMENTS_PER_SEQ`` parallel-softmax segments tile only the
    # sliding window's tile range rather than the full sequence, so
    # segment parallelism is not collapsed to 1/NSEG at long context.
    # Default False preserves the full-sequence (global / existing)
    # segmentation byte-for-byte.
    WINDOW_SEG_3D: tl.constexpr = False,
):
    USE_PER_TOKEN_HEAD_SCALES: tl.constexpr = KV_QUANT_MODE >= 2

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2) if IS_3D else 0

    (
        seq_idx,
        q_block_local_idx,
        cur_batch_in_all_start_index,
        cur_batch_query_len,
        seq_len,
    ) = resolve_seq_and_query_len(
        query_start_len_ptr, seq_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q
    )

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    # context_len is needed both for window segmentation (3D) and the
    # tile loop below; compute it once here.
    context_len = seq_len - cur_batch_query_len

    if IS_3D:
        if WINDOW_SEG_3D:
            # Window-relative segmentation (OP-003): segments tile only
            # the sliding window's tile range for THIS q-block.  The
            # early-return below skips segments past the window's active
            # count; those segment buffers are left untouched and are
            # masked out by ``reduce_segments`` (which recomputes the
            # SAME act_num_segments via compute_window_segments).
            _ws_tile_start, tiles_per_segment, _ws_act_segments = (
                compute_window_segments(
                    context_len,
                    seq_len,
                    cur_batch_query_len,
                    q_block_local_idx,
                    NUM_SEGMENTS_PER_SEQ,
                    TILE_SIZE,
                    BLOCK_M,
                    BLOCK_Q,
                    num_queries_per_kv,
                    SLIDING_WINDOW,
                    USE_MM_PREFIX,
                    CHUNK_LOOKBACK,
                    CHUNK_SIZE,
                )
            )
            if segm_idx >= _ws_act_segments:
                return
        else:
            tiles_per_segment = cdiv_fn(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
            if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
                return
    else:
        tiles_per_segment = 0

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0[:, None] * query_stride_0
        + query_offset_1[:, None] * query_stride_1
        + offs_d[None, :]
    )

    dim_mask = tl.where(offs_d < HEAD_SIZE, 1, 0).to(tl.int1)
    query_mask_0 = tl.where(query_pos < cur_batch_query_len, 1, 0).to(tl.int1)
    query_mask_1 = tl.where(query_offset_1 < num_query_heads, 1, 0).to(tl.int1)

    # Q : (BLOCK_M, HEAD_SIZE_PADDED)
    Q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )

    block_table_offset = seq_idx * block_table_stride

    M = init_softmax_M(
        sink_ptr, query_offset_1, query_mask_1, segm_idx, BLOCK_M, USE_SINKS, IS_3D
    )
    L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    # acc : (BLOCK_M, HEAD_SIZE_PADDED)
    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(
            alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0
        )

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    loop_lo, loop_hi, max_seq_prefix_len = compute_tile_loop_bounds(
        context_len,
        seq_len,
        cur_batch_query_len,
        q_block_local_idx,
        segm_idx,
        tiles_per_segment,
        TILE_SIZE,
        BLOCK_M,
        BLOCK_Q,
        num_queries_per_kv,
        SLIDING_WINDOW,
        USE_MM_PREFIX,
        IS_3D,
        CHUNK_LOOKBACK,
        CHUNK_SIZE,
        WINDOW_SEG_3D,
    )

    # iterate through tiles (now limited to the sliding window range)
    for j in range(loop_lo, loop_hi):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE
        ).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )
        # K : (HEAD_SIZE, TILE_SIZE)
        K_load = tl.load(
            key_cache_ptr + k_offset,
            mask=dim_mask[:, None] & tile_mask[None, :],
            other=0.0,
        )
        K = _cast_kv_tile(K_load, Q, k_scale, KV_QUANT_MODE)
        # V : (TILE_SIZE, HEAD_SIZE)
        V_load = tl.load(
            value_cache_ptr + v_offset,
            mask=dim_mask[None, :] & tile_mask[:, None],
            other=0.0,
        )
        V = _cast_kv_tile(V_load, Q, v_scale, KV_QUANT_MODE)

        # Per-(token, head) scales for INT8 / FP8 per-token-head modes.
        if USE_PER_TOKEN_HEAD_SCALES:
            scale_idx = (
                physical_block_idx * stride_ks_blk
                + (seq_offset % BLOCK_SIZE) * stride_ks_slot
                + kv_head_idx * stride_ks_head
            )
            k_token_head_scales = tl.load(
                k_scale_cache_ptr + scale_idx, mask=tile_mask, other=1.0
            )
            v_scale_idx = (
                physical_block_idx * stride_vs_blk
                + (seq_offset % BLOCK_SIZE) * stride_vs_slot
                + kv_head_idx * stride_vs_head
            )
            v_token_head_scales = tl.load(
                v_scale_cache_ptr + v_scale_idx, mask=tile_mask, other=1.0
            )

        query_abs_pos = context_len + query_pos[:, None]
        seq_mask = compute_kv_seq_mask(
            query_abs_pos,
            seq_offset,
            seq_idx,
            mm_prefix_range_ptr,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            MAX_MM_RANGES,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )

        # S : (BLOCK_M, TILE_SIZE)
        S = tl.zeros(shape=(BLOCK_M, TILE_SIZE), dtype=tl.float32)
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: fuse softmax_scale with per-head k_scale
            # to avoid a separate BLOCK_M × TILE_SIZE multiply on S.
            S += tl.dot(Q, K) * (scale * k_token_head_scales[None, :])
        else:
            S += scale * tl.dot(Q, K)

        if USE_SOFTCAP:
            S = apply_softcap(S, softcap)

        S = tl.where(
            query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, S, float("-inf")
        )

        if USE_ALIBI_SLOPES:
            S = apply_alibi_to_score(
                S, alibi_slope, seq_offset, context_len, query_pos, USE_ALIBI_SQRT
            )

        if USE_QQ_BIAS:
            S += load_qq_bias_tile(
                qq_bias_row_ptrs, seq_offset, context_len, qq_bias_stride_0
            )

        M, L, P, alpha = softmax_step(S, M, L)
        acc = acc * alpha[:, None]

        if SLIDING_WINDOW:
            qpos_lo = q_block_local_idx * BLOCK_Q
            V = tl.where(
                (context_len + qpos_lo - seq_offset[:, None]) < SLIDING_WINDOW,
                V,
                0.0,
            )
        if USE_PER_TOKEN_HEAD_SCALES:
            # Per-token-head quant: apply v_scale to P instead of V.
            P_v = (P * v_token_head_scales[None, :]).to(V.dtype)
            acc += tl.dot(P_v, V)
        else:
            acc += tl.dot(P.to(V.dtype), V)

    # ---- Epilogue ---------------------------------------------------------
    if IS_3D:
        # Store per-segment partials; finalized by ``reduce_segments``.
        segm_output_offset = (
            query_offset_0[:, None].to(tl.int64)
            * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
            + segm_idx * HEAD_SIZE_PADDED
            + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
        )
        tl.store(
            segm_output_ptr + segm_output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )
        store_segm_reduce_scalars(
            segm_max_ptr,
            segm_expsum_ptr,
            query_offset_0,
            query_offset_1,
            segm_idx,
            M,
            L,
            query_mask_0,
            query_mask_1,
            num_query_heads,
            NUM_SEGMENTS_PER_SEQ,
        )
    else:
        acc = acc / L[:, None]
        if USE_FP8:
            acc = acc * tl.load(out_scale)
            acc = tl.clamp(acc, FP8_MIN, FP8_MAX)
        output_offset = (
            query_offset_0[:, None] * output_stride_0
            + query_offset_1[:, None] * output_stride_1
            + offs_d[None, :]
        )
        tl.store(
            output_ptr + output_offset,
            acc,
            mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        )


@triton.jit
def reduce_segments(
    output_ptr,  # [num_tokens, num_query_heads, head_size]
    segm_output_ptr,
    # [num_tokens, num_query_heads, max_num_segments, head_size]
    segm_max_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    segm_expsum_ptr,  # [num_tokens, num_query_heads, max_num_segments]
    seq_lens_ptr,  # [num_seqs]
    num_seqs,  # int
    num_query_heads: tl.constexpr,  # int
    out_scale_inv,  # float32
    output_stride_0: tl.int64,  # int
    output_stride_1: tl.int64,  # int, should be equal to head_size
    block_table_stride: tl.int64,  # int
    TILE_SIZE: tl.constexpr,  # int
    HEAD_SIZE: tl.constexpr,  # int, must be power of 2
    HEAD_SIZE_PADDED: tl.constexpr,  # int, must be power of 2
    query_start_len_ptr,  # [num_seqs+1]
    BLOCK_Q: tl.constexpr,  # int
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,  # int
    USE_FP8: tl.constexpr,  # bool
    FP8_MIN: tl.constexpr = float8_info.min,
    FP8_MAX: tl.constexpr = float8_info.max,
    # OP-003 window-relative segmentation: when WINDOW_SEG_3D is True the
    # active-segment count is recomputed from the SAME per-q-block window
    # range the mainloop used (compute_window_segments) instead of the
    # window-blind full-seq formula.  These constexprs mirror the
    # mainloop's so the two views CANNOT drift.  Defaults reproduce the
    # original window-blind behavior byte-for-byte.
    BLOCK_M: tl.constexpr = 0,
    num_queries_per_kv: tl.constexpr = 0,
    SLIDING_WINDOW: tl.constexpr = 0,
    USE_MM_PREFIX: tl.constexpr = False,
    CHUNK_LOOKBACK: tl.constexpr = -1,
    CHUNK_SIZE: tl.constexpr = -1,
    WINDOW_SEG_3D: tl.constexpr = False,
):
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    seq_idx = find_seq_idx(
        query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False
    )

    # sequence len for this particular sequence
    seq_len = tl.load(seq_lens_ptr + seq_idx)

    # number of segments for this particular sequence
    if WINDOW_SEG_3D:
        # Reconstruct the SAME q-block geometry the mainloop used for this
        # query token, then derive the identical window-relative active
        # segment count.  cur_start / cur_batch_query_len mirror
        # resolve_seq_and_query_len; q_block_local_idx is the token's
        # q-block within its sequence (all tokens in a q-block share it).
        cur_start = tl.load(query_start_len_ptr + seq_idx)
        cur_stop = tl.load(query_start_len_ptr + seq_idx + 1)
        cur_batch_query_len = cur_stop - cur_start
        q_block_local_idx = (query_token_idx - cur_start) // BLOCK_Q
        context_len = seq_len - cur_batch_query_len
        _ws_tile_start, tiles_per_segment, act_num_segments = compute_window_segments(
            context_len,
            seq_len,
            cur_batch_query_len,
            q_block_local_idx,
            NUM_SEGMENTS_PER_SEQ,
            TILE_SIZE,
            BLOCK_M,
            BLOCK_Q,
            num_queries_per_kv,
            SLIDING_WINDOW,
            USE_MM_PREFIX,
            CHUNK_LOOKBACK,
            CHUNK_SIZE,
        )
    else:
        num_segments = NUM_SEGMENTS_PER_SEQ
        tiles_per_segment = cdiv_fn(seq_len, num_segments * TILE_SIZE)
        # create masks for subsequent loads
        act_num_segments = cdiv_fn(seq_len, tiles_per_segment * TILE_SIZE)

    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full(
        [NUM_SEGMENTS_PER_SEQ], act_num_segments, dtype=tl.int32
    )
    dim_mask = tl.where(tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE, 1, 0).to(tl.int1)

    # load segment maxima
    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    # load and rescale segment exp sums
    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    # load, rescale, and add segment attention outputs
    segm_output_offset = (
        query_token_idx.to(tl.int64)
        * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(
        segm_output_ptr + segm_output_offset,
        mask=segm_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    # safely divide by overall_expsum, returning 0.0 if overall_expsum is 0
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    if USE_FP8:
        acc = acc * tl.load(out_scale_inv)
        acc = tl.clamp(acc, FP8_MIN, FP8_MAX)

    # write result
    output_offset = (
        query_token_idx * output_stride_0
        + query_head_idx * output_stride_1
        + tl.arange(0, HEAD_SIZE_PADDED)
    )
    tl.store(output_ptr + output_offset, acc, mask=dim_mask)


def _is_gemma3_attention(head_size: int, sliding_window: int) -> bool:
    """Detect Gemma3 models via unique (head_size, sliding_window) signature.

    Gemma3 models are the only ones using sliding_window=1024 with
    head_size 128 (27B) or 256 (1B, 4B, 12B). Other SWA models use
    different window sizes (Mistral=4096, Phi-3=2047).
    """
    return sliding_window == 1024 and head_size in (128, 256)


def _get_tile_size(
    head_size: int,
    sliding_window: int,
    element_size: int,
    is_prefill: bool,
) -> int:
    """Select tile size with Gemma3-specific optimization."""
    if _is_gemma3_attention(head_size, sliding_window):
        # Gemma3: use 32 for decode (default is 16)
        return 32

    # Default behavior
    if is_prefill:
        return 32
    # Note: tile size must be at least 32 for fp8 (element_size == 1).
    return 16 if element_size >= 2 else 32


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    seq_threshold_3D=None,
    # Max per-sequence query length admitted to the 3D flash-decoding path.
    # Defaults to 1 (today's pure-decode behavior).  Under MTP/EAGLE
    # speculative decode this is ``1 + num_speculative_tokens`` so that the
    # uniform spec-verify decode step (query_len == 1 + num_spec) is still
    # routed to 3D instead of falling back to the under-occupied 2D split-KV
    # path.  See ``TritonAttentionMetadataBuilder`` for how it is derived.
    decode_query_len: int = 1,
    num_par_softmax_segments=None,
    softmax_segm_output=None,
    softmax_segm_max=None,
    softmax_segm_expsum=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    # Optional tensor for sinks
    sinks=None,
    # Optional tensor for prefix lengths (PrefixLM support)
    mm_prefix_range=None,
    use_alibi_sqrt=False,
    # KV cache quantization mode and per-token-head scale caches.
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE,
    k_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    v_scale_cache=None,  # [num_blocks, block_size, num_kv_heads] float32
    # Chunked attention: restrict attention to aligned blocks with lookback.
    chunk_lookback=-1,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    use_per_token_head_scales = kv_quant_mode in (
        KVQuantMode.INT8_PER_TOKEN_HEAD,
        KVQuantMode.FP8_PER_TOKEN_HEAD,
    )
    if use_per_token_head_scales:
        assert k_scale_cache is not None and v_scale_cache is not None, (
            f"{kv_quant_mode.name} requires k_scale_cache / v_scale_cache"
        )

    use_mm_prefix = False
    max_mm_ranges = 0
    if mm_prefix_range is not None:
        if mm_prefix_range.ndim == 3:
            use_mm_prefix = True
            max_mm_ranges = mm_prefix_range.shape[1]
        else:
            raise ValueError(
                f"Unsupported mm_prefix_range shape: {mm_prefix_range.shape}"
            )

    use_alibi_slopes = alibi_slopes is not None
    use_qq_bias = qq_bias is not None

    block_size = v.shape[1]
    num_seqs = len(seqused_k)
    num_query_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size = q.shape[2]

    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )

    # AMMO OP-017: 2D-prefill global-attention launch reroute. Widen
    # BLOCK_M 16 -> 32 jointly with num_warps=8 for head_size==512 /
    # num_queries_per_kv==8 / 2D path (gemma-4 global, prefill/chunked-prefill).
    # Conditions are enforced inline so that:
    #   * head_size == 512 (global layers; sliding hd256 is unchanged — §6.4)
    #   * num_queries_per_kv == 8 (the gemma-4 global config; ANY other
    #     nqpkv<=16 path keeps BLOCK_M=16 byte-for-byte)
    #   * 2D path only (use_3d == False — i.e. real prefill / chunked-prefill
    #     with max_seqlen_q > decode_query_len, or fallback 2D split-KV decode)
    #     — re-uses the same predicate evaluated at the use_3d= line below;
    #     pre-computed here so BLOCK_Q / total_num_q_blocks observe the
    #     reroute.  Decode launches (3D, gridZ=16) keep BLOCK_M=16 unchanged.
    #
    # Eligibility (selection_rationale §6.8): BLOCK_M is a `tl.constexpr`
    # template parameter; widening it routes the production shape to a
    # structurally different compiled Triton kernel (grid (8200,4)->(4104,4),
    # BLOCK_Q 2->4, acc[32,512] vs [16,512], 8 warps vs 4) — a NEW GPU code
    # path that does not run in production today.  num_warps=8 is also a
    # Triton template parameter (re-partitions the MMA tile across the
    # 8-warp scheduler — different `mma.sync` partitioning vs the 4-warp
    # baseline) and is NOT the maxnreg launch-time register cap (which
    # would be config-only and INELIGIBLE).
    #
    # Per-CTA fragment dilution mechanism: `acc[BLOCK_M=32, 512]` fp32 spread
    # across 256 threads (nw=8) holds 64 fp32/thread for acc alone — vs 128
    # fp32/thread at nw=4, which would spill at the 255-reg cap.  Doubling
    # threads/CTA absorbs the BLOCK_M doubling (intrinsic +354 reg/thread
    # working-set growth) WITHOUT spills (measured: 254 regs / 0 spills).
    #
    # §6.7 BINDING (correctness-of-dispatch guard): the (num_warps=8,
    # BLOCK_M=16) cell measured 0.735x — a regression.  num_warps=8 MUST
    # only be set when BLOCK_M==32 is also set.  The condition below
    # determines BOTH simultaneously, so they cannot drift apart.  The
    # launch site reads BLOCK_M and applies num_warps=8 IFF BLOCK_M==32.
    _op017_2d_prefill_global = (
        envs.VLLM_OP017
        and head_size == 512
        and num_queries_per_kv == 8
        # Inline re-evaluation of the use_3d -> 2D condition (the canonical
        # use_3d= computation sits below at line ~747; this predicate is
        # the same condition negated and is checked again at the canonical
        # site for the 3D launch path).  We deliberately do NOT depend on
        # the canonical `use_3d` variable here, because BLOCK_M / BLOCK_Q
        # / total_num_q_blocks must be set BEFORE use_3d is computed.
        and (
            seq_threshold_3D is None
            or num_par_softmax_segments is None
            or softmax_segm_output is None
            or softmax_segm_max is None
            or softmax_segm_expsum is None
            or max_seqlen_q > decode_query_len
            or num_seqs > seq_threshold_3D
            or is_batch_invariant
        )
    )
    if _op017_2d_prefill_global:
        BLOCK_M = 32

    # AMMO OP-019: 2D-prefill sliding-window-attention launch reroute. Widen
    # BLOCK_M 16 -> 64 jointly with num_warps=8 for head_size==256 /
    # num_queries_per_kv==2 / sliding_window>=0 / 2D path (gemma-4 sliding,
    # prefill/chunked-prefill).  Predicate is head_size-disjoint from
    # OP-017's (256 vs 512), so the two cannot both fire on the same launch.
    # Conditions are enforced inline so that:
    #   * head_size == 256 (sliding layers; global hd512 is unchanged — OP-017)
    #   * num_queries_per_kv == 2 (the gemma-4 sliding config; ANY other
    #     nqpkv path keeps BLOCK_M=16 byte-for-byte)
    #   * sliding_window >= 0 (sliding-window layers; global window<0 layers
    #     are unchanged)
    #   * 2D path only (use_3d == False — i.e. real prefill / chunked-prefill
    #     with max_seqlen_q > decode_query_len, or fallback 2D split-KV decode)
    #     — re-uses the same predicate evaluated at the use_3d= line below;
    #     pre-computed here so BLOCK_Q / total_num_q_blocks observe the
    #     reroute.  Sliding 3D-decode launches keep BLOCK_M=16 unchanged.
    #
    # Eligibility (mirror OP-017 §6.8 cut): BLOCK_M is a `tl.constexpr`
    # template parameter; widening it routes the production shape to a
    # structurally different compiled Triton kernel (grid (2056,16,1) ->
    # (520,16,1), BLOCK_Q 8->32, acc[64,256] vs [16,256], 8 warps vs 4) — a
    # NEW GPU code path that does not run in production today.  num_warps=8
    # is also a Triton template parameter (re-partitions the MMA tile across
    # the 8-warp scheduler — different `mma.sync` partitioning vs the 4-warp
    # baseline) and is NOT the maxnreg launch-time register cap (which would
    # be config-only and INELIGIBLE).
    #
    # Per-CTA fragment dilution mechanism: at (BLOCK_M=64, num_warps=8) the
    # acc[64,256] fp32 tile spread across 256 threads holds 64 fp32/thread
    # for acc — actually CHEAPER in regs (176 measured) than the (64,4) cell
    # (246 regs) because doubling warps absorbs the BLOCK_M-doubling
    # working-set growth (no spills at either cell, but headroom is wider at
    # nw=8).
    #
    # BINDING (correctness-of-dispatch guard, mirrors OP-017 §6.7): the
    # (BLOCK_M=16, num_warps=8) cell measured 0.608x — a regression — same
    # MMA-fragmentation pathology OP-017 documented for hd512.  num_warps=8
    # MUST only be set when BLOCK_M==64 is also set.  The condition below
    # determines BOTH simultaneously, so they cannot drift apart.  The
    # launch site reads BLOCK_M and applies num_warps=8 IFF BLOCK_M==64 and
    # the OP-019 predicate is live.
    _op019_2d_prefill_sliding = (
        envs.VLLM_OP019
        and head_size == 256
        and num_queries_per_kv == 2
        and window_size[0] >= 0
        # Inline re-evaluation of the use_3d -> 2D condition (the canonical
        # use_3d= computation sits below; this predicate is the same
        # condition negated and is checked again at the canonical site for
        # the 3D launch path).  We deliberately do NOT depend on the
        # canonical `use_3d` variable here, because BLOCK_M / BLOCK_Q /
        # total_num_q_blocks must be set BEFORE use_3d is computed.
        and (
            seq_threshold_3D is None
            or num_par_softmax_segments is None
            or softmax_segm_output is None
            or softmax_segm_max is None
            or softmax_segm_expsum is None
            or max_seqlen_q > decode_query_len
            or num_seqs > seq_threshold_3D
            or is_batch_invariant
        )
    )
    if _op019_2d_prefill_sliding:
        BLOCK_M = 64

    # AMMO OP-033: TILE_SIZE 32 -> 64 widening on the post-OP-019 sliding-
    # hd256 2D-prefill kernel_unified_attention launch.  Stacks structurally
    # on top of OP-019 — strict reslice of OP-019's coverage set: the TILE=64
    # widening fires IFF VLLM_OP033 is on AND the OP-019 BLOCK_M=64 reroute
    # is fired (BLOCK_M==64 binding).  This binding ensures TILE=64 only
    # runs on the post-OP-019 cubin (BLOCK_M=64, num_warps=8, head_size=256,
    # nqpkv=2, sliding_window>=0, 2D-prefill); a TILE=64 launch on the
    # pre-OP-019 (BLOCK_M=16, num_warps=4) cell measured 250 regs and is
    # NOT the supported working point.
    #
    # Eligibility (mirror OP-017 §6.8 / OP-019 cuts): TILE_SIZE is a
    # `tl.constexpr` Triton template parameter; widening it produces a
    # structurally different compiled cubin (different K-loop iteration count
    # ~33 -> ~17 per q-block, different SMEM 82,228 B -> 131,636 B / CTA at
    # num_stages=3, different reg footprint 180 -> 200) — a NEW GPU code
    # path that does not run in production today.  Same eligibility framing
    # as OP-017 / OP-019, both shipped on this exact precedent.
    #
    # Mechanism: the sliding-window mainloop is a flash-style online-softmax
    # recurrence; per K-tile iteration the kernel performs `acc = acc *
    # alpha[:, None]` — the binding latency-path serial recurrence hop.
    # Doubling TILE_SIZE halves the iteration count (and therefore halves
    # the recurrence hops) while keeping the MMA work and KV bytes loaded
    # per q-block invariant (TILE_SIZE × #iters is invariant).  LOSSLESS:
    # TILE_SIZE does not change tl.dot operand types or softmax precision.
    #
    # BINDING (correctness-of-dispatch guard, mirrors OP-017 §6.7 / OP-019):
    # the (BLOCK_M=16, num_warps=4, TILE=64) cell measures 250 regs and is
    # NOT the supported working point.  The reroute MUST only fire when
    # BLOCK_M==64 is also selected.  The condition below ties it to
    # `_op019_2d_prefill_sliding` (OP-019's predicate that drives BLOCK_M=64).
    _op033_2d_prefill_sliding_tile64 = (
        envs.VLLM_OP033 and _op019_2d_prefill_sliding
    )

    # AMMO OP-017: first-fire / first-passthru log (opt-arm fastpath
    # evidence).  Memory `[ammo-opt-arm-silently-disabled-masquerades-as-
    # clean]`: the OP-011 monitors were fooled twice by an opt arm that
    # silently ran with the optimization OFF — a no-crash, latency-near-
    # baseline "clean" run.  Emit a single human-grep-able line on first
    # eligible reroute and on first env-off pass-through so the sweep log
    # tells us whether the opt arm actually took the new code path.
    global _OP017_FIRST_FIRE_LOGGED, _OP017_FIRST_PASSTHRU_LOGGED
    if envs.VLLM_OP017:
        if _op017_2d_prefill_global and not _OP017_FIRST_FIRE_LOGGED:
            logger.info(
                "OP017_ACTIVE BLOCK_M=32 num_warps=8 head_size=%d "
                "num_queries_per_kv=%d use_3d=False",
                head_size,
                num_queries_per_kv,
            )
            _OP017_FIRST_FIRE_LOGGED = True
        # Per-launch coverage tally — gated to avoid per-launch overhead
        # on the production (env-off) hot path.
        _is_global = head_size == 512 and num_queries_per_kv == 8
        # Mirror the use_3d predicate evaluated above (no canonical use_3d
        # variable available yet — it's set further below).  This is a
        # tally tag, not a launch decision.
        _is_2d = (
            seq_threshold_3D is None
            or num_par_softmax_segments is None
            or softmax_segm_output is None
            or softmax_segm_max is None
            or softmax_segm_expsum is None
            or max_seqlen_q > decode_query_len
            or num_seqs > seq_threshold_3D
            or is_batch_invariant
        )
        if _op017_2d_prefill_global:
            _op017_dispatch_counters["rerouted_global_2d_prefill"] += 1
        elif _is_global and not _is_2d:
            _op017_dispatch_counters["passthru_global_3d_decode"] += 1
        elif head_size == 256 and _is_2d:
            _op017_dispatch_counters["passthru_sliding_2d"] += 1
        elif head_size == 256:
            _op017_dispatch_counters["passthru_sliding_3d"] += 1
        elif head_size != 512:
            _op017_dispatch_counters["passthru_other_head_size"] += 1
        else:  # head_size==512 but nqpkv != 8
            _op017_dispatch_counters["passthru_other_nqpkv"] += 1
    else:
        _op017_dispatch_counters["env_off_total_launches"] += 1

    # AMMO OP-019: first-fire / first-passthru log + dispatch counter tally
    # (mirrors the OP-017 block above).  Same opt-arm-fastpath-evidence
    # rationale per memory `[ammo-opt-arm-silently-disabled-masquerades-as-
    # clean]`.  Reuses _is_2d computed in the OP-017 block.
    global _OP019_FIRST_FIRE_LOGGED, _OP019_FIRST_PASSTHRU_LOGGED
    if envs.VLLM_OP019:
        if _op019_2d_prefill_sliding and not _OP019_FIRST_FIRE_LOGGED:
            logger.info(
                "OP019_ACTIVE BLOCK_M=64 num_warps=8 head_size=%d "
                "num_queries_per_kv=%d sliding_window=%d use_3d=False",
                head_size,
                num_queries_per_kv,
                window_size[0],
            )
            _OP019_FIRST_FIRE_LOGGED = True
        # Per-launch coverage tally — gated to keep the production
        # (env-off) hot path overhead-free.  We need a local _is_2d in case
        # VLLM_OP017 is OFF (the OP-017 block above only computes _is_2d
        # when its env is ON).
        _is_2d_op019 = (
            seq_threshold_3D is None
            or num_par_softmax_segments is None
            or softmax_segm_output is None
            or softmax_segm_max is None
            or softmax_segm_expsum is None
            or max_seqlen_q > decode_query_len
            or num_seqs > seq_threshold_3D
            or is_batch_invariant
        )
        _is_sliding_hd256 = head_size == 256 and window_size[0] >= 0
        if _op019_2d_prefill_sliding:
            _op019_dispatch_counters["rerouted_sliding_2d_prefill"] += 1
        elif _is_sliding_hd256 and num_queries_per_kv == 2 and not _is_2d_op019:
            _op019_dispatch_counters["passthru_sliding_3d_decode"] += 1
        elif head_size == 256 and window_size[0] < 0:
            _op019_dispatch_counters["passthru_sliding_global"] += 1
        elif head_size == 256 and num_queries_per_kv != 2:
            _op019_dispatch_counters["passthru_sliding_other_nqpkv"] += 1
        elif head_size == 512:
            _op019_dispatch_counters["passthru_global_head_size"] += 1
        else:
            _op019_dispatch_counters["passthru_other_head_size"] += 1
    else:
        _op019_dispatch_counters["env_off_total_launches"] += 1

    # AMMO OP-033: first-fire log + dispatch counter tally (mirrors OP-019).
    # Same opt-arm-fastpath-evidence rationale per memory
    # `[ammo-opt-arm-silently-disabled-masquerades-as-clean]`.  The OP-033
    # eligibility set is a strict reslice of OP-019's: rerouted_tile64 IFF
    # the OP-019 BLOCK_M=64 reroute fired AND VLLM_OP033 is on.
    global _OP033_FIRST_FIRE_LOGGED
    if envs.VLLM_OP033:
        if _op033_2d_prefill_sliding_tile64 and not _OP033_FIRST_FIRE_LOGGED:
            logger.info(
                "OP033_ACTIVE TILE_SIZE=64 BLOCK_M=64 num_warps=8 "
                "head_size=%d num_queries_per_kv=%d sliding_window=%d "
                "use_3d=False",
                head_size,
                num_queries_per_kv,
                window_size[0],
            )
            _OP033_FIRST_FIRE_LOGGED = True
        # Per-launch coverage tally — gated to keep the production
        # (env-off) hot path overhead-free.  Local _is_2d (independent of
        # whether VLLM_OP019 is on, since OP-033 may run with OP-019 off in
        # the misconfiguration bucket "passthru_op019_off_2d_prefill").
        _is_2d_op033 = (
            seq_threshold_3D is None
            or num_par_softmax_segments is None
            or softmax_segm_output is None
            or softmax_segm_max is None
            or softmax_segm_expsum is None
            or max_seqlen_q > decode_query_len
            or num_seqs > seq_threshold_3D
            or is_batch_invariant
        )
        _is_sliding_hd256_op033 = head_size == 256 and window_size[0] >= 0
        if _op033_2d_prefill_sliding_tile64:
            _op033_dispatch_counters["rerouted_tile64_2d_prefill"] += 1
        elif (
            _is_sliding_hd256_op033
            and num_queries_per_kv == 2
            and _is_2d_op033
            and not _op019_2d_prefill_sliding
        ):
            # head_size==256 + nqpkv==2 + sliding + 2D, but VLLM_OP019 is OFF
            # so BLOCK_M stayed at 16 — would have been eligible for TILE=64
            # but cannot be (binding).  This bucket is non-zero only when
            # VLLM_OP033 is ON and VLLM_OP019 is OFF — a misconfiguration we
            # want to catch in the report.
            _op033_dispatch_counters["passthru_op019_off_2d_prefill"] += 1
        elif (
            _is_sliding_hd256_op033
            and num_queries_per_kv == 2
            and not _is_2d_op033
        ):
            _op033_dispatch_counters["passthru_sliding_3d_decode"] += 1
        elif head_size == 256 and window_size[0] < 0:
            _op033_dispatch_counters["passthru_sliding_global"] += 1
        elif head_size == 256 and num_queries_per_kv != 2:
            _op033_dispatch_counters["passthru_sliding_other_nqpkv"] += 1
        elif head_size == 512:
            _op033_dispatch_counters["passthru_global_head_size"] += 1
        else:
            _op033_dispatch_counters["passthru_other_head_size"] += 1
    else:
        _op033_dispatch_counters["env_off_total_launches"] += 1

    BLOCK_Q = BLOCK_M // num_queries_per_kv

    # Ideally we would launch with kernel with:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)] blocks.
    # However, it is slow to realize the query_lens on cpu.
    # Instead we use upper-bound:
    # \sum_i[ceil(query_len[i] / BLOCK_Q)]
    #   <= \sum_i[floor(query_len[i] / BLOCK_Q) + 1]
    #    = \sum_i[floor(query_len[i] / BLOCK_Q)] + num_seqs
    #   <= floor(\sum_i(query_len[i]) / BLOCK_Q) + num_seqs
    #    = floor(q.shape[0] / BLOCK_Q) + num_seqs
    total_num_q_blocks = q.shape[0] // BLOCK_Q + num_seqs

    sliding_window_val = 1 + window_size[0] if window_size[0] >= 0 else 0

    # Compute chunked block size from sliding window if needed.
    chunk_size = -1
    if sliding_window_val > 0 and chunk_lookback > -1:
        chunk_size = sliding_window_val // (chunk_lookback + 1)
        assert chunk_size > 0, "sliding_window must be > chunk_lookback+1"
    elif sliding_window_val <= 0:
        chunk_lookback = -1

    TILE_SIZE_PREFILL = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=True
    )
    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )

    # Launch the 2D kernel if
    # 1. No intermediate tiled softmax buffers for the 3D kernel have been allocated, or
    # 2. The batch includes a query longer than the (spec-)decode query length, i.e.
    #    a real prefill/chunked-prefill request (``max_seqlen_q > decode_query_len``).
    #    Under MTP spec-decode ``decode_query_len = 1 + num_spec`` (e.g. 5), so a
    #    uniform spec-verify decode step (max_seqlen_q == 5) is admitted to 3D rather
    #    than falling to the under-occupied 2D split-KV path.  With the default
    #    ``decode_query_len == 1`` this disjunct is byte-for-byte the old
    #    ``max_seqlen_q > 1`` test (backward-safe no-op for non-spec callers), or
    # 3. The number of sequences exceeds the configured threshold, or
    # 4. Batch invariance is enabled
    #
    # Sliding-window layers under spec-decode (``decode_query_len > 1`` and
    # ``window_size[0] >= 0``) ARE admitted to 3D (OP-003): the
    # window-relative segmentation (``WINDOW_SEG_3D`` below) tiles the
    # segments over the sliding window's tile range instead of the full
    # sequence, so segment parallelism is not collapsed.  Without
    # WINDOW_SEG_3D the window-blind segmentation would mis-segment sliding
    # layers; the flag and the matched ``reduce_segments`` recompute keep
    # the mainloop and reduction consistent.  ``mm_prefix`` (bidirectional)
    # sliding layers are NOT window-segmented (the window pruning is
    # disabled for them in the helpers), so they fall back to full-seq 3D.
    use_3d = not (
        seq_threshold_3D is None
        or num_par_softmax_segments is None
        or softmax_segm_output is None
        or softmax_segm_max is None
        or softmax_segm_expsum is None
        or max_seqlen_q > decode_query_len
        or num_seqs > seq_threshold_3D
        or is_batch_invariant
    )

    # OP-003: window-relative 3D segmentation applies only to sliding-window
    # layers in the spec-decode regime.  Global layers (window_size[0] < 0)
    # and pure-decode callers (decode_query_len == 1) keep the original
    # full-sequence segmentation byte-for-byte.  mm_prefix sliding layers are
    # excluded (window tile-pruning is a no-op under USE_MM_PREFIX, so the
    # window range would equal the full sequence and segmentation must stay
    # full-seq to match).
    window_seg_3d = (
        use_3d
        and decode_query_len > 1
        and window_size[0] >= 0
        and not use_mm_prefix
    )

    # The kernel signature is the same for 2D and 3D — only the launch
    # grid + a handful of constexpr toggles differ.  Per-token-head scale
    # caches and their strides are required arguments; non-per-token-head
    # modes pass dummy zeros (the code path is dead-code eliminated by
    # the ``USE_PER_TOKEN_HEAD_SCALES`` constexpr branch in the kernel).
    if use_per_token_head_scales:
        ks_strides = k_scale_cache.stride()
        vs_strides = v_scale_cache.stride()
        ks_blk, ks_slot, ks_head = ks_strides[0], ks_strides[1], ks_strides[2]
        vs_blk, vs_slot, vs_head = vs_strides[0], vs_strides[1], vs_strides[2]
        k_scale_ptr = k_scale_cache
        v_scale_ptr = v_scale_cache
    else:
        ks_blk = ks_slot = ks_head = 0
        vs_blk = vs_slot = vs_head = 0
        # Pass the K cache as a stand-in pointer; never dereferenced.
        k_scale_ptr = k
        v_scale_ptr = v

    # 3D needs real segm tensors; 2D never touches them but Triton wants
    # a non-null pointer.  Reuse ``out`` as the placeholder.
    segm_output_ptr = softmax_segm_output if use_3d else out
    segm_max_ptr = softmax_segm_max if use_3d else out
    segm_expsum_ptr = softmax_segm_expsum if use_3d else out
    num_segments = num_par_softmax_segments if use_3d else 1

    grid: tuple[Any, ...]
    if not use_3d:
        grid = (total_num_q_blocks, num_kv_heads)
        tile_size = TILE_SIZE_PREFILL
    else:
        grid = (total_num_q_blocks, num_kv_heads, num_par_softmax_segments)
        tile_size = TILE_SIZE_DECODE

    # AMMO OP-033: TILE_SIZE 32 -> 64 widening on the post-OP-019 sliding-
    # hd256 2D-prefill kernel_unified_attention launch.  Override tile_size
    # at the launch site (binds to BLOCK_M==64 + 2D-prefill predicate that
    # OP-019 already enforces).  Disjoint from the 3D path (use_3d=True)
    # and from non-OP-019 launches (BLOCK_M=16): only the (BLOCK_M=64,
    # num_warps=8, sliding-hd256, 2D-prefill) cubin gets TILE=64.
    if not use_3d and BLOCK_M == 64 and _op033_2d_prefill_sliding_tile64:
        tile_size = 64

    # AMMO OP-017 / OP-019: gate num_warps=8 on the matched BLOCK_M widening
    # (BINDING).
    #   - OP-017 §6.7: (num_warps=8, BLOCK_M=16) measured 0.735x for hd512
    #     2D-prefill global => num_warps=8 MUST be conditioned on BLOCK_M==32
    #     AND _op017_2d_prefill_global.
    #   - OP-019 (mirrors OP-017 §6.7): (num_warps=8, BLOCK_M=16) measured
    #     0.608x for hd256 2D-prefill sliding => num_warps=8 MUST be
    #     conditioned on BLOCK_M==64 AND _op019_2d_prefill_sliding.
    # The two predicates are head_size-disjoint (512 vs 256) so cannot both
    # fire on the same launch.  Reading the actual BLOCK_M value here ensures
    # the parameters cannot drift apart even under future predicate edits.
    # Without this kwarg, Triton uses its JIT default (4 warps), which is the
    # production baseline for all other launches (decode 3D, any 2D where
    # BLOCK_M stayed at 16).
    _op017_kernel_kwargs: dict[str, Any] = {}
    if BLOCK_M == 32 and _op017_2d_prefill_global:
        _op017_kernel_kwargs["num_warps"] = 8
    elif BLOCK_M == 64 and _op019_2d_prefill_sliding:
        _op017_kernel_kwargs["num_warps"] = 8

    kernel_unified_attention[grid](
        output_ptr=out,
        segm_output_ptr=segm_output_ptr,
        segm_max_ptr=segm_max_ptr,
        segm_expsum_ptr=segm_expsum_ptr,
        query_ptr=q,
        key_cache_ptr=k,
        value_cache_ptr=v,
        sink_ptr=sinks,
        block_tables_ptr=block_table,
        seq_lens_ptr=seqused_k,
        alibi_slopes_ptr=alibi_slopes,
        qq_bias_ptr=qq_bias,
        k_scale_cache_ptr=k_scale_ptr,
        v_scale_cache_ptr=v_scale_ptr,
        scale=softmax_scale,
        k_scale=k_descale,
        v_scale=v_descale,
        out_scale=1 / output_scale if output_scale is not None else 1.0,
        softcap=softcap,
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        block_table_stride=block_table.stride(0),
        query_stride_0=q.stride(0),
        query_stride_1=q.stride(1),
        output_stride_0=out.stride(0),
        output_stride_1=out.stride(1),
        qq_bias_stride_0=qq_bias.stride(0) if use_qq_bias else 0,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
        USE_ALIBI_SLOPES=use_alibi_slopes,
        USE_ALIBI_SQRT=use_alibi_sqrt,
        USE_QQ_BIAS=use_qq_bias,
        USE_SOFTCAP=(softcap > 0),
        USE_SINKS=(sinks is not None),
        USE_MM_PREFIX=use_mm_prefix,
        MAX_MM_RANGES=max_mm_ranges,
        mm_prefix_range_ptr=mm_prefix_range,
        SLIDING_WINDOW=(1 + window_size[0]),
        stride_k_cache_0=k.stride(0),
        stride_k_cache_1=k.stride(1),
        stride_k_cache_2=k.stride(2),
        stride_k_cache_3=k.stride(3),
        stride_v_cache_0=v.stride(0),
        stride_v_cache_1=v.stride(1),
        stride_v_cache_2=v.stride(2),
        stride_v_cache_3=v.stride(3),
        stride_ks_blk=ks_blk,
        stride_ks_slot=ks_slot,
        stride_ks_head=ks_head,
        stride_vs_blk=vs_blk,
        stride_vs_slot=vs_slot,
        stride_vs_head=vs_head,
        query_start_len_ptr=cu_seqlens_q,
        BLOCK_Q=BLOCK_Q,
        num_seqs=num_seqs,
        BLOCK_M=BLOCK_M,
        NUM_SEGMENTS_PER_SEQ=num_segments,
        USE_FP8=output_scale is not None,
        IS_3D=use_3d,
        KV_QUANT_MODE=kv_quant_mode,
        CHUNK_LOOKBACK=chunk_lookback,
        CHUNK_SIZE=chunk_size,
        WINDOW_SEG_3D=window_seg_3d,
        **_op017_kernel_kwargs,
    )

    if use_3d:
        reduce_segments[(q.shape[0], num_query_heads)](
            output_ptr=out,
            segm_output_ptr=softmax_segm_output,
            segm_max_ptr=softmax_segm_max,
            segm_expsum_ptr=softmax_segm_expsum,
            seq_lens_ptr=seqused_k,
            num_seqs=num_seqs,
            num_query_heads=num_query_heads,
            out_scale_inv=1 / output_scale if output_scale is not None else 1.0,
            output_stride_0=out.stride(0),
            output_stride_1=out.stride(1),
            block_table_stride=block_table.stride(0),
            TILE_SIZE=TILE_SIZE_DECODE,
            HEAD_SIZE=head_size,
            HEAD_SIZE_PADDED=triton.next_power_of_2(head_size),
            query_start_len_ptr=cu_seqlens_q,
            BLOCK_Q=BLOCK_Q,
            NUM_SEGMENTS_PER_SEQ=num_par_softmax_segments,
            USE_FP8=output_scale is not None,
            # OP-003: mirror the mainloop's window-segmentation inputs so
            # reduce_segments recomputes the IDENTICAL active-segment count.
            BLOCK_M=BLOCK_M,
            num_queries_per_kv=num_queries_per_kv,
            SLIDING_WINDOW=(1 + window_size[0]),
            USE_MM_PREFIX=use_mm_prefix,
            CHUNK_LOOKBACK=chunk_lookback,
            CHUNK_SIZE=chunk_size,
            WINDOW_SEG_3D=window_seg_3d,
        )
