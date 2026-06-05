# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# OP-012: Fused Mamba2 SSD prefill `_chunk_state_fwd ⊕ _state_passing_fwd`.
#
# The baseline two-kernel chain materializes a per-layer fp32 ``states``
# tensor (~424 MiB at IL=27000 / nheads=128 / hdim=64 / dstate=128) which is
# written by ``_chunk_state_fwd`` and immediately read+overwritten by
# ``_state_passing_fwd``. This fused kernel walks the chunks of a sequence
# inside a single Triton program, keeping the per-(M, N) state tile resident
# in registers across the recurrence, so the fp32 round-trip never lands in
# HBM. The on-disk per-chunk ``states`` output is written in ``out_dtype``
# (C.dtype, typically bf16/fp16) — that single bf16 write is what the
# downstream ``_chunk_scan_fwd`` consumes for its ``prev_states`` reads.
#
# Numerics-equivalent to the baseline chain: same fp32 accumulation order
# (chunk-local dot first, then a single ``fast_exp(dA_cs_last) * state``
# recurrence step), same final cast to ``out_dtype`` on store. Outputs
# match the baseline within the ``atol=1e-3, rtol=1e-3`` tolerance the
# baseline already exhibits across re-launches.
#
# ruff: noqa: E501

import torch

from vllm.model_executor.layers.mamba.ops.triton_helpers import fast_exp
from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        # Production shape (Nemotron-3 Super 120B-A12B-NVFP4): hdim=64, dstate=128.
        # The chunk-walk loop is the dominant cost — minimizing the inner K-loop
        # trip count via a large BLOCK_SIZE_K (=128, half-chunk) is what brings
        # the fused kernel below the baseline chain time. The probe in the
        # OP-012 autotune investigation picked BM=64, BN=64, BK=128, nw=4, ns=3
        # at production shape (1.143× over baseline).
        #
        # Config list is intentionally minimal — every config Triton benchmarks
        # at autotune time costs ~50ms of cold-start latency on the first
        # forward pass after process start. With 4 configs we pay ~200ms once
        # per process; with 10+ configs the iter-1 outlier dominates the
        # measured E2E avg at small BS.
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 128},
            num_stages=3,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
            num_stages=2,
            num_warps=8,
        ),
        # K=64 fallback (for chunk_size < 128 or shapes where BK=128 spills).
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=4,
        ),
        # Large-tile fallback for non-production shapes (large hdim/dstate).
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=3,
            num_warps=4,
        ),
    ],
    key=["hdim", "dstate", "chunk_size", "nheads_ngroups_ratio"],
)
@triton.jit
def _chunk_state_passing_fused_fwd_kernel(
    # Pointers
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    last_chunk_indices_ptr,
    initstates_ptr,
    out_ptr,
    # Dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    # x strides — (seqlen, nheads, hdim)
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    # B strides — (seqlen, ngroups, dstate)
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    # dt strides — (nheads, nchunks, chunk_size)
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    # dA_cumsum strides — (nheads, nchunks, chunk_size)
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # initial_states strides — (batch, nheads, hdim, dstate); zero when HAS_INITSTATES is False
    stride_initstates_batch: tl.int64,
    stride_initstates_head: tl.int64,
    stride_initstates_hdim: tl.int64,
    stride_initstates_dstate: tl.constexpr,
    # out strides — (nchunks, nheads, hdim, dstate)
    stride_out_chunk: tl.int64,
    stride_out_head: tl.int64,
    stride_out_hdim: tl.int64,
    stride_out_dstate: tl.constexpr,
    # Meta
    HAS_INITSTATES: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    # Sequence chunk range from last_chunk_indices.
    chunk_end = tl.load(last_chunk_indices_ptr + pid_b) + 1
    chunk_start = (
        tl.load(last_chunk_indices_ptr + pid_b - 1, mask=pid_b > 0, other=-1) + 1
    )
    nchunks_this_seq = chunk_end - chunk_start

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize carry state. fp32 (BM, BN) tile, register-resident across the chunk loop.
    if HAS_INITSTATES:
        initstates_ptrs = (
            initstates_ptr
            + pid_b * stride_initstates_batch
            + pid_h * stride_initstates_head
            + offs_m[:, None] * stride_initstates_hdim
            + offs_n[None, :] * stride_initstates_dstate
        )
        state = tl.load(
            initstates_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
    else:
        state = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Walk this sequence's chunks. For each chunk: compute the local chunk state
    # in registers (mirroring `_chunk_state_fwd_kernel`'s K-loop), then apply the
    # `state_passing` recurrence, then store the post-recurrence state.
    for c in range(nchunks_this_seq):
        chunk_id = chunk_start + c
        chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + chunk_id)
        chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + chunk_id + 1)
        chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

        # Per-chunk pointers.
        x_chunk_ptr = (
            x_ptr + chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
        )
        b_chunk_ptr = (
            b_ptr
            + chunk_seqlen_start * stride_b_seqlen
            + (pid_h // nheads_ngroups_ratio) * stride_b_head
        )
        dt_chunk_ptr = dt_ptr + pid_h * stride_dt_head + chunk_id * stride_dt_chunk
        dA_cs_chunk_ptr = (
            dA_cumsum_ptr
            + pid_h * stride_dA_cs_head
            + chunk_id * stride_dA_cs_chunk
        )

        dA_cs_last = tl.load(
            dA_cs_chunk_ptr + (chunk_size - 1) * stride_dA_cs_csize
        ).to(tl.float32)

        # K-loop: compute chunk-local state into `chunk_state_local` (BM, BN).
        x_ptrs = x_chunk_ptr + (
            offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
        )
        b_ptrs = b_chunk_ptr + (
            offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
        )
        dt_ptrs = dt_chunk_ptr + offs_k * stride_dt_csize
        dA_cs_ptrs = dA_cs_chunk_ptr + offs_k * stride_dA_cs_csize

        chunk_state_local = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
            x_k = tl.load(
                x_ptrs,
                mask=(offs_m[:, None] < hdim)
                & (offs_k[None, :] < chunk_size_limit - k),
                other=0.0,
            )
            b_k = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < chunk_size_limit - k)
                & (offs_n[None, :] < dstate),
                other=0.0,
            ).to(tl.float32)
            dA_cs_k = tl.load(
                dA_cs_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
            ).to(tl.float32)
            dt_k = tl.load(
                dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
            ).to(tl.float32)
            scale_k = fast_exp(tl.minimum(dA_cs_last - dA_cs_k, 0.0)) * dt_k
            b_k *= scale_k[:, None]
            b_k = b_k.to(x_ptr.dtype.element_ty)
            chunk_state_local += tl.dot(x_k, b_k)

            x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
            b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
            dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
            dA_cs_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

        # Recurrence: state = exp(dA_cs_last) * state + chunk_state_local.
        state = fast_exp(dA_cs_last) * state + chunk_state_local

        # Store post-recurrence state to out[chunk_id, head, m, n] in out_dtype.
        out_chunk_ptr = (
            out_ptr + chunk_id * stride_out_chunk + pid_h * stride_out_head
        )
        out_ptrs = out_chunk_ptr + (
            offs_m[:, None] * stride_out_hdim
            + offs_n[None, :] * stride_out_dstate
        )
        tl.store(
            out_ptrs,
            state.to(out_ptr.dtype.element_ty),
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
        )


def _chunk_state_state_passing_fused_fwd(
    B,
    x,
    dt,
    dA_cumsum,
    cu_chunk_seqlens,
    last_chunk_indices,
    initial_states=None,
    out_dtype=None,
):
    """Fused replacement for the `_chunk_state_fwd → _state_passing_fwd` chain.

    Returns ``out`` of shape ``(nchunks, nheads, hdim, dstate)`` in ``out_dtype``.
    Output is the per-chunk post-recurrence state — semantically identical to
    the baseline chain's reshaped output:
        ``rearrange(_state_passing_fwd(rearrange(_chunk_state_fwd(...))), ...)``.
    """
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert last_chunk_indices.dim() == 1
    batch = last_chunk_indices.shape[0]

    if out_dtype is None:
        out_dtype = B.dtype  # baseline chain uses C.dtype; B and C share dtype.

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
        initstates_strides = (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
    else:
        initstates_strides = (0, 0, 0, 0)

    out = torch.empty(
        (nchunks, nheads, headdim, dstate), device=x.device, dtype=out_dtype
    )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch,
        nheads,
    )
    with torch.accelerator.device_index(x.device.index):
        _chunk_state_passing_fused_fwd_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            last_chunk_indices_ptr=last_chunk_indices,
            initstates_ptr=initial_states,
            out_ptr=out,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_initstates_batch=initstates_strides[0],
            stride_initstates_head=initstates_strides[1],
            stride_initstates_hdim=initstates_strides[2],
            stride_initstates_dstate=initstates_strides[3],
            stride_out_chunk=out.stride(0),
            stride_out_head=out.stride(1),
            stride_out_hdim=out.stride(2),
            stride_out_dstate=out.stride(3),
            HAS_INITSTATES=initial_states is not None,
        )
    return out
