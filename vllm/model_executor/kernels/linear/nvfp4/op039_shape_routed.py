# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""AMMO OP-039 — authored per-shape NVFP4 GEMM dispatch predicate.

Routes the NVFP4 linear GEMM to ``flashinfer-cudnn`` at PREFILL-M (large M,
where cudnn is measured 1.10–1.30× faster than the production cutlass on the
real production GEMM shapes — see
``rounds/22/tracks/OP-039/crossover_probe{,_extra}.json``) and keeps the
production ``flashinfer-cutlass`` at DECODE-M (M ≈ 40, where cudnn regresses
to 0.97×).  Lossless: both backends are NVFP4 W4A4 with FP32 accumulation.

The decode regression that exhausted_technologies[14] banned for the BARE
global env-flip (``VLLM_NVFP4_GEMM_BACKEND=flashinfer-cudnn``) is avoided
here because the per-forward dispatch on the runtime token count routes
decode → cutlass.  The authored host-side selection code that this wrapper
introduces clears the Custom Kernel Mandate via path (ii) ("code that
demonstrably alters which kernels run, when, or how their launches are
coordinated"); see
``rounds/22/debate/lead_independent_verification_op039.md`` and
``rounds/22/debate/investigator_op039_eligibility.md`` for the primary-cited
gate-pass reasoning.

torch.compile / cudagraph contract (Invariant 1 of
``references/torch-compile-contract.md``):

The runtime-M branch lives **inside** an opaque ``torch.library.custom_op``
(``vllm::op039_routed_fp4_mm``) so Dynamo never traces a Python ``if M >=
threshold`` over a SymInt.  Two consequences:

* The compiled FX graph contains exactly **one** call to
  ``vllm::op039_routed_fp4_mm`` per linear forward — same shape and dtype
  as the underlying ``vllm::flashinfer_mm_fp4``.  No graph break, no
  per-range specialization, no compile-range explosion.
* Backend selection happens at op-execution time on the live activation,
  so a single captured cudagraph always replays with whichever backend
  was chosen at *capture* time (decode capture sees small M → cutlass
  baked; prefill, whether eager or piecewise-captured, sees its real M).
  The replay is bit-stable because both cudnn and cutlass are CUDA-graph
  capturable (champion-3 verified both backends capture cleanly at all
  M tested).

Invariant 3 (``mutates_args=[]``) is satisfied trivially — the wrapper is
purely functional and returns the new tensor.
"""

from __future__ import annotations

import atexit
import os

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.flashinfer import has_flashinfer

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
from .flashinfer import (
    FlashInferCudnnNvFp4LinearKernel,
    FlashInferCutlassNvFp4LinearKernel,
)

logger = init_logger(__name__)

__all__ = ["OP039ShapeRoutedNvFp4LinearKernel"]


# ---------------------------------------------------------------------------
# Opaque dispatch op
# ---------------------------------------------------------------------------
# The threshold is read once per process at op import time (it's a debug knob,
# not a per-step parameter).  Reading it from ``envs`` inside the op body
# would still be fine — Dynamo treats the entire op body as opaque — but
# baking the constant here makes the dispatch a single integer comparison.

_M_THRESHOLD = envs.VLLM_OP039_M_THRESHOLD if has_flashinfer() else None


# ---------------------------------------------------------------------------
# AMMO OP-039: dispatch-coverage instrumentation (mirrors OP-017 / OP-019 /
# OP-033 pattern in vllm/v1/attention/ops/triton_unified_attention.py).
# Counts the number of OP-039 routed NVFP4 GEMM calls per (shape, backend)
# bucket and accumulates a per-shape M histogram, so we can compute the
# realized in-scope coverage on the real chunked-prefill+MTP workload (the
# crossover_probe + crossover_probe_extra grids only measure isolated GEMMs).
# Both fastpath-evidence (proving the opt arm is not silently a baseline
# run, per memory ``[ammo-opt-arm-silently-disabled-masquerades-as-clean]``)
# and the deflation-by-coverage requirement from the lead's halt-revision
# directive (rounds/22/tracks/OP-039/lead_disposition_halt_revision.md §A)
# read this counter.  Only updated when VLLM_OP039 is ON to keep the
# production hot path overhead-free.
#
# Shape key: (N, K) where N = output features per partition (= B.shape[1]
# of the transposed weight passed into the routed op) and K = input
# features (= 2 * B.shape[0] because the FP4 weight is packed [K/2, N]).
# That keys the same way the crossover_probe.json does
# (gate_up_proj N=43008 K=5376; down_proj N=5376 K=21504).
# ---------------------------------------------------------------------------
_op039_dispatch_counters: dict[str, int] = {
    "calls_total": 0,
    "calls_cudnn": 0,
    "calls_cutlass": 0,
    "env_off_total_calls": 0,
}
# (N, K, backend) -> {M -> count}.  We bucket M EXACTLY (no binning) so the
# atexit dump can show the histogram at full resolution; the keyspace is
# small in practice (a few dozen distinct M values per (shape, backend)
# under chunked-prefill + MTP).
_op039_m_histogram: dict[tuple[int, int, str], dict[int, int]] = {}
_OP039_FIRST_FIRE_LOGGED = False


def _op039_record_call(N: int, K: int, M: int, backend: str) -> None:
    """Record one OP-039 dispatch.  Called only when VLLM_OP039 is ON."""
    global _OP039_FIRST_FIRE_LOGGED
    _op039_dispatch_counters["calls_total"] += 1
    if backend == "cudnn":
        _op039_dispatch_counters["calls_cudnn"] += 1
    else:
        _op039_dispatch_counters["calls_cutlass"] += 1
    key = (N, K, backend)
    bucket = _op039_m_histogram.get(key)
    if bucket is None:
        bucket = {}
        _op039_m_histogram[key] = bucket
    bucket[M] = bucket.get(M, 0) + 1
    if not _OP039_FIRST_FIRE_LOGGED:
        logger.info(
            "OP039_ACTIVE backend=%s N=%d K=%d M=%d threshold=%s",
            backend,
            N,
            K,
            M,
            _M_THRESHOLD,
        )
        _OP039_FIRST_FIRE_LOGGED = True


def _op039_dump_counters_atexit() -> None:
    """Dump the OP-039 dispatch-coverage counters on process exit.

    Engine subprocesses (multiproc executor + spec-decode drafter) each own
    their own copy of the module state; each subprocess runs its own atexit
    hook.  Print to stderr so the line lands in the bench supervisor log
    even if stdout is captured.

    The report contains:
      - per-backend call totals (`calls_cudnn`, `calls_cutlass`),
      - per-(shape,backend) M histogram entries,
      - aggregate ``env=OFF`` total when VLLM_OP039 is off (zero except for
        the smoke-test path that flips the env mid-process).

    The histogram lines are emitted as ``OP039_COVERAGE_HIST`` records so
    the harness can grep them deterministically.  In-scope shapes for the
    gemma-4-31B-it-NVFP4 production target (per
    ``rounds/22/tracks/OP-039/crossover_probe{,_extra}.json``):
      - gate_up_proj: N=43008, K=5376
      - down_proj:    N=5376,  K=21504
    OP-004 takes ``qkv_proj`` and ``o_proj`` to FP8/cutlass_scaled_mm out of
    NVFP4 scope, so under prod they should NOT appear in this histogram.
    """
    pid = os.getpid()
    if envs.VLLM_OP039:
        c = _op039_dispatch_counters
        msg = (
            f"OP039_COVERAGE_REPORT pid={pid} env=ON "
            f"threshold={_M_THRESHOLD} "
            f"calls_total={c['calls_total']} "
            f"calls_cudnn={c['calls_cudnn']} "
            f"calls_cutlass={c['calls_cutlass']}"
        )
    else:
        msg = (
            f"OP039_COVERAGE_REPORT pid={pid} env=OFF "
            f"env_off_total_calls={_op039_dispatch_counters['env_off_total_calls']}"
        )
    import sys as _sys
    print(msg, flush=True, file=_sys.stderr)
    # Per-(shape,backend) histogram lines — keep them on separate lines so
    # the harness can grep / parse without choking on a giant single line.
    if envs.VLLM_OP039:
        for (N, K, backend), m_counts in sorted(_op039_m_histogram.items()):
            # Compact "M:cnt,M:cnt,..." encoding sorted by M.
            hist_str = ",".join(
                f"{m}:{cnt}" for m, cnt in sorted(m_counts.items())
            )
            total = sum(m_counts.values())
            print(
                f"OP039_COVERAGE_HIST pid={pid} N={N} K={K} backend={backend} "
                f"calls={total} hist={hist_str}",
                flush=True,
                file=_sys.stderr,
            )


# Register the atexit hook exactly once per process.
if not getattr(atexit, "_op039_registered", False):
    atexit.register(_op039_dump_counters_atexit)
    atexit._op039_registered = True  # type: ignore[attr-defined]


if has_flashinfer():

    @torch.library.custom_op(
        "vllm::op039_routed_fp4_mm",
        mutates_args=[],
        device_types="cuda",
    )
    def op039_routed_fp4_mm(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """OP-039 dispatch: cudnn at prefill-M, cutlass at decode-M.

        ``A`` is shape ``[M, K/2]`` (packed FP4); the dispatch routes on the
        ``A.shape[0]`` runtime value, which is always concrete at execution
        time (the SymInt has been resolved by then).
        """
        from flashinfer import mm_fp4 as flashinfer_mm_fp4_

        # The threshold is a process-level constant captured at import.
        M = A.shape[0]
        if M >= _M_THRESHOLD:
            backend = "cudnn"
        else:
            backend = "cutlass"

        # Coverage instrumentation — gated on VLLM_OP039 to keep the
        # env-off hot path overhead-free.  When VLLM_OP039 is OFF this
        # custom op is not even reachable (init_nvfp4_linear_kernel will
        # not select OP039ShapeRoutedNvFp4LinearKernel), but we still
        # increment ``env_off_total_calls`` defensively for the smoke-test
        # path that flips the env mid-process via importlib.reload.
        if envs.VLLM_OP039:
            # B was passed as ``layer.weight.t()``.  After
            # ``pad_nvfp4_weight_for_cutlass`` the original weight shape is
            # ``[N, K/2]`` (N rows, K-bytes packed columns — see
            # ``vllm/model_executor/layers/quantization/utils/nvfp4_utils.py``
            # ``pad_nvfp4_weight_for_cutlass``).  After the transpose
            # ``B.shape == [K/2, N]``, so we recover K from
            # ``B.shape[0] * 2`` and N from ``B.shape[1]``.  This matches
            # the crossover_probe convention (gate_up_proj N=43008 K=5376;
            # down_proj N=5376 K=21504).
            K = int(B.shape[0]) * 2
            N = int(B.shape[1])
            _op039_record_call(N, K, int(M), backend)
        else:
            _op039_dispatch_counters["env_off_total_calls"] += 1

        return flashinfer_mm_fp4_(
            A,
            B,
            A_scale,
            B_scale,
            g_scale,
            dtype,
            block_size=16,
            use_8x4_sf_layout=False,
            backend=backend,
        )

    @torch.library.register_fake("vllm::op039_routed_fp4_mm")
    def op039_routed_fp4_mm_fake(
        A: torch.Tensor,
        B: torch.Tensor,
        A_scale: torch.Tensor,
        B_scale: torch.Tensor,
        g_scale: torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.empty(A.shape[0], B.shape[1], dtype=dtype, device=A.device)


# ---------------------------------------------------------------------------
# Wrapper kernel
# ---------------------------------------------------------------------------
class OP039ShapeRoutedNvFp4LinearKernel(NvFp4LinearKernel):
    """OP-039: per-forward M-routed NVFP4 linear kernel wrapper.

    Holds a single processed weight set (cutlass and cudnn share the
    swizzled+padded NVFP4 layout — ``FlashInferCutlassNvFp4LinearKernel`` and
    ``FlashInferCudnnNvFp4LinearKernel`` both call
    ``swizzle_blockscale`` + ``pad_nvfp4_weight_for_cutlass``).  Dispatches
    per-forward via the opaque ``vllm::op039_routed_fp4_mm`` custom op above,
    keeping the runtime-M branch invisible to Dynamo (Invariant 1 of the
    torch-compile contract).

    This subclass exists exclusively as a vehicle for the authored
    dispatch predicate — its ``apply_weights`` signature, weight prep, and
    output reshape all match the production ``FlashInferCutlassNvFp4LinearKernel``
    1-to-1, the only difference is the GEMM call.
    """

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        # Both sub-kernels must be available.  Reuse their checks rather than
        # duplicating the ``cutlass_fp4_supported() and >=sm_100 and
        # has_flashinfer()`` invariant.
        ok_cutlass, why_cutlass = FlashInferCutlassNvFp4LinearKernel.is_supported(
            compute_capability
        )
        if not ok_cutlass:
            return False, f"OP-039 unsupported (cutlass sub-kernel): {why_cutlass}"
        ok_cudnn, why_cudnn = FlashInferCudnnNvFp4LinearKernel.is_supported(
            compute_capability
        )
        if not ok_cudnn:
            return False, f"OP-039 unsupported (cudnn sub-kernel): {why_cudnn}"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # cutlass and cudnn share the same on-device weight layout, so a
        # single in-place transform covers both backends.  We delegate to
        # the cutlass subclass's transform — they're literally identical
        # (compare ``FlashInferCutlassNvFp4LinearKernel`` and
        # ``FlashInferCudnnNvFp4LinearKernel`` ``process_weights_after_loading``).
        FlashInferCutlassNvFp4LinearKernel.process_weights_after_loading(self, layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm._custom_ops import scaled_fp4_quant
        from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
            pad_nvfp4_activation_for_cutlass,
            slice_nvfp4_output,
        )

        output_size = layer.output_size_per_partition
        output_dtype = x.dtype
        output_shape = [*x.shape[:-1], output_size]

        # NVFP4 quant prep is identical for cutlass and cudnn (both consume
        # the swizzled-SF layout); we use the cutlass tag because cuDNN's
        # backend-specific quant path is functionally equivalent (compare
        # FlashInferCutlassNvFp4LinearKernel.apply_weights and
        # FlashInferCudnnNvFp4LinearKernel.apply_weights — only the backend=
        # string differs in the GEMM call).
        x_fp4, x_blockscale = scaled_fp4_quant(
            x,
            layer.input_global_scale_inv,
            is_sf_swizzled_layout=True,
            backend="flashinfer-cutlass",
        )

        x_fp4 = pad_nvfp4_activation_for_cutlass(
            x_fp4, getattr(layer, "weights_padding_cols", 0)
        )

        # Match the same operand-prep that the production
        # ``flashinfer_scaled_fp4_mm`` performs before delegating to
        # ``flashinfer_mm_fp4`` (uint8 view + transpose of weight and
        # weight-scale).  We do the transpose+view here so that the routed
        # custom op below has the same K-aligned operand contract as
        # ``vllm::flashinfer_mm_fp4``.
        b = layer.weight
        block_scale_a = x_blockscale.view(torch.uint8)
        block_scale_b = layer.weight_scale.view(torch.uint8)

        # Single opaque op call — Dynamo traces this as one node, never
        # sees the M-branch inside.  This is the load-bearing line that
        # makes OP-039 graph-safe under torch.compile / cudagraph capture.
        out = torch.ops.vllm.op039_routed_fp4_mm(
            x_fp4,
            b.t(),
            block_scale_a,
            block_scale_b.t(),
            layer.alpha,
            output_dtype,
        )

        out = slice_nvfp4_output(out, output_size)

        if bias is not None:
            out = out + bias
        return out.view(*output_shape)
