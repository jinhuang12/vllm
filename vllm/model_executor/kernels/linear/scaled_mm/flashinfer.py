# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.flashinfer import (
    flashinfer_fp8_blockscale_gemm,
    flashinfer_scaled_fp8_mm,
    has_flashinfer,
    is_flashinfer_fp8_blockscale_gemm_supported,
    should_use_flashinfer_for_blockscale_fp8_gemm,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .BlockScaledMMLinearKernel import (
    Fp8BlockScaledDynamicMMLinearKernel,
    Fp8BlockScaledMMLinearKernel,
)
from .deep_gemm import DeepGemmFp8BlockScaledMMKernel, fp8_gemm_nt
from .ScaledMMLinearKernel import (
    FP8ScaledMMLinearKernel,
    FP8ScaledMMLinearLayerConfig,
)

logger = init_logger(__name__)


class FlashInferFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):
    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "requires CUDA."

        if not has_flashinfer():
            return False, "requires FlashInfer to be installed."

        if compute_capability is not None and compute_capability < 100:
            return False, "requires compute capability 100 and above."

        return True, None

    @classmethod
    def can_implement(cls, c: FP8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:
        per_tensor_activation_scales = (
            c.activation_quant_key.scale.group_shape.is_per_tensor()
        )
        per_tensor_weight_scales = c.weight_quant_key.scale.group_shape.is_per_tensor()

        if not (per_tensor_activation_scales and per_tensor_weight_scales):
            return False, "requires per tensor activation and weight scales."

        return True, None

    def apply_scaled_mm(
        self,
        *,
        A: torch.Tensor,
        B: torch.Tensor,
        out_dtype: torch.dtype,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: torch.Tensor | None,
        output_shape: list,
    ) -> torch.Tensor:
        # AMMO tracks dense_fp8_decode_gemm_sm100 (R1) and
        # dense_fp8_prefill_gemm_sm100 (R6).
        # When either flag is set, route through the per-M-bucket custom-kernel
        # dispatcher. The dispatcher does call-time per-bucket selection (custom
        # skinny-M CUTLASS kernel for decode M<=8 when the decode flag is set;
        # custom per-output-N TileN=256 CUTLASS kernel for prefill M>256 when the
        # prefill flag is set; the production FlashInfer path otherwise), so the
        # baseline is byte-for-byte preserved for any M-bucket whose flag is
        # unset, and identically when both flags are unset. The bias case is
        # never routed here (the Nemotron-3 dense FP8 layers carry no bias); if a
        # bias is present we fall back to the unmodified production path.
        if (
            (
                envs.VLLM_NEMOTRON3_FP8_DECODE_GEMM_SM100
                or envs.VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100
                or envs.VLLM_NEMOTRON3_FP8_PREFILL_C3X_REROUTE_SM100
            )
            and bias is None
            and out_dtype == torch.bfloat16
        ):
            return torch.ops.vllm.nemotron3_fp8_decode_gemm(A, B, As, Bs)
        return flashinfer_scaled_fp8_mm(
            A, B, out_dtype=out_dtype, scale_a=As, scale_b=Bs, bias=bias
        )


class FlashInferFp8BlockScaledMMKernel(Fp8BlockScaledMMLinearKernel):
    # FlashInfer accepts BF16 input and handles FP8 conversion internally.
    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig) -> None:
        super().__init__(config)

    @classmethod
    def can_implement(cls, config: FP8ScaledMMLinearLayerConfig):
        can_implement_base, reason = super().can_implement(config)
        if not can_implement_base:
            return can_implement_base, reason

        act_quant_desc = config.activation_quant_key.scale
        if act_quant_desc.group_shape != GroupShape(1, 128):
            return (
                False,
                "Supports only dynamic per token group activation "
                "quantization with group_shape=(1,128).",
            )

        if not should_use_flashinfer_for_blockscale_fp8_gemm(
            is_flashinfer_fp8_blockscale_gemm_supported(),
            config.out_dtype,
            config.input_dtype,
            config.weight_quant_key.dtype,
            config.weight_shape,
        ):
            return (
                False,
                "The provided metadata is not supported.",
            )

        return True, None

    @classmethod
    def is_supported(cls, compute_capability=None):
        if not current_platform.is_cuda():
            return False, "only cuda devices are supported."

        if not is_flashinfer_fp8_blockscale_gemm_supported():
            return False, "FlashInfer block-scale FP8 GEMM is not available."

        return True, None

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        # A is BF16 — FlashInfer handles FP8 conversion internally.
        # As is a placeholder (apply_input_quant=False) and is not used here.
        return torch.ops.vllm.flashinfer_fp8_blockscale_gemm(
            A,  # BF16 input
            B,  # FP8 weight
            Bs,  # Weight scales
        )


class FlashInferFp8DeepGEMMDynamicBlockScaledKernel(
    Fp8BlockScaledDynamicMMLinearKernel
):
    """
    Conditional FlashInfer / DeepGEMM FP8 block-scaled GEMM.

    Dispatches between two kernels based on input batch size:
    - Small batches (M < 32): FlashInfer's swapAB trick for better utilisation.
    - Large batches (M >= 32): DeepGEMM for peak throughput.

    apply_input_quant is False because FlashInfer accepts BF16 input and
    handles FP8 conversion internally.  The DeepGEMM branch therefore
    quantises BF16→FP8 inside apply_mm via a closure before dispatching to
    the DeepGEMM kernel — keeping both branches compatible with the single
    BF16 tensor operand list passed by torch.cond.
    """

    base_type: ClassVar[type[FlashInferFp8BlockScaledMMKernel]] = (
        FlashInferFp8BlockScaledMMKernel
    )
    fallback_type: ClassVar[type[DeepGemmFp8BlockScaledMMKernel]] = (
        DeepGemmFp8BlockScaledMMKernel
    )
    apply_input_quant: ClassVar[bool] = False

    def __init__(self, config: FP8ScaledMMLinearLayerConfig):
        super().__init__(config)
        self.base: FlashInferFp8BlockScaledMMKernel
        self.fallback: DeepGemmFp8BlockScaledMMKernel

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # DeepGEMM need post-processing; both kernels share the same
        # parameter tensor layout so processing once is sufficient.
        self.fallback.process_weights_after_loading(layer)

    def apply_block_scaled_mm(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
    ) -> torch.Tensor:
        group_size = self.weight_group_shape.col
        use_deep_gemm_e8m0 = self.fallback.use_deep_gemm_e8m0

        return torch.ops.vllm.dynamic_flashinfer_deepgemm_blockscale_gemm(
            A, B, Bs, group_size, use_deep_gemm_e8m0
        )


def _flashinfer_fp8_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    return flashinfer_fp8_blockscale_gemm(
        input=input,
        weight=weight,
        weight_scale=weight_scale,
        out_dtype=torch.bfloat16,
    )


def _flashinfer_fp8_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """
    Required fake/meta implementation for torch.compile graph tracing.
    """
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "flashinfer_fp8_blockscale_gemm",
    _flashinfer_fp8_blockscale_gemm_impl,
    fake_impl=_flashinfer_fp8_blockscale_gemm_fake,
)


# Decode M-bucket threshold for the AMMO dense_fp8_decode_gemm_sm100 track.
# Measured head-to-head cold speedup over the production FlashInfer bmm_fp8
# "auto" path (champion1_baseline_headtohead.log): M=1 -> 1.39-1.48x, M=8 ->
# 1.23-1.34x, M=32 -> ~1.07x (collapses). M <= 8 routes to the custom kernel;
# M > 8 (prefill, BS=32) keeps the production path. This crossover is gated
# (crossover_probe obligation) and may be tightened during validation.
_NEMOTRON3_FP8_DECODE_GEMM_MAX_M = 8

# Prefill thresholds for the AMMO dense_fp8_prefill_gemm_sm100 track.
#
# The custom kernel uses a per-output-N tile (mirrored here in the dispatch so
# the gate matches the kernel's internal selector at csrc .../
# cutlass_fp8_prefill_gemm_sm100.cu: `n > 8192 -> Tile<256,256,128>` else
# `Tile<128,256,128>`). The two tile families have DIFFERENT M-crossovers vs c3x
# (Stage-4 crossover_probe obligation; measured by the kernel validator at
# matched-clock interleaved cold-L2, M=10624 chunk):
#
#   TileN=256 path  (N > 8192; in_proj N=18560): custom wins at ALL M > 256
#     (M=257 -> 1.10x ... M=10624 -> 1.16x). Crossover M* = 257.
#   TileN=128 path  (N <= 8192; out_proj/o_proj/shared_up/shared_down):
#     custom LOSES to c3x for M in (256, ~4096) -- worst 0.892x at M=2048 --
#     then breaks even ~M=4096 and wins (M=10624 -> 1.05-1.09x). Crossover
#     M* ~= 4096. o_proj (smallest K=4096) is the worst-case / latest-crossover
#     TileN=128 shape, so its breakeven is used as a CONSERVATIVE universal
#     TileN=128 threshold (out_proj/shared_up/shared_down win at <= this M).
#
# REROUTE lower bound: any prefill GEMM with M > 256 is rerouted off FlashInfer
# (config B / C). The custom kernel is then selected ONLY where it beats c3x
# (_prefill_custom_kernel_wins below); otherwise we fall back to stock c3x (the
# rerouted target), so config C is never slower than config B at any M -- the
# custom kernel never fires in its losing regime, and chunked-prefill tail
# chunks landing in M in (256, 4096) on TileN=128 shapes get c3x, not a
# regression.
_NEMOTRON3_FP8_PREFILL_REROUTE_MIN_M = 256
_NEMOTRON3_FP8_PREFILL_GEMM_TILEN256_MIN_M = 256  # N > 8192 (in_proj)
_NEMOTRON3_FP8_PREFILL_GEMM_TILEN128_MIN_M = 4096  # N <= 8192 (others)
# Tile-family N boundary -- MUST match the kernel's internal selector.
_NEMOTRON3_FP8_PREFILL_TILEN256_N_THRESHOLD = 8192


def _prefill_custom_kernel_wins(n: int, m: int) -> bool:
    """True iff the custom prefill kernel beats c3x at this (N, M).

    Mirrors the kernel's per-output-N tile selector and the validator's measured
    per-tile M-crossovers (crossover_probe). N is the output dim (weight is
    [K, N] -> N = weight.shape[1]).
    """
    if n > _NEMOTRON3_FP8_PREFILL_TILEN256_N_THRESHOLD:
        return m > _NEMOTRON3_FP8_PREFILL_GEMM_TILEN256_MIN_M
    return m > _NEMOTRON3_FP8_PREFILL_GEMM_TILEN128_MIN_M


def _nemotron3_fp8_decode_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Per-M-bucket dispatcher for the dense FP8 GEMM (decode + prefill tracks).

    This is an opaque custom op (registered via ``direct_register_custom_op``),
    so Dynamo does not trace into its body: the ``input.shape[0]`` branches below
    resolve at call time. For the decode M-buckets this is CUDA-graph capture
    time, independently per bucket under ``cudagraph_mode=FULL_AND_PIECEWISE``
    (each of {1,8,32} is captured as its own FULL graph). For prefill, the model
    runs eager (``CUDAGraphMode.NONE``) so the branch resolves at the eager
    Python call with a concrete M (chunk token count ~= 10624). Either way this
    is per-bucket selection -- NOT a runtime ``torch.cond`` inside a captured
    region.

    Each branch is gated by its OWN env flag, so the production baseline is
    preserved byte-for-byte for any M-bucket whose flag is unset:

    - decode M-buckets (M <= 8), VLLM_NEMOTRON3_FP8_DECODE_GEMM_SM100 set:
      custom skinny-M CUTLASS SM100 kernel (tileN=128 ~1-wave schedule).
      [AMMO track dense_fp8_decode_gemm_sm100, R1-shipped]
    - prefill large-M (M > 256), VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100 set:
      custom per-output-N tiled CUTLASS SM100 kernel, but ONLY where it beats
      c3x (N-aware crossover: N>8192 at M>256; N<=8192 at M>4096). In the
      rerouted-but-custom-loses region the prefill GEMM falls back to stock c3x
      (never to FlashInfer), so config C is never slower than config B.
      [AMMO track dense_fp8_prefill_gemm_sm100, R6]
    - everything else (or the relevant flag unset): the unmodified production
      FlashInfer ``bmm_fp8 "auto"`` path.
    """
    from vllm import _custom_ops as ops

    m = input.shape[0]
    n = weight.shape[1]  # weight is [K, N] col-major -> N = output dim

    # --- Decode bucket (AMMO R1 track dense_fp8_decode_gemm_sm100) ----------
    # Independent of the prefill reroute mechanism; unchanged from R1.
    if (
        envs.VLLM_NEMOTRON3_FP8_DECODE_GEMM_SM100
        and m <= _NEMOTRON3_FP8_DECODE_GEMM_MAX_M
    ):
        # Fast-path evidence marker (asserted by the E2E sweep require_patterns).
        logger.info_once(
            "AMMO[dense_fp8_decode_gemm_sm100]: custom skinny-M CUTLASS kernel "
            "ACTIVE (decode M-bucket M<=%d)",
            _NEMOTRON3_FP8_DECODE_GEMM_MAX_M,
            scope="global",
        )
        return ops.cutlass_fp8_decode_gemm_sm100(
            input, weight, scale_a, scale_b, out_dtype=torch.bfloat16
        )

    # --- Prefill bucket (AMMO R6 track dense_fp8_prefill_gemm_sm100) --------
    # SHARED MECHANISM: the FlashInfer->c3x reroute. It is feature-agnostic
    # (config B = stock c3x mainloop). Any prefill feature flag IMPLIES the
    # reroute is on; the feature flags then select WHICH kernel runs on the
    # rerouted path. Stage 6 composes:
    #   reroute (B)  +  PREFILL_GEMM (custom mainloop, this track)
    #                +  RELU2_EPILOGUE (peer EVT epilogue) -> one kernel.
    # The peer EVT track ORs its VLLM_NEMOTRON3_FP8_RELU2_EPILOGUE_SM100 flag
    # into `prefill_reroute` and adds its branch below at integration time.
    prefill_reroute = (
        envs.VLLM_NEMOTRON3_FP8_PREFILL_C3X_REROUTE_SM100
        or envs.VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100
    )
    if prefill_reroute and m > _NEMOTRON3_FP8_PREFILL_REROUTE_MIN_M:
        if envs.VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100 and _prefill_custom_kernel_wins(
            n, m
        ):
            # Config C (shipped): custom per-output-N tiled CUTLASS mainloop, in
            # the N-aware M-range where it beats c3x (crossover_probe).
            # Fast-path evidence marker (asserted by the sweep require_patterns).
            logger.info_once(
                "AMMO[dense_fp8_prefill_gemm_sm100]: custom CUTLASS kernel "
                "ACTIVE (prefill M=%d N=%d, %s tile)",
                m,
                n,
                "TileN256"
                if n > _NEMOTRON3_FP8_PREFILL_TILEN256_N_THRESHOLD
                else "TileN128",
                scope="global",
            )
            return ops.cutlass_fp8_prefill_gemm_sm100(
                input, weight, scale_a, scale_b, out_dtype=torch.bfloat16
            )
        # Config B (eligibility reference) AND the custom-loses fallback: bare
        # reroute to stock in-tree c3x cutlass_scaled_mm, NO custom kernel.
        #   - When only REROUTE is set: this isolates the free/ineligible
        #     FlashInfer->c3x reroute slice (eligible E2E = config B - config C).
        #   - When PREFILL_GEMM is set but the custom kernel would LOSE at this
        #     (N, M) (TileN=128 shapes at M in (256, 4096)): fall back to c3x so
        #     config C is never slower than config B (no tail-chunk regression).
        reason = (
            "config B eligibility reference"
            if not envs.VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100
            else "custom-loses fallback (below N-aware crossover)"
        )
        logger.info_once(
            "AMMO[dense_fp8_prefill_gemm_sm100]: stock c3x reroute ACTIVE "
            "(prefill M=%d N=%d, %s)",
            m,
            n,
            reason,
            scope="global",
        )
        return ops.cutlass_scaled_mm(
            input, weight, scale_a, scale_b, out_dtype=torch.bfloat16, bias=None
        )

    # --- Production path (config A baseline, or out-of-bucket M) ------------
    logger.info_once(
        "AMMO[dense_fp8_gemm_sm100]: routing M=%d (outside custom decode<=%d / "
        "prefill>%d buckets, or flag unset) to production flashinfer bmm_fp8 path",
        m,
        _NEMOTRON3_FP8_DECODE_GEMM_MAX_M,
        _NEMOTRON3_FP8_PREFILL_REROUTE_MIN_M,
        scope="global",
    )
    return flashinfer_scaled_fp8_mm(
        input, weight, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b
    )


def _nemotron3_fp8_decode_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Required fake/meta implementation for torch.compile graph tracing."""
    return torch.empty(
        input.shape[0], weight.shape[1], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "nemotron3_fp8_decode_gemm",
    _nemotron3_fp8_decode_gemm_impl,
    mutates_args=[],
    fake_impl=_nemotron3_fp8_decode_gemm_fake,
)


# ---------------------------------------------------------------------------
# AMMO track fp8_relu2_requant_epilogue_sm100 (Round 6).
#
# Two additive opaque ops layered on top of the R1 decode dispatch:
#
#  (B) nemotron3_fp8_prefill_reroute_gemm  -- the "config B" baseline. Routes
#      the prefill (M > 8) dense FP8 GEMM through the stock c3x CUTLASS
#      ``cutlass_scaled_mm`` (bf16 out) instead of the production FlashInfer
#      bmm_fp8 path, while PRESERVING the R1 skinny-M decode kernel at M <= 8
#      (and FlashInfer when the R1 flag is unset). The reroute (A - B) is the
#      "free"/ineligible delta; it exists so the eligible fusion benefit can be
#      isolated as (config B - config C) with c3x common to BOTH arms.
#
#  (C) nemotron3_fp8_relu2_fused_up        -- the "config C" fused path. Folds
#      the shared/dense MLP up_proj GEMM + ReLUSquared activation + requant-to-
#      fp8 into a single c3x EVT epilogue (ScaledEpilogueReLUSquared), emitting
#      a pre-scaled fp8 tensor ready for the consuming down_proj (which then
#      SKIPS its own input quant -- the down-proj input_scale is folded into the
#      epilogue out_scale = 1 / down_input_scale).
#
# Both ops M-branch INTERNALLY (capture-time, per the R1 pattern) so torch.compile
# never bakes a shape-dependent Python branch into the traced forward. The
# forward-level gate is a Python ENV constant only (no tensor-shape branch).
# ---------------------------------------------------------------------------


def _nemotron3_fp8_prefill_reroute_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    """Config-B reroute: prefill dense FP8 GEMM through stock c3x (bf16 out).

    Opaque custom op -> Dynamo does not trace into the ``input.shape[0]`` branch,
    which resolves at CUDA-graph capture time independently per bucket.

      - prefill (M > 8): stock c3x ``cutlass_scaled_mm`` bf16 out. This is the
        same mainloop config C fuses on, so (B - C) isolates exactly the
        epilogue fusion (eliminated glue + fp8-write-save + down-quant elim).
      - decode (M <= 8): PRESERVE the R1 skinny-M custom kernel when its flag is
        set; otherwise the unmodified production FlashInfer path. The reroute
        NEVER touches decode-shape numerics or kernel selection.
    """
    from vllm import _custom_ops as ops

    if input.shape[0] > _NEMOTRON3_FP8_DECODE_GEMM_MAX_M:
        logger.info_once(
            "AMMO[fp8_relu2_requant_epilogue_sm100]: c3x prefill REROUTE ACTIVE "
            "(config B, M>%d)",
            _NEMOTRON3_FP8_DECODE_GEMM_MAX_M,
            scope="global",
        )
        return ops.cutlass_scaled_mm(
            input, weight, scale_a, scale_b, torch.bfloat16
        )
    if envs.VLLM_NEMOTRON3_FP8_DECODE_GEMM_SM100:
        return ops.cutlass_fp8_decode_gemm_sm100(
            input, weight, scale_a, scale_b, out_dtype=torch.bfloat16
        )
    return flashinfer_scaled_fp8_mm(
        input, weight, out_dtype=torch.bfloat16, scale_a=scale_a, scale_b=scale_b
    )


def _nemotron3_fp8_prefill_reroute_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
) -> torch.Tensor:
    return torch.empty(
        input.shape[0], weight.shape[1], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "nemotron3_fp8_prefill_reroute_gemm",
    _nemotron3_fp8_prefill_reroute_gemm_impl,
    mutates_args=[],
    fake_impl=_nemotron3_fp8_prefill_reroute_gemm_fake,
)


# =====================================================================
# AMMO track shared_expert_relu2_quant_fusion_decode (R13)
# Fused {relu^2 + static per-tensor fp8 requant} for the shared-expert FP8
# DECODE path (M <= _NEMOTRON3_FP8_DECODE_GEMM_MAX_M = 8). Replaces the 3
# eager ATen kernels at the decode branch (torch.relu + torch.square +
# ops.scaled_fp8_quant) with ONE Triton kernel. The producer up_proj GEMM is
# UNTOUCHED. M > 8 (prefill / BS32) routes to the fused CUTLASS EVT epilogue
# above and never reaches this kernel.
#
# BIT-EXACT CONTRACT (lossless classification, Gate 5.1a nmis == 0):
# The kernel must reproduce production's rounding chain element-for-element:
#   prod: relu(up_bf16) -> bf16; square -> bf16 (torch.square output dtype);
#         scaled_fp8_quant upcasts bf16 -> fp32, multiplies by
#         inv_scale = (1.0f / input_scale_down) computed ONCE in fp32, clamps
#         to [-448, 448], casts to e4m3 with round-to-nearest-even (SATFINITE).
# Primary source for the quant arithmetic:
#   csrc/quantization/w8a8/fp8/common.cu:32-34 (inv = 1.0f / scale[0], fp32)
#   csrc/quantization/w8a8/fp8/common.cuh:40-51 (val*inv, clamp +-448)
#   csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:22-31
#       (__nv_cvt_float_to_fp8(r, __NV_SATFINITE, __NV_E4M3) == RNE)
# Production-faithfulness traps this kernel avoids (vs the debate V2 micro,
# which left nmis=3 sub-ULP mismatches):
#   1. inv_scale is recomputed INSIDE the kernel as 1.0 / input_scale_down in
#      fp32 (NOT a literal pre-baked constant). Both arms then derive the
#      reciprocal from the SAME scale tensor with the SAME fp32 division, so
#      the eager path's 1.0f/scale and this path's 1.0/scale are bit-identical
#      -> no RNE-tie flips. (The V2 micro passed a literal 60.0 to the kernel
#      while the eager arm got 1/60 and production recomputed 1.0/(1/60) !=
#      60.0 in fp32 -> 3 tie flips. That artifact does not exist here.)
#   2. The square is forced to round to bf16 BEFORE the fp32 upcast, matching
#      torch.square(bf16) -> bf16 (7-bit mantissa) exactly, rather than relying
#      on Triton bf16*bf16 promotion semantics.


@triton.jit
def _fused_relu2_quant_kernel(
    x_ptr,            # bf16 [M, N] input (up_bf16, the producer GEMM output)
    scale_ptr,        # fp32 scalar input_scale_down (the down_proj input scale)
    o_ptr,            # fp8_e4m3 [M, N] output, pre-scaled by 1/input_scale_down
    M,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    # Match CUDA: inv = 1.0f / scale[0], computed once per row in fp32.
    scale = tl.load(scale_ptr).to(tl.float32)
    inv_scale = 1.0 / scale
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0)
    # relu in the input (bf16) domain -> bf16 (torch.relu preserves dtype).
    r = tl.where(x > 0, x, 0.0).to(tl.bfloat16)
    # square: torch.square(bf16) computes in fp32 (acc_type<bf16> == float) and
    # rounds the result back to bf16. Make the fp32 intermediate EXPLICIT so we
    # match torch's rounding regardless of Triton's native-bf16-mul lowering:
    #   h_bf16 = round_to_bf16( float(r) * float(r) ).
    rf = r.to(tl.float32)
    h_bf16 = (rf * rf).to(tl.bfloat16)
    # scaled_fp8_quant then upcasts bf16 -> fp32: static_cast<float>(src).
    h = h_bf16.to(tl.float32)
    # static per-tensor fp8 quant (is_scale_inverted=true): val * inv_scale.
    q = h * inv_scale
    q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
    tl.store(o_ptr + row * N + offs, q.to(o_ptr.dtype.element_ty), mask=mask)


def _fused_relu2_quant(
    up_bf16: torch.Tensor, input_scale_down: torch.Tensor
) -> torch.Tensor:
    """One Triton kernel == relu^2 + static per-tensor fp8 requant.

    ``up_bf16`` is the 2D producer GEMM output ``[M, N]`` (M <= 8). Bit-for-bit
    equivalent to the eager decode chain it replaces:
        relu2 = torch.square(torch.relu(up_bf16))
        h_q, _ = ops.scaled_fp8_quant(relu2, input_scale_down)
    Returns fp8_e4m3 ``[M, N]`` pre-scaled by 1 / input_scale_down (so the
    consuming down_proj, whose static input_scale == input_scale_down, recovers
    relu(up)^2 exactly). Caller applies the final ``.view(*out_shape)``, exactly
    as it did for the eager ``h_q``.
    """
    assert up_bf16.dtype == torch.bfloat16
    assert up_bf16.dim() == 2, "decode branch GEMM output is 2D [M, N]"
    m, n = up_bf16.shape
    out = torch.empty(
        (m, n), dtype=current_platform.fp8_dtype(), device=up_bf16.device
    )
    # FP8 e4m3 finite max (quant_type_max_v<Float8_e4m3fn> == 448.0).
    fp8_max = torch.finfo(current_platform.fp8_dtype()).max
    block_n = triton.next_power_of_2(n)
    _fused_relu2_quant_kernel[(m,)](
        up_bf16,
        input_scale_down,
        out,
        m,
        n,
        BLOCK_N=block_n,
        FP8_MAX=fp8_max,
        num_warps=8,
    )
    return out


def _nemotron3_fp8_relu2_fused_up_impl(
    x: torch.Tensor,
    weight_up: torch.Tensor,
    weight_scale_up: torch.Tensor,
    input_scale_up: torch.Tensor,
    input_scale_down: torch.Tensor,
) -> torch.Tensor:
    """Config-C fused up_proj: GEMM + ReLUSquared + requant-to-fp8 in one epilogue.

    Returns an fp8_e4m3 tensor pre-scaled by ``1 / input_scale_down`` so the
    consuming down_proj GEMM skips its own input quant (its static input_scale
    recovers ``relu(up)**2`` exactly).

    Input quant is bit-identical to production: ``ops.scaled_fp8_quant(x,
    input_scale_up)`` is exactly what ``QuantFP8`` (static, per-tensor) computes
    in the stock ``apply_weights`` path.

      - prefill (M > 8): single c3x ``cutlass_scaled_mm_relu2_fp8out_sm100`` --
        relu^2 is computed in f32 in-register (MORE accurate than production's
        bf16 round-trip through ReLUSquaredActivation), then requantized to fp8.
      - decode (M <= 8): production-faithful fallback -- R1 skinny-M kernel (or
        FlashInfer) bf16 up, bf16 ReLUSquared, then the SAME static fp8 requant
        the down_proj would have applied. Bit-identical to production-with-R1.

    Output dtype is fp8 in BOTH branches (consistent for torch.compile / the
    fake impl). The down_proj input-quant SKIP is therefore unconditional on
    this route.
    """
    from vllm import _custom_ops as ops

    x_2d = x.view(-1, x.shape[-1])
    # Bit-identical to production up_proj input quant (static per-tensor).
    a_fp8, _ = ops.scaled_fp8_quant(x_2d, input_scale_up)
    out_shape = (*x.shape[:-1], weight_up.shape[1])

    if x_2d.shape[0] > _NEMOTRON3_FP8_DECODE_GEMM_MAX_M:
        logger.info_once(
            "AMMO[fp8_relu2_requant_epilogue_sm100]: FUSED relu^2 epilogue "
            "ACTIVE (config C, M>%d)",
            _NEMOTRON3_FP8_DECODE_GEMM_MAX_M,
            scope="global",
        )
        # out_scale = 1 / input_scale_down  -> epilogue emits relu(up)^2 / scale_down
        recip_down = torch.reciprocal(input_scale_down)
        h_q = ops.cutlass_scaled_mm_relu2_fp8out_sm100(
            a_fp8, weight_up, input_scale_up, weight_scale_up, recip_down
        )
        return h_q.view(*out_shape)

    # Decode: preserve R1 (or FlashInfer), bf16 relu^2, then static requant
    # exactly as the down_proj would have done -> bit-identical to prod-with-R1.
    if envs.VLLM_NEMOTRON3_FP8_DECODE_GEMM_SM100:
        up_bf16 = ops.cutlass_fp8_decode_gemm_sm100(
            a_fp8, weight_up, input_scale_up, weight_scale_up, out_dtype=torch.bfloat16
        )
    else:
        up_bf16 = flashinfer_scaled_fp8_mm(
            a_fp8,
            weight_up,
            out_dtype=torch.bfloat16,
            scale_a=input_scale_up,
            scale_b=weight_scale_up,
        )
    if envs.VLLM_NEMOTRON3_FP8_RELU2_DECODE_FUSION_SM100:
        # AMMO track shared_expert_relu2_quant_fusion_decode (R13): fuse the
        # 3 eager kernels (relu, square, scaled_fp8_quant) into ONE Triton
        # kernel. Bit-identical to the eager path below (Gate 5.1a nmis == 0);
        # fires ONLY here (decode M <= 8) -- M > 8 took the EVT epilogue above.
        logger.info_once(
            "AMMO[shared_expert_relu2_quant_fusion_decode]: FUSED relu^2+quant "
            "Triton kernel ACTIVE (decode M<=%d)",
            _NEMOTRON3_FP8_DECODE_GEMM_MAX_M,
            scope="global",
        )
        h_q = _fused_relu2_quant(up_bf16, input_scale_down)
        return h_q.view(*out_shape)

    # Baseline arm (flag OFF): production-faithful 3 eager kernels.
    relu2 = torch.square(torch.relu(up_bf16))
    h_q, _ = ops.scaled_fp8_quant(relu2, input_scale_down)
    return h_q.view(*out_shape)


def _nemotron3_fp8_relu2_fused_up_fake(
    x: torch.Tensor,
    weight_up: torch.Tensor,
    weight_scale_up: torch.Tensor,
    input_scale_up: torch.Tensor,
    input_scale_down: torch.Tensor,
) -> torch.Tensor:
    out_shape = (*x.shape[:-1], weight_up.shape[1])
    return torch.empty(
        out_shape, dtype=current_platform.fp8_dtype(), device=x.device
    )


direct_register_custom_op(
    "nemotron3_fp8_relu2_fused_up",
    _nemotron3_fp8_relu2_fused_up_impl,
    mutates_args=[],
    fake_impl=_nemotron3_fp8_relu2_fused_up_fake,
)


def _dynamic_flashinfer_deepgemm_blockscale_gemm_impl(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """
    Conditional FlashInfer FP8 blockscale GEMM with batch-size-dependent selection.

    This function switches between two optimized kernels based on the input batch size:
    - For small batches (M < 32): Uses FlashInfer's DeepGEMM swapAB optimization.
    - For larger batches (M >= 32): Uses the official DeepGEMM kernel.

    The conditional logic must use torch.cond() instead of a simple if-else statement
    to maintain compatibility with torch.compile graph compilation.

    This batch-size-dependent selection is essential for maintaining model accuracy.
    Benchmarks on GSM8K show a significant accuracy gap (88% vs 95%) for DeepSeek-V3.1
    when using FlashInfer's DeepGEMM on M>=32. The M < 32 strategy fixes the accuracy
    drop.

    Args:
        input: Input tensor of shape (batch_size, input_dim) in FP8 format
        weight: Weight tensor of shape (output_dim, input_dim) in FP8 format
        weight_scale: Scale factors for weight quantization (per-group)
        group_size: Quantization group size for the weight tensor
        use_deep_gemm_e8m0: Whether to use the E8M0 format in DeepGEMM quantization

    Returns:
        Output tensor of shape (batch_size, output_dim) in bfloat16 format
    """

    def run_flashinfer_deepgemm_swapAB(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        return flashinfer_fp8_blockscale_gemm(
            input=input,
            weight=weight,
            weight_scale=weight_scale,
            out_dtype=torch.bfloat16,
        )

    def run_deepgemm(
        input: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
    ) -> torch.Tensor:
        q_input, input_scale = per_token_group_quant_fp8(
            input,
            group_size=group_size,
            column_major_scales=True,
            use_ue8m0=use_deep_gemm_e8m0,
        )
        output = torch.empty(
            (q_input.shape[0], weight.shape[0]),
            dtype=torch.bfloat16,
            device=q_input.device,
        )
        fp8_gemm_nt(
            (q_input, input_scale),
            (weight, weight_scale),
            output,
            is_deep_gemm_e8m0_used=use_deep_gemm_e8m0,
        )
        return output

    if envs.VLLM_BATCH_INVARIANT:
        return run_deepgemm(input, weight, weight_scale)

    condition = input.shape[0] < 32

    # PyTorch's torch.compile cannot handle input-dependent control flow in standard
    # Python conditionals. torch.cond() explicitly registers both code paths in the
    # computation graph, allowing torch.compile to capture both branches.
    # without torch.cond, the M < 32 condition won't be able to be captured by torch
    # compile
    return torch.cond(
        condition,
        run_flashinfer_deepgemm_swapAB,
        run_deepgemm,
        (input, weight, weight_scale),
    )


def _dynamic_flashinfer_deepgemm_blockscale_gemm_fake(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
    use_deep_gemm_e8m0: bool,
) -> torch.Tensor:
    """
    Required fake/meta implementation for torch.compile graph tracing.
    """
    return torch.empty(
        input.shape[0], weight.shape[0], dtype=torch.bfloat16, device=input.device
    )


direct_register_custom_op(
    "dynamic_flashinfer_deepgemm_blockscale_gemm",
    _dynamic_flashinfer_deepgemm_blockscale_gemm_impl,
    fake_impl=_dynamic_flashinfer_deepgemm_blockscale_gemm_fake,
)
