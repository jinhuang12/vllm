# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AMMO OP-004: FP8 (W8A8) replacement for the bf16 self-attention QKV/O
projection GEMMs.

Context
-------
The ``nvidia/gemma-4-31B-it-NVFP4`` checkpoint quantizes only the MLP Linears
to NVFP4. All 60 layers' ``self_attn*`` projections (and ``lm_head`` / vision
tower) are in the modelopt ``quantization_config.ignore`` list, so they fall
through to :class:`UnquantizedLinearMethod` and run bf16 on cuBLASLt nvJet.

At decode (M = 5-160 tokens) these GEMMs are **weight-load HBM-bandwidth
bound** (NCU: bf16 at 65.7% of HBM peak, 179.6 MB/QKV-layer). bf16 cannot move
fewer weight bytes, so the only physical lever is precision reduction. FP8-e4m3
weights halve the weight stream (NCU: 179.6 -> 90.6 MB, 1.98x) and route through
vLLM's SM100-native ``cutlass_scaled_mm`` (``SM100_MMA_F8F6F4``).

This is a **LOSSY** optimization (bf16 -> fp8-e4m3, per-channel weight scale +
per-token dynamic activation scale). It is gated behind ``VLLM_OP004_FP8_ATTN``
(default on); set ``VLLM_OP004_FP8_ATTN=0`` to fall back to the original bf16
path for clean A/B ablation.

Design (production-parity, torch.compile-safe)
----------------------------------------------
* ``create_weights`` is inherited from :class:`UnquantizedLinearMethod`, so the
  bf16 weight loads exactly as in the baseline (no checkpoint-format change).
* ``process_weights_after_loading`` quantizes the loaded bf16 ``[N, K]`` weight
  to fp8-e4m3 **once** with a per-output-channel scale, then stores it
  column-major ``[K, N]`` as ``cutlass_scaled_mm`` requires (``b.stride(0)==1``).
* ``apply`` per-token-dynamic-quantizes the bf16 activation via the registered
  ``QuantFP8`` CustomOp (so the existing ``RMSNormQuantFusionPass`` can fuse the
  preceding ``input_layernorm`` RMSNorm into the QKV act-quant at TP=1), then
  calls the registered ``cutlass_scaled_mm`` op. Both are graph-visible
  registered ops -- no opaque wrapper, no Python shape branch, no in-place
  mutation -- satisfying torch-compile-contract invariants 1/3/4/5.
"""

import torch

from vllm import _custom_ops as ops
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.layer_utils import (
    replace_parameter,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

logger = init_logger(__name__)


def op004_fp8_attn_enabled() -> bool:
    """Master enable flag for the OP-004 FP8 attention-projection path."""
    return bool(getattr(envs, "VLLM_OP004_FP8_ATTN", False))


def is_op004_attn_projection(prefix: str) -> bool:
    """True iff ``prefix`` names a self-attention QKV or O projection.

    Scope is deliberately narrow: ONLY language-model ``*.self_attn.qkv_proj``
    and ``*.self_attn.o_proj``. This must NOT match ``lm_head`` (also excluded
    from FP4) or vision-tower attention, which are out of OP-004's scope (the
    bottleneck is the 60 language-model decoder layers).
    """
    if "vision_tower" in prefix or "vision_model" in prefix:
        return False
    if ".self_attn." not in prefix:
        return False
    return prefix.endswith("qkv_proj") or prefix.endswith("o_proj")


class Op004Fp8AttnLinearMethod(UnquantizedLinearMethod):
    """W8A8 FP8 linear method for the bf16 self-attention QKV/O projections.

    Inherits ``create_weights`` (bf16 weight allocation + standard weight
    loader) from :class:`UnquantizedLinearMethod`; overrides
    ``process_weights_after_loading`` (one-time weight quant) and ``apply``
    (per-token act-quant + ``cutlass_scaled_mm``).
    """

    def __init__(self, prefix: str = "") -> None:
        super().__init__()
        self.prefix = prefix
        self.fp8_dtype = current_platform.fp8_dtype()
        # Per-token dynamic activation quantization. Using the registered
        # QuantFP8 CustomOp (rather than a raw ops.scaled_fp8_quant call) keeps
        # the lowered graph node identical to production fp8 paths, so the
        # RMSNorm->dynamic-per-token-fp8-quant fusion pattern can match.
        self.act_quant = QuantFP8(static=False, group_shape=GroupShape.PER_TOKEN)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        weight = layer.weight
        # Idempotency guard: only quantize a still-bf16 weight once.
        if weight.dtype == self.fp8_dtype:
            return
        if weight.dim() != 2:
            # Not a standard 2D Linear weight; leave untouched (defensive).
            logger.warning(
                "OP-004: skipping FP8 quant for %s (weight ndim=%d)",
                self.prefix,
                weight.dim(),
            )
            return

        # weight is [N, K] (output_channels, input). Quantize per-output-channel
        # (treat each output row as a "token" -> one fp8 scale per output
        # channel). scaled_fp8_quant(use_per_token_if_dynamic=True) returns
        # (fp8 weight [N, K], scale [N, 1] float32).
        w_bf16 = weight.data.contiguous()
        qweight, weight_scale = ops.scaled_fp8_quant(
            w_bf16, scale=None, use_per_token_if_dynamic=True
        )

        # cutlass_scaled_mm requires b column-major ([K, N], stride(0)==1).
        # qweight is [N, K] row-major -> .t() is [K, N] with stride (1, K).
        qweight_t = qweight.t()

        replace_parameter(layer, "weight", qweight_t)
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(weight_scale.contiguous(), requires_grad=False),
        )

        # cutlass_scaled_mm requires b column-major (b.stride(0) == 1). The
        # transpose above yields that layout; assert it survived
        # replace_parameter so any future regression fails loudly at load time
        # rather than as a cryptic CUDA stride-check crash in the first forward.
        assert layer.weight.stride(0) == 1, (
            f"OP-004 {self.prefix}: fp8 weight is not column-major "
            f"(stride={layer.weight.stride()}); cutlass_scaled_mm requires "
            f"b.stride(0)==1. Do NOT call .contiguous() on the transposed weight."
        )

        # Fast-path activation proof (the E2E sweep matches this as a
        # require_pattern to prove the FP8 path actually fired in the opt run).
        logger.info(
            "OP004_FP8_ACTIVE cutlass_scaled_mm fp8 attn-proj quantized: %s "
            "[K=%d,N=%d]",
            self.prefix,
            layer.weight.shape[0],
            layer.weight.shape[1],
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x: [..., K] bf16. Flatten to 2D for the GEMM.
        x_2d = x.view(-1, x.shape[-1])

        # Per-token dynamic fp8 activation quant: x_fp8 [M, K], x_scale [M, 1].
        x_fp8, x_scale = self.act_quant(x_2d)

        # layer.weight is fp8 [K, N] column-major; weight_scale is [N, 1] f32.
        weight = layer.weight
        weight_scale = layer.weight_scale
        n = weight.shape[1]

        out = ops.cutlass_scaled_mm(
            x_fp8,
            weight,
            scale_a=x_scale,
            scale_b=weight_scale,
            out_dtype=x.dtype,
            bias=bias,
        )
        return out.view(*x.shape[:-1], n)
