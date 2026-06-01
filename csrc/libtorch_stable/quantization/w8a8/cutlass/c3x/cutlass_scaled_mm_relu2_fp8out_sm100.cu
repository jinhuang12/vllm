// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fused ReLUSquared + requant-to-fp8 epilogue on the dense FP8 GEMM (SM100).
//
// AMMO track: fp8_relu2_requant_epilogue_sm100
// Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (B300, sm_103)
//
// Motivation
// ----------
// The production Nemotron-3 shared-expert MLP runs:
//   up_proj FP8 GEMM (bf16 out)
//     -> Inductor-fused vectorized_elementwise_kernel (ReLUSquared + requant
//        to fp8, one full HBM round-trip of the [M,N] up_proj output)
//     -> down_proj FP8 GEMM (consumes fp8 directly)
// The down_proj input requant scale is a STATIC per-tensor scalar (modelopt
// kFp8StaticTensorSym), so the whole activation+requant is a pure per-element
// op with no grid-wide amax reduction -- cleanly foldable into a CUTLASS EVT
// epilogue.
//
// This op fuses that activation+requant into the up_proj GEMM epilogue via the
// ScaledEpilogueReLUSquared EVT struct, writing fp8 (float_e4m3fn) directly. It
// eliminates the separate elementwise kernel's launch + HBM round-trip, and
// halves the output write (fp8 vs bf16).
//
// Operand contract (mirrors cutlass_scaled_mm but fp8 OUTPUT + an extra
// out_scale):
//   a        : [M, K] fp8_e4m3 row-major (the quantized up_proj input)
//   b        : [K, N] fp8_e4m3 column-major (the up_proj weight, b.stride(0)==1)
//   a_scales : per-tensor scalar f32 (up_proj input_scale)
//   b_scales : per-tensor scalar f32 (up_proj weight_scale)
//   out_scale: per-tensor scalar f32 = 1 / down_proj.input_scale (the static
//              requant scalar)
//   out      : [M, N] fp8_e4m3 row-major  (pre-scaled for down_proj's GEMM)
//
// This is the FUSED form: out = saturate_fp8( out_scale * relu(scaleA*scaleB*
// acc)^2 ). It is ADDITIVE -- it does not touch the R1-shipped decode kernel
// (cutlass_fp8_decode_gemm_sm100) or the stock cutlass_scaled_mm path. The EVT
// struct is mainloop-agnostic, so Stage-6 can graft it onto a custom mainloop.
//
// Precision: fp8_e4m3 operands, fp32 accumulate, in-register f32 dequant +
// ReLUSquared + requant scalar, final NumericArrayConverter saturating cast to
// fp8 (+-448, native). Bit-equivalent to the production separate kernels (same
// dtype boundary, in-register instead of via HBM) -> lossless.

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm100_fp8_dispatch.cuh"

namespace vllm {

namespace {

// Fused-epilogue config structs, one per M bucket. These mirror the TileShape /
// ClusterShape / swap_ab of the stock sm100_fp8_config_* (no-bias variants) in
// scaled_mm_sm100_fp8_dispatch.cuh, but plug in the supplied fused Epilogue and
// force ElementD = float_e4m3_t. There is no bias on the shared-expert MLP path
// (Nemotron-3 dense FP8 layers carry no bias), so no EnableBias variant.
//
// Epilogue is a template-template parameter so the SAME config / dispatch is
// shared by the full relu2 op and the cast-only ablation op -- guaranteeing the
// two arms differ ONLY by the ReLUSquared node (identical tiles, cluster,
// swap_ab, memory traffic) for the Gate-5.2 attribution-by-ablation.

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_fused_config_default {
  // M in (256, inf) -- the prefill large-M target (M ~= 10624).
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_256, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_2, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, Epilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_fused_config_M256 {
  // M in (64, 256]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_128, cute::_128, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, Epilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_fused_config_M64 {
  // M = 64 and K < 4096 (no swap AB)
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_64, cute::_64, cute::_128>;
  using ClusterShape = cute::Shape<cute::_1, cute::_1, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, Epilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_fused_config_M64_swap_ab {
  // M in (16, 64] and K >= 4096
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_128, cute::_64, cute::_256>;
  using ClusterShape = cute::Shape<cute::_4, cute::_1, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, Epilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                /*swap_ab=*/true>;
};

template <typename InType, typename OutType,
          template <typename, typename, typename> typename Epilogue>
struct sm100_fp8_fused_config_M16_swap_ab {
  // M in [1, 16]
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_128, cute::_32, cute::_128>;
  using ClusterShape = cute::Shape<cute::_4, cute::_1, cute::_1>;
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, Epilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                /*swap_ab=*/true>;
};

// M-bucket dispatch, mirroring cutlass_gemm_sm100_fp8_dispatch but with the
// supplied fused Epilogue + an explicit out_scale epilogue arg. The swap_ab
// buckets pass (b_scales, a_scales) in swapped order, matching the in-tree
// swap_ab convention; out_scale is a per-tensor scalar applied as the outermost
// EVT node (order-invariant).
template <typename OutType,
          template <typename, typename, typename> typename Epilogue>
void cutlass_scaled_mm_fused_fp8out_sm100_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale) {
  using InType = cutlass::float_e4m3_t;
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);

  using GemmDefault = typename sm100_fp8_fused_config_default<
      InType, OutType, Epilogue>::Cutlass3xGemm;
  using GemmM256 = typename sm100_fp8_fused_config_M256<InType, OutType,
                                                        Epilogue>::Cutlass3xGemm;
  using GemmM64 = typename sm100_fp8_fused_config_M64<InType, OutType,
                                                      Epilogue>::Cutlass3xGemm;
  using GemmM64SwapAB = typename sm100_fp8_fused_config_M64_swap_ab<
      InType, OutType, Epilogue>::Cutlass3xGemm;
  using GemmM16SwapAB = typename sm100_fp8_fused_config_M16_swap_ab<
      InType, OutType, Epilogue>::Cutlass3xGemm;

  uint32_t const m = a.size(0);
  uint32_t const k = a.size(1);

  if (m <= 16) {
    return cutlass_gemm_caller_sm100_fp8<GemmM16SwapAB>(
        out, a, b, b_scales, a_scales, out_scale);
  } else if (m <= 64) {
    if (m == 64 && k < 4096) {
      return cutlass_gemm_caller_sm100_fp8<GemmM64>(out, a, b, a_scales,
                                                    b_scales, out_scale);
    }
    return cutlass_gemm_caller_sm100_fp8<GemmM64SwapAB>(
        out, a, b, b_scales, a_scales, out_scale);
  } else if (m <= 256) {
    return cutlass_gemm_caller_sm100_fp8<GemmM256>(out, a, b, a_scales,
                                                   b_scales, out_scale);
  } else {
    return cutlass_gemm_caller_sm100_fp8<GemmDefault>(out, a, b, a_scales,
                                                      b_scales, out_scale);
  }
}

// Shared operand/scale checks for both fused fp8-out ops.
void check_relu2_fp8out_args(torch::stable::Tensor const& out,
                             torch::stable::Tensor const& a,
                             torch::stable::Tensor const& b,
                             torch::stable::Tensor const& a_scales,
                             torch::stable::Tensor const& b_scales,
                             torch::stable::Tensor const& out_scale) {
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(out.scalar_type() ==
                      torch::headeronly::ScalarType::Float8_e4m3fn,
                  "fused fp8out op requires fp8 (e4m3) output");
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1 &&
                      out_scale.numel() == 1,
                  "fused fp8out op requires per-tensor scalar scales");
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous() &&
                  out_scale.is_contiguous());
}

}  // namespace

// Full fused op: out = saturate_fp8( out_scale * relu(scaleA*scaleB*acc)^2 ).
void cutlass_scaled_mm_relu2_fp8out_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale) {
  check_relu2_fp8out_args(out, a, b, a_scales, b_scales, out_scale);
  cutlass_scaled_mm_fused_fp8out_sm100_dispatch<cutlass::float_e4m3_t,
                                                c3x::ScaledEpilogueReLUSquared>(
      out, a, b, a_scales, b_scales, out_scale);
}

// Attribution-by-ablation op: out = saturate_fp8( out_scale * scaleA*scaleB*acc
// ) -- same as above MINUS the ReLUSquared node. Gate-5.2 ablation only; NOT a
// production path.
void cutlass_scaled_mm_cast_fp8out_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale) {
  check_relu2_fp8out_args(out, a, b, a_scales, b_scales, out_scale);
  cutlass_scaled_mm_fused_fp8out_sm100_dispatch<cutlass::float_e4m3_t,
                                                c3x::ScaledEpilogueCastFp8>(
      out, a, b, a_scales, b_scales, out_scale);
}

}  // namespace vllm
