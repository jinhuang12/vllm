// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Custom prefill-shape (large-M) CUTLASS SM100 FP8 dense GEMM with per-shape
// tuned TileN=256 mainloop schedules.
//
// AMMO track: dense_fp8_prefill_gemm_sm100  (Round 6)
// Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (B300, sm_103)
//
// Motivation
// ----------
// At prefill the dense per-tensor FP8 GEMM family runs at the chunk token count
// M ~= 10624 (compute-bound, eager / graphId=NULL). The in-tree c3x dispatch
// (scaled_mm_sm100_fp8_dispatch.cuh:99-115, 286-294) routes every M>256 GEMM to
// a SINGLE fixed config -- sm100_fp8_config_default = TileShape<256,128,128> /
// ClusterShape<2,2,1>. c3x's whole SM100 FP8 menu tops out at TileN=128; it has
// NO TileN=256 config and does NOT tune the tile per output-N. For the
// Nemotron-3 prefill FP8 shapes a wider-N tile (TileN=256) packs the grid more
// efficiently and beats c3x's fixed default on all 5 shapes.
//
// A drift-immune INTERLEAVED A/B/A/B decider at matched steady-state clocks,
// cold-L2, correctness-gated (champion2_fp8_INTERLEAVED_decider.log) measured
// c3x/custom speedups (all bit-identical, max_abs=0.0 => lossless):
//   in_proj    [N=18560,K=4096]  Tile<256,256,128> Clu<2,1,1>  1.1605x
//   out_proj   [N= 4096,K=8192]  Tile<128,256,128> Clu<2,1,1>  1.1177x
//   o_proj     [N= 4096,K=4096]  Tile<128,256,128> Clu<2,1,1>  1.0979x
//   shared_up  [N= 5376,K=4096]  Tile<128,256,128> Clu<2,1,1>  1.1105x
//   shared_down[N= 4096,K=5376]  Tile<128,256,128> Clu<2,1,1>  1.1004x
//
// Tile selection rule (per output-N):
//   N >  8192  -> sm100_fp8_prefill_config_largeN  Tile<256,256,128> Clu<2,1,1>
//                 (in_proj N=18560 is the only family member with N>8192)
//   N <= 8192  -> sm100_fp8_prefill_config_default Tile<128,256,128> Clu<2,1,1>
//                 (out_proj/o_proj/shared_up/shared_down, all N in [4096,5376])
//
// Both configs are NON swap_ab (large-M compute-bound; swap_ab is a skinny-M
// trick and is irrelevant here) and use the stock c3x::ScaledEpilogue, so the
// numerics are byte-for-byte identical to c3x cutlass_scaled_mm (fp8_e4m3
// operands, fp32 accumulate, per-tensor scale_a*scale_b folded in the epilogue,
// bf16/fp16 out). Lossless. No bias path (the Nemotron-3 dense FP8 layers carry
// no bias; the Python wrapper routes the bias case back to production).
//
// IMPORTANT (baseline integrity): this is an ADDITIVE new kernel. It does NOT
// touch the R1-shipped cutlass_fp8_decode_gemm_sm100.cu (decode M<=8) nor the
// production FlashInfer path. The Python dispatcher gates this kernel behind
// VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100 and only routes prefill large-M GEMMs
// (M>256) here; decode and the M in (8,256] regime are unchanged.

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm100_fp8_dispatch.cuh"

namespace vllm {

namespace {

// Large-N prefill config: in_proj [N=18560]. TileN=256 (CUTLASS SM100 max valid
// TileN) widens the N tiling; TileM=256 with ClusterShape<2,1,1> routes to the
// 2SM cooperative UMMA schedule (cluster_M=2 even, TileM=256 divisible by 128).
template <typename InType, typename OutType>
struct sm100_fp8_prefill_config_largeN {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_256, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  // No-bias path: stock ScaledEpilogue applies per-tensor scale_a * scale_b
  // (bit-identical to c3x). swap_ab=false (default) -- large-M compute-bound.
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule>;
};

// Default prefill config: out_proj / o_proj / shared_up / shared_down
// (N in [4096, 5376]). TileM=128 + TileN=256, ClusterShape<2,1,1> -> 2SM UMMA
// (cluster_M=2 even, TileM=128 divisible by 128).
template <typename InType, typename OutType>
struct sm100_fp8_prefill_config_default {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_128, cute::_256, cute::_128>;
  using ClusterShape = cute::Shape<cute::_2, cute::_1, cute::_1>;

  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule>;
};

// Per-output-N tile dispatch. Correctness is tile-invariant (every config is
// the same lossless ScaledEpilogue GEMM); the tile only changes performance.
template <typename OutType>
void cutlass_fp8_prefill_gemm_sm100_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  using LargeN =
      typename sm100_fp8_prefill_config_largeN<cutlass::float_e4m3_t,
                                               OutType>::Cutlass3xGemm;
  using Default =
      typename sm100_fp8_prefill_config_default<cutlass::float_e4m3_t,
                                                OutType>::Cutlass3xGemm;

  // b is [K, N] column-major; N = b.size(1).
  int64_t const n = b.size(1);
  if (n > 8192) {
    // in_proj (N=18560): the only prefill FP8 shape with N>8192.
    return cutlass_gemm_caller_sm100_fp8<LargeN>(out, a, b, a_scales, b_scales);
  }
  // out_proj / o_proj / shared_up / shared_down (N in [4096, 5376]).
  return cutlass_gemm_caller_sm100_fp8<Default>(out, a, b, a_scales, b_scales);
}

}  // namespace

void cutlass_fp8_prefill_gemm_sm100(torch::stable::Tensor& out,
                                    torch::stable::Tensor const& a,
                                    torch::stable::Tensor const& b,
                                    torch::stable::Tensor const& a_scales,
                                    torch::stable::Tensor const& b_scales) {
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1,
                  "prefill_gemm requires per-tensor scalar scales");
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_fp8_prefill_gemm_sm100_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_fp8_prefill_gemm_sm100_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
