// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Custom decode-shape (skinny-M) CUTLASS SM100 FP8 dense GEMM.
//
// AMMO track: dense_fp8_decode_gemm_sm100
// Model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 (B300, sm_103)
//
// Motivation
// ----------
// The production Nemotron-3 dense per-tensor FP8 path dispatches through
// FlashInfer `bmm_fp8(..., "auto")`, which selects the general-purpose CUTLASS
// SM100 e4m3 template with TileShape <64,64,128>. At decode shapes (M = batch =
// 1/8/32) this collapses the M dimension (tileM=64 >= M) and tiles only N by 64,
// producing gridX=1, gridY=ceil(N/64). For Mamba in_proj (N=18560) that is 290
// CTAs ~= 1.96 waves on 148 SMs (a half-empty 2nd wave); for out_proj (N=4096 in
// the [4096,8192] layout) only 64 CTAs ~= 0.43 waves.
//
// A measured head-to-head (champion1_baseline_headtohead.log) showed cuBLAS-Lt
// selects a tileN=128 / skinny-tileM "nvjet_sm103_qqtst_128x8" schedule giving
// ~146 CTAs ~= 1.0 wave at M=1 and beating production 1.39-1.48x cold @ M=1.
//
// This kernel instantiates exactly that winning schedule as a *custom* CUTLASS
// GemmUniversal collective, pinned to the decode M-buckets. It reuses the same
// SM100 MainloopSm100TmaUmmaWarpSpecialized collective the baseline already uses
// internally (the proven, CUDA-graph-safe path), specialized via the swap_ab
// skinny-M config: TileShape <128,32,128> with swap_ab=true tiles the ORIGINAL N
// dimension by 128 (~1 wave). This is the same schedule vLLM already ships in
// `sm100_fp8_config_M16_swap_ab`, which the production Nemotron-3 path bypasses.
//
// Precision is identical to baseline: fp8_e4m3 operands, fp32 accumulate,
// bf16/fp16 output, per-tensor scale_a/scale_b folded in the epilogue. No new
// precision reduction => lossless. No bias path (the Nemotron-3 dense FP8 layers
// carry no bias; the Python wrapper routes the bias case back to production).

#include "scaled_mm_kernels.hpp"
#include "scaled_mm_sm100_fp8_dispatch.cuh"

namespace vllm {

namespace {

// Decode-shape skinny-M config: the cuBLAS-Lt-equivalent winning schedule.
//
// swap_ab=true => the GEMM problem is presented to the collective as (N, M, K),
// so TileShape's leading mode (128) tiles the ORIGINAL N dimension. For the
// Nemotron-3 decode shapes (N in {4096, 4608, 5376, 8192, 18560}) this yields
// ceil(N/128) CTAs, i.e. ~1 wave for the large-N projections on 148 SMs --
// matching cuBLAS-Lt's measured (146,1,1) grid. ClusterShape <4,1,1> matches the
// in-tree M16_swap_ab config that demonstrably compiles and is correct on SM100.
template <typename InType, typename OutType>
struct sm100_fp8_decode_gemm_config {
  static_assert(std::is_same<InType, cutlass::float_e4m3_t>());
  using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;
  using EpilogueSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
  using TileShape = cute::Shape<cute::_128, cute::_32, cute::_128>;
  using ClusterShape = cute::Shape<cute::_4, cute::_1, cute::_1>;

  // No-bias decode path: ScaledEpilogue applies per-tensor scale_a * scale_b.
  // swap_ab=true (last template arg) so the operands/layouts are swapped to the
  // (N, M, K) presentation that gives the skinny-M ~1-wave schedule.
  using Cutlass3xGemm =
      cutlass_3x_gemm_sm100_fp8<InType, OutType, c3x::ScaledEpilogue, TileShape,
                                ClusterShape, KernelSchedule, EpilogueSchedule,
                                /*swap_ab=*/true>;
};

template <typename OutType>
void cutlass_fp8_decode_gemm_sm100_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales) {
  using Cutlass3xGemm =
      typename sm100_fp8_decode_gemm_config<cutlass::float_e4m3_t,
                                            OutType>::Cutlass3xGemm;
  // swap_ab caller: pass (b_scales, a_scales) in swapped order, matching the
  // in-tree swap_ab convention in scaled_mm_sm100_fp8_dispatch.cuh.
  return cutlass_gemm_caller_sm100_fp8<Cutlass3xGemm>(out, a, b, b_scales,
                                                      a_scales);
}

}  // namespace

void cutlass_fp8_decode_gemm_sm100(torch::stable::Tensor& out,
                                   torch::stable::Tensor const& a,
                                   torch::stable::Tensor const& b,
                                   torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales) {
  STD_TORCH_CHECK(a.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(b.scalar_type() ==
                  torch::headeronly::ScalarType::Float8_e4m3fn);
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1,
                  "decode_gemm requires per-tensor scalar scales");
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  if (out.scalar_type() == torch::headeronly::ScalarType::BFloat16) {
    return cutlass_fp8_decode_gemm_sm100_dispatch<cutlass::bfloat16_t>(
        out, a, b, a_scales, b_scales);
  } else {
    STD_TORCH_CHECK(out.scalar_type() == torch::headeronly::ScalarType::Half);
    return cutlass_fp8_decode_gemm_sm100_dispatch<cutlass::half_t>(
        out, a, b, a_scales, b_scales);
  }
}

}  // namespace vllm
