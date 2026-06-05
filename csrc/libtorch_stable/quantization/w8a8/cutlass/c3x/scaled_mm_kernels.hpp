#pragma once

#include <torch/csrc/stable/tensor.h>

namespace vllm {

void cutlass_scaled_mm_sm90_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_sm90_int8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm90_int8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm90_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales);

void cutlass_scaled_mm_sm100_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias);

// AMMO track dense_fp8_decode_gemm_sm100: custom skinny-M (decode-shape) FP8
// dense GEMM instantiating the cuBLAS-Lt-equivalent tileN=128 ~1-wave schedule.
// No bias (Nemotron-3 dense FP8 layers carry no bias).
void cutlass_fp8_decode_gemm_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales);

// AMMO track dense_fp8_prefill_gemm_sm100: custom prefill-shape (large-M) FP8
// dense GEMM with per-output-N tuned TileN=256 schedules (in_proj
// Tile<256,256,128>; others Tile<128,256,128>; both Cluster<2,1,1>). No bias.
void cutlass_fp8_prefill_gemm_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales);

// AMMO track fp8_relu2_requant_epilogue_sm100: dense FP8 GEMM with a fused
// ReLUSquared + static per-tensor requant-to-fp8 epilogue. fp8 (e4m3) output
// pre-scaled by out_scale. Per-tensor scalar scales, no bias.
void cutlass_scaled_mm_relu2_fp8out_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale);

// Attribution-by-ablation sibling (Gate 5.2 only): same epilogue MINUS the
// ReLUSquared node (dequant -> requant -> fp8). NOT a production path.
void cutlass_scaled_mm_cast_fp8out_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale);

void cutlass_scaled_mm_sm120_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_blockwise_sm100_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales);

void cutlass_scaled_mm_blockwise_sm120_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales);
}  // namespace vllm
