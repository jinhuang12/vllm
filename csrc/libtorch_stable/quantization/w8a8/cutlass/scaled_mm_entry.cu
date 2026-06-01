#include <cudaTypedefs.h>

#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/torch_utils.h"

#include "cutlass_extensions/common.hpp"

void cutlass_scaled_mm_sm75(torch::stable::Tensor& c,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_sm80(torch::stable::Tensor& c,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_sm89(torch::stable::Tensor& c,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_sm90(torch::stable::Tensor& c,
                            torch::stable::Tensor const& a,
                            torch::stable::Tensor const& b,
                            torch::stable::Tensor const& a_scales,
                            torch::stable::Tensor const& b_scales,
                            std::optional<torch::stable::Tensor> const& bias);
#endif
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
void cutlass_moe_mm_sm90(torch::stable::Tensor& out_tensors,
                         torch::stable::Tensor const& a_tensors,
                         torch::stable::Tensor const& b_tensors,
                         torch::stable::Tensor const& a_scales,
                         torch::stable::Tensor const& b_scales,
                         torch::stable::Tensor const& expert_offsets,
                         torch::stable::Tensor const& problem_sizes,
                         torch::stable::Tensor const& a_strides,
                         torch::stable::Tensor const& b_strides,
                         torch::stable::Tensor const& c_strides,
                         bool per_act_token, bool per_out_ch);

#endif

#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100
void cutlass_moe_mm_sm100(torch::stable::Tensor& out_tensors,
                          torch::stable::Tensor const& a_tensors,
                          torch::stable::Tensor const& b_tensors,
                          torch::stable::Tensor const& a_scales,
                          torch::stable::Tensor const& b_scales,
                          torch::stable::Tensor const& expert_offsets,
                          torch::stable::Tensor const& problem_sizes,
                          torch::stable::Tensor const& a_strides,
                          torch::stable::Tensor const& b_strides,
                          torch::stable::Tensor const& c_strides,
                          bool per_act_token, bool per_out_ch);
#endif

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
void cutlass_scaled_mm_sm120(torch::stable::Tensor& c,
                             torch::stable::Tensor const& a,
                             torch::stable::Tensor const& b,
                             torch::stable::Tensor const& a_scales,
                             torch::stable::Tensor const& b_scales,
                             std::optional<torch::stable::Tensor> const& bias);
#endif

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
void cutlass_scaled_mm_sm100(torch::stable::Tensor& c,
                             torch::stable::Tensor const& a,
                             torch::stable::Tensor const& b,
                             torch::stable::Tensor const& a_scales,
                             torch::stable::Tensor const& b_scales,
                             std::optional<torch::stable::Tensor> const& bias);

// AMMO track dense_fp8_decode_gemm_sm100: custom skinny-M decode-shape kernel
// launcher (defined in c3x/cutlass_fp8_decode_gemm_sm100.cu).
namespace vllm {
void cutlass_fp8_decode_gemm_sm100(torch::stable::Tensor& out,
                                   torch::stable::Tensor const& a,
                                   torch::stable::Tensor const& b,
                                   torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales);

// AMMO track dense_fp8_prefill_gemm_sm100: custom prefill-shape large-M kernel
// launcher (defined in c3x/cutlass_fp8_prefill_gemm_sm100.cu).
void cutlass_fp8_prefill_gemm_sm100(torch::stable::Tensor& out,
                                    torch::stable::Tensor const& a,
                                    torch::stable::Tensor const& b,
                                    torch::stable::Tensor const& a_scales,
                                    torch::stable::Tensor const& b_scales);

// AMMO track fp8_relu2_requant_epilogue_sm100: fused ReLUSquared+requant-to-fp8
// epilogue launchers (defined in c3x/cutlass_scaled_mm_relu2_fp8out_sm100.cu).
void cutlass_scaled_mm_relu2_fp8out_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale);
void cutlass_scaled_mm_cast_fp8out_sm100(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale);
}  // namespace vllm
#endif

#if (defined(ENABLE_CUTLASS_MOE_SM90) && ENABLE_CUTLASS_MOE_SM90) ||   \
    (defined(ENABLE_CUTLASS_MOE_SM100) && ENABLE_CUTLASS_MOE_SM100) || \
    (defined(ENABLE_CUTLASS_MOE_SM120) && ENABLE_CUTLASS_MOE_SM120)
void get_cutlass_moe_mm_data_caller(
    const torch::stable::Tensor& topk_ids,
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    torch::stable::Tensor& input_permutation,
    torch::stable::Tensor& output_permutation, const int64_t num_experts,
    const int64_t n, const int64_t k,
    const std::optional<torch::stable::Tensor>& blockscale_offsets,
    const bool is_gated);

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets_caller(
    const torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2, const int64_t n, const int64_t k,
    const bool swap_ab);

void get_cutlass_batched_moe_mm_data_caller(
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    const torch::stable::Tensor& expert_num_tokens,
    const int64_t num_local_experts, const int64_t padded_m, const int64_t n,
    const int64_t k);
#endif

void cutlass_scaled_mm_azp_sm75(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm80(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias);

void cutlass_scaled_mm_azp_sm89(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias);

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
void cutlass_scaled_mm_azp_sm90(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales, torch::stable::Tensor const& azp_adj,
    std::optional<torch::stable::Tensor> const& azp,
    std::optional<torch::stable::Tensor> const& bias);
#endif

bool cutlass_scaled_mm_supports_fp8(int64_t cuda_device_capability) {
  // CUTLASS FP8 kernels need at least
  //   CUDA 12.0 on SM90 systems (Hopper)
  //   CUDA 12.4 on SM89 systems (Lovelace)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  } else if (cuda_device_capability >= 89) {
    return CUDA_VERSION >= 12040;
  }
#endif

  return false;
}

bool cutlass_scaled_mm_supports_block_fp8(int64_t cuda_device_capability) {
  // CUTLASS block-quantized FP8 kernels need at least CUDA 12.0
  // and at least SM90 (Hopper)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 100) {
    return CUDA_VERSION >= 12080;
  } else if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12000;
  }
#endif

  return false;
}

bool cutlass_group_gemm_supported(int64_t cuda_device_capability) {
  // CUTLASS grouped FP8 kernels need at least CUDA 12.3 and SM90 (Hopper)
  // or CUDA 12.8 and SM100 (Blackwell)

#if defined CUDA_VERSION
  if (cuda_device_capability >= 100) {
    return CUDA_VERSION >= 12080;
  }
  if (cuda_device_capability >= 90) {
    return CUDA_VERSION >= 12030;
  }
#endif

  return false;
}

void cutlass_scaled_mm(torch::stable::Tensor& c, torch::stable::Tensor const& a,
                       torch::stable::Tensor const& b,
                       torch::stable::Tensor const& a_scales,
                       torch::stable::Tensor const& b_scales,
                       std::optional<torch::stable::Tensor> const& bias) {
  // Checks for conformality
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));

  // Check for strides and alignment
  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(c.stride(0) % 16 == 0 &&
                  b.stride(1) % 16 == 0);  // 16 Byte Alignment

  if (bias) {
    STD_TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous() &&
                    bias->dim() == 1);
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM120 && ENABLE_SCALED_MM_SM120
  if (version_num >= 120) {
    cutlass_scaled_mm_sm120(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    cutlass_scaled_mm_sm100(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  // Guard against compilation issues for sm90 kernels
#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90 && version_num < 100) {
    // Hopper
    cutlass_scaled_mm_sm90(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_sm89(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_sm80(c, a, b, a_scales, b_scales, bias);
    return;
  }

  if (version_num >= 75) {
    // Turing
    cutlass_scaled_mm_sm75(c, a, b, a_scales, b_scales, bias);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}

// AMMO track dense_fp8_decode_gemm_sm100.
// Custom skinny-M (decode-shape) FP8 dense GEMM. Same operand contract as
// cutlass_scaled_mm (A=[M,K] fp8 row-major, B=[K,N] fp8 col-major, scalar f32
// scales, bf16/fp16 out), but specialized to the cuBLAS-Lt-equivalent tileN=128
// ~1-wave schedule and pinned to decode shapes. SM100-only; per-tensor scales;
// no bias. Caller (Python) gates this behind VLLM_NEMOTRON3_FP8_DECODE_GEMM_SM100
// and only routes the decode M-buckets here.
void cutlass_fp8_decode_gemm_sm100(torch::stable::Tensor& c,
                                   torch::stable::Tensor const& a,
                                   torch::stable::Tensor const& b,
                                   torch::stable::Tensor const& a_scales,
                                   torch::stable::Tensor const& b_scales) {
  // Checks for conformality (mirror cutlass_scaled_mm).
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1,
                  "decode_gemm requires per-tensor scalar scales");

  // Check for strides and alignment.
  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(c.stride(0) % 16 == 0 &&
                  b.stride(1) % 16 == 0);  // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    vllm::cutlass_fp8_decode_gemm_sm100(c, a, b, a_scales, b_scales);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "cutlass_fp8_decode_gemm_sm100 is only compiled for SM100 (Blackwell). "
      "CUDA device capability: ",
      version_num);
}

// AMMO track dense_fp8_prefill_gemm_sm100.
// Custom prefill-shape (large-M) FP8 dense GEMM. Same operand contract as
// cutlass_scaled_mm (A=[M,K] fp8 row-major, B=[K,N] fp8 col-major, scalar f32
// scales, bf16/fp16 out), specialized to per-output-N tuned TileN=256
// schedules and pinned to prefill large-M. SM100-only; per-tensor scales; no
// bias. Caller (Python) gates this behind VLLM_NEMOTRON3_FP8_PREFILL_GEMM_SM100
// and only routes the prefill large-M GEMMs (M>256) here.
void cutlass_fp8_prefill_gemm_sm100(torch::stable::Tensor& c,
                                    torch::stable::Tensor const& a,
                                    torch::stable::Tensor const& b,
                                    torch::stable::Tensor const& a_scales,
                                    torch::stable::Tensor const& b_scales) {
  // Checks for conformality (mirror cutlass_scaled_mm).
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1,
                  "prefill_gemm requires per-tensor scalar scales");

  // Check for strides and alignment.
  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(c.stride(0) % 16 == 0 &&
                  b.stride(1) % 16 == 0);  // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    vllm::cutlass_fp8_prefill_gemm_sm100(c, a, b, a_scales, b_scales);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "cutlass_fp8_prefill_gemm_sm100 is only compiled for SM100 (Blackwell). "
      "CUDA device capability: ",
      version_num);
}

// AMMO track fp8_relu2_requant_epilogue_sm100.
// Dense FP8 GEMM with a fused ReLUSquared + static per-tensor requant-to-fp8
// epilogue. Operand contract mirrors cutlass_scaled_mm (A=[M,K] fp8 row-major,
// B=[K,N] fp8 col-major, scalar f32 a/b scales) but the OUTPUT is fp8 (e4m3),
// pre-scaled by the per-tensor scalar out_scale (= 1 / down_proj.input_scale).
// SM100-only; per-tensor scales; no bias. Caller (Python) gates this behind
// VLLM_NEMOTRON3_FP8_RELU2_EPILOGUE_SM100 and only routes the fused
// shared-expert up_proj here.
void cutlass_scaled_mm_relu2_fp8out_sm100(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale) {
  // Conformality (mirror cutlass_scaled_mm).
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1 &&
                      out_scale.numel() == 1,
                  "relu2_fp8out requires per-tensor scalar scales");

  // Strides and alignment.
  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(b.stride(1) % 16 == 0);                 // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous() &&
                  out_scale.is_contiguous());

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    vllm::cutlass_scaled_mm_relu2_fp8out_sm100(c, a, b, a_scales, b_scales,
                                               out_scale);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "cutlass_scaled_mm_relu2_fp8out_sm100 is only compiled for SM100 "
      "(Blackwell). CUDA device capability: ",
      version_num);
}

// AMMO track fp8_relu2_requant_epilogue_sm100 (attribution-by-ablation only):
// the same fused op MINUS the ReLUSquared node (dequant -> requant -> fp8).
// Gate-5.2 harness use only; NOT a production path.
void cutlass_scaled_mm_cast_fp8out_sm100(
    torch::stable::Tensor& c, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales,
    torch::stable::Tensor const& out_scale) {
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));
  STD_TORCH_CHECK(a_scales.numel() == 1 && b_scales.numel() == 1 &&
                      out_scale.numel() == 1,
                  "cast_fp8out requires per-tensor scalar scales");

  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(b.stride(1) % 16 == 0);                 // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous() &&
                  out_scale.is_contiguous());

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());
  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM100 && ENABLE_SCALED_MM_SM100
  if (version_num >= 100 && version_num < 120) {
    vllm::cutlass_scaled_mm_cast_fp8out_sm100(c, a, b, a_scales, b_scales,
                                              out_scale);
    return;
  }
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "cutlass_scaled_mm_cast_fp8out_sm100 is only compiled for SM100 "
      "(Blackwell). CUDA device capability: ",
      version_num);
}

void cutlass_moe_mm(torch::stable::Tensor& out_tensors,
                    torch::stable::Tensor const& a_tensors,
                    torch::stable::Tensor const& b_tensors,
                    torch::stable::Tensor const& a_scales,
                    torch::stable::Tensor const& b_scales,
                    torch::stable::Tensor const& expert_offsets,
                    torch::stable::Tensor const& problem_sizes,
                    torch::stable::Tensor const& a_strides,
                    torch::stable::Tensor const& b_strides,
                    torch::stable::Tensor const& c_strides, bool per_act_token,
                    bool per_out_ch) {
  int32_t version_num = get_sm_version_num();
#if defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100
  if (version_num >= 100 && version_num < 110) {
    cutlass_moe_mm_sm100(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                         expert_offsets, problem_sizes, a_strides, b_strides,
                         c_strides, per_act_token, per_out_ch);
    return;
  }
#endif
#if defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90
  if (version_num >= 90 && version_num < 100) {
    cutlass_moe_mm_sm90(out_tensors, a_tensors, b_tensors, a_scales, b_scales,
                        expert_offsets, problem_sizes, a_strides, b_strides,
                        c_strides, per_act_token, per_out_ch);
    return;
  }
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm for CUDA device capability: ", version_num,
      ". Required capability: 90 or 100");
}

void get_cutlass_moe_mm_data(
    const torch::stable::Tensor& topk_ids,
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    torch::stable::Tensor& input_permutation,
    torch::stable::Tensor& output_permutation, const int64_t num_experts,
    const int64_t n, const int64_t k,
    const std::optional<torch::stable::Tensor>& blockscale_offsets,
    const bool is_gated) {
  // This function currently gets compiled only if we have a valid cutlass moe
  // mm to run it for.
  int32_t version_num = get_sm_version_num();
#if (defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90) ||   \
    (defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100) || \
    (defined ENABLE_CUTLASS_MOE_SM120 && ENABLE_CUTLASS_MOE_SM120)
  get_cutlass_moe_mm_data_caller(topk_ids, expert_offsets, problem_sizes1,
                                 problem_sizes2, input_permutation,
                                 output_permutation, num_experts, n, k,
                                 blockscale_offsets, is_gated);
  return;
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_data: no cutlass_scaled_mm kernel for "
      "CUDA device capability: ",
      version_num, ". Required capability: 90, 100, or 120");
}

void get_cutlass_moe_mm_problem_sizes_from_expert_offsets(
    const torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2, const int64_t n, const int64_t k,
    const bool swap_ab) {
  int32_t version_num = get_sm_version_num();
#if (defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90) ||   \
    (defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100) || \
    (defined ENABLE_CUTLASS_MOE_SM120 && ENABLE_CUTLASS_MOE_SM120)
  get_cutlass_moe_mm_problem_sizes_from_expert_offsets_caller(
      expert_first_token_offset, problem_sizes1, problem_sizes2, n, k, swap_ab);
  return;
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_moe_mm_problem_sizes_from_expert_offsets: "
      "no cutlass_scaled_mm kernel for CUDA device capability: ",
      version_num, ". Required capability: 90, 100, or 120");
}

void get_cutlass_batched_moe_mm_data(
    torch::stable::Tensor& expert_offsets,
    torch::stable::Tensor& problem_sizes1,
    torch::stable::Tensor& problem_sizes2,
    const torch::stable::Tensor& expert_num_tokens,
    const int64_t num_local_experts, const int64_t padded_m, const int64_t n,
    const int64_t k) {
  // This function currently gets compiled only if we have a valid cutlass moe
  // mm to run it for.
  int32_t version_num = get_sm_version_num();
#if (defined ENABLE_CUTLASS_MOE_SM90 && ENABLE_CUTLASS_MOE_SM90) ||   \
    (defined ENABLE_CUTLASS_MOE_SM100 && ENABLE_CUTLASS_MOE_SM100) || \
    (defined ENABLE_CUTLASS_MOE_SM120 && ENABLE_CUTLASS_MOE_SM120)
  get_cutlass_batched_moe_mm_data_caller(expert_offsets, problem_sizes1,
                                         problem_sizes2, expert_num_tokens,
                                         num_local_experts, padded_m, n, k);
  return;
#endif
  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled get_cutlass_batched_moe_mm_data: no "
      "cutlass_scaled_mm kernel "
      "for CUDA device capability: ",
      version_num, ". Required capability: 90, 100, or 120");
}

void cutlass_scaled_mm_azp(torch::stable::Tensor& c,
                           torch::stable::Tensor const& a,
                           torch::stable::Tensor const& b,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b_scales,
                           torch::stable::Tensor const& azp_adj,
                           std::optional<torch::stable::Tensor> const& azp,
                           std::optional<torch::stable::Tensor> const& bias) {
  // Checks for conformality
  STD_TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && c.dim() == 2);
  STD_TORCH_CHECK(c.size(0) == a.size(0) && a.size(1) == b.size(0) &&
                  b.size(1) == c.size(1));
  STD_TORCH_CHECK(a_scales.numel() == 1 || a_scales.numel() == a.size(0));
  STD_TORCH_CHECK(b_scales.numel() == 1 || b_scales.numel() == b.size(1));

  // Check for strides and alignment
  STD_TORCH_CHECK(a.stride(1) == 1 && c.stride(1) == 1);  // Row-major
  STD_TORCH_CHECK(b.stride(0) == 1);                      // Column-major
  STD_TORCH_CHECK(c.stride(0) % 16 == 0 &&
                  b.stride(1) % 16 == 0);  // 16 Byte Alignment
  STD_TORCH_CHECK(a_scales.is_contiguous() && b_scales.is_contiguous());

  // bias, azp, azp_adj are all 1d
  // bias and azp_adj have n elements, azp has m elements
  if (bias) {
    STD_TORCH_CHECK(bias->numel() == b.size(1) && bias->is_contiguous());
  }
  if (azp) {
    STD_TORCH_CHECK(azp->numel() == a.size(0) && azp->is_contiguous());
  }
  STD_TORCH_CHECK(azp_adj.numel() == b.size(1) && azp_adj.is_contiguous());

  // azp & bias types
  STD_TORCH_CHECK(azp_adj.scalar_type() == torch::headeronly::ScalarType::Int);
  STD_TORCH_CHECK(!azp ||
                  azp->scalar_type() == torch::headeronly::ScalarType::Int);
  STD_TORCH_CHECK(!bias || bias->scalar_type() == c.scalar_type(),
                  "currently bias dtype must match output dtype ",
                  c.scalar_type());

  const torch::stable::accelerator::DeviceGuard device_guard(
      a.get_device_index());

  int32_t version_num = get_sm_version_num();

#if defined ENABLE_SCALED_MM_SM90 && ENABLE_SCALED_MM_SM90
  if (version_num >= 90) {
    cutlass_scaled_mm_azp_sm90(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }
#endif

#if defined ENABLE_SCALED_MM_C2X && ENABLE_SCALED_MM_C2X
  if (version_num == 89) {
    // Ada Lovelace
    cutlass_scaled_mm_azp_sm89(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  if (version_num >= 80) {
    // Ampere
    cutlass_scaled_mm_azp_sm80(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
    return;
  }

  // Turing
  STD_TORCH_CHECK(version_num >= 75);
  cutlass_scaled_mm_azp_sm75(c, a, b, a_scales, b_scales, azp_adj, azp, bias);
  return;
#endif

  STD_TORCH_CHECK_NOT_IMPLEMENTED(
      false,
      "No compiled cutlass_scaled_mm_azp for a compute capability less than "
      "CUDA device capability: ",
      version_num);
}
