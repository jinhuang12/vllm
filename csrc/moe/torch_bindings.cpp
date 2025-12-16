#include "core/registration.h"
#include "moe_ops.h"

#ifndef USE_ROCM
#include <ATen/cuda/CUDAContext.h>
#include "moe_monokernel/moe_monokernel_wrapper.h"

// ============================================================================
// MoE Monokernel for Qwen3-Coder-30B-A3B-Instruct-FP8 (block quantization)
// ============================================================================
// These functions support 128x128 block-scaled FP8 quantization.
// Scale tensor shapes are 3D: w13_scale [E, 12, 16], w2_scale [E, 16, 6]

bool moe_monokernel_supported()
{
    return moe_monokernel::check_device_supported();
}

void moe_monokernel_qwen3_block_quant_bs8_impl(
    torch::Tensor& activations,
    torch::Tensor& router_logits,
    torch::Tensor& w13,
    torch::Tensor& w13_scale,
    torch::Tensor& w2,
    torch::Tensor& w2_scale,
    torch::Tensor& output,
    torch::Tensor& scratchpad)
{
    TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be a CUDA tensor");
    TORCH_CHECK(w13.is_cuda(), "w13 must be a CUDA tensor");
    TORCH_CHECK(w13_scale.is_cuda(), "w13_scale must be a CUDA tensor");
    TORCH_CHECK(w13_scale.dim() == 3, "w13_scale must be 3D for block quantization [E, row_blocks, k_blocks]");
    TORCH_CHECK(w2.is_cuda(), "w2 must be a CUDA tensor");
    TORCH_CHECK(w2_scale.is_cuda(), "w2_scale must be a CUDA tensor");
    TORCH_CHECK(w2_scale.dim() == 3, "w2_scale must be 3D for block quantization [E, k_blocks, n_blocks]");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(scratchpad.is_cuda(), "scratchpad must be a CUDA tensor");

    uint32_t batch_size = static_cast<uint32_t>(activations.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    moe_monokernel::launch_moe_monokernel_qwen3_block_quant_bs8(
        activations.data_ptr(),
        batch_size,
        router_logits.data_ptr(),
        w13.data_ptr(),
        w13_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        output.data_ptr(),
        scratchpad.data_ptr(),
        stream);
}

void moe_monokernel_qwen3_block_quant_bs64_impl(
    torch::Tensor& activations,
    torch::Tensor& router_logits,
    torch::Tensor& w13,
    torch::Tensor& w13_scale,
    torch::Tensor& w2,
    torch::Tensor& w2_scale,
    torch::Tensor& output,
    torch::Tensor& scratchpad)
{
    TORCH_CHECK(activations.is_cuda(), "activations must be a CUDA tensor");
    TORCH_CHECK(router_logits.is_cuda(), "router_logits must be a CUDA tensor");
    TORCH_CHECK(w13.is_cuda(), "w13 must be a CUDA tensor");
    TORCH_CHECK(w13_scale.is_cuda(), "w13_scale must be a CUDA tensor");
    TORCH_CHECK(w13_scale.dim() == 3, "w13_scale must be 3D for block quantization [E, row_blocks, k_blocks]");
    TORCH_CHECK(w2.is_cuda(), "w2 must be a CUDA tensor");
    TORCH_CHECK(w2_scale.is_cuda(), "w2_scale must be a CUDA tensor");
    TORCH_CHECK(w2_scale.dim() == 3, "w2_scale must be 3D for block quantization [E, k_blocks, n_blocks]");
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(scratchpad.is_cuda(), "scratchpad must be a CUDA tensor");

    uint32_t batch_size = static_cast<uint32_t>(activations.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    moe_monokernel::launch_moe_monokernel_qwen3_block_quant_bs64(
        activations.data_ptr(),
        batch_size,
        router_logits.data_ptr(),
        w13.data_ptr(),
        w13_scale.data_ptr(),
        w2.data_ptr(),
        w2_scale.data_ptr(),
        output.data_ptr(),
        scratchpad.data_ptr(),
        stream);
}

int64_t moe_monokernel_qwen3_block_quant_scratchpad_size(int64_t batch_size)
{
    if (batch_size <= 8) {
        return moe_monokernel::get_scratchpad_size_block_quant_bs8();
    } else {
        return moe_monokernel::get_scratchpad_size_block_quant_bs64();
    }
}

torch::Tensor moe_monokernel_block_quant_get_timing(torch::Tensor& scratchpad, int64_t batch_size)
{
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU);
    torch::Tensor timing = torch::empty({10}, options);

    if (batch_size <= 8) {
        moe_monokernel::get_monokernel_timing_block_quant_bs8(
            scratchpad.data_ptr(),
            timing.data_ptr<int64_t>());
    } else {
        moe_monokernel::get_monokernel_timing_block_quant_bs64(
            scratchpad.data_ptr(),
            timing.data_ptr<int64_t>());
    }

    return timing;
}
#endif

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
  // Apply topk softmax to the gating outputs.
  m.def(
      "topk_softmax(Tensor! topk_weights, Tensor! topk_indices, Tensor! "
      "token_expert_indices, Tensor gating_output, bool renormalize) -> ()");
  m.impl("topk_softmax", torch::kCUDA, &topk_softmax);

  // Calculate the result of moe by summing up the partial results
  // from all selected experts.
  m.def("moe_sum(Tensor input, Tensor! output) -> ()");
  m.impl("moe_sum", torch::kCUDA, &moe_sum);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_align_block_size(Tensor topk_ids, int num_experts,"
      "                     int block_size, Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size, but for the batched case.
  m.def(
      "batched_moe_align_block_size(int max_tokens_per_batch,"
      "                     int block_size, Tensor expert_num_tokens,"
      "                     Tensor! sorted_token_ids,"
      "                     Tensor! experts_ids,"
      "                     Tensor! num_tokens_post_pad) -> ()");
  m.impl("batched_moe_align_block_size", torch::kCUDA,
         &batched_moe_align_block_size);

  // Aligning the number of tokens to be processed by each expert such
  // that it is divisible by the block size.
  m.def(
      "moe_lora_align_block_size(Tensor topk_ids,"
      "                     Tensor token_lora_mapping,"
      "                     int num_experts,"
      "                     int block_size, int max_loras, "
      "                     int max_num_tokens_padded, "
      "                     int max_num_m_blocks, "
      "                     Tensor !sorted_token_ids,"
      "                     Tensor !experts_ids,"
      "                     Tensor !num_tokens_post_pad,"
      "                     Tensor !adapter_enabled,"
      "                     Tensor !lora_ids) -> () ");
  m.impl("moe_lora_align_block_size", torch::kCUDA, &moe_lora_align_block_size);

#ifndef USE_ROCM
  m.def(
      "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
      "Tensor b_scales, Tensor? b_qzeros, "
      "Tensor? topk_weights, Tensor sorted_token_ids, "
      "Tensor expert_ids, Tensor num_tokens_post_pad, "
      "int top_k, int BLOCK_SIZE_M, int BLOCK_SIZE_N, int BLOCK_SIZE_K, "
      "int bit) -> Tensor");

  m.impl("moe_wna16_gemm", torch::kCUDA, &moe_wna16_gemm);

  m.def(
      "moe_wna16_marlin_gemm(Tensor! a, Tensor? c_or_none,"
      "Tensor! b_q_weight, Tensor? b_bias_or_none,"
      "Tensor! b_scales, Tensor? global_scale, Tensor? "
      "b_zeros_or_none,"
      "Tensor? g_idx_or_none, Tensor? perm_or_none, Tensor! workspace,"
      "Tensor sorted_token_ids,"
      "Tensor! expert_ids, Tensor! num_tokens_past_padded,"
      "Tensor! topk_weights, int moe_block_size, int top_k, "
      "bool mul_topk_weights, bool is_ep, int b_q_type_id,"
      "int size_m, int size_n, int size_k,"
      "bool is_full_k, bool use_atomic_add,"
      "bool use_fp32_reduce, bool is_zp_float) -> Tensor");
  m.def(
      "marlin_gemm_moe(Tensor! a, Tensor! b_q_weights, Tensor! sorted_ids, "
      "Tensor! topk_weights, Tensor! topk_ids, Tensor! b_scales, Tensor! "
      "b_zeros, Tensor! g_idx, Tensor! perm, Tensor! workspace, "
      "int b_q_type, SymInt size_m, "
      "SymInt size_n, SymInt size_k, bool is_k_full, int num_experts, int "
      "topk, "
      "int moe_block_size, bool replicate_input, bool apply_weights)"
      " -> Tensor");

  m.def(
      "moe_permute(Tensor input, Tensor topk_ids,"
      "Tensor token_expert_indices, Tensor? expert_map, int n_expert,"
      "int n_local_expert,"
      "int topk, int? align_block_size,Tensor! permuted_input, Tensor! "
      "expert_first_token_offset, Tensor! inv_permuted_idx, Tensor! "
      "permuted_idx, Tensor! m_indices)->()");

  m.def(
      "moe_unpermute(Tensor permuted_hidden_states, Tensor topk_weights,"
      "Tensor inv_permuted_idx, Tensor? expert_first_token_offset, "
      "int topk, Tensor! hidden_states)->()");

  m.def("moe_permute_unpermute_supported() -> bool");
  m.impl("moe_permute_unpermute_supported", &moe_permute_unpermute_supported);

  // Row shuffle for MoE
  m.def(
      "shuffle_rows(Tensor input_tensor, Tensor dst2src_map, Tensor! "
      "output_tensor) -> ()");
  m.impl("shuffle_rows", torch::kCUDA, &shuffle_rows);

  // Apply grouped topk routing to select experts.
  m.def(
      "grouped_topk(Tensor scores, int n_group, int "
      "topk_group, int topk, bool renormalize, float "
      "routed_scaling_factor, Tensor bias, int scoring_func) -> (Tensor, "
      "Tensor)");
  m.impl("grouped_topk", torch::kCUDA, &grouped_topk);

  // Check if MoE monokernel is supported
  m.def("moe_monokernel_supported() -> bool");
  m.impl("moe_monokernel_supported", &moe_monokernel_supported);

  // ============================================================================
  // Block Quantization Support (Qwen3-Coder-30B-A3B-Instruct-FP8)
  // ============================================================================
  // These ops support 128x128 block-scaled FP8 quantization.
  // Scale tensor shapes are 3D: w13_scale [E, 12, 16], w2_scale [E, 16, 6]

  // MoE Monokernel with block quant for Qwen3 (BS <= 8)
  m.def(
      "moe_monokernel_qwen3_block_quant_bs8(Tensor! activations, Tensor! router_logits, "
      "Tensor! w13, Tensor! w13_scale, Tensor! w2, Tensor! w2_scale, "
      "Tensor! output, Tensor! scratchpad) -> ()");
  m.impl("moe_monokernel_qwen3_block_quant_bs8", torch::kCUDA, &moe_monokernel_qwen3_block_quant_bs8_impl);

  // MoE Monokernel with block quant for Qwen3 (BS <= 64)
  m.def(
      "moe_monokernel_qwen3_block_quant_bs64(Tensor! activations, Tensor! router_logits, "
      "Tensor! w13, Tensor! w13_scale, Tensor! w2, Tensor! w2_scale, "
      "Tensor! output, Tensor! scratchpad) -> ()");
  m.impl("moe_monokernel_qwen3_block_quant_bs64", torch::kCUDA, &moe_monokernel_qwen3_block_quant_bs64_impl);

  // Get scratchpad size for block quant MoE monokernel
  m.def("moe_monokernel_qwen3_block_quant_scratchpad_size(int batch_size) -> int");
  m.impl("moe_monokernel_qwen3_block_quant_scratchpad_size", &moe_monokernel_qwen3_block_quant_scratchpad_size);

  // Get per-stage timing data from block quant monokernel (for profiling)
  m.def("moe_monokernel_block_quant_get_timing(Tensor! scratchpad, int batch_size) -> Tensor");
  m.impl("moe_monokernel_block_quant_get_timing", &moe_monokernel_block_quant_get_timing);

  // ============================================================================
  // MoE Monokernel for gpt-oss-120b (BF16, TP=4, L40S)
  // ============================================================================
  // Native BF16 monokernel for openai/gpt-oss-120b with TP=4 on L40S
  // No quantization, simpler than FP8 but larger SMEM footprint

  // Check if gpt-oss-120b monokernel is supported
  m.def("moe_monokernel_gpt_oss_120b_supported() -> bool");
  m.impl("moe_monokernel_gpt_oss_120b_supported", &moe_monokernel_gpt_oss_120b::moe_monokernel_gpt_oss_120b_supported);

  // MoE Monokernel for gpt-oss-120b (BS <= 8)
  m.def(
      "moe_monokernel_gpt_oss_120b_BS8(Tensor activations_in, Tensor router_logits, "
      "Tensor expert_weights_up, Tensor expert_weights_down, "
      "Tensor! activations_out, Tensor! scratchpad) -> ()");
  m.impl("moe_monokernel_gpt_oss_120b_BS8", torch::kCUDA,
         &moe_monokernel_gpt_oss_120b::moe_monokernel_gpt_oss_120b_BS8_impl);

  // MoE Monokernel for gpt-oss-120b (BS <= 32)
  m.def(
      "moe_monokernel_gpt_oss_120b_BS32(Tensor activations_in, Tensor router_logits, "
      "Tensor expert_weights_up, Tensor expert_weights_down, "
      "Tensor! activations_out, Tensor! scratchpad) -> ()");
  m.impl("moe_monokernel_gpt_oss_120b_BS32", torch::kCUDA,
         &moe_monokernel_gpt_oss_120b::moe_monokernel_gpt_oss_120b_BS32_impl);

  // Get scratchpad size for gpt-oss-120b monokernel (BS=8)
  m.def("moe_monokernel_gpt_oss_120b_scratchpad_size_BS8() -> int");
  m.impl("moe_monokernel_gpt_oss_120b_scratchpad_size_BS8",
         &moe_monokernel_gpt_oss_120b::moe_monokernel_gpt_oss_120b_scratchpad_size_BS8);

  // Get scratchpad size for gpt-oss-120b monokernel (BS=32)
  m.def("moe_monokernel_gpt_oss_120b_scratchpad_size_BS32() -> int");
  m.impl("moe_monokernel_gpt_oss_120b_scratchpad_size_BS32",
         &moe_monokernel_gpt_oss_120b::moe_monokernel_gpt_oss_120b_scratchpad_size_BS32);
#endif
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
