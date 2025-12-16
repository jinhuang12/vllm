#pragma once

#include <torch/all.h>

void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices,
                  torch::Tensor& token_expert_indices,
                  torch::Tensor& gating_output, bool renormalize);

void moe_sum(torch::Tensor& input, torch::Tensor& output);

void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts,
                          int64_t block_size, torch::Tensor sorted_token_ids,
                          torch::Tensor experts_ids,
                          torch::Tensor num_tokens_post_pad);

void batched_moe_align_block_size(int64_t max_tokens_per_batch,
                                  int64_t block_size,
                                  torch::Tensor const& expert_num_tokens,
                                  torch::Tensor sorted_ids,
                                  torch::Tensor expert_ids,
                                  torch::Tensor num_tokens_post_pad);

void moe_lora_align_block_size(
    torch::Tensor topk_ids, torch::Tensor token_lora_mapping,
    int64_t num_experts, int64_t block_size, int64_t max_loras,
    int64_t max_num_tokens_padded, int64_t max_num_m_blocks,
    torch::Tensor sorted_token_ids, torch::Tensor expert_ids,
    torch::Tensor num_tokens_post_pad, torch::Tensor adapter_enabled,
    torch::Tensor lora_ids);
#ifndef USE_ROCM
torch::Tensor moe_wna16_gemm(torch::Tensor input, torch::Tensor output,
                             torch::Tensor b_qweight, torch::Tensor b_scales,
                             std::optional<torch::Tensor> b_qzeros,
                             std::optional<torch::Tensor> topk_weights,
                             torch::Tensor sorted_token_ids,
                             torch::Tensor expert_ids,
                             torch::Tensor num_tokens_post_pad, int64_t top_k,
                             int64_t BLOCK_SIZE_M, int64_t BLOCK_SIZE_N,
                             int64_t BLOCK_SIZE_K, int64_t bit);

std::tuple<torch::Tensor, torch::Tensor> grouped_topk(
    torch::Tensor const& scores, int64_t n_group, int64_t topk_group,
    int64_t topk, bool renormalize, double routed_scaling_factor,
    torch::Tensor const& bias, int64_t scoring_func);
#endif

bool moe_permute_unpermute_supported();

void shuffle_rows(const torch::Tensor& input_tensor,
                  const torch::Tensor& dst2src_map,
                  torch::Tensor& output_tensor);

// MoE Monokernel for gpt-oss-120b (BF16, TP=4, L40S)
namespace moe_monokernel_gpt_oss_120b {
void moe_monokernel_gpt_oss_120b_BS8_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_weights_down,
    torch::Tensor& activations_out,
    torch::Tensor& scratchpad);

void moe_monokernel_gpt_oss_120b_BS32_impl(
    const torch::Tensor& activations_in,
    const torch::Tensor& router_logits,
    const torch::Tensor& expert_weights_up,
    const torch::Tensor& expert_weights_down,
    torch::Tensor& activations_out,
    torch::Tensor& scratchpad);

int64_t moe_monokernel_gpt_oss_120b_scratchpad_size_BS8();
int64_t moe_monokernel_gpt_oss_120b_scratchpad_size_BS32();
bool moe_monokernel_gpt_oss_120b_supported();
}