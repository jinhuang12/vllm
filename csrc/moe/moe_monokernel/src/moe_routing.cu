
#pragma once
#ifndef MOE_ROUTING_CU
#define MOE_ROUTING_CU

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>
#include <cfloat>
#include <cuda_bf16.h>

#include "moe_internal.h"

namespace moe_monokernel {

/**
 * @brief Computes the sigmoid activation function for a given input.
 *
 * @param x The input value.
 * @return The sigmoid of the input value.
 */
__device__ static inline float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * @brief Computes softmax over TOP_K values and stores normalized weights
 *
 * @param values Array of TOP_K values
 * @param weights Output array for normalized weights
 */
template <typename Dims>
__device__ static inline void softmax_topk(float* values, float* weights)
{
    constexpr uint32_t TOP_K = Dims::TOP_K;

    // Find max for numerical stability
    float max_val = values[0];
    #pragma unroll
    for (uint32_t i = 1; i < TOP_K; ++i) {
        max_val = fmaxf(max_val, values[i]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    #pragma unroll
    for (uint32_t i = 0; i < TOP_K; ++i) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    #pragma unroll
    for (uint32_t i = 0; i < TOP_K; ++i) {
        weights[i] = values[i] * inv_sum;
    }
}

/**
 * @brief Insert a value into a sorted top-k array (descending order)
 *
 * Uses insertion sort to maintain the top-k values.
 * Returns true if the value was inserted.
 */
template <uint32_t TOP_K>
__device__ static inline bool insert_topk(
    float* topk_values,
    uint32_t* topk_indices,
    float value,
    uint32_t index)
{
    // Find insertion position (array is sorted descending)
    if (value <= topk_values[TOP_K - 1]) {
        return false;  // Value not in top-k
    }

    // Find correct position
    uint32_t pos = TOP_K - 1;
    while (pos > 0 && value > topk_values[pos - 1]) {
        topk_values[pos] = topk_values[pos - 1];
        topk_indices[pos] = topk_indices[pos - 1];
        pos--;
    }

    topk_values[pos] = value;
    topk_indices[pos] = index;
    return true;
}

/**
 * @brief Selects the top-8 experts for each of up to 64 tokens based on router logits.
 *
 * This device function processes a batch of up to 64 tokens, each with up to 128 experts.
 * It takes as input a pointer to the router logits (in bfloat16 format) and determines,
 * for each token, the 8 experts with the highest routing scores.
 *
 * The routing weights are normalized using softmax over the top-8 values.
 *
 * The selection is done on each CUDA block redundantly such that the result can be placed in shared memory.
 *
 * @param router_logits Pointer to the input router logits array of shape [num_tokens, experts] in row-major order.
 *                      Individual elements are in __nv_bfloat16 format.
 * @param num_tokens Number of tokens
 * @param shmem Shared Memory struct to store the result to.
 */
template <typename Dims>
__device__ static void top8_BS64(const __nv_bfloat16 *__restrict__ router_logits,
                uint32_t num_tokens,
                MoE_SHM<Dims>* shmem)
{
    static_assert(Dims::TOP_K == 8, "This function is only for top-k=8 routing");
    static_assert(Dims::BS <= 64, "Dispatch to incorrect implementation");
    static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX, "Batch size or number of experts too high for uint32 indices.");

    constexpr uint32_t TOP_K = 8;
    uint32_t thread_idx = threadIdx.x;

    // Each thread processes exactly one token (1:1 mapping)
    // Only threads with thread_idx < num_tokens do work
    if (thread_idx < num_tokens) {
        uint32_t tokidx = thread_idx;

        // Initialize top-k tracking arrays (in registers)
        float topk_values[TOP_K];
        uint32_t topk_indices[TOP_K];

        #pragma unroll
        for (uint32_t i = 0; i < TOP_K; ++i) {
            topk_values[i] = -FLT_MAX;
            topk_indices[i] = 0;
        }

        // Scan all experts for this token
        for (uint32_t idx = 0; idx < Dims::NUM_EXPERTS; idx++) {
            uint32_t index = tokidx * Dims::NUM_EXPERTS + idx;
            float value = (float)router_logits[index];

            // Insert into top-k if value is large enough
            insert_topk<TOP_K>(topk_values, topk_indices, value, idx);
        }

        // Compute softmax over top-8 values
        float weights[TOP_K];
        softmax_topk<Dims>(topk_values, weights);

        // Store results for this token
        // Layout: [token_0_expert_0, token_0_expert_1, ..., token_0_expert_7, token_1_expert_0, ...]
        uint32_t base_idx = tokidx * TOP_K;

        #pragma unroll
        for (uint32_t k = 0; k < TOP_K; ++k) {
            shmem->topk_ids[base_idx + k] = (uint8_t)topk_indices[k];
            shmem->topk_weights[base_idx + k] = weights[k];
        }
    }
}

/**
 * @brief Dispatch function for top-8 routing
 *
 * Only calc warps (threads 0-255) should execute this function.
 * Prefetch warps (threads 256-383) return early to avoid assertions.
 */
template <typename Dims>
__device__ __forceinline__ void topk_route(const __nv_bfloat16 *__restrict__ router_logits,
                uint32_t num_tokens,
                MoE_SHM<Dims>* shmem)
{
    static_assert(Dims::TOP_K == 8, "This implementation is for top-k=8");

    // Only calc warps (threads 0-255) should execute routing
    // Prefetch warps would trigger assertion in get_calc_warp()
    if (!is_calc_warp<Dims>()) {
        return;
    }

    // Always use BS64 variant - it works with any CALC_WARP_COUNT
    // The BS8 variant assumes 8 calc warps (warp-per-token), which doesn't work
    // when benchmarking with fewer calc warps (e.g., 4c4p config)
    top8_BS64(router_logits, num_tokens, shmem);
}

// Store total number of pairs for this batch
template <typename Dims>
__device__ __forceinline__ void set_total_pairs(
    uint32_t num_tokens,
    MoE_SHM<Dims>* shmem)
{
    if (threadIdx.x == 0) {
        shmem->total_pairs = num_tokens * Dims::TOP_K;
    }
}

} // namespace moe_monokernel

#endif
