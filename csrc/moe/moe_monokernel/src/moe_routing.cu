
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
 * @brief Warp-parallel top-k routing (E=128, TOP_K=8).
 *
 * Each warp processes one token at a time:
 * - Each lane loads 4 logits (128 / 32 = 4)
 * - Iteratively selects the global max 8 times using warp reductions
 * - Computes softmax over the selected 8 logits
 */
template <typename Dims>
__device__ static void top8_warp_parallel(
    const __nv_bfloat16* __restrict__ router_logits,
    uint32_t num_tokens,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    static_assert(Dims::TOP_K == 8, "This function is only for top-k=8 routing");
    static_assert(Dims::NUM_EXPERTS == 128, "This implementation assumes E=128.");

    using CoreDims = MoECoreDims<Dims>;

    constexpr uint32_t TOP_K = Dims::TOP_K;
    constexpr uint32_t VALUES_PER_LANE = Dims::NUM_EXPERTS / CoreDims::THREADS_PER_WARP;  // 4
    static_assert(VALUES_PER_LANE == 4, "Unexpected values per lane");

    const uint32_t lane = get_thread<Dims>();
    const uint32_t warp = get_any_warp<Dims>();

    // One warp per token, striding by total warps in the block (calc + prefetch).
    for (uint32_t tokidx = warp; tokidx < num_tokens;
         tokidx += CoreDims::TOTAL_WARP_COUNT) {
        float local_vals[VALUES_PER_LANE];
        uint32_t local_idxs[VALUES_PER_LANE];

        #pragma unroll
        for (uint32_t i = 0; i < VALUES_PER_LANE; ++i) {
            uint32_t expert = lane * VALUES_PER_LANE + i;
            local_idxs[i] = expert;
            local_vals[i] = (float)router_logits[tokidx * Dims::NUM_EXPERTS + expert];
        }

        // Lane 0 will write results for this token.
        float topk_vals[TOP_K];
        uint32_t topk_idxs[TOP_K];

        #pragma unroll
        for (uint32_t k = 0; k < TOP_K; ++k) {
            // Select the best remaining value in this lane (4 candidates).
            float best = local_vals[0];
            uint32_t best_idx = local_idxs[0];
            uint32_t best_pos = 0;

            #pragma unroll
            for (uint32_t i = 1; i < VALUES_PER_LANE; ++i) {
                float v = local_vals[i];
                if (v > best) {
                    best = v;
                    best_idx = local_idxs[i];
                    best_pos = i;
                }
            }

            // Warp reduce (max) to find the best expert among all 128.
            float warp_best = best;
            uint32_t warp_best_idx = best_idx;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xFFFFFFFF, warp_best, offset);
                uint32_t other_idx =
                    __shfl_down_sync(0xFFFFFFFF, warp_best_idx, offset);
                if (other > warp_best) {
                    warp_best = other;
                    warp_best_idx = other_idx;
                }
            }

            float global_best =
                __shfl_sync(0xFFFFFFFF, warp_best, 0 /*srcLane*/);
            uint32_t global_best_idx =
                __shfl_sync(0xFFFFFFFF, warp_best_idx, 0 /*srcLane*/);

            if (lane == 0) {
                topk_vals[k] = global_best;
                topk_idxs[k] = global_best_idx;
            }

            // Remove selected expert from the lane that owns it.
            #pragma unroll
            for (uint32_t i = 0; i < VALUES_PER_LANE; ++i) {
                if (local_idxs[i] == global_best_idx) {
                    local_vals[i] = -FLT_MAX;
                }
            }
        }

        if (lane == 0) {
            float weights[TOP_K];
            softmax_topk<Dims>(topk_vals, weights);

            uint32_t base_idx = tokidx * TOP_K;
            #pragma unroll
            for (uint32_t k = 0; k < TOP_K; ++k) {
                shmem->topk_ids[base_idx + k] =
                    static_cast<uint8_t>(topk_idxs[k]);
                shmem->topk_weights[base_idx + k] = weights[k];
            }
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
    top8_warp_parallel<Dims>(router_logits, num_tokens, shmem);
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
