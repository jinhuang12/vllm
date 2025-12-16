
#pragma once
#ifndef MOE_PREPARE_CU
#define MOE_PREPARE_CU

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>

#include "moe_internal.h"

#ifndef __SIZEOF_INT128__
static_assert(false, "This module currently needs int128. Your host compiler does not support it.")
#endif

#define FULL_MASK 0xFFFFFFFFU

namespace moe_monokernel {

// We use an uint128 to store 16 uint8
typedef __uint128_t uint8x16_t;

/**
 * @brief 16-byte allreduce summation within a warp
 *
 * Sums a 16-byte integer across all threads and returns the sum.
 * This operation is collective and needs to be called by all threads within a warp.
 */
__device__ static inline uint8x16_t allreduce_sum_across_warp(uint8x16_t val)
{
    uint64_t val_lo = val & 0xFFFFFFFFFFFFFFFFU;
    uint64_t val_hi = val >> 64;
    for (int offset = 16; offset > 0; offset /= 2) {
        val_lo += __shfl_xor_sync(FULL_MASK, val_lo, offset, 32);
        val_hi += __shfl_xor_sync(FULL_MASK, val_hi, offset, 32);
    }
    return val_lo | ((uint8x16_t) val_hi << 64);
}

/**
 * @brief Prefix sum of all uint8s in an uint8x16_t
 *
 * Computes a prefix sum over 16 uint8. First element is the least significant byte.
 */
__device__ static inline uint8x16_t prefix_sum_over_bytes(uint8x16_t val)
{
    val += val << 8;
    val += val << 16;
    val += val << 32;
    val += val << 64;
    return val;
}

/**
 * @brief Prepares the MoE computation for top-k=8 routing.
 *
 * For top-k=8, each token has 8 expert assignments. This function:
 * 1. Counts how many times each expert is used across all token-expert pairs
 * 2. Builds the expert reference list (first_token, last_token for each used expert)
 * 3. Sorts token-expert pairs by expert ID into token_indexes
 *
 * The layout is:
 * - topk_ids[BS * TOP_K]: expert ID for each token-expert pair
 * - token_indexes[BS * TOP_K]: maps sorted position to original (token, k) pair index
 *
 * @param batch_size The batch size (number of tokens)
 * @param shm Pointer to shared memory struct
 */
template <typename Dims>
__device__ static void prepare_moe_topk(
    std::uint32_t batch_size,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    using CoreDims = MoECoreDims<Dims>;
    using MoE_SHM = MoE_SHM<Dims>;

    static_assert(Dims::TOP_K == 8, "This function is for top-k=8 routing");

    typename MoE_SHM::U::SortData* shm = &shmem->u.sorting;

    auto& counters = shm->counters;
    auto& total_counts = shm->total_counts;

    const uint32_t total_pairs = batch_size * Dims::TOP_K;

    // Fast path for small batches: for tiny `total_pairs`, the full radix-style
    // per-expert prefix scan is overkill. Use shared-memory atomics instead.
    if (total_pairs <= 32) {
        if (threadIdx.x < CoreDims::THREADS_PER_WARP) {
            const std::uint32_t lane = threadIdx.x;

            // Initialize per-expert counts.
            for (unsigned e = lane; e < Dims::NUM_EXPERTS; e += CoreDims::THREADS_PER_WARP) {
                total_counts[e] = 0;
            }
            __syncwarp();

            // Count pairs per expert.
            for (unsigned i = lane; i < total_pairs; i += CoreDims::THREADS_PER_WARP) {
                uint8_t expert_id = shmem->topk_ids[i];
                if (expert_id < Dims::NUM_EXPERTS) {
                    atomicAdd(&total_counts[expert_id], 1u);
                }
            }
            __syncwarp();

            // Build expert ranges (sorted by expert_id) and initialize per-expert write cursors.
            if (lane == 0) {
                std::uint32_t sum = 0;
                std::uint32_t expert_count = 0;
                for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
                    std::uint32_t count = total_counts[e];
                    if (count > 0) {
                        shmem->experts[expert_count].first_token = sum;
                        shmem->experts[expert_count].last_token = sum + count;
                        shmem->experts[expert_count].id = e;
                        expert_count++;

                        counters[e][0] = sum;  // cursor
                        sum += count;
                    }
                }
                shmem->expert_count = expert_count;
                shmem->total_pairs = total_pairs;
            }
            __syncwarp();

            // Write sorted token-expert pair indices (order within expert is unspecified).
            std::uint16_t* ordered = shmem->token_indexes;
            for (unsigned i = lane; i < total_pairs; i += CoreDims::THREADS_PER_WARP) {
                uint8_t expert_id = shmem->topk_ids[i];
                if (expert_id < Dims::NUM_EXPERTS) {
                    unsigned index = atomicAdd(&counters[expert_id][0], 1u);
                    ordered[index] = (std::uint16_t)i;
                }
            }
        }
        return;
    }

    // Implements a Radix sort on the first warp of each CUDA block.
    if (threadIdx.x < CoreDims::THREADS_PER_WARP) {
        // initialize counters
        for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
            counters[e][threadIdx.x] = 0;
        }

        // count token-expert pairs per expert
        // Each thread counts a subset of pairs
        for (unsigned i = threadIdx.x; i < total_pairs; i += CoreDims::THREADS_PER_WARP) {
            uint8_t expert_id = shmem->topk_ids[i];
            if (expert_id < Dims::NUM_EXPERTS) {  // Skip padding (0xFF)
                counters[expert_id][threadIdx.x]++;
            }
        }

        __syncwarp();

        // sum up counts per expert. counters become offsets
        for (unsigned e = threadIdx.x; e < Dims::NUM_EXPERTS; e += CoreDims::THREADS_PER_WARP) {
            std::uint32_t sum = 0;
            for (unsigned i = 0; i < CoreDims::THREADS_PER_WARP; ++i) {
                std::uint32_t prior = sum;
                sum += counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP];
                counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP] = prior;
            }
            total_counts[e] = sum;
        }

        __syncwarp();

        // Compute global offsets and build expert ranges
        if (threadIdx.x == 0) {
            std::uint32_t sum = 0;
            std::uint32_t expert_count = 0;

            for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
                std::uint32_t local_count = total_counts[e];

                if (local_count > 0) {
                    std::uint32_t prior = sum;
                    total_counts[e] = prior;
                    sum += local_count;

                    shmem->experts[expert_count].first_token = prior;
                    shmem->experts[expert_count].last_token = sum;
                    shmem->experts[expert_count].id = e;
                    expert_count++;
                }
            }

            shmem->expert_count = expert_count;
            shmem->total_pairs = total_pairs;
        }

        __syncwarp();

        // Write sorted token-expert pair indexes
        // token_indexes[i] = original pair index that should be at sorted position i
        std::uint16_t* ordered = shmem->token_indexes;
        for (unsigned i = threadIdx.x; i < total_pairs; i += CoreDims::THREADS_PER_WARP) {
            uint8_t expert_id = shmem->topk_ids[i];
            if (expert_id < Dims::NUM_EXPERTS) {
                unsigned offset = counters[expert_id][threadIdx.x];
                unsigned index = total_counts[expert_id] + offset;
                counters[expert_id][threadIdx.x] = offset + 1;

                ordered[index] = (std::uint16_t)i;
            }
        }
    }
}

/**
 * @brief Dispatch function for prepare
 */
template <typename Dims>
__device__ __forceinline__ void prepare_moe(
    std::uint32_t batch_size,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    static_assert(Dims::TOP_K == 8, "This implementation is for top-k=8");

    // Always use the full prepare function for top-k routing
    // since we need to handle BS * TOP_K pairs
    prepare_moe_topk<Dims>(batch_size, shmem);
}

} // namespace moe_monokernel

#endif
