
#pragma once
#ifndef MOE_UP_PROJECTION_CU
#define MOE_UP_PROJECTION_CU

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include <cstdint>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "moe_internal.h"
#include "ptx_utils.h"

namespace moe_monokernel {

///////////////////////////////////////////////////////////////////////////////
//
// Design Considerations (adapted from Llama4 monokernel for Qwen3)
//
// Qwen3 dimensions: K=2048 (hidden), N=768 (intermediate), TOP_K=8
//
// * MMA m16n8k32 computes W[16,K] × A^T[K,8] = C[16,8]
//   - 8 tokens (A_TILE) per iteration
//   - 16 weight rows (W_UP_TILE) split: first 8 rows for x, next 8 for gate
//   - Accumulated over K dimension (2048) in chunks of 32
//
// * Each block handles W_UP_TILE/2 = 8 rows of N dimension
//   - With GRID_SIZE=128 blocks × 8 rows = 1024 rows covered
//   - But N=768, so some blocks will be idle or handle edge cases
//
// * For 2*N=1536 total rows (x + gate), need ~192 blocks
//   - Each block iteration: 8 rows, so 192 iterations per pair chunk
//
// * All 8 calc warps collaborate on same MMA, then warp 0 reduces
//
///////////////////////////////////////////////////////////////////////////////

/**
 * @brief Asynchronously loads activation rows for one chunk of tokens.
 *
 * Loads A_TILE (8) activation rows, each with HIDDEN_STATES/2 columns,
 * into shared memory with 32-byte swizzling for bank conflict avoidance.
 *
 * @tparam Dims The dimension template for the kernel
 * @param source Global memory pointer to quantized activations [BS, HIDDEN_STATES]
 * @param token_indexes Array mapping positions to original token indices
 * @param dest Shared memory destination [A_TILE][K_DIM_HALF_PADDED_A]
 * @param max_count Maximum valid tokens to load
 * @param pipe Pipeline for async memory operations
 */
template <typename Dims>
__device__ static void moe_request_input_tokens(
    const AQ_element* __restrict__ source,
    const std::uint16_t* __restrict__ token_indexes,
    AQ_element (&dest)[MoECoreDims<Dims>::A_TILE][MoECoreDims<Dims>::K_DIM_HALF_PADDED_A],
    std::uint32_t max_count,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    using CoreDims = MoECoreDims<Dims>;
    const unsigned thread = get_thread<Dims>();
    const unsigned warp = get_prefetch_warp<Dims>();

    // async transfers are 16 bytes / thread
    const unsigned chunk_size = 16 / sizeof(*source);

    for (unsigned row = warp; row < CoreDims::A_TILE; row += CoreDims::PREFETCH_WARP_COUNT) {
        if (row < max_count) {
            // Get the pair index, then extract original token index
            std::uint16_t pair_idx = token_indexes[row];
            std::uint16_t token_idx = pair_idx / Dims::TOP_K;

            const AQ_element* a = source + token_idx * Dims::HIDDEN_STATES;
            for (unsigned col = thread * chunk_size; col < Dims::HIDDEN_STATES / 2; col += CoreDims::THREADS_PER_WARP * chunk_size) {
                unsigned dest_col = rotate_col_32(col, row);
                copy128(dest[row][dest_col], a[col], pipe);
            }
        }
    }
}

/**
 * @brief Asynchronously loads up-projection weights for one expert.
 *
 * Loads W_UP_TILE (16) weight rows from global memory. The rows are loaded
 * in two halves (for x and gate) with 32-byte swizzling matching Llama4 pattern.
 *
 * @tparam Dims The dimension template for the kernel
 * @tparam CopyCols Number of columns to copy (HIDDEN_STATES/2 for half-K tiling)
 * @param source Pointer to expert weights [NUM_EXPERTS, 2*N, HIDDEN_STATES]
 * @param id Expert index
 * @param dest Shared memory destination [W_UP_TILE][K_DIM_HALF_PADDED_W]
 * @param pipe Pipeline for async memory operations
 */
template <typename Dims, std::size_t CopyCols, std::size_t Rows, std::size_t Cols>
__device__ static void moe_request_up_expert(
    const W_element* __restrict__ source,
    std::uint32_t id,
    W_element (&dest)[Rows][Cols],
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    static_assert(CopyCols <= Cols);

    using CoreDims = MoECoreDims<Dims>;
    const unsigned thread = get_thread<Dims>();
    const unsigned warp = get_prefetch_warp<Dims>();
    const unsigned chunk_size = 16 / sizeof(*source);

    // Starting row for this block (each block handles W_UP_TILE/2 rows of x and gate)
    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

    // Columns to copy per iteration
    const unsigned item_cols_per_iteration = CoreDims::THREADS_PER_WARP * chunk_size;

    // Each row must be multiple of copy size
    static_assert(Dims::HIDDEN_STATES % (32 * 16) == 0);

    const OpaqueElement* weights = (const OpaqueElement*)(source + id * 2 * Dims::N * Dims::HIDDEN_STATES);

    // Even warps fetch 8 rows from lower N (x), odd warps fetch 8 rows from upper N (gate)
    #pragma unroll 1
    for (unsigned row = warp / 2; row < CoreDims::W_UP_TILE / 2; row += CoreDims::PREFETCH_WARP_COUNT / 2) {
        for (unsigned col = 0; col < CopyCols; col += item_cols_per_iteration) {
            unsigned source_col = thread;
            unsigned is_upper = warp & 1;

            // Skip if this block's base_row is beyond N (edge case for Qwen3 N=768)
            if (base_row + row >= Dims::N) continue;

            copy128(dest[row + is_upper * CoreDims::W_UP_TILE / 2][rotate_col_32(col + source_col * chunk_size, row)],
                    weights[((row + base_row + is_upper * Dims::N) * Dims::HIDDEN_STATES + col + source_col * chunk_size) / sizeof(OpaqueElement)],
                    pipe);
        }
    }
}

/**
 * @brief Pre-loads block scales for up-projection into shared memory.
 *
 * For block quantization, scale depends on both row-block AND K-block.
 * This function pre-loads all K-block scales for the current row-block
 * into shared memory before the MMA loop begins.
 *
 * @tparam Dims The dimension template
 * @param expert_scales_up Expert scales pointer
 * @param expert_id Current expert ID
 * @param base_row Base row for this block (N dimension)
 * @param shm Shared memory Gemm1Data
 */
template <typename Dims>
__device__ static void moe_preload_up_block_scales(
    const S_element* __restrict__ expert_scales_up,
    std::uint32_t expert_id,
    std::uint32_t base_row,
    typename MoE_SHM<Dims>::U::Gemm1Data* __restrict__ shm)
{
    if constexpr (!Dims::USE_BLOCK_QUANT) {
        return;  // No-op for per-tensor/channel quantization
    }

    using CoreDims = MoECoreDims<Dims>;
    const unsigned thread = get_thread<Dims>();
    const unsigned warp = get_prefetch_warp<Dims>();

    // Only prefetch warps load scales
    if (!is_prefetch_warp<Dims>()) return;

    // Calculate row-blocks for x and gate
    // x uses rows [0, N), gate uses rows [N, 2*N)
    const std::uint32_t row_block_up = base_row / Dims::BLOCK_SIZE_QUANT;
    const std::uint32_t row_block_gate = (base_row + Dims::N) / Dims::BLOCK_SIZE_QUANT;

    // Scale layout: [expert_id][row_block][k_block]
    const S_element* scales_base = expert_scales_up +
        expert_id * Dims::UP_SCALE_ROW_BLOCKS * Dims::UP_SCALE_K_BLOCKS;

    // Load all K-block scales for up (x) row-block
    for (std::uint32_t k_block = thread + warp * CoreDims::THREADS_PER_WARP;
         k_block < Dims::UP_SCALE_K_BLOCKS;
         k_block += CoreDims::PREFETCH_WARP_COUNT * CoreDims::THREADS_PER_WARP) {
        if (k_block < Dims::UP_SCALE_K_BLOCKS) {
            shm->block_scales_up[k_block] = scales_base[row_block_up * Dims::UP_SCALE_K_BLOCKS + k_block];
            shm->block_scales_gate[k_block] = scales_base[row_block_gate * Dims::UP_SCALE_K_BLOCKS + k_block];
        }
    }
}

/**
 * @brief Gets the appropriate scale for a K-column in block quantization mode.
 *
 * @tparam Dims The dimension template
 * @param shm Shared memory Gemm1Data with pre-loaded block scales
 * @param k_col Current K column being processed
 * @param is_gate Whether this is for gate (upper N rows) or x (lower N rows)
 * @return The scale to apply
 */
template <typename Dims>
__device__ static __forceinline__ float moe_get_up_block_scale(
    const typename MoE_SHM<Dims>::U::Gemm1Data* __restrict__ shm,
    std::uint32_t k_col,
    bool is_gate)
{
    if constexpr (Dims::USE_BLOCK_QUANT) {
        std::uint32_t k_block = k_col / Dims::BLOCK_SIZE_QUANT;
        return is_gate ? shm->block_scales_gate[k_block] : shm->block_scales_up[k_block];
    } else {
        return 1.0f;  // Unused in per-tensor mode
    }
}

/**
 * @brief Performs MMA result reduction and sigmoid step of up-projection.
 *
 * Sums partial scalar products from all calc warps, applies scales, computes
 * sigmoid (SiLU), and stores results to global memory.
 *
 * Key insight from Llama4: The MMA computes W[16,K] × A^T[K,8] = C[16,8]
 * where rows 0-7 of W are for 'x' and rows 8-15 are for 'gate'.
 * Only warp 0 does the final reduction.
 *
 * Output layout: Uses original pair indices from token_indexes to maintain
 * compatibility with scalar down-projection which expects temp[pair_idx * N + n].
 *
 * @tparam Dims The dimension template
 * @param partial_result Array of MMA results from all warps [WARPS, 4 * THREADS]
 * @param d0-d3 This warp's MMA outputs
 * @param ws0, ws1 Weight scales for x and gate parts
 * @param ts0, ts1 Token scales for two consecutive rows
 * @param store_row0, store_row1 Whether to store each row
 * @param pair_idx0, pair_idx1 Original pair indices (from token_indexes)
 * @param result Output pointer to global temp buffer (spec->temp)
 */
template <typename Dims, std::size_t Rows, std::size_t Cols>
__device__ static void moe_up_reduction(
    const float (&partial_result)[Rows][Cols],
    float d0, float d1, float d2, float d3,
    float ws0, float ws1,
    float ts0, float ts1,
    bool store_row0,
    bool store_row1,
    std::uint16_t pair_idx0,
    std::uint16_t pair_idx1,
    T_element* __restrict__ result)
{
    using CoreDims = MoECoreDims<Dims>;
    const unsigned thread = get_thread<Dims>();

    // Starting column for this block (each block handles W_UP_TILE/2 = 8 N-columns)
    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

    // Skip if this block handles columns beyond N
    if (base_row >= Dims::N) return;

    // Combine results from all calc warps
    for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
        d0 += partial_result[i][thread + 0];
        d1 += partial_result[i][thread + 32];
        d2 += partial_result[i][thread + 64];
        d3 += partial_result[i][thread + 96];
    }

#ifdef MOE_MONOKERNEL_DEBUG_OUTPUT
    // Debug: print first thread's values to understand MMA output
    if (blockIdx.x == 0 && thread == 0) {
        printf("DEBUG moe_up_reduction: d0=%.4f d1=%.4f d2=%.4f d3=%.4f ws0=%.4f ws1=%.4f ts0=%.4f ts1=%.4f pair0=%d pair1=%d\n",
               d0, d1, d2, d3, ws0, ws1, ts0, ts1, (int)pair_idx0, (int)pair_idx1);
    }
#endif

    // Apply weight scales (NOT topk_weight yet!)
    // x (d0, d1): rows 0-7 of weight matrix
    // gate (d2, d3): rows 8-15 of weight matrix
    // For TOP_K > 1, topk_weight must be applied AFTER SiLU, not before!
    // Llama4 (TOP_K=1) applies ts before SiLU because ts is sigmoid(logit), but
    // Qwen3 (TOP_K=8) uses softmax-normalized weights that should multiply the final result.
    //
    // For block quantization: scales were already applied per-K-tile during MMA accumulation,
    // so ws0/ws1 are not used (set to 1.0 effectively by using d0/d1 directly).
    float x0 = Dims::USE_BLOCK_QUANT ? d0 : (d0 * ws0);
    float x1 = Dims::USE_BLOCK_QUANT ? d1 : (d1 * ws0);
    float gate0 = Dims::USE_BLOCK_QUANT ? d2 : (d2 * ws1);
    float gate1 = Dims::USE_BLOCK_QUANT ? d3 : (d3 * ws1);

    // SiLU activation: gate * (x * sigmoid(x)) = gate * x / (1 + exp(-x))
    float sig0 = (gate0 * x0) / (1 + expf(-x0));
    float sig1 = (gate1 * x1) / (1 + expf(-x1));

    // NOW apply topk_weight (routing weight) - only once!
    sig0 *= ts0;
    sig1 *= ts1;

    // Write to temporary buffer using original pair indices
    // This matches the scalar down-projection's expectation of temp[pair_idx * N + n]
    unsigned out_col = base_row + (thread / 4);
    if (out_col < Dims::N) {
        if (store_row0) {
            result[pair_idx0 * Dims::N + out_col] = sig0;
#ifdef MOE_MONOKERNEL_DEBUG_OUTPUT
            if (blockIdx.x == 0 && thread == 0) {
                printf("DEBUG output: pair=%d col=%d sig0=%.4f x0=%.4f gate0=%.4f\n",
                       (int)pair_idx0, out_col, sig0, x0, gate0);
            }
#endif
        }
        if (store_row1) {
            result[pair_idx1 * Dims::N + out_col] = sig1;
        }
    }
}

/**
 * @brief Standard kernel for up-projection ("GEMM1"), combined with SiLU reduction.
 *
 * Adapted from Llama4's moe_up_projection_normal for Qwen3 dimensions with TOP_K=8.
 *
 * Processing strategy:
 * - Each block handles W_UP_TILE/2 = 8 columns of the N dimension
 * - For each expert, process tokens in chunks of A_TILE=8
 * - All calc warps collaborate on K dimension, warp 0 does final reduction
 * - Triple buffering for weight and activation prefetching
 *
 * @tparam Dims The dimension template
 * @param expert_weights_up Up-projection weights [NUM_EXPERTS, 2*N, HIDDEN_STATES]
 * @param expert_scales_up Up-projection scales [NUM_EXPERTS, 2*N]
 * @param spec Global memory scratchpad with quantized activations
 * @param shmem Shared memory with routing data and local scratch
 */
template <typename Dims>
__device__ void moe_up_projection(
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    MoEGemmSpec<Dims>* __restrict__ spec,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    using CoreDims = MoECoreDims<Dims>;
    using MoE_SHM = MoE_SHM<Dims>;

    const unsigned thread = get_thread<Dims>();
    const unsigned warp = get_any_warp<Dims>();

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    // Starting column for this block (handles 8 columns of N each iteration)
    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;

    // Early exit if this block's columns are beyond N
    if (base_row >= Dims::N) return;

    typename MoE_SHM::U::Gemm1Data* shm = &shmem->u.gemm1;
    std::uint32_t expert_count = shmem->expert_count;
    const AQ_element* activations = spec->activations[0];

    // Triple-buffering: queue first 2 tiles
    if (is_prefetch_warp<Dims>() && expert_count > 0) {
        const ExpertRef& expert = shmem->experts[0];

        // First half of K
        pipe.producer_acquire();
        moe_request_input_tokens<Dims>(
            activations,
            &shmem->token_indexes[expert.first_token],
            shm->a[0],
            expert.last_token - expert.first_token,
            pipe);
        moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
            expert_weights_up,
            expert.id,
            shm->w[0],
            pipe);
        pipe.producer_commit();

        // Second half of K
        pipe.producer_acquire();
        moe_request_input_tokens<Dims>(
            activations + Dims::HIDDEN_STATES / 2,
            &shmem->token_indexes[expert.first_token],
            shm->a[1],
            expert.last_token - expert.first_token,
            pipe);
        moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
            expert_weights_up + Dims::HIDDEN_STATES / 2,
            expert.id,
            shm->w[1],
            pipe);
        pipe.producer_commit();
    }

    // Triple-buffer indices
    std::uint32_t t_index_read = 0;
    std::uint32_t t_index_write = 2;
    std::uint32_t w_index_read = 0;
    std::uint32_t w_index_write = 2;

    // Process all experts
    for (std::uint32_t e = 0; e < expert_count; ++e) {
        const ExpertRef& expert = shmem->experts[e];
        std::uint32_t id = expert.id;
        unsigned int a_rows = expert.last_token - expert.first_token;

        // Weight scales for this block's columns (base_row and base_row + N for gate)
        // For per-tensor/channel: load scales directly
        // For block quantization: pre-load K-block scales to shared memory
        float ws0, ws1;
        if constexpr (Dims::USE_BLOCK_QUANT) {
            // Pre-load block scales for this expert
            moe_preload_up_block_scales<Dims>(expert_scales_up, id, base_row, shm);
            __syncthreads();  // Ensure scales are loaded before calc warps use them
            // ws0/ws1 will be loaded per K-tile inside MMA loop
            ws0 = 0.0f;  // Placeholder, actual scaling done per-K-tile
            ws1 = 0.0f;
        } else {
            const S_element* scales = expert_scales_up + id * 2 * Dims::N;
            ws0 = (base_row + thread / 4 < Dims::N) ?
                        scales[base_row + thread / 4] : 0.0f;
            ws1 = (base_row + thread / 4 < Dims::N) ?
                        scales[base_row + thread / 4 + Dims::N] : 0.0f;
        }

        // Process tokens in chunks of A_TILE=8
        for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::A_TILE) {
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

            // Wait for first K-half tile
            cuda::pipeline_consumer_wait_prior<1>(pipe);
            __syncthreads();

            if (is_prefetch_warp<Dims>()) {
                // Request next tiles
                pipe.producer_acquire();
                if (e + 1 < expert_count && a_row == 0) {
                    moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
                        expert_weights_up,
                        shmem->experts[e + 1].id,
                        shm->w[w_index_write],
                        pipe);
                    w_index_write = w_index_write == 2 ? 0 : w_index_write + 1;
                }
                if (a_row + CoreDims::A_TILE < a_rows) {
                    moe_request_input_tokens<Dims>(
                        activations,
                        &shmem->token_indexes[expert.first_token + CoreDims::A_TILE + a_row],
                        shm->a[t_index_write],
                        a_rows - CoreDims::A_TILE - a_row,
                        pipe);
                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
                } else if (e + 1 < expert_count) {
                    const ExpertRef& next_expert = shmem->experts[e + 1];
                    moe_request_input_tokens<Dims>(
                        activations,
                        &shmem->token_indexes[next_expert.first_token],
                        shm->a[t_index_write],
                        next_expert.last_token - next_expert.first_token,
                        pipe);
                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
                }
                pipe.producer_commit();
            } else {
                // Compute first half of K dimension
                for (unsigned base_col = warp * CoreDims::K_TILE;
                     base_col < Dims::HIDDEN_STATES / 2;
                     base_col += CoreDims::BLOCK_STRIDE) {

                    unsigned row = thread / 4;
                    unsigned col = 4 * (thread % 4);

                    __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col + 0, row)];
                    __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col + 0, row)];
                    __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col + 16, row)];
                    __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col + 16, row)];

                    // Clamp activation row for MMA (16 rows needed, only 8 available)
                    unsigned a_row_clamped = min(row, (unsigned)(CoreDims::A_TILE - 1));
                    __nv_fp8x4_e4m3 a02 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][a_row_clamped][rotate_col_32(base_col + col + 0, row)]);
                    __nv_fp8x4_e4m3 a13 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][a_row_clamped][rotate_col_32(base_col + col + 16, row)]);

                    // For block quantization: apply per-K-block scale during accumulation
                    // The MMA computes partial products that need to be scaled by the
                    // block scale for this K-range before being accumulated
                    if constexpr (Dims::USE_BLOCK_QUANT) {
                        // Compute MMA result into temporaries
                        float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
                        mma_fp8_fp8(t0, t1, t2, t3, w0, w1, w2, w3, a02, a13, 0.f, 0.f, 0.f, 0.f);

                        // Get scale for this K-block (first half of K: columns 0 to K/2-1)
                        float block_scale_up = moe_get_up_block_scale<Dims>(shm, base_col, false);
                        float block_scale_gate = moe_get_up_block_scale<Dims>(shm, base_col, true);

                        // Apply scale and accumulate
                        // t0, t1 correspond to rows 0-7 (up/x), t2, t3 to rows 8-15 (gate)
                        d0 += t0 * block_scale_up;
                        d1 += t1 * block_scale_up;
                        d2 += t2 * block_scale_gate;
                        d3 += t3 * block_scale_gate;
                    } else {
                        mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
                    }
                }
            }

            __syncthreads();
            w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
            t_index_read = t_index_read == 2 ? 0 : t_index_read + 1;

            // Wait for second K-half tile
            cuda::pipeline_consumer_wait_prior<1>(pipe);
            __syncthreads();

            if (is_prefetch_warp<Dims>()) {
                pipe.producer_acquire();
                if (a_row + CoreDims::A_TILE < a_rows) {
                    moe_request_input_tokens<Dims>(
                        activations + Dims::HIDDEN_STATES / 2,
                        &shmem->token_indexes[expert.first_token + CoreDims::A_TILE + a_row],
                        shm->a[t_index_write],
                        a_rows - CoreDims::A_TILE - a_row,
                        pipe);
                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
                } else if (e + 1 < expert_count) {
                    const ExpertRef& next_expert = shmem->experts[e + 1];
                    moe_request_input_tokens<Dims>(
                        activations + Dims::HIDDEN_STATES / 2,
                        &shmem->token_indexes[next_expert.first_token],
                        shm->a[t_index_write],
                        next_expert.last_token - next_expert.first_token,
                        pipe);
                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;

                    moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
                        expert_weights_up + Dims::HIDDEN_STATES / 2,
                        next_expert.id,
                        shm->w[w_index_write],
                        pipe);
                    w_index_write = w_index_write == 2 ? 0 : w_index_write + 1;
                }
                pipe.producer_commit();
            } else {
                // Compute second half of K dimension
                for (unsigned base_col = warp * CoreDims::K_TILE;
                     base_col < Dims::HIDDEN_STATES / 2;
                     base_col += CoreDims::BLOCK_STRIDE) {

                    unsigned row = thread / 4;
                    unsigned col = 4 * (thread % 4);

                    __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col + 0, row)];
                    __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col + 0, row)];
                    __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col + 16, row)];
                    __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col + 16, row)];

                    unsigned a_row_clamped = min(row, (unsigned)(CoreDims::A_TILE - 1));
                    __nv_fp8x4_e4m3 a02 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][a_row_clamped][rotate_col_32(base_col + col + 0, row)]);
                    __nv_fp8x4_e4m3 a13 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][a_row_clamped][rotate_col_32(base_col + col + 16, row)]);

                    // For block quantization: apply per-K-block scale during accumulation
                    if constexpr (Dims::USE_BLOCK_QUANT) {
                        float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
                        mma_fp8_fp8(t0, t1, t2, t3, w0, w1, w2, w3, a02, a13, 0.f, 0.f, 0.f, 0.f);

                        // Second half of K: actual K-column is base_col + HIDDEN_STATES/2
                        std::uint32_t actual_k_col = base_col + Dims::HIDDEN_STATES / 2;
                        float block_scale_up = moe_get_up_block_scale<Dims>(shm, actual_k_col, false);
                        float block_scale_gate = moe_get_up_block_scale<Dims>(shm, actual_k_col, true);

                        d0 += t0 * block_scale_up;
                        d1 += t1 * block_scale_up;
                        d2 += t2 * block_scale_gate;
                        d3 += t3 * block_scale_gate;
                    } else {
                        mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
                    }
                }

                // Store partial results
                shm->partial_result[warp][thread + 0] = d0;
                shm->partial_result[warp][thread + 32] = d1;
                shm->partial_result[warp][thread + 64] = d2;
                shm->partial_result[warp][thread + 96] = d3;

#ifdef MOE_MONOKERNEL_DEBUG_OUTPUT
                // Debug: print partial results after MMA
                if (blockIdx.x == 0 && threadIdx.x == 0 && warp == 0) {
                    printf("DEBUG MMA warp0: d0=%.4f d1=%.4f d2=%.4f d3=%.4f a_row=%d\n",
                           d0, d1, d2, d3, a_row);
                }
#endif
            }

            __syncthreads();
            w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
            t_index_read = t_index_read == 2 ? 0 : t_index_read + 1;

            if (a_row + CoreDims::A_TILE < a_rows) {
                w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
            }

            // Warp 0 does final reduction and writes output
            if (warp == 0) {
                std::uint32_t local_row0 = (thread % 4) * 2 + 0;
                std::uint32_t local_row1 = (thread % 4) * 2 + 1;
                std::uint32_t row0 = a_row + local_row0;
                std::uint32_t row1 = a_row + local_row1;
                const std::uint16_t* token_indexes = &shmem->token_indexes[expert.first_token];

                // Get original pair indices from token_indexes
                // These are the indices the scalar down-projection will use
                std::uint16_t pair_idx0 = token_indexes[row0 < a_rows ? row0 : 0];
                std::uint16_t pair_idx1 = token_indexes[row1 < a_rows ? row1 : 0];

                // Get token scales (topk_weights) using original pair indices
                float ts0 = shmem->topk_weights[pair_idx0];
                float ts1 = shmem->topk_weights[pair_idx1];

                moe_up_reduction<Dims>(
                    shm->partial_result,
                    d0, d1, d2, d3,
                    ws0, ws1,
                    ts0, ts1,
                    row0 < a_rows,
                    row1 < a_rows,
                    pair_idx0,
                    pair_idx1,
                    spec->temp);
            }
        }
    }
}

} // namespace moe_monokernel

#endif
