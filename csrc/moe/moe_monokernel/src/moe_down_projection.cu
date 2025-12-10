
#pragma once
#ifndef MOE_DOWN_PROJECTION_CU
#define MOE_DOWN_PROJECTION_CU

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

/**
 * @brief Loads activation tile from scratchpad for down-projection.
 *
 * Loads T_TILE (8) activation rows from scratchpad to shared memory.
 * Uses 16-byte aligned copy128 operations for async memory transfer.
 *
 * @tparam Dims The dimension template
 * @param scratchpad Global memory scratchpad
 * @param shm Pointer to shared memory
 * @param pipeline Pipeline for async memory operations
 * @param tokens Array of pair indices
 * @param t_buffer_idx Buffer index for double buffering (0 or 1)
 * @param num_valid_tokens Number of valid tokens in this tile
 */
template <typename Dims>
__device__ static void moe_request_down_activations(
    const MoEGemmSpec<Dims>* __restrict__ scratchpad,
    MoE_SHM<Dims>* shm,
    cuda::pipeline<cuda::thread_scope_thread>& pipeline,
    const std::uint16_t* tokens,
    std::uint32_t t_buffer_idx,
    std::uint32_t num_valid_tokens)
{
    using CoreDims = MoECoreDims<Dims>;

    auto* t_load_dest = t_buffer_idx == 0 ? shm->u.gemm2.t_g0 : shm->u.gemm2.t_g1;

    // Each prefetch warp loads some rows
    std::uint32_t warp = get_prefetch_warp<Dims>();
    std::uint32_t lane = get_thread<Dims>();

    // Load T_TILE rows (8 rows for Qwen3, N=768 float elements each)
    // With 4 prefetch warps and T_TILE=8, each warp handles 2 rows
    for (std::uint32_t row = warp; row < CoreDims::T_TILE; row += CoreDims::PREFETCH_WARP_COUNT) {
        // Clamp to valid tokens to avoid OOB access
        std::uint32_t safe_row = min(row, num_valid_tokens > 0 ? num_valid_tokens - 1 : 0u);
        std::uint16_t pair_idx = tokens[safe_row];

        const T_element* src = &scratchpad->temp[pair_idx * Dims::N];
        T_element* dest = t_load_dest[row];

        // T_element is float (4 bytes), copy128 copies 16 bytes = 4 floats
        // N = 768 floats, 768 / 4 = 192 vectors per row
        // Each of 32 threads handles multiple vectors
        constexpr std::uint32_t n_vec = Dims::N / 4;  // 4 floats per 16-byte copy

        for (std::uint32_t vec = lane; vec < n_vec; vec += CoreDims::THREADS_PER_WARP) {
            std::uint32_t col = vec * 4;  // 4 floats = 16 bytes, ALWAYS 16-byte aligned
            copy128(dest[col], src[col], pipeline);
        }
    }
}

/**
 * @brief Loads down-projection weights for one expert.
 *
 * Loads W_DOWN_TILE (16) rows of weights from global memory to shared memory.
 * Each row has N elements (768 for Qwen3). Uses 16-byte aligned copy128 operations.
 *
 * For block quantization, also loads the N-block scales for the current K-block.
 *
 * @tparam Dims The dimension template
 * @param expert_weights_down Pointer to expert down-projection weights [E, K, N]
 * @param expert_scales_down Pointer to expert down-projection scales [E, K] or [E, K_blocks, N_blocks]
 * @param shm Pointer to shared memory
 * @param pipeline Pipeline for async memory operations
 * @param expert_id Expert ID to load
 * @param k_offset Offset into the K dimension (blockIdx.x * W_DOWN_TILE)
 * @param w_buffer_idx Buffer index for double buffering (0 or 1)
 */
template <typename Dims>
__device__ static void moe_request_down_expert(
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    MoE_SHM<Dims>* shm,
    cuda::pipeline<cuda::thread_scope_thread>& pipeline,
    std::uint32_t expert_id,
    std::uint32_t k_offset,
    std::uint32_t w_buffer_idx)
{
    using CoreDims = MoECoreDims<Dims>;

    auto* w_load_dest = w_buffer_idx == 0 ? shm->u.gemm2.w_g0 : shm->u.gemm2.w_g1;
    auto* scale_load_dest = w_buffer_idx == 0 ? shm->u.gemm2.scale_g0 : shm->u.gemm2.scale_g1;

    std::uint32_t warp = get_prefetch_warp<Dims>();
    std::uint32_t lane = get_thread<Dims>();

    // Load W_DOWN_TILE rows of weights using 16-byte aligned accesses
    // W_DOWN_TILE = HIDDEN_STATES / GRID_SIZE = 2048 / 128 = 16 for Qwen3
    // With 4 prefetch warps, each warp handles 4 rows (16 / 4 = 4)
    std::uint32_t rows_per_warp = CoreDims::W_DOWN_TILE / CoreDims::PREFETCH_WARP_COUNT;
    std::uint32_t row_start = warp * rows_per_warp;
    std::uint32_t row_end = row_start + rows_per_warp;

    for (std::uint32_t row = row_start; row < row_end && row < CoreDims::W_DOWN_TILE; ++row) {
        std::uint32_t global_row = k_offset + row;

        const W_element* src_row = expert_weights_down
            + (expert_id * Dims::HIDDEN_STATES + global_row) * Dims::N;
        W_element* dst_row = w_load_dest[row];

        // Load columns using 16-byte aligned copy128 operations
        // N = 768 for Qwen3, 768 / 16 = 48 vectors per row
        // Each thread in warp (32 threads) handles multiple vectors
        constexpr std::uint32_t n_vec = Dims::N / 16;

        for (std::uint32_t vec = lane; vec < n_vec; vec += CoreDims::THREADS_PER_WARP) {
            std::uint32_t col = vec * 16;  // ALWAYS 16-byte aligned
            copy128(dst_row[col], src_row[col], pipeline);
        }
    }

    // Load scales for this K tile
    // Per-tensor/channel: One scale per K row, W_DOWN_TILE = 16 scales
    // Block quant: scale[k_block][n_block], need to load all N-block scales for current K-block
    if constexpr (Dims::USE_BLOCK_QUANT) {
        // For block quantization, scale layout is [expert_id][k_block][n_block]
        // k_offset / W_DOWN_TILE gives which K-block we're processing
        // For Qwen3: W_DOWN_TILE=16, BLOCK_SIZE_QUANT=128, so each K-block spans 8 W_DOWN_TILEs
        std::uint32_t k_block = k_offset / Dims::BLOCK_SIZE_QUANT;

        // Load all N-block scales for this K-block
        if (warp == 0 && lane < Dims::DOWN_SCALE_N_BLOCKS) {
            scale_load_dest[lane] = expert_scales_down[
                expert_id * Dims::DOWN_SCALE_K_BLOCKS * Dims::DOWN_SCALE_N_BLOCKS
                + k_block * Dims::DOWN_SCALE_N_BLOCKS + lane];
        }
    } else {
        // Per-tensor/channel: one scale per K row
        if (warp == 0 && lane < CoreDims::W_DOWN_TILE) {
            std::uint32_t global_row = k_offset + lane;
            scale_load_dest[lane] = expert_scales_down[expert_id * Dims::HIDDEN_STATES + global_row];
        }
    }
}

/**
 * @brief Helper to compute far_row for edge cases in MMA layout.
 *
 * Handles the case where W_DOWN_TILE is not evenly divisible by 16.
 */
template <typename CoreDims>
__device__ static inline std::uint32_t far_row_static(std::uint32_t w_row)
{
    return (CoreDims::W_DOWN_TILE % 16 == 8 && w_row + 8 == CoreDims::W_DOWN_TILE)
        ? w_row : w_row + 8;
}

/**
 * @brief Computes GEMM2 (down-projection) using Tensor Core MMA instructions.
 *
 * This is the Tensor Core version following Llama4's moe_down_mult pattern.
 * Computes: partial_result[w_row, :] = sum_n(activation[:, n] * weight[w_row, n]) * scale[w_row]
 *
 * Uses mma_fp8_tf32() for FP8 weights × FP32 activations → FP32 output.
 *
 * Key differences from Llama4:
 * - Each thread processes activations from shm->t[t_index][thread/4] - single token row
 * - MMA accumulates partial products from all 6 calc warps
 * - Results stored in partial_result for later reduction
 *
 * For block quantization:
 * - Scale depends on both K-block (determined by blockIdx) and N-block (varies in MMA loop)
 * - Scales are pre-loaded per N-block, applied during MMA accumulation
 *
 * @tparam Dims The dimension template
 * @param shm Pointer to shared memory
 * @param buffer_idx Buffer index for double buffering (0 or 1)
 */
template <typename Dims>
__device__ static void moe_down_gemm_tile_tc(
    MoE_SHM<Dims>* shm,
    std::uint32_t buffer_idx)
{
    using CoreDims = MoECoreDims<Dims>;

    // Use buffer accessors for double buffering
    auto& t = shm->u.gemm2.t(buffer_idx);
    auto& w = shm->u.gemm2.w(buffer_idx);
    auto& scale = shm->u.gemm2.scale(buffer_idx);
    auto* partial_result = shm->u.gemm2.partial_result;  // Single partial result buffer

    const std::uint32_t thread = get_thread<Dims>();
    const std::uint32_t warp = get_calc_warp<Dims>();

    // thread / 4 determines which token row (0-7) this thread computes for
    const std::uint32_t t_row = thread / 4;

    // Iterate over weight rows in chunks of W_DOWN_MMA_TILE=16
    for (std::uint32_t w_row = 0; w_row < CoreDims::W_DOWN_TILE; w_row += CoreDims::W_DOWN_MMA_TILE) {
        // Initialize accumulators
        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

        // Handle edge case where w_row + 8 might be out of bounds
        std::uint32_t far_row = far_row_static<CoreDims>(w_row);

        // Iterate over N dimension in chunks of K_TILE=32
        // Each warp handles different starting offsets, cycling through N
        // BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE = 6 * 32 = 192
        for (std::uint32_t base_col = warp * CoreDims::K_TILE;
             base_col < Dims::N;
             base_col += CoreDims::BLOCK_STRIDE) {

            // Load FP8 weights: 4 bytes per thread, covering 16 weight rows × 32 N columns
            // thread / 4 selects row offset (0-7), thread % 4 selects column offset
            __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&w[w_row   + thread / 4][base_col + 4 * (thread % 4) +  0];
            __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&w[far_row + thread / 4][base_col + 4 * (thread % 4) +  0];
            __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&w[w_row   + thread / 4][base_col + 4 * (thread % 4) + 16];
            __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&w[far_row + thread / 4][base_col + 4 * (thread % 4) + 16];

            // Load FP32 activations: 16 bytes (4 floats) per thread
            // t_row (thread / 4) selects which of the 8 token rows to use
            float4 b0 = *(float4*)&t[t_row][base_col + 4 * (thread % 4) +  0];
            float4 b1 = *(float4*)&t[t_row][base_col + 4 * (thread % 4) + 16];

            // For block quantization: apply per-N-block scale during accumulation
            if constexpr (Dims::USE_BLOCK_QUANT) {
                // Compute MMA result into temporaries
                float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
                mma_fp8_tf32(t0, t1, t2, t3, w0, w1, w2, w3, b0, b1, 0.f, 0.f, 0.f, 0.f);

                // Get scale for this N-block
                // N-block index = base_col / BLOCK_SIZE_QUANT
                std::uint32_t n_block = base_col / Dims::BLOCK_SIZE_QUANT;
                float block_scale = scale[n_block];

                // Apply scale and accumulate
                d0 += t0 * block_scale;
                d1 += t1 * block_scale;
                d2 += t2 * block_scale;
                d3 += t3 * block_scale;
            } else {
                // Per-tensor/channel: accumulate without scaling here
                mma_fp8_tf32(d0, d1, d2, d3, w0, w1, w2, w3, b0, b1, d0, d1, d2, d3);
            }
        }

        // Apply weight scales for per-tensor/channel mode (scales already applied for block quant)
        if constexpr (!Dims::USE_BLOCK_QUANT) {
            float s0 = scale[w_row + thread / 4];
            float s1 = scale[far_row + thread / 4];

            d0 *= s0;
            d1 *= s0;  // d1 is same weight row as d0
            d2 *= s1;
            d3 *= s1;  // d3 is same weight row as d2
        }

        // Store partial results
        // Layout: partial_result[warp][thread + offset] (6 warps total)
        partial_result[warp][thread +  0] = d0;
        partial_result[warp][thread + 32] = d1;
        partial_result[warp][thread + 64] = d2;
        partial_result[warp][thread + 96] = d3;
    }
}

/**
 * @brief Reduces partial results and accumulates to output.
 *
 * This is a simplified version that correctly maps MMA outputs to output indices.
 * The MMA produces outputs for 8 token rows × 16 K columns per tile.
 *
 * @tparam Dims The dimension template
 * @param shm Pointer to shared memory
 * @param scratchpad Global memory scratchpad (contains output_accum)
 * @param tokens Array of pair indices (T_TILE pairs)
 * @param k_offset Offset into K dimension (blockIdx.x * W_DOWN_TILE)
 * @param num_valid_tokens Number of valid tokens in this tile
 */
template <typename Dims>
__device__ static void moe_down_accumulate_tc(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const std::uint16_t* tokens,
    std::uint32_t k_offset,
    std::uint32_t num_valid_tokens)
{
    using CoreDims = MoECoreDims<Dims>;

    // Only calc warps participate in reduction
    if (!is_calc_warp<Dims>()) {
        return;
    }

    auto* partial_result = shm->u.gemm2.partial_result;  // Use unified partial result buffer

    const std::uint32_t thread = get_thread<Dims>();
    const std::uint32_t warp = get_calc_warp<Dims>();

    // Warp 0 does the reduction and output
    //
    // MMA m16n8k8 OUTPUT layout - Llama4 interpretation:
    // The partial_result stores d0-d3 in linear order per thread.
    // Based on Llama4's moe_down_reduction pattern:
    //   - Token offset: (thread % 4) * 2 gives rows 0, 2, 4, 6 (row0)
    //                   (thread % 4) * 2 + 1 gives rows 1, 3, 5, 7 (row1)
    //   - K offset: (thread / 4) gives 0-7 within each MMA tile
    //
    // d0 → token_row0, k_idx = (thread/4)
    // d1 → token_row1, k_idx = (thread/4)
    // d2 → token_row0, k_idx = (thread/4) + 8
    // d3 → token_row1, k_idx = (thread/4) + 8
    if (warp == 0) {
        // Following Llama4's mapping:
        std::uint32_t token_row0 = (thread % 4) * 2;      // 0, 2, 4, 6
        std::uint32_t token_row1 = (thread % 4) * 2 + 1;  // 1, 3, 5, 7
        std::uint32_t k_local = thread / 4;               // 0-7 for K offset within tile

        // Check validity for both token rows
        bool row0_valid = (token_row0 < num_valid_tokens);
        bool row1_valid = (token_row1 < num_valid_tokens);

        // Get token info (pair_idx -> token_idx)
        std::uint16_t pair_idx0 = row0_valid ? tokens[token_row0] : tokens[0];
        std::uint16_t pair_idx1 = row1_valid ? tokens[token_row1] : tokens[0];
        std::uint16_t token_idx0 = pair_idx0 / Dims::TOP_K;
        std::uint16_t token_idx1 = pair_idx1 / Dims::TOP_K;

        // For W_DOWN_TILE=16, we have one iteration (w_row=0)
        for (std::uint32_t w_row = 0; w_row < CoreDims::W_DOWN_TILE; w_row += CoreDims::W_DOWN_MMA_TILE) {
            // Sum partial results from all calc warps (6 warps)
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

            for (std::uint32_t i = 0; i < CoreDims::CALC_WARP_COUNT; ++i) {
                d0 += partial_result[i][thread +  0];
                d1 += partial_result[i][thread + 32];
                d2 += partial_result[i][thread + 64];
                d3 += partial_result[i][thread + 96];
            }

            // K indices: base + w_row + local offset (0-7 for d0/d1, +8 for d2/d3)
            std::uint32_t k_idx0 = k_offset + w_row + k_local;      // d0: token_row0
            std::uint32_t k_idx1 = k_offset + w_row + k_local;      // d1: token_row1 (same K)
            std::uint32_t k_idx2 = k_offset + w_row + k_local + 8;  // d2: token_row0
            std::uint32_t k_idx3 = k_offset + w_row + k_local + 8;  // d3: token_row1 (same K)

            // Write outputs - d0/d2 go to token_row0, d1/d3 go to token_row1
            if (row0_valid) {
                if (k_idx0 < Dims::HIDDEN_STATES) {
                    atomicAdd(&scratchpad->output_accum[token_idx0 * Dims::HIDDEN_STATES + k_idx0], d0);
                }
                if (k_idx2 < Dims::HIDDEN_STATES) {
                    atomicAdd(&scratchpad->output_accum[token_idx0 * Dims::HIDDEN_STATES + k_idx2], d2);
                }
            }
            if (row1_valid) {
                if (k_idx1 < Dims::HIDDEN_STATES) {
                    atomicAdd(&scratchpad->output_accum[token_idx1 * Dims::HIDDEN_STATES + k_idx1], d1);
                }
                if (k_idx3 < Dims::HIDDEN_STATES) {
                    atomicAdd(&scratchpad->output_accum[token_idx1 * Dims::HIDDEN_STATES + k_idx3], d3);
                }
            }
        }
    }
}

/**
 * @brief Sequential expert-tile down-projection with Tensor Cores.
 *
 * Processes each expert's token-expert pairs sequentially using:
 * - Prefetch warps for async data loading via cp.async
 * - Compute warps for Tensor Core GEMM
 * - Single buffer (no double buffering for simplicity)
 *
 * @tparam Dims The dimension template
 * @param shm Pointer to shared memory
 * @param scratchpad Global memory scratchpad
 * @param expert_weights_down Down-projection weights [E, K, N]
 * @param expert_scales_down Down-projection scales [E, K] or [E, K_blocks, N_blocks]
 * @param k_offset K dimension offset for this block
 */
template <typename Dims>
__device__ static void moe_down_projection_sequential(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    std::uint32_t k_offset)
{
    using CoreDims = MoECoreDims<Dims>;

    // Note: We use __syncthreads() for block-wide synchronization instead of
    // cuda::pipeline because each thread has its own pipeline instance and
    // cross-warp synchronization requires thread_scope_block.

    // Process each expert sequentially
    for (std::uint32_t expert_ref_idx = 0; expert_ref_idx < shm->expert_count; ++expert_ref_idx) {
        ExpertRef& expert_ref = shm->experts[expert_ref_idx];
        std::uint32_t expert_id = expert_ref.id;
        std::uint32_t first_pair = expert_ref.first_token;
        std::uint32_t last_pair = expert_ref.last_token;

        // Process pairs in chunks of T_TILE
        for (std::uint32_t pair_offset = first_pair; pair_offset < last_pair; pair_offset += CoreDims::T_TILE) {
            const std::uint16_t* pair_indices = &shm->token_indexes[pair_offset];
            std::uint32_t num_valid_tokens = min((std::uint32_t)CoreDims::T_TILE, last_pair - pair_offset);

            // Prefetch warps load data using cp.async
            if (is_prefetch_warp<Dims>()) {
                cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
                pipeline.producer_acquire();
                moe_request_down_activations(scratchpad, shm, pipeline, pair_indices, 0, num_valid_tokens);
                moe_request_down_expert(expert_weights_down, expert_scales_down, shm, pipeline,
                                        expert_id, k_offset, 0);
                pipeline.producer_commit();
                // Wait for our own async copies to complete
                cuda::pipeline_consumer_wait_prior<0>(pipeline);
            }

            // Block-wide sync ensures all data is in shared memory
            __syncthreads();

            // Compute warps perform Tensor Core GEMM
            if (is_calc_warp<Dims>()) {
                moe_down_gemm_tile_tc<Dims>(shm, 0);
            }

            __syncthreads();

            // Accumulate to FP32 output buffer
            moe_down_accumulate_tc<Dims>(shm, scratchpad, pair_indices, k_offset, num_valid_tokens);

            __syncthreads();
        }
    }
}

/**
 * @brief Main down-projection function for top-k routing.
 *
 * Processes all token-expert pairs and accumulates results to output using
 * Tensor Core GEMM with shared memory tiling.
 *
 * The up-projection already computes gate * silu(x) * topk_weight and stores
 * it in scratchpad->temp, so no additional SiLU computation is needed here.
 *
 * @tparam Dims The dimension template
 * @param shm Pointer to shared memory
 * @param scratchpad Global memory scratchpad (contains up-projection results)
 * @param expert_weights_down Down-projection weights [E, K, N]
 * @param expert_scales_down Down-projection scales [E, K] or [E, K_blocks, N_blocks]
 * @param output_activations Output buffer [BS, K]
 */
template <typename Dims>
__device__ void moe_down_projection_topk(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ output_activations)
{
    using CoreDims = MoECoreDims<Dims>;

    // K offset for this block
    std::uint32_t k_offset = blockIdx.x * CoreDims::W_DOWN_TILE;

    // Use sequential Tensor Core down-projection
    moe_down_projection_sequential<Dims>(shm, scratchpad, expert_weights_down,
                                         expert_scales_down, k_offset);
}

} // namespace moe_monokernel

#endif
