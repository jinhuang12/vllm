#ifndef MOE_INTERFACE_H
#define MOE_INTERFACE_H

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>

namespace moe_monokernel {

/**
 * @brief Template struct for MoE dimensions.
 *
 * @tparam m Maximum batch size (number of tokens)
 * @tparam n Intermediate size (after TP split)
 * @tparam k Hidden size
 * @tparam num_experts Number of experts
 * @tparam topk Number of experts per token (new for Qwen3)
 * @tparam block_size Block size for block quantization (0 = per-tensor/per-channel)
 */
template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts,
          uint32_t topk = 1, uint32_t block_size = 0>
struct MoEDimensions {
    static constexpr uint32_t HIDDEN_STATES = k;
    static constexpr uint32_t K = k;
    static constexpr uint32_t N = n;
    static constexpr uint32_t BS = m;
    static constexpr uint32_t M = m;
    static constexpr uint32_t NUM_EXPERTS = num_experts;
    static constexpr uint32_t TOP_K = topk;

    // Block quantization parameters
    // When block_size=0, use per-tensor/per-channel scales (existing behavior)
    // When block_size>0 (e.g., 128), use 2D block scales
    static constexpr uint32_t BLOCK_SIZE_QUANT = block_size;
    static constexpr bool USE_BLOCK_QUANT = (block_size > 0);

    // Scale grid dimensions for block quantization
    // Up-projection (W13): weight shape [E, 2*N, K], scale shape [E, row_blocks, k_blocks]
    static constexpr uint32_t UP_SCALE_ROW_BLOCKS = USE_BLOCK_QUANT ?
        ((2 * N + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT) : (2 * N);  // 12 for Qwen3-30B
    static constexpr uint32_t UP_SCALE_K_BLOCKS = USE_BLOCK_QUANT ?
        ((K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT) : 1;            // 16 for Qwen3-30B

    // Down-projection (W2): weight shape [E, K, N], scale shape [E, k_blocks, n_blocks]
    static constexpr uint32_t DOWN_SCALE_K_BLOCKS = USE_BLOCK_QUANT ?
        ((K + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT) : K;            // 16 for Qwen3-30B
    static constexpr uint32_t DOWN_SCALE_N_BLOCKS = USE_BLOCK_QUANT ?
        ((N + BLOCK_SIZE_QUANT - 1) / BLOCK_SIZE_QUANT) : 1;            // 6 for Qwen3-30B

    struct KernelConfig {
        // Grid size: one block per 16 rows of the down-projection result
        // Down-projection outputs K elements per token, so we tile by K/16
        // For Qwen3: HIDDEN_STATES=2048, so GRID_SIZE = 2048/16 = 128
        // For Llama4: HIDDEN_STATES=5120, but uses 2*N/16 since 2*1024/16 = 128
        static constexpr std::uint32_t GRID_SIZE = k / 16;
        // Block size: 6 calc + 2 prefetch = 256 threads (optimal for L40S)
        static constexpr std::uint32_t BLOCK_SIZE = 256;
    };
};

// ============================================================================
// Pre-defined dimensions for Qwen3-Coder-30B-A3B-Instruct-FP8 (block quantization)
// ============================================================================
// Model config: hidden_size=2048, moe_intermediate_size=768, num_experts=128, top_k=8
// Block quantization: weight_block_size=[128, 128]
// Up-projection scale shape: [E, 12, 16] = [128, 12, 16]
// Down-projection scale shape: [E, 16, 6] = [128, 16, 6]
// Note: W13 shape is [E, 2*N, K] = [128, 1536, 2048]
// Note: W2 shape is [E, K, N] = [128, 2048, 768]
using Dims_Qwen3_BS8_E128_TP1_BQ128 = MoEDimensions<8, 768, 2048, 128, 8, 128>;
using Dims_Qwen3_BS64_E128_TP1_BQ128 = MoEDimensions<64, 768, 2048, 128, 8, 128>;

// ============================================================================
// Element types
// ============================================================================
using W_element = __nv_fp8_e4m3;    // expert weights (FP8)
using A_element = __nv_bfloat16;    // activations input (BF16)
using AQ_element = __nv_fp8_e4m3;   // activations after quantization (FP8)
using S_element = float;            // scaling factors (FP32)
using R_element = __nv_bfloat16;    // MoE output (BF16)

/**
 * @brief Returns the maximum amount of shared memory necessary to run moe_kernel()
 */
constexpr size_t get_moe_max_shmem_size();

/**
 * @brief Returns the maximum amount of global scratchpad memory to run moe_kernel()
 */
constexpr size_t get_moe_max_scratchpad_size();

/**
 * @brief W8A8 MoE kernel with top-k routing
 *
 * This function implements a W8A8 Mixture-of-Experts kernel.
 * It routes each input token to the top-k experts as determined by the @p routing_logits.
 *
 * For Qwen3: top_k=8, meaning each token is processed by 8 experts and results are combined.
 *
 * Inputs:
 * - Activations are provided as bfloat16. They are quantized to FP8 E4M3 before the matrix multiplies.
 * - Expert weights are provided as FP8 E4M3.
 *
 * This is a cooperative kernel. It needs to be launched via cudaLaunchCooperativeKernel().
 * This kernel needs at least get_moe_max_shmem_size() shared memory.
 *
 * In the parameter descriptions, we use the following shorthand constants:
 * - The batch size M, @c Dims::BS
 * - The number of experts E, @c Dims::NUM_EXPERTS
 * - The number of hidden states K, @c Dims::HIDDEN_STATES
 * - The up-projection dimension N, @c Dims::N
 * - The number of experts per token, @c Dims::TOP_K
 *
 * @param [in] activations_in Input activations. Shape: [M, K]
 * @param [in] token_count Number of active tokens
 * @param [in] router_logits Result of routing matrix multiply. Shape: [M, E]
 * @param [in] expert_weights_up Up-projection weights. Shape: [E, 2*N, K]
 * @param [in] expert_scales_up Scales for up-projection weights. Shape: [E, 2*N]
 * @param [in] expert_weights_down Down-projection weights. Shape: [E, K, N]
 * @param [in] expert_scales_down Scales for down-projection weights. Shape: [E, K]
 * @param [out] activations_out Pointer to the output buffer. Shape: [M, K]
 * @param [out] scratchpad Global memory to use for temporary data
 * @param [in] scratchpad_size Size of the scratchpad
 * @param [in] shmem_size Size of the shared memory
 */
template <typename Dims>
__global__ extern void moe_kernel(
    const A_element* __restrict__ activations_in,
    std::uint32_t token_count,
    const __nv_bfloat16 *__restrict__ router_logits,
    const W_element* __restrict expert_weights_up,
    const S_element* __restrict expert_scales_up,
    const W_element* __restrict expert_weights_down,
    const S_element* __restrict expert_scales_down,
    R_element* __restrict activations_out,
    void* __restrict__ scratchpad,
    size_t scratchpad_size,
    size_t shmem_size);

} // namespace moe_monokernel

#endif
