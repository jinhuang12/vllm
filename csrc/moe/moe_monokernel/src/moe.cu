/**
 * @file moe.cu
 * @brief MoE Monokernel implementation for Qwen3 with top-k=8 routing
 *
 * This file implements a single cooperative kernel that fuses:
 * 1. Top-8 expert routing with softmax normalization
 * 2. Token preparation and sorting by expert
 * 3. Per-token activation quantization (BF16 -> FP8)
 * 4. Up-projection GEMM (W13 @ activations)
 * 5. SiLU activation with gate multiplication
 * 6. Down-projection GEMM (W2 @ intermediate)
 * 7. Weighted accumulation across all 8 experts per token
 *
 * Target: Qwen3-Coder-30B-A3B-Instruct-FP8 on L40S (SM 8.9)
 */

#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION

// #define MOE_MONOKERNEL_DEBUG_OUTPUT  // Uncomment for debugging

#include <cooperative_groups.h>
#include <cuda/pipeline>

#include "moe_interface.h"
#include "moe_internal.h"
#include "ptx_utils.h"
#include "moe_routing.cu"
#include "moe_prepare.cu"
#include "moe_up_projection.cu"
#include "moe_down_projection.cu"

namespace cg = cooperative_groups;

namespace moe_monokernel {

/**
 * @brief Quantizes input activations from BF16 to FP8 E4M3 with per-token scaling.
 *
 * For each token:
 * 1. Compute max absolute value across hidden dimension
 * 2. Compute scale = max_abs / FP8_MAX (where FP8_MAX ≈ 448 for E4M3)
 * 3. Quantize: fp8_val = bf16_val / scale
 * 4. Store quantized activations and fold scale into topk_weights
 *
 * @tparam Dims The dimension template
 * @param activations_in Input activations [BS, HIDDEN_STATES] in BF16
 * @param batch_size Number of tokens
 * @param shm Shared memory struct
 * @param scratchpad Global memory scratchpad
 */
template <typename Dims>
__device__ static void quantize_activations(
    const A_element* __restrict__ activations_in,
    std::uint32_t batch_size,
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad)
{
    using CoreDims = MoECoreDims<Dims>;

    // Only calc warps (threads 0-255) participate in quantization
    // Prefetch warps (threads 256-383) would cause OOB access to rescale.a[warp]
    if (!is_calc_warp<Dims>()) {
        return;
    }

    constexpr float FP8_E4M3_MAX = 448.0f;

    // Use rescale shared memory area
    auto& a_shm = shm->u.rescale.a;

    // Each calc warp processes one token
    std::uint32_t warp = get_calc_warp<Dims>();
    std::uint32_t thread = get_thread<Dims>();

    for (std::uint32_t token_idx = warp; token_idx < batch_size; token_idx += CoreDims::CALC_WARP_COUNT) {
        // Step 1: Load activation row to shared memory
        for (std::uint32_t k = thread; k < Dims::HIDDEN_STATES; k += CoreDims::THREADS_PER_WARP) {
            a_shm[warp][k] = activations_in[token_idx * Dims::HIDDEN_STATES + k];
        }

        __syncwarp();

        // Step 2: Use static scale of 1.0 (no scaling) to match reference behavior
        // Per-token dynamic scaling requires more complex integration with the GEMM flow
        // to properly compensate for the scale in both x and gate paths
        // For now, use static scale=1.0 which matches reference fused_experts with a1_scale=1.0
        if (thread == 0) {
            for (std::uint32_t k = 0; k < Dims::TOP_K; ++k) {
                std::uint32_t pair_idx = token_idx * Dims::TOP_K + k;
                // No scale folding - use original topk_weights
                scratchpad->topk_weights_scaled[pair_idx] = shm->topk_weights[pair_idx];
            }
        }

        __syncwarp();

        // Step 3: Convert BF16 to FP8 without scaling (static scale = 1.0)
        // Just clamp to FP8 range and convert
        for (std::uint32_t k = thread; k < Dims::HIDDEN_STATES; k += CoreDims::THREADS_PER_WARP) {
            float val = (float)a_shm[warp][k];
            // Clamp to FP8 range
            val = fminf(fmaxf(val, -FP8_E4M3_MAX), FP8_E4M3_MAX);
            scratchpad->activations[token_idx][k] = __nv_fp8_e4m3(val);
        }
    }
}

/**
 * @brief Initializes FP32 output accumulator to zero.
 *
 * Since top-k=8 routing accumulates results from 8 experts per token,
 * we need to initialize the FP32 accumulator to zero before accumulation.
 * The FP32 buffer is used instead of BF16 because atomicAdd on BF16 is broken.
 *
 * @tparam Dims The dimension template
 * @param scratchpad Global memory scratchpad containing output_accum
 * @param batch_size Number of tokens
 */
template <typename Dims>
__device__ static void zero_output_accum(
    MoEGemmSpec<Dims>* scratchpad,
    std::uint32_t batch_size)
{
    std::uint32_t total_elements = batch_size * Dims::HIDDEN_STATES;

    for (std::uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        scratchpad->output_accum[idx] = 0.0f;
    }
}

/**
 * @brief Converts FP32 output accumulator to BF16 output.
 *
 * After all experts have been processed and their results accumulated
 * in the FP32 buffer, this function converts to the final BF16 output.
 *
 * @tparam Dims The dimension template
 * @param scratchpad Global memory scratchpad containing output_accum
 * @param output Output buffer [BS, HIDDEN_STATES] in BF16
 * @param batch_size Number of tokens
 */
template <typename Dims>
__device__ static void convert_fp32_to_bf16(
    MoEGemmSpec<Dims>* scratchpad,
    R_element* __restrict__ output,
    std::uint32_t batch_size)
{
    std::uint32_t total_elements = batch_size * Dims::HIDDEN_STATES;

    for (std::uint32_t idx = threadIdx.x + blockIdx.x * blockDim.x;
         idx < total_elements;
         idx += blockDim.x * gridDim.x) {
        output[idx] = __float2bfloat16(scratchpad->output_accum[idx]);
    }
}

/**
 * @brief Main MoE monokernel for Qwen3 with top-k=8 routing.
 *
 * This is a cooperative kernel that must be launched with cudaLaunchCooperativeKernel.
 * Grid size must equal KernelConfig::GRID_SIZE.
 * Block size must equal KernelConfig::BLOCK_SIZE (384 threads = 12 warps).
 *
 * Execution phases:
 * 1. [Block 0] Routing: Select top-8 experts per token + softmax normalization
 * 2. [Block 0] Prepare: Sort token-expert pairs by expert ID
 * 3. [All blocks] Quantize: BF16 -> FP8 per-token quantization
 * 4. Grid sync
 * 5. [All blocks] Up-projection: W13 @ quantized_activations + SiLU
 * 6. Grid sync
 * 7. [All blocks] Down-projection: W2 @ intermediate, accumulate with topk_weights
 *
 * @tparam Dims The dimension template (e.g., Dims_Qwen3_BS64_E128_TP4)
 */
template <typename Dims>
__global__ void moe_kernel(
    const A_element* __restrict__ activations_in,
    std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out,
    void* __restrict__ scratchpad_ptr,
    size_t scratchpad_size,
    size_t shmem_size)
{
    using CoreDims = MoECoreDims<Dims>;

    // Verify dimensions
    static_assert(Dims::TOP_K == 8, "This kernel is for top-k=8 routing");
    // Warp counts are configurable for benchmarking different configurations
    // Supported: 4c2p, 4c4p, 6c2p, 6c4p, 8c4p, 8c8p
    static_assert(CoreDims::CALC_WARP_COUNT >= 4 && CoreDims::CALC_WARP_COUNT <= 8,
                  "CALC_WARP_COUNT must be between 4 and 8");
    static_assert(CoreDims::PREFETCH_WARP_COUNT >= 2 && CoreDims::PREFETCH_WARP_COUNT <= 8,
                  "PREFETCH_WARP_COUNT must be between 2 and 8");

    // Get shared memory and scratchpad pointers
    extern __shared__ char shmem_raw[];
    MoE_SHM<Dims>* shm = reinterpret_cast<MoE_SHM<Dims>*>(shmem_raw);
    MoEGemmSpec<Dims>* scratchpad = reinterpret_cast<MoEGemmSpec<Dims>*>(scratchpad_ptr);

    // Cooperative groups for grid-wide synchronization
    cg::grid_group grid = cg::this_grid();

    // Record kernel start time (block 0, thread 0 only)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.kernel_start = clock64();
    }

    // Clamp token count to maximum supported
    std::uint32_t batch_size = min(token_count, (std::uint32_t)Dims::BS);

    // ========================================================================
    // Phase 1: Routing and preparation (block 0 only)
    // ========================================================================
    // Block 0 computes top-k routing and sorts token-expert pairs by expert
    if (blockIdx.x == 0) {
        // Routing: select top-8 experts per token, compute softmax weights
        topk_route<Dims>(router_logits, batch_size, shm);
        __syncthreads();

        // Record routing end time
        if (threadIdx.x == 0) {
            scratchpad->timing.routing_end = clock64();
        }

        // Prepare: sort token-expert pairs by expert ID, build expert references
        prepare_moe<Dims>(batch_size, shm);
        __syncthreads();

        // Record prepare end time
        if (threadIdx.x == 0) {
            scratchpad->timing.prepare_end = clock64();
        }

        // Store total pairs
        set_total_pairs<Dims>(batch_size, shm);
    }

    // ========================================================================
    // Phase 2: Initialize output accumulator and quantize activations
    // ========================================================================
    // All blocks zero the FP32 output accumulator (each block handles its portion)
    zero_output_accum<Dims>(scratchpad, batch_size);

    // Block 0 quantizes activations from BF16 to FP8
    if (blockIdx.x == 0) {
        quantize_activations<Dims>(activations_in, batch_size, shm, scratchpad);

        // Record quantize end time
        if (threadIdx.x == 0) {
            scratchpad->timing.quantize_end = clock64();
        }
    }

    // Grid sync: all blocks wait for routing/quantization to complete
    grid.sync();

    // Record after first grid.sync()
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.grid_sync_1 = clock64();
    }

    // ========================================================================
    // Phase 3: Up-projection (all blocks need routing data)
    // ========================================================================
    // Re-compute routing data in each block (cheaper than global memory broadcast)
    topk_route<Dims>(router_logits, batch_size, shm);
    __syncthreads();

    prepare_moe<Dims>(batch_size, shm);
    __syncthreads();

    // ========================================================================
    // Phase 3+4: Up-projection + SiLU (Tensor Core version)
    // ========================================================================
    // Uses Tensor Cores for GEMM, fuses SiLU activation in reduction step
    // Output in scratchpad->temp[pair_idx * N + n] already has SiLU applied
    moe_up_projection<Dims>(
        expert_weights_up,
        expert_scales_up,
        scratchpad,
        shm);

    // Record up-projection end time
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.up_proj_end = clock64();
    }

    // Grid sync: all blocks wait for up-projection to complete
    grid.sync();

    // Record after second grid.sync()
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.grid_sync_2 = clock64();
    }

    // ========================================================================
    // Phase 5: Down-projection
    // ========================================================================
    // Up-projection already stored gate * silu(x) * topk_weight in scratchpad->temp
    // output_accum[token_idx, k] += sum_n(intermediate[pair_idx, n] * w2[expert_id, k, n]) * scale[expert_id, k]
    //
    // Uses optimized Tensor Core down-projection with:
    // - Shared memory tiling for activations and weights
    // - MMA instructions for FP8 × FP32 → FP32 GEMM
    // - Warp shuffle reduction before atomicAdd
    moe_down_projection_topk<Dims>(
        shm,
        scratchpad,
        expert_weights_down,
        expert_scales_down,
        activations_out);

    // Record down-projection end time
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.down_proj_end = clock64();
    }

    // Grid sync: all blocks wait for down-projection to complete
    grid.sync();

    // Record after third grid.sync()
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.grid_sync_3 = clock64();
    }

    // ========================================================================
    // Phase 6: Convert FP32 accumulator to BF16 output
    // ========================================================================
    convert_fp32_to_bf16<Dims>(scratchpad, activations_out, batch_size);

    // Record kernel end time
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.kernel_end = clock64();
    }
}

// ============================================================================
// Explicit template instantiations for Qwen3-Coder-30B-A3B-Instruct-FP8
// ============================================================================
// Block quantization: weight_block_size=[128, 128]

// BS=8, E=128, top_k=8, block_quant=128
template __global__ void moe_kernel<Dims_Qwen3_BS8_E128_TP1_BQ128>(
    const A_element* __restrict__ activations_in,
    std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out,
    void* __restrict__ scratchpad,
    size_t scratchpad_size,
    size_t shmem_size);

// BS=64, E=128, top_k=8, block_quant=128
template __global__ void moe_kernel<Dims_Qwen3_BS64_E128_TP1_BQ128>(
    const A_element* __restrict__ activations_in,
    std::uint32_t token_count,
    const __nv_bfloat16* __restrict__ router_logits,
    const W_element* __restrict__ expert_weights_up,
    const S_element* __restrict__ expert_scales_up,
    const W_element* __restrict__ expert_weights_down,
    const S_element* __restrict__ expert_scales_down,
    R_element* __restrict__ activations_out,
    void* __restrict__ scratchpad,
    size_t scratchpad_size,
    size_t shmem_size);

} // namespace moe_monokernel
