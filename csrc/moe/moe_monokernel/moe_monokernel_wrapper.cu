/**
 * @file moe_monokernel_wrapper.cu
 * @brief CUDA implementation for MoE Monokernel wrapper for Qwen3-Coder-30B-A3B-Instruct-FP8
 *
 * This file includes all kernel implementations directly to avoid
 * linker issues with CUDA template instantiation across translation units.
 *
 * Supports 128x128 block-scaled FP8 quantization.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "moe_monokernel_wrapper.h"

// Include the full kernel implementation (templates must be in same translation unit)
#include "src/moe.cu"

namespace moe_monokernel {

/**
 * @brief Launches the MoE monokernel with the appropriate configuration.
 *
 * For small batch sizes (BS ≤ 4), uses dynamic grid sizing based on the
 * Split-H optimization formula. The grid size is computed to achieve ~80%
 * SM utilization while maintaining correctness.
 *
 * Note: The kernel's down-projection requires at least STANDARD_GRID_SIZE
 * blocks to cover the full hidden dimension. For small batches, we still
 * use the standard grid size but the Split-H config enables future
 * optimizations in the kernel itself.
 */
template <typename Dims>
static void launch_moe_kernel_impl(
    const void* activations_in,
    std::uint32_t token_count,
    const void* router_logits,
    const void* expert_weights_up,
    const void* expert_scales_up,
    const void* expert_weights_down,
    const void* expert_scales_down,
    void* activations_out,
    void* scratchpad,
    cudaStream_t stream)
{
    using KernelConfig = typename Dims::KernelConfig;

    // Compute required shared memory size
    size_t shmem_size = get_moe_shmem_size<Dims>();
    size_t scratchpad_size = get_moe_scratchpad_size<Dims>();

    // Determine grid size
    // For now, use standard grid size for correctness
    // The dynamic get_grid_size() function is available for future Split-H kernel variants
    // where the kernel internally handles different work distribution
    std::uint32_t grid_size = KernelConfig::STANDARD_GRID_SIZE;

    // Debug: Log if we could use Split-H optimization
    // (Future: implement Split-H kernel variant and use get_grid_size here)
    #ifdef DEBUG_MOE_SPLIT_H
    if (KernelConfig::use_split_h(token_count)) {
        std::uint32_t split_h_grid = KernelConfig::get_grid_size(token_count);
        printf("MoE Monokernel: BS=%u would benefit from Split-H (grid %u -> %u)\n",
               token_count, grid_size, split_h_grid);
    }
    #endif

    // Configure shared memory
    cudaFuncSetAttribute(
        moe_kernel<Dims>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shmem_size);

    // Prepare kernel arguments
    void* kernel_args[] = {
        (void*)&activations_in,
        (void*)&token_count,
        (void*)&router_logits,
        (void*)&expert_weights_up,
        (void*)&expert_scales_up,
        (void*)&expert_weights_down,
        (void*)&expert_scales_down,
        (void*)&activations_out,
        (void*)&scratchpad,
        (void*)&scratchpad_size,
        (void*)&shmem_size
    };

    // Launch cooperative kernel
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)moe_kernel<Dims>,
        dim3(grid_size),
        dim3(KernelConfig::BLOCK_SIZE),
        kernel_args,
        shmem_size,
        stream);

    // Check for launch errors
    if (err != cudaSuccess) {
        printf("MoE Monokernel launch error: %s (grid=%u, block=%u, shmem=%zu)\n",
               cudaGetErrorString(err), grid_size, KernelConfig::BLOCK_SIZE, shmem_size);
    }
}

bool check_device_supported()
{
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // Require SM 8.9+ (Ada) or SM 9.0+ (Hopper) for FP8 support
    int sm_version = prop.major * 10 + prop.minor;
    return sm_version >= 89;
}

// ============================================================================
// Block Quantization Launch Functions (Qwen3-Coder-30B-A3B-Instruct-FP8)
// ============================================================================

void launch_moe_monokernel_qwen3_block_quant_bs8(
    const void* activations_in,
    uint32_t token_count,
    const void* router_logits,
    const void* expert_weights_up,
    const void* expert_scales_up,
    const void* expert_weights_down,
    const void* expert_scales_down,
    void* activations_out,
    void* scratchpad,
    void* stream)
{
    launch_moe_kernel_impl<Dims_Qwen3_BS8_E128_TP1_BQ128>(
        activations_in, token_count, router_logits,
        expert_weights_up, expert_scales_up,
        expert_weights_down, expert_scales_down,
        activations_out, scratchpad,
        static_cast<cudaStream_t>(stream));
}

void launch_moe_monokernel_qwen3_block_quant_bs64(
    const void* activations_in,
    uint32_t token_count,
    const void* router_logits,
    const void* expert_weights_up,
    const void* expert_scales_up,
    const void* expert_weights_down,
    const void* expert_scales_down,
    void* activations_out,
    void* scratchpad,
    void* stream)
{
    launch_moe_kernel_impl<Dims_Qwen3_BS64_E128_TP1_BQ128>(
        activations_in, token_count, router_logits,
        expert_weights_up, expert_scales_up,
        expert_weights_down, expert_scales_down,
        activations_out, scratchpad,
        static_cast<cudaStream_t>(stream));
}

int64_t get_scratchpad_size_block_quant_bs8()
{
    return static_cast<int64_t>(get_moe_scratchpad_size<Dims_Qwen3_BS8_E128_TP1_BQ128>());
}

int64_t get_scratchpad_size_block_quant_bs64()
{
    return static_cast<int64_t>(get_moe_scratchpad_size<Dims_Qwen3_BS64_E128_TP1_BQ128>());
}

void get_monokernel_timing_block_quant_bs8(const void* scratchpad, int64_t* timing_out)
{
    using Dims = Dims_Qwen3_BS8_E128_TP1_BQ128;
    const MoEGemmSpec<Dims>* spec = reinterpret_cast<const MoEGemmSpec<Dims>*>(scratchpad);
    cudaMemcpy(timing_out, &spec->timing, sizeof(spec->timing), cudaMemcpyDeviceToHost);
}

void get_monokernel_timing_block_quant_bs64(const void* scratchpad, int64_t* timing_out)
{
    using Dims = Dims_Qwen3_BS64_E128_TP1_BQ128;
    const MoEGemmSpec<Dims>* spec = reinterpret_cast<const MoEGemmSpec<Dims>*>(scratchpad);
    cudaMemcpy(timing_out, &spec->timing, sizeof(spec->timing), cudaMemcpyDeviceToHost);
}

} // namespace moe_monokernel
