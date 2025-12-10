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
        dim3(KernelConfig::GRID_SIZE),
        dim3(KernelConfig::BLOCK_SIZE),
        kernel_args,
        shmem_size,
        stream);

    // Check for launch errors
    if (err != cudaSuccess) {
        printf("MoE Monokernel launch error: %s (grid=%u, block=%u, shmem=%zu)\n",
               cudaGetErrorString(err), KernelConfig::GRID_SIZE, KernelConfig::BLOCK_SIZE, shmem_size);
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
