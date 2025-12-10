/**
 * @file moe_monokernel_wrapper.h
 * @brief Header for MoE Monokernel wrapper functions for Qwen3-Coder-30B-A3B-Instruct-FP8
 *
 * Supports 128x128 block-scaled FP8 quantization.
 * Scale tensor shapes:
 *   - expert_scales_up: [E, 12, 16] for Qwen3-30B
 *   - expert_scales_down: [E, 16, 6] for Qwen3-30B
 */

#ifndef MOE_MONOKERNEL_WRAPPER_H
#define MOE_MONOKERNEL_WRAPPER_H

#include <cstdint>
#include <cstddef>

namespace moe_monokernel {

// Check if the device supports the monokernel
bool check_device_supported();

// Launch functions for block quantization (BS <= 8 and BS <= 64)
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
    void* stream);

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
    void* stream);

// Scratchpad size functions
int64_t get_scratchpad_size_block_quant_bs8();
int64_t get_scratchpad_size_block_quant_bs64();

// Per-stage timing functions - output is 10 int64_t values (clock64 cycles)
// Indices: 0=kernel_start, 1=routing_end, 2=prepare_end, 3=quantize_end,
//          4=grid_sync_1, 5=up_proj_end, 6=grid_sync_2, 7=down_proj_end,
//          8=grid_sync_3, 9=kernel_end
void get_monokernel_timing_block_quant_bs8(const void* scratchpad, int64_t* timing_out);
void get_monokernel_timing_block_quant_bs64(const void* scratchpad, int64_t* timing_out);

} // namespace moe_monokernel

#endif
