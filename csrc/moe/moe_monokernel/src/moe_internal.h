
#pragma once
#ifndef MOE_INTERNAL_H
#define MOE_INTERNAL_H

#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
#error Do not include this file directly.
#endif

#include "moe_interface.h"

namespace moe_monokernel {

using T_element = float; //< Type of GEMM1 (up projection) result as well as sigmoid
using OpaqueElement = std::uint32_t; //< Auxiliary 32-bit type for better assembly code

/**
 * @brief Offsets into the token_indexes field for expert routing
 *
 * For top-k routing, each token has TOP_K entries (e.g., 8 for Qwen3).
 * This struct describes the token range assigned to each expert.
 */
struct ExpertRef
{
    std::uint16_t first_token;  // First index in sorted token list for this expert
    std::uint16_t last_token;   // One past last index
    std::uint32_t id;           // Expert ID
};

/**
 * @brief Token-expert assignment for top-k routing
 *
 * For each token-expert pair, stores the token index and the normalized weight.
 */
struct TokenExpertPair
{
    std::uint16_t token_idx;    // Original token index
    std::uint16_t expert_id;    // Expert assigned to this token
    float weight;               // Normalized routing weight for this expert
};

/**
 * @brief Work item for interleaved expert-tile processing in down-projection.
 *
 * The interleaved processing flattens all (expert, tile) pairs into a work queue
 * and processes them in round-robin fashion to hide memory latency across expert
 * boundaries via double buffering.
 */
struct WorkItem
{
    std::uint32_t expert_idx;   // Index into experts[] array (not expert ID)
    std::uint32_t pair_offset;  // Start index into token_indexes[]
};

/**
 * @brief Maximum number of work items for interleaved down-projection.
 *
 * For Qwen3: BS=64, TOP_K=8, T_TILE=8 => 512/8 = 64 work items.
 * We use 128 to provide headroom for fragmentation and larger batches.
 * If the actual work count exceeds this, we fall back to sequential processing.
 */
constexpr std::uint32_t MAX_WORK_ITEMS = 128;

/**
 * @brief Scratchpad memory for use within the monokernel.
 *
 * Place in global memory. Adapted for top-k routing.
 */
template <typename Dims>
struct alignas(16) MoEGemmSpec
{
    // For top-k, we have BS * TOP_K token-expert pairs
    static constexpr std::uint32_t MAX_PAIRS = Dims::BS * Dims::TOP_K;

#ifdef DEBUG_MOE
    // Debug information
    std::int32_t token_indexes[MAX_PAIRS];
    T_element gemm1[(Dims::BS + 8) * 2 * Dims::N];
#endif
    // All arrays aligned to 16 bytes for efficient vectorized access (copy128)
    alignas(16) AQ_element activations[Dims::BS][Dims::HIDDEN_STATES]; //< Quantized activations (one per token)
    // Up projection produces 2*N values (x and gate) per token-expert pair.
    // Layout: [0..MAX_PAIRS*N) = x, [MAX_PAIRS*N..2*MAX_PAIRS*N) = gate
    alignas(16) T_element temp[2 * MAX_PAIRS * Dims::N]; //< Up projection results (x and gate)
    alignas(16) float topk_weights_scaled[MAX_PAIRS]; //< topk_weights multiplied with activation quantization
    // FP32 accumulator for atomic adds (BF16 atomicAdd is broken/unsupported)
    // Used by down-projection to accumulate results before converting to BF16
    alignas(16) float output_accum[Dims::BS * Dims::HIDDEN_STATES];

    // Per-stage timing (recorded by block 0, thread 0 for profiling)
    // Uses clock64() to record GPU cycle counts at each stage boundary
    struct {
        std::uint64_t kernel_start;     // Kernel entry
        std::uint64_t routing_end;      // After topk_route()
        std::uint64_t prepare_end;      // After prepare_moe()
        std::uint64_t quantize_end;     // After quantize_activations()
        std::uint64_t grid_sync_1;      // After first grid.sync()
        std::uint64_t up_proj_end;      // After moe_up_projection()
        std::uint64_t grid_sync_2;      // After second grid.sync()
        std::uint64_t down_proj_end;    // After down-projection loop
        std::uint64_t grid_sync_3;      // After third grid.sync()
        std::uint64_t kernel_end;       // After convert_fp32_to_bf16()
    } timing;
};


// Maximum supported dimensions for shared memory and scratchpad allocation sizes
// Qwen3: smaller K and N but top_k=8
using Dims_Max = MoEDimensions<1024, 1024, 5120, 128, 8>;

/**
 * @brief Contains various constants used within the MoE monokernel.
 */
template <typename Dims>
struct MoECoreDims {
    using MoEDims = Dims;

    // GPU configuration
    // Optimal configuration determined by benchmarking on L40S (SM 8.9)
    // 6c2p (6 calc + 2 prefetch = 256 threads) provides best performance
    // Benchmark results (BS=1/8/64 ms):
    //   4c2p: 0.764/1.032/3.619 - good
    //   4c4p: 0.786/1.064/3.777 - good
    //   6c2p: 0.754/1.018/3.605 - BEST
    //   6c4p: 0.822/1.088/3.840 - okay
    //   8c4p: 1.122/1.447/4.744 - original baseline
    //   8c8p: 1.355/1.683/6.404 - worst (too many threads)
    static constexpr std::uint32_t THREADS_PER_WARP = 32;
    static constexpr std::uint32_t CALC_WARP_COUNT = 6;  // Optimal for L40S
    static constexpr std::uint32_t PREFETCH_WARP_COUNT = 2;  // Minimal prefetch sufficient
    static constexpr std::uint32_t TOTAL_WARP_COUNT = CALC_WARP_COUNT + PREFETCH_WARP_COUNT;

    // MMA 1 matrix tile dimensions
    // A_TILE=8 is fixed: MMA m16n8k16 requires 8 rows per half-warp
    static constexpr std::uint32_t A_TILE = 8;
    static constexpr std::uint32_t W_UP_TILE = 16;
    static constexpr std::uint32_t K_TILE = 32;

    // GEMM 2 matrix tile dimensions
    // T_TILE=8 is fixed: MMA m16n8k8 output layout uses thread/4 to index 8 token rows
    static constexpr std::uint32_t W_DOWN_MMA_TILE = 16;
    static constexpr std::uint32_t W_DOWN_TILE = Dims::HIDDEN_STATES / Dims::KernelConfig::GRID_SIZE;
    static constexpr std::uint32_t T_TILE = 8;

    static constexpr std::uint32_t W_DIM = 2 * Dims::N;

    static constexpr unsigned BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE;

    static constexpr unsigned PADDING = 32;
    static constexpr unsigned K_DIM_PADDED_A = Dims::HIDDEN_STATES;
    static constexpr unsigned K_DIM_PADDED_W = Dims::HIDDEN_STATES;
    static constexpr unsigned K_DIM_HALF_PADDED_A = Dims::HIDDEN_STATES / 2;
    static constexpr unsigned K_DIM_HALF_PADDED_W = Dims::HIDDEN_STATES / 2;

    // For top-k routing
    static constexpr std::uint32_t MAX_PAIRS = Dims::BS * Dims::TOP_K;
};

/**
 * @brief Shared memory structure for the monokernel
 *
 * Adapted for top-k routing with larger token-expert arrays.
 */
template <typename Dims>
struct alignas(16) MoE_SHM {
    using CoreDims = MoECoreDims<Dims>;
    static constexpr std::uint32_t MAX_PAIRS = Dims::BS * Dims::TOP_K;

    // Union is aligned to 16 bytes for copy128 operations
    union alignas(16) U {
        struct SortData {
            std::uint32_t counters[Dims::NUM_EXPERTS][CoreDims::THREADS_PER_WARP];
            std::uint32_t total_counts[Dims::NUM_EXPERTS];
        } sorting;
        struct RescaleData {
            alignas(16) A_element a[CoreDims::CALC_WARP_COUNT][Dims::HIDDEN_STATES];
        } rescale;
        struct Gemm1Data {
            // prefetch & process tile in 2 halves
            alignas(16) AQ_element a[3][CoreDims::A_TILE][CoreDims::K_DIM_HALF_PADDED_A];
            alignas(16) W_element w[3][CoreDims::W_UP_TILE][CoreDims::K_DIM_HALF_PADDED_W];
            alignas(16) T_element partial_result[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];

            // Block quantization: pre-loaded K-block scales for up-projection
            // For block quant, scale depends on both row and K-column block
            // We pre-load all K-block scales for the current row-block into shared memory
            // Size: UP_SCALE_K_BLOCKS floats (e.g., 16 for Qwen3-30B with block_size=128)
            // For non-block-quant, this is unused (size=1 placeholder)
            alignas(16) S_element block_scales_up[Dims::USE_BLOCK_QUANT ? Dims::UP_SCALE_K_BLOCKS : 1];
            alignas(16) S_element block_scales_gate[Dims::USE_BLOCK_QUANT ? Dims::UP_SCALE_K_BLOCKS : 1];
        } gemm1;
        struct TinyData {
            union alignas(16) {
                AQ_element up[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];
                T_element down[CoreDims::T_TILE][Dims::N];
            } a;
            union alignas(16) {
                A_element orig[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];
                W_element up[CoreDims::W_UP_TILE][CoreDims::K_DIM_PADDED_W];
                W_element down[CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)];
            } w[2];
            alignas(16) S_element scale[2][CoreDims::W_DOWN_TILE + CoreDims::PADDING];
            union alignas(16) {
                T_element up[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];
                T_element down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2][CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
            } partial_result;
        } tiny;
        struct Gemm2Data {
            // Down-projection shared memory buffers for sequential processing.
            // Uses single-buffered approach for simplicity.

            // Activation buffers (double-buffered for potential future use)
            // Total: 2 × T_TILE × N × sizeof(float) = 2 × 8 × 768 × 4 = 49,152 bytes
            alignas(16) T_element t_g0[CoreDims::T_TILE][Dims::N];
            alignas(16) T_element t_g1[CoreDims::T_TILE][Dims::N];

            // Weight buffers (double-buffered for potential future use)
            // Total: 2 × W_DOWN_TILE × (N + padding) × sizeof(W_element) ≈ 24,576 bytes
            alignas(16) W_element w_g0[CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)];
            alignas(16) W_element w_g1[CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)];

            // Scale buffers
            // For block quant: scale indexed by N-block (DOWN_SCALE_N_BLOCKS scales, e.g., 6)
            // Total: 2 × DOWN_SCALE_N_BLOCKS × sizeof(S_element)
            static constexpr std::uint32_t SCALE_BUFFER_SIZE = Dims::USE_BLOCK_QUANT ?
                Dims::DOWN_SCALE_N_BLOCKS : (CoreDims::W_DOWN_TILE + CoreDims::PADDING);
            alignas(16) S_element scale_g0[SCALE_BUFFER_SIZE];
            alignas(16) S_element scale_g1[SCALE_BUFFER_SIZE];

            // Partial results buffer (single, reused each iteration)
            // All 6 calc warps write to this, then warp 0 reduces and does atomicAdd.
            // Layout: [warp][W_DOWN_MMA_TILE * T_TILE]
            // Total: 6 × 16 × 8 × sizeof(float) = 3,072 bytes
            alignas(16) T_element partial_result[CoreDims::CALC_WARP_COUNT][CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];

            // Buffer accessors (idx = 0 or 1)
            __device__ auto& t(std::uint32_t idx) {
                return idx == 0 ? t_g0 : t_g1;
            }
            __device__ auto& w(std::uint32_t idx) {
                return idx == 0 ? w_g0 : w_g1;
            }
            __device__ auto& scale(std::uint32_t idx) {
                return idx == 0 ? scale_g0 : scale_g1;
            }
        } gemm2;
    } u;

    // For top-k=8: each token has 8 expert assignments
    // topk_ids stores the expert ID for each token-expert pair
    // topk_weights stores the normalized weight for each pair
    static_assert(Dims::NUM_EXPERTS < 255, "Number of experts too high, cannot store as uint8 anymore.");

    // Top-k routing data: [BS, TOP_K] flattened
    // All arrays aligned to 8 bytes for efficient vectorized access and to avoid
    // misaligned address errors in cooperative kernel launches
    alignas(8) uint8_t topk_ids[MAX_PAIRS < 8 ? 8 : MAX_PAIRS];
    alignas(8) std::uint16_t token_indexes[MAX_PAIRS + CoreDims::PADDING];
    alignas(8) S_element topk_weights[MAX_PAIRS];

    // Expert reference list (sorted by expert ID)
    ExpertRef experts[Dims::NUM_EXPERTS];

    // For tiny kernel path (BS <= 8)
    std::uint64_t expert_mask;
    std::uint64_t expert_ids;
    std::uint32_t expert_count;

    // For top-k: total number of token-expert pairs to process
    std::uint32_t total_pairs;
};

/**
 * @brief Returns the amount of shared memory necessary to run moe_kernel with template parameter Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_shmem_size()
{
    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
    return sizeof(MoE_SHM<Dims>);
}

constexpr size_t get_moe_max_shmem_size()
{
    return sizeof(MoE_SHM<Dims_Max>);
}

/**
 * @brief Returns the amount of global scratchpad memory necessary to run moe_kernel() with template parameter Dims
 */
template <typename Dims>
__device__ __host__ constexpr size_t get_moe_scratchpad_size()
{
    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
    return sizeof(MoEGemmSpec<Dims>);
}

constexpr size_t get_moe_max_scratchpad_size()
{
    return sizeof(MoEGemmSpec<Dims_Max>);
}

// ============================================================================
// Helper functions for warp/thread identification
// ============================================================================

template <typename Dims>
inline __device__ bool is_calc_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x < CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ bool is_prefetch_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x >= CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_thread()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x % CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_any_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_calc_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    assert(is_calc_warp<Dims>());
    return threadIdx.x / CoreDims::THREADS_PER_WARP;
}

template <typename Dims>
inline __device__ unsigned get_prefetch_warp()
{
    using CoreDims = MoECoreDims<Dims>;
    assert(is_prefetch_warp<Dims>());
    return threadIdx.x / CoreDims::THREADS_PER_WARP - CoreDims::CALC_WARP_COUNT;
}

/**
 * @brief Synchronizes the calc threads of the calling CUDA block
 * The number of threads is determined by CALC_WARP_COUNT * THREADS_PER_WARP
 */
template <typename Dims>
__device__ __forceinline__ void sync_calc_threads()
{
    using CoreDims = MoECoreDims<Dims>;
    constexpr std::uint32_t calc_threads = CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
    // Use appropriate barrier based on calc thread count
    if constexpr (calc_threads == 128) {
        __asm volatile("bar.sync  15, 128;\n");
    } else if constexpr (calc_threads == 192) {
        __asm volatile("bar.sync  15, 192;\n");
    } else if constexpr (calc_threads == 256) {
        __asm volatile("bar.sync  15, 256;\n");
    } else {
        __syncthreads();  // Fallback for other configurations
    }
}

/**
 * @brief Computes the maximum value within a warp
 */
__device__ static inline float warp_reduce_max_float(float value)
{
    for (int i = 16; i >= 1; i /= 2) {
        value = fmaxf(__shfl_xor_sync(0xffffffff, value, i, 32), value);
    }
    return value;
}

/**
 * @brief Warp-level sum reduction
 */
__device__ static inline float warp_reduce_sum_float(float value)
{
    for (int i = 16; i >= 1; i /= 2) {
        value += __shfl_xor_sync(0xffffffff, value, i, 32);
    }
    return value;
}

/**
 * @brief Reinterprets the bit-pattern of x to type To
 */
template <typename To, typename From>
__device__ static __forceinline__ To type_pun(From x)
{
    static_assert(sizeof(To) == sizeof(From), "Types of different size");
    To y;
    memcpy(&y, &x, sizeof(From));
    return y;
}

} // namespace moe_monokernel

#endif
