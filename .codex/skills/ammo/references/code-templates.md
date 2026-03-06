# Code Patterns Reference

This document describes **patterns** to follow when generating GPU kernel optimization code. Many examples use MoE as a concrete case, but the patterns (MMA, tiling, shared memory, CUDA graphs) apply to any kernel target.

Note: While many examples below use MoE kernels as concrete illustrations, the patterns (MMA instructions, SRAM budgeting, tiling, activation templates, CUDA graph safety) apply to any GPU kernel optimization target.

## Contents
- Key Principle
- C++ Header Structure
- Main Kernel Structure
- Token‑Major Template
- Expert‑Major Template
- Activation Templates
- MMA Templates
- Shared Memory Layouts

## Search anchors
GPU kernel, optimization, MMA templates, token-major, expert-major, shared memory, cp.async, TMA, activation, weight placement.

## Key Principle

The templates here show structural patterns. When generating actual code, Codex should:
1. **Reason through dimensions** - Calculate SMEM usage, tile sizes, K-chunking needs
2. **Adapt to edge cases** - Shared experts, unusual E counts, hybrid dense+MoE
3. **Explain tradeoffs** - Why triple vs double buffer, why specific tile sizes

---

## C++ Header Structure

The header defines compile-time constants that drive kernel specialization.

```cpp
#pragma once
#include <cuda_fp8.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>

namespace vllm::moe {

// === Hardware Configuration ===
// Set based on target GPU (from gpu-configs.md)
static constexpr bool HAS_TMA = ...;        // true for sm_90a only
static constexpr bool HAS_FP8 = ...;        // true for sm_89+
static constexpr uint32_t MONOKERNEL_THRESHOLD = ...; // 128/64/32 by GPU

// === Algorithmic Branching ===
// Set based on model topology (from algorithmic-branching.md)
enum class Ownership { TokenMajor, ExpertMajor, Hybrid };
static constexpr Ownership OWNERSHIP = ...;
enum class AccumulationStrategy { DirectWrite, BlockReduce, GlobalAtomic };
static constexpr AccumulationStrategy ACCUM = ...;
static constexpr bool OUTPUT_OVERLAP = (OWNERSHIP != Ownership::TokenMajor);
static constexpr bool USE_ATOMICS = OUTPUT_OVERLAP;  // only if output overlaps
enum class WeightPlacement { BeforeActivation, AfterActivation };
static constexpr WeightPlacement WEIGHT_PLACEMENT = ...;
static constexpr uint32_t TOKENS_PER_WARP = ...; // from sorter strategy
static constexpr bool WEIGHT_AFTER_ACTIVATION =
    (WEIGHT_PLACEMENT == WeightPlacement::AfterActivation);

// Per-phase ownership/atomics (for hybrid designs)
static constexpr Ownership UP_OWNERSHIP = ...;
static constexpr Ownership DOWN_OWNERSHIP = ...;
static constexpr bool USE_ATOMICS_UP = (UP_OWNERSHIP != Ownership::TokenMajor);
static constexpr bool USE_ATOMICS_DOWN = (DOWN_OWNERSHIP != Ownership::TokenMajor);

// === Dimension Template ===
template <uint32_t bs, uint32_t n, uint32_t k, uint32_t e>
struct Dims {
    static constexpr uint32_t BS = bs;
    static constexpr uint32_t N = n;           // N_local after TP split
    static constexpr uint32_t K = k;           // Hidden size
    static constexpr uint32_t E = e;           // E_local after EP split
};

// === Tile Configuration ===
// These values come from SRAM Tetris analysis
template <typename D>
struct TileConfig {
    // Warp organization
    static constexpr uint32_t CALC_WARPS = ...;      // 6-10 typically
    static constexpr uint32_t PREFETCH_WARPS = ...;  // 2-4 typically
    static constexpr uint32_t BLOCK_SIZE = (CALC_WARPS + PREFETCH_WARPS) * 32;
    
    // Tile sizes (from SRAM Tetris)
    static constexpr uint32_t M_TILE = ...;    // 8 or 16
    static constexpr uint32_t N_TILE = ...;    // 16, 32, or 64
    static constexpr uint32_t K_TILE = 32;     // MMA accumulation chunk
    static constexpr uint32_t K_CHUNKS = ...;  // 1, 2, or 4 if SMEM tight
    
    // Buffering depth (from SRAM Tetris)
    static constexpr uint32_t UP_BUFFERS = ...; // 2 or 3
    static constexpr uint32_t DOWN_BUFFERS = 2;
    
    // Grid sizing
    static constexpr uint32_t GRID_SIZE = ...;  // ≤ SM count
};

} // namespace vllm::moe
```

---

## Split-Kernel Interface (Optional)

When routing is done outside the monokernel, pass top-k data explicitly:

```cpp
// Inputs produced by router/prepare kernels
const uint16_t* topk_ids;       // [BS, TOP_K]
const float* topk_weights;      // [BS, TOP_K]
const uint16_t* expert_offsets; // [E+1] prefix sums for grouped GEMM (optional)
const uint16_t* pair_indices;   // [BS * TOP_K] packed pairs (optional)
```

This avoids controller blocks and grid.sync when the GEMM kernel only consumes precomputed routing.

---

## SRAM Tetris: How Codex Should Reason Through Tile Sizes

Don't use a formula blindly. Walk through this reasoning:

### Step 1: Know Your Budget
```
SMEM_budget = SMEM_per_SM - 8KB_margin
  H100/H200: 228KB - 8KB = 220KB
  L40S:      100KB - 8KB = 92KB
  A100:      164KB - 8KB = 156KB
```

### Step 2: Calculate Memory Per Tile Choice
```
For a given (M_tile, N_tile, K_eff, buffers):

dtype_bytes = 1 (FP8) or 2 (BF16/FP16)

Up-projection phase:
  A_bytes = buffers × M_tile × K_eff × dtype_bytes
  W_bytes = buffers × N_tile × (K_eff + 32) × dtype_bytes  // 32B padding
  partial_bytes = calc_warps × N_tile × M_tile × 4         // FP32 accumulators

Down-projection phase:
  T_bytes = 2 × M_tile × N × 4                             // FP32 intermediates
  W_down_bytes = 2 × W_down_tile × (N + 32) × dtype_bytes
  partial_down = calc_warps × W_down_tile × M_tile × 4

Total = max(up_phase, down_phase)  // Phases don't overlap
```

### Step 3: Search With Fallback Cascade
```
Try in order:
1. buffers=3, M=16, N=64  → if fits, done (best latency hiding)
2. buffers=3, M=16, N=32  → reduce N
3. buffers=2, M=16, N=32  → reduce buffering
4. buffers=2, M=8,  N=32  → reduce M (tiny kernel)
5. buffers=2, M=8,  N=16, K_chunks=2  → chunk K dimension
6. buffers=2, M=8,  N=16, K_chunks=4  → more K chunking

Pick the first that fits. Explain WHY it was chosen.
```

### Example Reasoning (Qwen3 on L40S)

```
Model: K=8192, N_local=2048, E=64
Hardware: L40S, SMEM=92KB
dtype: FP8 (1 byte)

Try buffers=3, M=16:
  A = 3 × 16 × 4096 × 1 = 192KB  ← Already exceeds 92KB!
  
Try buffers=2, M=8, K_chunks=4 (so K_eff=2048):
  A = 2 × 8 × 2048 × 1 = 32KB
  W = 2 × 32 × (2048 + 32) × 1 = 133KB ← Still too big
  
Try buffers=2, M=8, N=16, K_chunks=4:
  A = 2 × 8 × 2048 × 1 = 32KB
  W = 2 × 16 × (2048 + 32) × 1 = 66KB
  partial = 8 × 16 × 8 × 4 = 4KB
  Total = 32 + 66 + 4 = 102KB ← Still too big
  
Try buffers=2, M=8, N=16, K_chunks=8:
  K_eff = 1024
  A = 2 × 8 × 1024 × 1 = 16KB
  W = 2 × 16 × 1056 × 1 = 34KB
  partial = 4KB
  Total = 54KB ← FITS!

Result: M=8, N=16, K_chunks=8, double buffer
Tradeoff: More K iterations, but fits in SMEM
```

---

## MMA Instruction Templates

### FP8 MMA (m16n8k32)

Used by: FP8 quantized models (Llama4-FP8, Qwen3-FP8, etc.)

```cpp
/**
 * @brief FP8 E4M3 matrix multiply-accumulate using Tensor Cores
 * 
 * Computes: D = A @ B + C where A is [16,32], B is [32,8], C/D are [16,8]
 * Each thread holds fragments, warp cooperatively computes full tile.
 */
__device__ __forceinline__ void mma_fp8_fp8(
    float& d0, float& d1, float& d2, float& d3,
    __nv_fp8x4_e4m3 a0, __nv_fp8x4_e4m3 a1,
    __nv_fp8x4_e4m3 a2, __nv_fp8x4_e4m3 a3,
    __nv_fp8x4_e4m3 b0, __nv_fp8x4_e4m3 b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(*reinterpret_cast<uint32_t*>(&a0)),
          "r"(*reinterpret_cast<uint32_t*>(&a1)),
          "r"(*reinterpret_cast<uint32_t*>(&a2)),
          "r"(*reinterpret_cast<uint32_t*>(&a3)),
          "r"(*reinterpret_cast<uint32_t*>(&b0)),
          "r"(*reinterpret_cast<uint32_t*>(&b1)),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}
```

### BF16 MMA (m16n8k16)

Used by: BF16 models without quantization (native precision inference)

**NOTE**: BF16 uses m16n8k16 (not k32 like FP8), so K_TILE iterations differ.

```cpp
/**
 * @brief BF16 matrix multiply-accumulate using Tensor Cores
 * 
 * Computes: D = A @ B + C where A is [16,16], B is [16,8], C/D are [16,8]
 * K dimension is 16 (half of FP8's 32), so need 2x iterations for same K.
 * 
 * Fragment layout per thread (warp of 32 threads):
 *   A fragments: 4 x uint32_t (each holds 2 BF16 values)
 *   B fragments: 2 x uint32_t (each holds 2 BF16 values)
 *   C/D fragments: 4 x float
 */
__device__ __forceinline__ void mma_bf16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c0), "f"(c1), "f"(c2), "f"(c3)
    );
}

/**
 * @brief Load A matrix fragment from shared memory for BF16 MMA
 * 
 * Loads 8 BF16 values (4 uint32_t) for one thread's A fragment.
 * SMEM layout assumed: [M_TILE, K_TILE] with 32-byte row stride for swizzling.
 */
template <uint32_t BS_MAX>
__device__ __forceinline__ void load_A_fragment_bf16(
    const __nv_bfloat16* __restrict__ smem_a,
    uint32_t row,
    uint32_t k_offset,
    uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3)
{
    const uint32_t lane = threadIdx.x % 32;
    
    // m16n8k16 A fragment: each thread loads from specific rows
    // Thread mapping: lane -> (row_in_tile, k_in_tile)
    uint32_t frag_row = (lane % 16);
    uint32_t frag_k = (lane / 16) * 8;  // 0 or 8
    
    // Load 4 pairs of BF16 (8 values total)
    const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
        &smem_a[(row + frag_row) * K_TILE_STRIDE + k_offset + frag_k]);
    
    a0 = *reinterpret_cast<const uint32_t*>(&src[0]);
    a1 = *reinterpret_cast<const uint32_t*>(&src[1]);
    a2 = *reinterpret_cast<const uint32_t*>(&src[2]);
    a3 = *reinterpret_cast<const uint32_t*>(&src[3]);
}

/**
 * @brief Load B matrix fragment from shared memory for BF16 MMA
 * 
 * Loads 4 BF16 values (2 uint32_t) for one thread's B fragment.
 * SMEM layout assumed: [N_TILE, K_TILE] with swizzling.
 */
template <uint32_t BS_MAX>
__device__ __forceinline__ void load_B_fragment_bf16(
    const __nv_bfloat16* __restrict__ smem_b,
    uint32_t n_row,
    uint32_t k_offset,
    uint32_t& b0, uint32_t& b1)
{
    const uint32_t lane = threadIdx.x % 32;
    
    // m16n8k16 B fragment: threads load from k dimension
    uint32_t frag_n = lane % 8;
    uint32_t frag_k = (lane / 8) * 4;  // 0, 4, 8, or 12
    
    const __nv_bfloat162* src = reinterpret_cast<const __nv_bfloat162*>(
        &smem_b[(n_row + frag_n) * K_TILE_STRIDE + k_offset + frag_k]);
    
    b0 = *reinterpret_cast<const uint32_t*>(&src[0]);
    b1 = *reinterpret_cast<const uint32_t*>(&src[1]);
}
```

### Complete BF16 MMA Loop Example

This is a **complete, working example** - adapt dimensions, don't rewrite from scratch:

```cpp
/**
 * @brief Complete K-chunked MMA loop for BF16 up-projection
 * 
 * This is a REFERENCE IMPLEMENTATION. Adapt dimensions, don't derive from scratch.
 * 
 * @param gemm1 Shared memory struct with double-buffered a[] and w[]
 * @param K_CHUNKS Number of K-dimension chunks (ceil(K / K_TILE))
 * @param d0-d7 FP32 accumulators (4 for x path, 4 for gate path)
 */
template <uint32_t BS_MAX, uint32_t K_CHUNKS>
__device__ void bf16_up_projection_mma_loop(
    GemmSharedMem& gemm1,
    uint32_t n_row,    // Which N-tile this warp handles
    float& d0_x, float& d1_x, float& d2_x, float& d3_x,
    float& d0_g, float& d1_g, float& d2_g, float& d3_g)
{
    constexpr uint32_t K_TILE = 64;      // Adjust based on SMEM budget
    constexpr uint32_t K_MMA = 16;       // BF16 MMA processes 16 K elements
    constexpr uint32_t K_ITERS = K_TILE / K_MMA;  // 4 iterations per K-chunk
    
    uint32_t read_buf = 0;
    
    // Initialize accumulators to zero
    d0_x = d1_x = d2_x = d3_x = 0.0f;
    d0_g = d1_g = d2_g = d3_g = 0.0f;
    
    for (uint32_t k_chunk = 0; k_chunk < K_CHUNKS; k_chunk++) {
        // Wait for async memory loads to complete
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();
        
        if (is_calc_warp<BS_MAX>()) {
            // Process K_TILE in K_MMA-sized chunks
            #pragma unroll
            for (uint32_t k_iter = 0; k_iter < K_ITERS; k_iter++) {
                uint32_t k_offset = k_iter * K_MMA;
                
                // Load A fragment (activations)
                uint32_t a0, a1, a2, a3;
                load_A_fragment_bf16<BS_MAX>(
                    gemm1.a[read_buf], 
                    0,  // row offset (handle externally for multi-row tiles)
                    k_offset,
                    a0, a1, a2, a3);
                
                // Load B fragment for x path (rows 0 to N-1)
                uint32_t b0_x, b1_x;
                load_B_fragment_bf16<BS_MAX>(
                    gemm1.w[read_buf],
                    n_row,  // x weights
                    k_offset,
                    b0_x, b1_x);
                
                // Load B fragment for gate path (rows N to 2N-1)
                uint32_t b0_g, b1_g;
                load_B_fragment_bf16<BS_MAX>(
                    gemm1.w[read_buf],
                    n_row + N_TILE,  // gate weights (upper half of W13)
                    k_offset,
                    b0_g, b1_g);
                
                // MMA for x path: accumulate into d0_x..d3_x
                mma_bf16_bf16(d0_x, d1_x, d2_x, d3_x,
                              a0, a1, a2, a3,
                              b0_x, b1_x,
                              d0_x, d1_x, d2_x, d3_x);
                
                // MMA for gate path: accumulate into d0_g..d3_g
                mma_bf16_bf16(d0_g, d1_g, d2_g, d3_g,
                              a0, a1, a2, a3,
                              b0_g, b1_g,
                              d0_g, d1_g, d2_g, d3_g);
            }
        }
        
        // Swap double buffer for next iteration
        read_buf = 1 - read_buf;
        __syncthreads();
    }
    
    // After loop: d0_x..d3_x contain x results, d0_g..d3_g contain gate results
    // Next step: apply activation function (SiLU, GELU, etc.)
}
```

### FP8 vs BF16 MMA Comparison

| Aspect | FP8 (m16n8k32) | BF16 (m16n8k16) |
|--------|----------------|-----------------|
| K per MMA | 32 | 16 |
| A fragment | 4 × fp8x4 (16 bytes) | 4 × uint32 (16 bytes) |
| B fragment | 2 × fp8x4 (8 bytes) | 2 × uint32 (8 bytes) |
| Iterations for K=1024 | 32 | 64 |
| SMEM per K-tile | K × dtype = K bytes | K × 2 bytes |
| Scale handling | Block or per-tensor | None needed |

**Key insight**: BF16 needs 2× the K iterations but doesn't require quantization scale handling, making the loop simpler despite more iterations.

---

For models with block-wise FP8 quantization (e.g., 128×128 blocks):

### Scale Tensor Shapes

```cpp
// Given weight W_up: [E, 2*N, K]
// Block size: 128×128
// Scale shape: [E, ceil(2*N/128), ceil(K/128)]

// Example for Qwen3: K=2048, N=768
// W_up: [128, 1536, 2048]
// W_up_scale: [128, 12, 16]  // ceil(1536/128)=12, ceil(2048/128)=16

// W_down: [E, K, N]
// W_down_scale: [E, ceil(K/128), ceil(N/128)]
// Example: [128, 16, 6]

// Helper to compute scale indices
constexpr uint32_t BLOCK_SIZE = 128;
uint32_t n_blocks = (2 * N + BLOCK_SIZE - 1) / BLOCK_SIZE;
uint32_t k_blocks = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
```

### Loading Block Scales to Shared Memory

```cpp
// During weight prefetch, also load corresponding scales
template <typename Dims>
__device__ void load_up_scales(
    const float* __restrict__ expert_scales,
    MoE_SHM<Dims>* shm,
    uint32_t expert_id,
    uint32_t n_block_start,
    uint32_t k_block_start)
{
    constexpr uint32_t BLOCK_SIZE = 128;
    constexpr uint32_t N_BLOCKS = (2 * Dims::N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    constexpr uint32_t K_BLOCKS = (Dims::K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Scale layout: [E, N_BLOCKS, K_BLOCKS]
    // Load scales needed for current tile
    if (threadIdx.x < N_TILE_BLOCKS * K_TILE_BLOCKS) {
        uint32_t nb = threadIdx.x / K_TILE_BLOCKS;
        uint32_t kb = threadIdx.x % K_TILE_BLOCKS;
        
        uint32_t scale_idx = expert_id * N_BLOCKS * K_BLOCKS
                           + (n_block_start + nb) * K_BLOCKS
                           + (k_block_start + kb);
        
        shm->up_scales[nb][kb] = expert_scales[scale_idx];
    }
}
```

### Applying Block Scales During MMA

```cpp
// In up-projection MMA loop:
template <typename Dims>
__device__ void mma_with_block_scale(
    float& d0, float& d1, float& d2, float& d3,
    __nv_fp8x4_e4m3 w0, __nv_fp8x4_e4m3 w1,
    __nv_fp8x4_e4m3 w2, __nv_fp8x4_e4m3 w3,
    __nv_fp8x4_e4m3 a02, __nv_fp8x4_e4m3 a13,
    const MoE_SHM<Dims>* shm,
    uint32_t base_col,  // K offset
    bool is_gate)       // upper half of W for gate
{
    if constexpr (Dims::USE_BLOCK_QUANT) {
        // Compute MMA into temporaries (unscaled)
        float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
        mma_fp8_fp8(t0, t1, t2, t3, w0, w1, w2, w3, a02, a13, 0.f, 0.f, 0.f, 0.f);
        
        // Get scale for this K-block
        uint32_t k_block_idx = base_col / 128;
        uint32_t n_block_idx = is_gate ? (Dims::N / 128) : 0;
        float block_scale = shm->up_scales[n_block_idx][k_block_idx];
        
        // Apply scale during accumulation
        d0 += t0 * block_scale;
        d1 += t1 * block_scale;
        d2 += t2 * block_scale;
        d3 += t3 * block_scale;
    } else {
        // Per-tensor scale: just accumulate, apply scale at end
        mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
    }
}
```

### Helper Function for Scale Lookup

```cpp
template <typename Dims>
__device__ __forceinline__ float moe_get_up_block_scale(
    const MoE_SHM<Dims>* shm,
    uint32_t k_offset,
    bool is_gate)
{
    constexpr uint32_t BLOCK_SIZE = 128;
    uint32_t k_block = k_offset / BLOCK_SIZE;
    // Gate weights are in upper half of W (rows N to 2N-1)
    uint32_t n_block = is_gate ? (Dims::N / BLOCK_SIZE) : 0;
    return shm->up_scales[n_block][k_block];
}
```

### Down-Projection Block Scale Handling (Qwen3 Pattern)

The down-projection has different scale indexing because the weight shape is `[E, K, N]` (not `[E, 2*N, K]` like up-projection).

```cpp
// Down-projection scale layout: [E, K_blocks, N_blocks]
// For Qwen3: [128, 16, 6] where K=2048, N=768, block=128

template <typename Dims>
struct DownScaleDims {
    static constexpr uint32_t K_BLOCKS = (Dims::K + Dims::BLOCK_SIZE_QUANT - 1) / Dims::BLOCK_SIZE_QUANT;
    static constexpr uint32_t N_BLOCKS = (Dims::N + Dims::BLOCK_SIZE_QUANT - 1) / Dims::BLOCK_SIZE_QUANT;
    // Qwen3 example: K_BLOCKS=16, N_BLOCKS=6
};

// Loading scales during down-projection weight prefetch
template <typename Dims>
__device__ void moe_request_down_expert_with_scale(
    const W_element* expert_weights_down,
    const S_element* expert_scales_down,
    MoE_SHM<Dims>* shm,
    cuda::pipeline<cuda::thread_scope_thread>& pipeline,
    uint32_t expert_id,
    uint32_t k_offset,      // Current K-slice: blockIdx.x * W_DOWN_TILE
    uint32_t w_buffer_idx)
{
    // ... weight loading code ...

    if constexpr (Dims::USE_BLOCK_QUANT) {
        // k_offset determines which K-block we're in
        // Each K-block spans BLOCK_SIZE_QUANT rows
        // W_DOWN_TILE (16) < BLOCK_SIZE_QUANT (128), so multiple tiles share one K-block
        uint32_t k_block = k_offset / Dims::BLOCK_SIZE_QUANT;

        // Load ALL N-block scales for this K-block (only N_BLOCKS=6 values)
        if (warp == 0 && lane < Dims::DOWN_SCALE_N_BLOCKS) {
            uint32_t scale_idx =
                expert_id * DownScaleDims<Dims>::K_BLOCKS * DownScaleDims<Dims>::N_BLOCKS
                + k_block * DownScaleDims<Dims>::N_BLOCKS
                + lane;

            shm->down_scales[w_buffer_idx][lane] = expert_scales_down[scale_idx];
        }
    }
}

// Applying N-block-varying scales during MMA
template <typename Dims>
__device__ void moe_down_gemm_with_block_scale(
    MoE_SHM<Dims>* shm,
    uint32_t buffer_idx)
{
    auto& scale = shm->down_scales[buffer_idx];  // [N_BLOCKS] preloaded

    for (uint32_t base_col = warp * K_TILE; base_col < Dims::N; base_col += BLOCK_STRIDE) {
        // Load FP8 weights and activations...
        __nv_fp8x4_e4m3 w0, w1, w2, w3, a02, a13;

        if constexpr (Dims::USE_BLOCK_QUANT) {
            // Compute MMA into temporaries (unscaled)
            float t0 = 0.f, t1 = 0.f, t2 = 0.f, t3 = 0.f;
            mma_fp8_fp8(t0, t1, t2, t3, w0, w1, w2, w3, a02, a13, 0.f, 0.f, 0.f, 0.f);

            // KEY: N-block determines which scale to use
            // base_col is the N-dimension offset, not K
            uint32_t n_block = base_col / Dims::BLOCK_SIZE_QUANT;
            float block_scale = scale[n_block];

            // Apply N-block-specific scale
            d0 += t0 * block_scale;
            d1 += t1 * block_scale;
            d2 += t2 * block_scale;
            d3 += t3 * block_scale;
        } else {
            // Per-tensor: accumulate, apply row-scale at end
            mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
        }
    }
}
```

### Block Quantization vs Per-Tensor Scale Summary

| Aspect | Per-Tensor Scale | Block Quantization (128x128) |
|--------|------------------|------------------------------|
| **Scale shape (up)** | `[E, 2*N]` or `[E]` | `[E, 2*N/128, K/128]` |
| **Scale shape (down)** | `[E, K]` or `[E]` | `[E, K/128, N/128]` |
| **Scale application** | After all K iterations | During each K-block MMA |
| **N-dimension handling** | Same scale for all N | Different scale per N-block |
| **SMEM overhead** | Minimal (1-2 scales) | O(K_blocks × N_blocks) per expert |
| **Models** | Llama 4 FP8 | Qwen3-FP8, DeepSeek-FP8 |

---

## Per-Stage Timing Infrastructure

For debugging performance without NCU overhead:

### Timing Struct

```cpp
struct KernelTiming {
    int64_t kernel_start;
    int64_t routing_end;
    int64_t prepare_end;
    int64_t quantize_end;
    int64_t grid_sync_1;
    int64_t up_proj_end;
    int64_t grid_sync_2;
    int64_t down_proj_end;
    int64_t grid_sync_3;
    int64_t kernel_end;
};

// Add to scratchpad struct
template <typename Dims>
struct MoEGemmSpec {
    // ... other fields ...
    KernelTiming timing;
};
```

### Recording in Kernel

```cpp
template <typename Dims>
__global__ void moe_kernel(...) {
    auto* scratchpad = reinterpret_cast<MoEGemmSpec<Dims>*>(scratch);
    
    // Block 0, thread 0 records timing
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.kernel_start = clock64();
    }
    
    // === Routing ===
    topk_route<Dims>(router_logits, batch_size, shm);
    __syncthreads();
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.routing_end = clock64();
    }
    
    // === Prepare ===
    prepare_moe<Dims>(batch_size, shm);
    __syncthreads();
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.prepare_end = clock64();
    }
    
    // === Quantize ===
    quantize_activations<Dims>(activations_in, batch_size, shm, scratchpad);
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.quantize_end = clock64();
    }
    
    // NOTE: cooperative grid sync requires gridDim.x <= max active blocks for this kernel.
    cooperative_groups::this_grid().sync();
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.grid_sync_1 = clock64();
    }
    
    // === Up-projection ===
    moe_up_projection<Dims>(expert_weights_up, expert_scales_up, scratchpad, shm);
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.up_proj_end = clock64();
    }
    
    cooperative_groups::this_grid().sync();
    
    // ... continue for down-projection ...
    
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        scratchpad->timing.kernel_end = clock64();
    }
}
```

### Extracting Timing (Host Side)

```cpp
// In wrapper
void get_monokernel_timing(const void* scratchpad, int64_t* timing_out) {
    cudaMemcpy(timing_out, 
               &(reinterpret_cast<const MoEGemmSpec<Dims>*>(scratchpad)->timing),
               sizeof(KernelTiming), 
               cudaMemcpyDeviceToHost);
}

// Python usage
def get_stage_latencies(scratchpad, gpu_clock_khz):
    timing = np.zeros(10, dtype=np.int64)
    ops.get_monokernel_timing(scratchpad, timing)
    
    stages = {
        'routing': (timing[1] - timing[0]) / gpu_clock_khz * 1000,  # µs
        'prepare': (timing[2] - timing[1]) / gpu_clock_khz * 1000,
        'quantize': (timing[3] - timing[2]) / gpu_clock_khz * 1000,
        'sync_1': (timing[4] - timing[3]) / gpu_clock_khz * 1000,
        'up_proj': (timing[5] - timing[4]) / gpu_clock_khz * 1000,
        'sync_2': (timing[6] - timing[5]) / gpu_clock_khz * 1000,
        'down_proj': (timing[7] - timing[6]) / gpu_clock_khz * 1000,
        'sync_3': (timing[8] - timing[7]) / gpu_clock_khz * 1000,
        'total': (timing[9] - timing[0]) / gpu_clock_khz * 1000,
    }
    return stages
```

### Debug Output Macro

```cpp
// Enable with -DMOE_MONOKERNEL_DEBUG_OUTPUT
#ifdef MOE_MONOKERNEL_DEBUG_OUTPUT
#define MOE_DEBUG(fmt, ...) \
    if (blockIdx.x == 0 && threadIdx.x == 0) { \
        printf("MOE_DEBUG: " fmt "\n", ##__VA_ARGS__); \
    }
#else
#define MOE_DEBUG(fmt, ...) do {} while(0)
#endif

// Usage
MOE_DEBUG("routing complete: top_ids[0]=%d, weights[0]=%.4f", 
          shm->topk_ids[0], shm->topk_weights[0]);
```

---

## Activation Function Templates

### SiLU (Swish) - Most Common

Used by: Llama, Qwen, Mistral, most modern LLMs

```cpp
// Formula: SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
// For gated: output = up * SiLU(gate) = up * gate / (1 + exp(-gate))

template <bool WeightAfterActivation>
__device__ __forceinline__ float apply_silu_gated(
    float gate,      // First N outputs from W13
    float up,        // Second N outputs from W13  
    float topk_weight)
{
    // SiLU on gate, multiply by up
    float silu = (gate * up) / (1.0f + expf(-gate));
    
    if constexpr (WeightAfterActivation) {
        // Apply weight after activation if model semantics require it
        return silu * topk_weight;
    } else {
        // Weight already folded earlier by model definition
        return silu;
    }
}
```

### GELU - Used by Some Models

Used by: GPT-2, BERT, some research models

```cpp
// Formula: GELU(x) = x * Φ(x) ≈ x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
// Fast approximation used in practice

__device__ __forceinline__ float gelu_fast(float x) {
    constexpr float SQRT_2_OVER_PI = 0.7978845608f;  // sqrt(2/π)
    constexpr float COEFF = 0.044715f;
    
    float x3 = x * x * x;
    float inner = SQRT_2_OVER_PI * (x + COEFF * x3);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// For gated GELU (GeGLU)
template <bool WeightAfterActivation>
__device__ __forceinline__ float apply_geglu(
    float gate,
    float up,
    float topk_weight)
{
    float gelu_gate = gelu_fast(gate);
    float result = up * gelu_gate;
    
    if constexpr (WeightAfterActivation) {
        return result * topk_weight;
    } else {
        return result;
    }
}
```

### ReLU - Simple but Less Common for MoE

```cpp
// Formula: ReLU(x) = max(0, x)

template <bool WeightAfterActivation>
__device__ __forceinline__ float apply_relu_gated(
    float gate,
    float up,
    float topk_weight)
{
    float relu_gate = fmaxf(0.0f, gate);
    float result = up * relu_gate;
    
    if constexpr (WeightAfterActivation) {
        return result * topk_weight;
    } else {
        return result;
    }
}

// Squared ReLU variant (used by some models)
__device__ __forceinline__ float squared_relu(float x) {
    float relu = fmaxf(0.0f, x);
    return relu * relu;
}
```

### Handling Unknown/Custom Activations

If the model uses an activation function not in templates above:

```markdown
**When encountering unknown activation in Phase 3 (up-projection):**

1. **Explore the activation function** using the prompt below (recommended: run in an isolated fresh session, e.g., via `codex exec`, to avoid polluting your main context):

   ```
   Investigate {activation_name} activation function for CUDA MoE kernel:

   1. Find the mathematical formula in vLLM source or model documentation
   2. Search for efficient CUDA implementations (check CUTLASS, cuDNN, Triton)
   3. Determine how it interacts with gated architecture (is it gate*act(up) or act(gate)*up?)
   4. Check for numerical stability concerns (overflow, underflow, NaN)
   5. Look for any model-specific variations

   Output findings to {artifact_dir}/activation_exploration.md with:
   - Mathematical formula
   - Recommended CUDA implementation
   - Test cases for numerical validation
   ```

2. Based on exploration findings, implement the activation function

3. Add numerical validation in Phase 4 to ensure correctness
```

### Activation Selection Pattern

```cpp
// Compile-time activation selection
enum class ActivationType { SILU, GELU, GEGLU, RELU, SQUARED_RELU };

template <ActivationType Act, bool WeightAfter>
__device__ __forceinline__ float apply_activation(
    float gate, float up, float weight)
{
    float result;
    
    if constexpr (Act == ActivationType::SILU) {
        result = (gate * up) / (1.0f + expf(-gate));
    } else if constexpr (Act == ActivationType::GELU) {
        result = up * gelu_fast(gate);
    } else if constexpr (Act == ActivationType::GEGLU) {
        result = up * gelu_fast(gate);
    } else if constexpr (Act == ActivationType::RELU) {
        result = up * fmaxf(0.0f, gate);
    } else if constexpr (Act == ActivationType::SQUARED_RELU) {
        float r = fmaxf(0.0f, gate);
        result = up * r * r;
    }
    
    if constexpr (WeightAfter) {
        return result * weight;
    } else {
        return result;
    }
}
```

---

## Kernel Entry Pattern

```cpp
template <typename Dims, typename Tiles>
__global__ void __launch_bounds__(Tiles::BLOCK_SIZE)
moe_monokernel(
    // Input tensors
    bf16 const* __restrict__ input,         // [BS, K]
    bf16 const* __restrict__ router_logits, // [BS, E]
    fp8 const* __restrict__ w13,            // [E, 2*N, K]
    fp8 const* __restrict__ w2,             // [E, K, N]
    float const* __restrict__ scales_13,
    float const* __restrict__ scales_2,
    // Output
    bf16* __restrict__ output,              // [BS, K]
    void* __restrict__ scratch
) {
    extern __shared__ char smem[];
    
    // Phase 1: In-kernel router (top-k selection)
    // Phase 2: Token grouping by expert
    // Phase 3: FP8 activation quantization
    // Phase 4: Up-projection + activation fusion
    // Grid sync
    // Phase 5: Down-projection + output
}
```

---

## Token-Major Kernel Skeleton

Use when output is token‑owned and you want to avoid atomics:

```cpp
template <typename Dims, typename Tiles>
__global__ void moe_token_major(
    const bf16* __restrict__ input,
    const uint16_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    const fp8* __restrict__ w13,
    const fp8* __restrict__ w2,
    bf16* __restrict__ output)
{
    // Each block owns one token and a K-slice
    uint32_t token = blockIdx.y;
    uint32_t k_slice = blockIdx.x;
    // Loop over top_k experts for this token; accumulate locally
    // Write output directly for this token + K-slice
}
```

---

## Output Strategy Pattern (Decision A)

```cpp
// Compile-time selection based on USE_ATOMICS

template <bool UseAtomics>
__device__ void write_output(bf16* out, float* accum, float val, uint32_t idx, float gate);

template <>
__device__ void write_output<false>(bf16* out, float* /*accum*/, float val, uint32_t idx, float gate) {
    // No output overlap: direct write
    out[idx] = __float2bfloat16(val * gate);
}

template <>
__device__ void write_output<true>(bf16* /*out*/, float* accum, float val, uint32_t idx, float gate) {
    // Output overlap: accumulate into FP32 scratchpad
    atomicAdd(&accum[idx], val * gate);
}
```

---

## Python Wrapper Pattern

```python
def moe_monokernel(
    x: torch.Tensor,           # [M, K] bf16
    router_logits: torch.Tensor,
    w13: torch.Tensor,         # [E, 2N, K] fp8
    w2: torch.Tensor,          # [E, K, N] fp8
    scales_13: torch.Tensor,
    scales_2: torch.Tensor,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    M, K = x.shape
    E = router_logits.size(1)
    
    # Validate (Codex fills in actual values based on model)
    assert M <= MONOKERNEL_THRESHOLD
    assert x.dtype == torch.bfloat16
    
    output = torch.empty_like(x)
    
    if scratch is None:
        scratch = allocate_scratch(M, K, E, x.device)
    
    torch.ops.vllm.moe_monokernel(
        x, router_logits, w13, w2, scales_13, scales_2, output, scratch
    )
    return output
```

---

## Integration Pattern

```python
# In the MoE method's forward()

def forward(self, layer, x, router_logits, top_k, ...):
    M = x.size(0)
    
    # Fast path: monokernel for small batches
    if M <= MONOKERNEL_THRESHOLD and self._dims_match(x, layer):
        return moe_monokernel(
            x, router_logits,
            layer.w13_weight, layer.w2_weight,
            layer.w13_scale, layer.w2_scale,
            self._scratch,
        )
    
    # Fallback: stock fused_moe for large batches
    return fused_moe(x, ...)
```

---

## Edge Cases Codex Should Handle

### Shared Experts (DeepSeek-style)
```cpp
// Some models have a "shared expert" that processes ALL tokens
// This requires special handling in the kernel
if constexpr (HAS_SHARED_EXPERT) {
    // Process shared expert separately, then combine
}
```

### Non-Power-of-2 Expert Counts
```cpp
// E=48, E=96, etc. need careful warp assignment
// May need padding or specialized sorting
```

### Very Large K (>16384)
```cpp
// May need more aggressive K-chunking
// Consider if monokernel is even beneficial vs dense GEMM
```

### Mixed Precision
```cpp
// Some models use BF16 weights instead of FP8
// Tile sizes change (2x bytes per element)
// MMA instructions change (mma.f32.bf16.bf16 vs mma.f32.f8.f8)
```
