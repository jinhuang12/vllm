# MoE Monokernel Optimization Guide for vLLM

## Overview

This document describes advanced CUDA kernel optimization techniques for Mixture-of-Experts (MoE) models, based on the monokernel implementation for Llama4 Scout/Maverick in vLLM 0.10.2. The patch replaces vLLM's standard multi-kernel MoE implementation with a **single cooperative kernel** that fuses routing, token preparation, input quantization, up-projection GEMM, SiLU activation, and down-projection GEMM.

**Key Insight**: For small batch sizes (≤64 tokens), kernel launch overhead and intermediate memory traffic dominate compute time.

---

## Execution Flow Comparison

### Standard vLLM (5-7 kernel launches)
```
[Routing] → GlobalMem → [Permute] → GlobalMem → [Quantize] → GlobalMem → [GEMM1] → GlobalMem → [Activation] → GlobalMem → [GEMM2]
```

### Monokernel (1 cooperative kernel)
```
[Single Kernel]
  ├─ Routing (shared memory)
  ├─ Prepare/Sort (shared memory)  
  ├─ Scale/Quantize (shared memory → scratchpad)
  ├─ Up-Projection + SiLU (shared memory)
  ├─ grid.sync()
  └─ Down-Projection (shared memory → output)
```

| Aspect | Standard vLLM | Monokernel | Impact |
|--------|---------------|------------|--------|
| Kernel launches | 5-7 | 1 | Eliminates ~15-30μs overhead |
| Global memory round-trips | 6+ | 2 | ~4x bandwidth reduction |
| Intermediate tensors | Multiple allocations | Single scratchpad | Reduced memory footprint |

**Decision Metric**: Fusion is beneficial when `kernel_launch_overhead × num_kernels > compute_time × 0.1`. For BS 8-64, the ~5μs per kernel launch is significant relative to ~50-100μs compute time.

---

## Key Dimensions

```cpp
template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts>
struct MoEDimensions {
    static constexpr uint32_t BS = m;            // Batch size (tokens)
    static constexpr uint32_t N = n;             // Intermediate dim / TP
    static constexpr uint32_t HIDDEN_STATES = k; // Hidden size
    static constexpr uint32_t NUM_EXPERTS = num_experts;
    
    struct KernelConfig {
        static constexpr uint32_t GRID_SIZE = (2*N) / 16;  // One block per output tile
        static constexpr uint32_t BLOCK_SIZE = 384;         // 12 warps × 32 threads
    };
};

// Llama4 configurations
using Dims_BS8_E16_TP8 = MoEDimensions<8, 1024, 5120, 16>;    // Scout
using Dims_BS8_E128_TP8 = MoEDimensions<8, 1024, 5120, 128>;  // Maverick
```

---

## Optimization Techniques

### 1. Batch-Size-Aware Kernel Specialization

Two distinct code paths based on batch size:

```cpp
if constexpr (Dims::BS <= 8) {
    moe_kernel_BS8(...);  // "Tiny" path
} else {
    moe_kernel_BS64(...); // "Normal" path
}
```

| Path | Characteristics |
|------|-----------------|
| **Tiny (BS≤8)** | Expert routing in 64-bit registers, single-pass weights, 1:1 warp-token mapping |
| **Normal (BS≤64)** | Full token sorting, triple-buffered tiles, K dimension split for shared memory |

**Generalizability**: Apply to any kernel with bimodal problem size distribution where small problems fit in registers/L1.

---

### 2. Warp Specialization: Compute vs. Prefetch

```cpp
// 12 warps total (384 threads)
static constexpr uint32_t CALC_WARP_COUNT = 8;      // Warps 0-7: MMA computation
static constexpr uint32_t PREFETCH_WARP_COUNT = 4;  // Warps 8-11: Async memory

if (is_prefetch_warp<Dims>()) {
    pipe.producer_acquire();
    moe_request_input_tokens<Dims>(...);
    moe_request_up_expert<Dims>(...);
    pipe.producer_commit();
} else {
    // MMA operations on current tile
    mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, ...);
}
```

**Why 8:4 ratio?** H100 has 4 Tensor Cores per SM. 8 warps saturate Tensor Cores while 4 warps keep memory pipeline full. Tune ratio based on `compute_time / memory_latency`.

**Sync only calc warps** (faster than `__syncthreads`):
```cpp
__asm volatile("bar.sync 15, 256;\n");  // Named barrier, 256 threads
```

---

### 3. Triple Buffering for Latency Hiding

```cpp
// Queue first 2 tiles, rotate through 3 buffers
std::uint32_t t_index_read = 0, t_index_write = 2;

for (/*...*/) {
    cuda::pipeline_consumer_wait_prior<1>(pipe);  // Wait for tile n-1
    __syncthreads();
    
    // Prefetch warps: request tile n+1
    // Compute warps: process tile n
    
    t_index_read = (t_index_read + 1) % 3;
    t_index_write = (t_index_write + 1) % 3;
}
```

**Why triple vs double?** Triple buffering absorbs memory latency variance from non-uniform expert access patterns and L2 cache contention.

---

### 4. Bank Conflict Mitigation

**Strategy A: 32-Byte Column Swizzling** (for fixed-size tiles)
```cpp
// Rotate column index based on row to map same column to different banks
__device__ inline uint32_t rotate_col_32(uint32_t col, uint32_t row) {
    uint32_t col_base = col & 0xff9f;
    uint32_t col_rot = (col + 0x20 * row) & 0x60;
    return col_base | col_rot;
}
```

**Strategy B: Padding** (for variable-size tiles)
```cpp
static constexpr unsigned PADDING = 32;
W_element w[TILE_ROWS][COLS + PADDING / sizeof(W_element)];
```

| Method | Pros | Cons |
|--------|------|------|
| Swizzling | No wasted memory | Complex addressing |
| Padding | Simple addressing | 3-6% memory overhead |

**Why 32-byte swizzle?** Balances bank conflict reduction, address calculation overhead, and alignment with FP8 vector loads (4 bytes × 8 = 32 bytes).

---

### 5. Custom FP8 MMA with Inline Conversion

```cpp
__device__ static inline void mma_fp8_fp8(
    float& d0, float& d1, float& d2, float& d3,
    __nv_fp8x4_e4m3 const& a0, ...,
    __nv_fp8x4_e4m3 const& b02, __nv_fp8x4_e4m3 const& b13,
    float const& c0, ...)
{
    asm volatile(
        "{"
        // Convert FP8 to FP16 in registers
        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
        // Two chained m16n8k16 MMAs for K=32
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{t0,t1,t2,t3}, {al0,al1,al2,al3}, {b0,b1}, {c0,c1,c2,c3};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {ah0,ah1,ah2,ah3}, {b2,b3}, {t0,t1,t2,t3};\n"
        "}\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : ...);
}
```

**Why custom vs cuBLAS?** Inline conversion avoids materializing FP16 intermediates, reduces register pressure, enables custom accumulation control.

---

### 6. Bitfield-Based Expert Deduplication ★★★★★

```cpp
// For ≤32 experts: encode presence in 32-bit integer
uint32_t expert_bitset = 0;
expert_bitset |= 1 << t0;  // Set bit for each token's expert
expert_bitset |= 1 << t1;
// ...
uint32_t expert_count = __popc(expert_bitset);      // Hardware popcount
uint32_t e0 = __ffs(expert_bitset) - 1;             // Find first set
expert_bitset &= expert_bitset - 1;                  // Clear lowest bit
uint32_t e1 = __ffs(expert_bitset) - 1;             // Next expert
```

**For 128 experts**: Use `__uint128_t` split into two 64-bit halves:
```cpp
__uint128_t expert_bitset = 0;
expert_bitset |= __uint128_t(1) << t0;
uint64_t b0 = expert_bitset & 0xFFFFFFFFFFFFFFFFU;
uint64_t b1 = expert_bitset >> 64;
uint32_t expert_count = __popcll(b0) + __popcll(b1);
```

| Method | Time | Space |
|--------|------|-------|
| Standard sort | O(n log n) | O(n) |
| Hash table | O(n) average | O(experts) |
| **Bitfield** | **O(n + experts)** | **O(1)** |

---

### 7. Scale Folding into Routing Weights ★★★★☆

Per-token quantization scale folded into `topk_weight` instead of stored separately:

```cpp
// During quantization
float scale = max_activation * FP8_MAX_INV;
topk_weight = topk_weight * scale;  // Fold scale into weight

// During down-projection output
// Mathematically: output = W_down @ (scaled_activation) / scale 
//                        = W_down @ activation * topk_weight
```

**Benefits**: No extra memory for scale factors, scale applied during output scaling.

---

### 8. Speculative Computation with Mask Filtering ★★★★☆

For small batches, compute all expert outputs for all tokens, filter at write:

```cpp
// Pack 8 expert IDs into 64-bit register
std::uint64_t expert_mask;  // Which expert handles each token

// During computation
for (each expert) {
    result = compute(all_tokens, expert);
    bool store = (expert_mask >> (row * 8) & 0xff) == expert_id;
    if (store) write(result);
}
```

**Trade-off**: Higher arithmetic intensity but eliminates control flow divergence.

---

### 9. Branchless Expert Selection

```cpp
for (uint32_t idx = 0; idx < NUM_EXPERTS; idx++) {
    float value = (float)router_logits[tokidx * NUM_EXPERTS + idx];
    max_value = fmaxf(max_value, value);
    int is_new = max_value == value;
    max_index = max_index * (1 - is_new) + idx * is_new;  // No branch!
}
```

**Performance Model**: Branchless wins when `divergence_probability > 3 / branch_penalty ≈ 15%`. For uniform routing across 128 experts: ~95% divergence.

---

### 10. Prefix Sum for Parallel Token Sorting

```cpp
// Prefix sum over 16 uint8 packed in uint128
__device__ static inline uint8x16_t prefix_sum_over_bytes(uint8x16_t val) {
    val += val << 8;   // Each byte += previous byte
    val += val << 16;  // Each pair += previous pair
    val += val << 32;  // Each quad += previous quad
    val += val << 64;  // Each octuple += previous octuple
    return val;
}
```

**Complexity**: O(1) parallel prefix sum + O(n/warp_size) scatter vs O(n log n) standard sort.

---

### 11. Sigmoid Fusion with GEMM Reduction

```cpp
// After MMA reduction, fuse SiLU activation
float x0 = d0 * ts0 * ws0;   // Scaled x
float w0 = d2 * ts0 * ws1;   // Scaled gate
float sig0 = (w0 * x0) / (1 + expf(-x0));  // Fused SiLU
result[row0 * N + ...] = sig0;
```

**Pattern**: Whenever activation follows GEMM, fuse in reduction phase to eliminate intermediate storage.

---

### 12. Vectorized BF16→FP8 with NaN Handling

```cpp
__device__ static __forceinline__ __nv_bfloat162 mask_NaNs_to_zero(__nv_bfloat162 xs) {
    // NaN == NaN is false by IEEE 754, so mask becomes 0 for NaN
    return type_pun<__nv_bfloat162>(type_pun<uint32_t>(xs) & __heq2_mask(xs, xs));
}
```

**Why explicit NaN handling?** FP8 has limited/no NaN representation. Explicit handling prevents undefined behavior without branching.

---

### 13. Cooperative Grid Synchronization

```cpp
// Host: Cooperative launch
cudaLaunchCooperativeKernel(moe_kernel<dims>, GRID_SIZE, BLOCK_SIZE, args, shmem, stream);

// Device: Grid-wide sync between stages
moe_up_projection<Dims>(...);
cooperative_groups::this_grid().sync();  // ~1-2μs vs ~5-10μs kernel launch
moe_down_projection<Dims>(...);
```

**Requirement**: Grid size must be ≤ number of SMs.

---

## Memory Layout

### Shared Memory (Union for Phase Reuse)

```cpp
template <typename Dims>
struct MoE_SHM {
    union U {
        struct SortData { ... } sorting;      // Phase 1: Routing
        struct RescaleData { ... } rescale;   // Phase 2: Quantization
        struct Gemm1Data {                    // Phase 3: Up-projection
            AQ_element a[3][A_TILE][K_DIM_HALF];     // Triple-buffered activations
            W_element w[3][W_TILE][K_DIM_HALF];      // Triple-buffered weights
            T_element partial_result[WARPS][...];
        } gemm1;
        struct Gemm2Data { ... } gemm2;       // Phase 4: Down-projection
        struct TinyData { ... } tiny;         // Alternative: BS≤8 path
    } u;
    
    // Persistent across phases
    uint8_t topk_ids[BS];
    uint16_t token_indexes[BS];
    float topk_weights[BS];
    ExpertRef experts[NUM_EXPERTS];
};
```

**Budget**: ~180KB of 224KB available on H100.

### Global Scratchpad

```cpp
template <typename Dims>
struct MoEGemmSpec {
    AQ_element activations[BS][HIDDEN_STATES];  // 320KB for BS=64
    T_element temp[BS * N];                      // 288KB for BS=64
    float topk_weights_scaled[BS];               // 256B
};  // Total ~608KB vs ~2.1MB standard vLLM
```

---

## Applying to New MoE Models

### Step 1: Define Dimensions
```cpp
using Dims_NewModel = MoEDimensions<
    64,     // BS (max tokens)
    2048,   // N = intermediate_size / TP
    4096,   // K = hidden_size
    32      // NUM_EXPERTS
>;
```

### Step 2: Verify Constraints
```cpp
static_assert(NUM_EXPERTS <= 128);
static_assert(BS <= 64);
static_assert(HIDDEN_STATES % 32 == 0);
static_assert(N % 16 == 0);
```

### Step 3: Calculate Shared Memory
```cpp
// Verify total < 220KB
size_t gemm1 = 3 * BS * (K/2) + 3 * W_TILE * (K/2) + 8 * W_TILE * 8 * 4;
size_t gemm2 = 2 * 8 * N * 4 + 2 * W_DOWN_TILE * N;
```

### Step 4: Instantiate Wrapper
```cpp
MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_NewModel_impl, Dims_NewModel)
```

### Step 5: Register with PyTorch
```cpp
m.def("moe_monokernel_NewModel(...) -> ()");
m.impl("moe_monokernel_NewModel", torch::kCUDA, &moe_monokernel_NewModel_impl);
```

---

## Applicability Matrix

| Technique | MoE | Attention | Conv | GEMM |
|-----------|:---:|:---------:|:----:|:----:|
| Monokernel fusion | ✓ | ✓ | ○ | ○ |
| Batch-size specialization | ✓ | ✓ | ✓ | ✓ |
| Warp specialization | ✓ | ✓ | ✓ | ✓ |
| Bank conflict mitigation | ✓ | ✓ | ✓ | ✓ |
| Triple buffering | ✓ | ✓ | ✓ | ✓ |
| Custom FP8 MMA | ✓ | ✓ | ✓ | ✓ |
| Per-token quantization | ✓ | ✓ | ○ | ○ |
| Bitfield dedup | ✓ | ○ | ○ | ○ |
| Packed expert mask | ✓ | ○ | ○ | ○ |
| Grid sync fusion | ✓ | ✓ | ○ | ○ |
| Activation fusion | ✓ | ✓ | ✓ | ✓ |

✓ = Directly applicable, ○ = May apply with modification

---

## Unique Insights (Rarity Rating)

| Technique | Rarity | Note |
|-----------|:------:|------|
| 128-bit integer expert tracking | ★★★★★ | Most implementations cap at 64 experts |
| Bitfield expert deduplication | ★★★★★ | Usually uses sorting or hash tables |
| Scale folding into routing weights | ★★★★☆ | Avoids explicit scale storage |
| Speculative compute + mask filter | ★★★★☆ | Trades compute for reduced divergence |
| Cooperative grid sync fusion | ★★★★☆ | Underutilized in production ML |
| 32-byte vs 16-byte swizzle choice | ★★★☆☆ | Empirical tuning insight |

---

## Full Implementation Reference

Full patch can be found [here](LLAMA4_MONOKERNEL_PATCH.md)