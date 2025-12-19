# Optimization Techniques Reference

Techniques in this file apply to **specific patterns**. Always choose ownership and fusion boundary first.

| Pattern | Best‑fit Techniques |
|---------|---------------------|
| Token‑major | T1 (conditional), T7, T8, T11, T12 |
| Expert‑major | T1, T5, T6, T7, T8, T11, T13 |
| Cooperative | U5, T3, T6, T7 |
| Split‑kernel | T5, T6, T7 (routing/prepare), T11 |

## Wrapper Guidance (Scaffolding)
Prefer Python‑level wrappers until a C++ wrapper introduces new behavior, reduces overhead, or unlocks graph capture.
Do not add C++ wrappers that only forward to existing ops unless required for integration.

## P0: Baseline Profiling (Decision Input)

Before choosing fusion, collect a reference per‑kernel breakdown under CUDA graphs / torch.compile:
- Capture routing + experts inside a **single CUDA graph** to match production.
- Compute delta‑to‑baseline requirements; if required savings are implausible, prefer split‑kernel or document the limitation.
Full monokernel is justified only when routing/prepare share is large enough to cover the delta (roughly ≥15–20%).
- If reference GEMMs already dominate and are near‑optimal, fusion gains are limited.
- If routing/prepare/quantization dominates, fusion or split‑kernel coalescing can help.

## T1: Full MoE Monokernel Fusion (Conditional)

Single cooperative kernel that fuses:
1. Router logits → top-k selection
2. Activation quantization (FP8 dynamic scaling)
3. Up-projection GEMM (W8A8)
4. Gated activation (SiLU/SwiGLU)
5. Down-projection GEMM → BF16 output

**Applicability**:
- Best for decode (M ≤ 64) **and** low M_avg with token-major ownership
- Avoid when M_avg is high or ownership is expert-major (prefer split-kernel / grouped GEMM)
- Less attractive for massive prefill (M > 256)

**Decision Metric**: Fusion beneficial when `kernel_launch_overhead × num_kernels > compute_time × 0.1` **and** `M_avg < M_AVG_THRESHOLD`. For BS 8-64, ~5μs per launch is significant vs ~50-100μs compute.

## T2: Template Specialization + Python Dispatch

```cpp
using Dims_BS8_E16_TP8 = MoEDimensions<8, 1024, 5120, 16>;
using Dims_BS64_E16_TP8 = MoEDimensions<64, 1024, 5120, 16>;

MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS8_E16_TP8_impl, Dims_BS8_E16_TP8)
MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS64_E16_TP8_impl, Dims_BS64_E16_TP8)
```

**How to Choose Specializations**:
1. Sample real decode workloads → histogram of M per MoE call
2. Pick 2-3 representative caps: BS8, BS32, BS64
3. For each (BS, E_local): instantiate `Dims_...` and wrapper
4. Python dispatch via if/else on `(M, E, TP)`

## T3: In-Kernel Router Selection

Use only when routing must be fused. If routing is done outside the kernel, skip T3 and pass top-k ids/weights in as inputs.

**Strategy by (M, E_local)**:

| Case | Strategy |
|------|----------|
| M ≤ 8, E ≤ 16 | One warp/token, scalar loop |
| M ≤ 8, E ≤ 128 | One warp/token, thread subset per expert |
| M ≤ 64, E ≤ 128 | Per-thread loops with warp reduction |
| E > 128 | Hierarchical: tile into 128-expert chunks |

**Performance Target**: Router ≤ 5-10% of total kernel time.

## T4: Per-Token FP8 Activation Quantization

```cpp
// For each token i:
float max_i = max_j |x_i[j]|;
float scale_i = max(max_i, eps) / FP8_MAX;  // FP8_MAX ≈ 448
float inv_scale_i = FP8_MAX / max(max_i, eps);

// Fold scale into gate weight
topk_weight_scaled_i = topk_weight_i * scale_i;
```

**Hardware Check**: Requires FP8 Tensor Core + conversion (Hopper, Ada). For A100/V100: use BF16 GEMM path.

## T5: Expert-Local Token Sorting

**BS ≤ 8 (Bitfield)**:
```cpp
uint64_t expert_mask;  // 8 bytes × 8 tokens
uint64_t expert_ids;   // unique experts packed
uint32_t expert_count = __popc(bitset);
```

**BS ≤ 64 (Histogram)**:
```cpp
__shared__ uint32_t counters[E_local][warp_size];
// Per-warp counting → prefix sum → token_indexes permutation
```

## T6: Tensor Core Tiling + Multi-Buffering

**SMEM Budget Formula**:
```
SMEM_block_max ≈ SMEM_SM / B_per_SM
// H100: 228KB, target 1 CTA/SM → ~228KB available
```

**Up-Projection (Triple Buffer)**:
```
bytes_A = 3 × M_tile × (K/2) × sizeof_fp8
bytes_W = 3 × N_tile × (K/2 + pad) × sizeof_fp8
bytes_partial = CALC_WARPS × N_tile × M_tile × sizeof_f32
```

**Down-Projection (Double Buffer)**:
```
bytes_T = 2 × T_TILE × N × sizeof_f32
bytes_Wd = 2 × W_DOWN_TILE × (N + pad) × sizeof_fp8
```

**Tile Size Search**:
- M_tile ∈ {8, 16} (matches m16n8k16 MMA)
- N_tile ∈ {16, 32, 64}
- Prefer largest N_tile that fits SMEM constraint

## T7: Warp Role Partitioning

```cpp
static constexpr uint32_t TOTAL_WARP_COUNT = 12;  // 384 threads
static constexpr uint32_t CALC_WARP_COUNT = 8;    // Warps 0-7: MMA
static constexpr uint32_t PREFETCH_WARP_COUNT = 4; // Warps 8-11: async copy
```

**Why 8:4?** H100 has 4 Tensor Cores/SM. 8 warps saturate TCs, 4 warps keep memory pipeline full.

**Sync Only Calc Warps**:
```cpp
__asm volatile("bar.sync 15, 256;\n");  // Named barrier, 256 threads
```

## T8: Tiny vs Normal Path Specialization

```cpp
if constexpr (Dims::BS <= 8) {
    moe_kernel_BS8(...);  // Everything in registers/SMEM
} else {
    moe_kernel_BS64(...); // Full sorting, triple buffer
}
```

**Tiny Path Characteristics**:
- 1:1 warp-token mapping
- Bitfield expert encoding
- Single-pass weight loading
- No global scratch for routing

## T9: Global Scratchpad Design

```cpp
template <typename Dims>
struct MoEGemmSpec {
    AQ_element activations[BS][K];      // Quantized FP8
    T_element temp[(BS+8) * N];         // Up-projection output FP32
    float topk_weights_scaled[BS];      // Scale-folded gate weights
};
```

**Sizing Formula**:
```
bytes ≈ BS_max × K × 1 + (BS_max + 8) × N × 4 + BS_max × 4
```

## Unique Insights (U1-U6)

### U1: Bit-Packing Expert IDs (BS ≤ 8)
Pack 8 expert IDs into `uint64_t`, use `__popc` for count, `__ffs` for iteration.

### U2: 32-Byte Column Swizzling
```cpp
uint32_t rotate_col_32(uint32_t col, uint32_t row) {
    uint32_t col_base = col & 0xff9f;
    uint32_t col_rot = (col + 0x20 * row) & 0x60;
    return col_base | col_rot;
}
```
Eliminates bank conflicts for columnar reads.

### U3: Fused Gate + Activation in GEMM Reduction
```cpp
// After MMA, fuse SiLU
float x0 = d0 * ts0 * ws0;
float w0 = d2 * ts0 * ws1;
float sig0 = (w0 * x0) / (1 + expf(-x0));
```

### U4: Minimal Global/Shared Split
Only persist what's needed across phases:
- Quantized activations (up → down)
- Temp `[M,N]` (up → down)
- Everything else in SMEM/registers

### U5: Cooperative Grid Sync
```cpp
// Host: cudaLaunchCooperativeKernel(...)
// Device: cooperative_groups::this_grid().sync();  // ~1-2μs vs ~5-10μs launch
```
Requirement: Grid size ≤ SM count.

Use only when per‑expert load is balanced and barriers are amortized; otherwise prefer split kernels.

### U6: Monokernel As Fast Path
Wire monokernel as a conditional fast path and keep fused MoE as a fallback for mismatched dims or regressions.

---

## T14: Split-H for Small Batch SM Utilization (★★★★☆)

**Problem**: For very small batch sizes (BS ≤ 4), standard grid sizing (one block per K-slice) leaves SMs underutilized.

**Example** (Qwen3 on L40S):
```
BS=1, top_k=8: only 8 token-expert pairs
Standard grid: 128 blocks (K/16 slices)
But only ~8-16 pairs to process → most blocks idle
SM utilization: 128 blocks / 142 SMs → each block touches few pairs
```

**Solution**: Use Split-H to have multiple blocks collaborate per (token, expert) pair.

### Configuration

```cpp
template <typename Dims>
struct KernelConfig {
    static constexpr uint32_t SM_COUNT = 142;  // L40S
    static constexpr uint32_t SPLIT_H_THRESHOLD = 4;
    static constexpr uint32_t MAX_SPLIT_FACTOR = 16;

    // Calculate optimal split factor
    __host__ __device__ static constexpr uint32_t get_split_factor(uint32_t batch_size) {
        if (batch_size > SPLIT_H_THRESHOLD) return 1;

        uint32_t total_pairs = batch_size * Dims::TOP_K;
        uint32_t target_blocks = (SM_COUNT * 8) / 10;  // 80% utilization target
        uint32_t split = (target_blocks + total_pairs - 1) / total_pairs;
        return split < MAX_SPLIT_FACTOR ? split : MAX_SPLIT_FACTOR;
    }

    // Dynamic grid size
    __host__ __device__ static constexpr uint32_t get_grid_size(uint32_t batch_size) {
        if (batch_size <= SPLIT_H_THRESHOLD) {
            return batch_size * Dims::TOP_K * get_split_factor(batch_size);
        }
        return STANDARD_GRID_SIZE;  // K / 16
    }
};
```

### Example Calculations

| BS | top_k | Pairs | Standard Grid | Split Factor | Split-H Grid | SM Util |
|----|-------|-------|---------------|--------------|--------------|---------|
| 1 | 8 | 8 | 128 | 14 | 112 | 79% |
| 2 | 8 | 16 | 128 | 7 | 112 | 79% |
| 4 | 8 | 32 | 128 | 4 | 128 | 90% |
| 8 | 8 | 64 | 128 | 1 | 128 | 90% |

### Implementation Pattern

```cpp
template <typename Dims>
__device__ void moe_down_projection_split_h(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const W_element* expert_weights_down,
    const S_element* expert_scales_down,
    uint32_t num_tokens)
{
    uint32_t split_factor = KernelConfig::get_split_factor(num_tokens);

    if (split_factor == 1) {
        // Standard path: one block per K-slice
        moe_down_projection_standard<Dims>(...);
        return;
    }

    // Split-H: multiple blocks per (token, expert) pair
    uint32_t total_pairs = num_tokens * Dims::TOP_K;

    // Map blockIdx to (pair_idx, split_idx)
    uint32_t pair_idx = blockIdx.x / split_factor;
    uint32_t split_idx = blockIdx.x % split_factor;

    if (pair_idx >= total_pairs) return;

    // Each split handles different K-range
    uint32_t k_per_split = Dims::K / split_factor;
    uint32_t k_start = split_idx * k_per_split;
    uint32_t k_end = (split_idx == split_factor - 1) ? Dims::K : (split_idx + 1) * k_per_split;

    // Process this K-slice for this pair
    // IMPORTANT: Output overlaps across blocks; require reduction (atomic or staged)
    for (uint32_t k = k_start; k < k_end; k += W_DOWN_TILE) {
        // ... compute partial result for K-slice ...

        // Atomic or staged accumulation required
        atomicAdd(&scratchpad->output_accum[token_idx * Dims::K + k], partial_result);
    }
}
```

### When to Use Split-H

```python
def should_use_split_h(batch_size: int, top_k: int, sm_count: int) -> bool:
    """
    Split-H beneficial when standard grid underutilizes SMs.
    """
    total_pairs = batch_size * top_k
    standard_utilization = total_pairs / sm_count

    # Below 50% utilization → Split-H helps
    return standard_utilization < 0.5 and batch_size <= 4
```

### Tradeoffs

| Pro | Con |
|-----|-----|
| Better SM utilization for tiny batches | Requires reduction (atomic/staged) |
| Hides memory latency with more blocks | More complex block assignment |
| Can achieve 80%+ SM utilization at BS=1 | Additional coordination overhead |

**Recommendation**: Use as optional optimization. Default to standard grid, enable Split-H via environment variable or when profiling shows underutilization.

```cpp
// Configuration flag
static constexpr bool ENABLE_SPLIT_H = true;

// Runtime check
if constexpr (ENABLE_SPLIT_H) {
    if (batch_size <= SPLIT_H_THRESHOLD) {
        moe_down_projection_split_h<Dims>(...);
        return;
    }
}
moe_down_projection_standard<Dims>(...);
```
