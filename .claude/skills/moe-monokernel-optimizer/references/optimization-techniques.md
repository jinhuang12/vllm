# Optimization Techniques Reference

## T1: Full MoE Monokernel Fusion

Single cooperative kernel that fuses:
1. Router logits → top-k selection
2. Activation quantization (FP8 dynamic scaling)
3. Up-projection GEMM (W8A8)
4. Gated activation (SiLU/SwiGLU)
5. Down-projection GEMM → BF16 output

**Applicability**:
- Best for decode (M ≤ 64), latency-critical
- Less attractive for massive prefill (M > 256)

**Decision Metric**: Fusion beneficial when `kernel_launch_overhead × num_kernels > compute_time × 0.1`. For BS 8-64, ~5μs per launch is significant vs ~50-100μs compute.

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

### U6: Monokernel Replaces (Not Adds To) Fused MoE
Wire directly into MoE method, disable cutlass/triton paths for matching dims.
