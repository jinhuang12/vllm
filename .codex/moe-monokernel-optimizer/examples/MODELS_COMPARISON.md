# MoE Monokernel Model Comparison

This document compares implementation patterns across different MoE models to help generalize the monokernel approach. Each section includes detailed code snippets from reference implementations.

## Reference Implementations

| Model | Location | Hardware | Key Characteristics |
|-------|----------|----------|---------------------|
| **Llama 4** | `assets/LLAMA4_MONOKERNEL_PATCH.md` | H100 (sm_90a) | top_k=1, per-tensor scales, BF16 activations |
| **Qwen3-Coder-30B-A3B** | `csrc/moe/moe_monokernel/` | L40S (sm_89) | top_k=8, 128x128 block scales, FP8 W8A8 |

## Architectural Decisions Comparison

**Context (reference runs)**:

| Model | E_local | EP Enabled | Notes |
|-------|---------|------------|-------|
| Llama 4 | 16 | No | TP=8 reference |
| Qwen3 | 128 | No | TP=1 reference |

| Decision | Llama 4 (top_k=1) | Qwen3 (top_k=8) | When to Use |
|----------|------------------|-----------------|-------------|
| **Routing** | Single-expert selection | Multi-expert with softmax | top_k determines approach |
| **Accumulation** | Direct write to output | FP32 scratchpad + accumulate (expert-major) | top_k > 1 requires accumulation, atomics only if output overlaps |
| **Weight Application** | Before activation (fold into scale) | After activation | Based on model semantics |
| **Scale Layout** | Per-tensor: `[E, K]` | Per-block: `[E, K/128, N/128]` | Model quantization format |
| **Ownership (recommended)** | Token‑major | Hybrid or expert‑major | Based on M_avg and EP |
| **Atomics required?** | No | Only if output overlaps | Token‑major avoids atomics |

---

## Pattern 1: Top-K Routing Implementation

### Llama 4: Single Expert Selection (top_k=1)

```cuda
// From LLAMA4_MONOKERNEL_PATCH.md - moe_routing.cu
// Each warp selects the single best expert for one token
template <typename Dims>
__device__ static void top1_BS64(
    const __nv_bfloat16* __restrict__ router_logits,
    uint32_t batch_size,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    const uint32_t lane = get_thread<Dims>();
    const uint32_t warp = get_calc_warp<Dims>();

    // One token per warp iteration
    for (uint32_t tok = warp; tok < batch_size; tok += CALC_WARP_COUNT) {
        float best_val = -FLT_MAX;
        uint32_t best_idx = 0;

        // Each lane loads multiple logits (E/32 per lane)
        for (uint32_t i = lane; i < Dims::NUM_EXPERTS; i += 32) {
            float val = (float)router_logits[tok * Dims::NUM_EXPERTS + i];
            if (val > best_val) {
                best_val = val;
                best_idx = i;
            }
        }

        // Warp reduction to find global max
        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_val = __shfl_down_sync(0xFFFFFFFF, best_val, offset);
            uint32_t other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
            if (other_val > best_val) {
                best_val = other_val;
                best_idx = other_idx;
            }
        }

        // Lane 0 stores result - top_k=1 often skips softmax
        // NOTE: Some models still apply non‑unit weights even for top_k=1.
        if (lane == 0) {
            shmem->topk_ids[tok] = best_idx;
            shmem->topk_weights[tok] = 1.0f;  // Weight is always 1.0
        }
    }
}
```

### Qwen3: Multi-Expert Selection (top_k=8)

```cuda
// From csrc/moe/moe_monokernel/src/moe_routing.cu
// Each warp selects top-8 experts with softmax normalization
template <typename Dims>
__device__ static void top8_warp_parallel(
    const __nv_bfloat16* __restrict__ router_logits,
    uint32_t num_tokens,
    MoE_SHM<Dims>* __restrict__ shmem)
{
    static_assert(Dims::TOP_K == 8, "This function is only for top-k=8 routing");
    constexpr uint32_t TOP_K = Dims::TOP_K;
    constexpr uint32_t VALUES_PER_LANE = Dims::NUM_EXPERTS / 32;  // 4 for E=128

    const uint32_t lane = get_thread<Dims>();
    const uint32_t warp = get_any_warp<Dims>();

    for (uint32_t tokidx = warp; tokidx < num_tokens; tokidx += TOTAL_WARP_COUNT) {
        float local_vals[VALUES_PER_LANE];
        uint32_t local_idxs[VALUES_PER_LANE];

        // Load 4 logits per lane (128 total across warp)
        for (uint32_t i = 0; i < VALUES_PER_LANE; ++i) {
            uint32_t expert = lane * VALUES_PER_LANE + i;
            local_idxs[i] = expert;
            local_vals[i] = (float)router_logits[tokidx * Dims::NUM_EXPERTS + expert];
        }

        float topk_vals[TOP_K];
        uint32_t topk_idxs[TOP_K];

        // Iteratively select top-8 using warp reduction
        for (uint32_t k = 0; k < TOP_K; ++k) {
            // Find best in lane
            float best = local_vals[0];
            uint32_t best_idx = local_idxs[0];
            for (uint32_t i = 1; i < VALUES_PER_LANE; ++i) {
                if (local_vals[i] > best) {
                    best = local_vals[i];
                    best_idx = local_idxs[i];
                }
            }

            // Warp reduce to global best
            for (int offset = 16; offset > 0; offset >>= 1) {
                float other = __shfl_down_sync(0xFFFFFFFF, best, offset);
                uint32_t other_idx = __shfl_down_sync(0xFFFFFFFF, best_idx, offset);
                if (other > best) { best = other; best_idx = other_idx; }
            }

            float global_best = __shfl_sync(0xFFFFFFFF, best, 0);
            uint32_t global_idx = __shfl_sync(0xFFFFFFFF, best_idx, 0);

            if (lane == 0) {
                topk_vals[k] = global_best;
                topk_idxs[k] = global_idx;
            }

            // Remove selected expert from candidates
            for (uint32_t i = 0; i < VALUES_PER_LANE; ++i) {
                if (local_idxs[i] == global_idx) local_vals[i] = -FLT_MAX;
            }
        }

        // CRITICAL: Apply softmax to get normalized weights
        if (lane == 0) {
            float weights[TOP_K];
            softmax_topk<Dims>(topk_vals, weights);  // Softmax over top-8

            for (uint32_t k = 0; k < TOP_K; ++k) {
                shmem->topk_ids[tokidx * TOP_K + k] = topk_idxs[k];
                shmem->topk_weights[tokidx * TOP_K + k] = weights[k];
            }
        }
    }
}
```

**Key Difference**: top_k=1 often skips softmax in some models, while top_k>1 typically uses softmax normalization over selected experts. Always verify model semantics.

---

## Pattern 2: Output Accumulation Strategy

### Llama 4: Direct Write (top_k=1)

```cuda
// From LLAMA4_MONOKERNEL_PATCH.md - moe_down_projection.cu
// Direct write to output - no atomic needed since each token has exactly one expert
template <typename Dims>
__device__ inline void moe_down_reduction(
    MoE_SHM<Dims>* shm,
    R_element* __restrict__ activations_out,
    uint32_t tok0, uint32_t tok1,
    bool store_row0, bool store_row1)
{
    const uint32_t thread = get_thread<Dims>();
    const uint32_t k_idx = blockIdx.x * W_DOWN_TILE + (thread / 4);

    // Sum partial results from all calc warps
    float d0 = 0.f, d1 = 0.f;
    for (uint32_t w = 0; w < CALC_WARP_COUNT; ++w) {
        d0 += shm->partial_result[w][thread + 0];
        d1 += shm->partial_result[w][thread + 32];
    }

    // DIRECT WRITE - no atomicAdd needed!
    if (store_row0 && k_idx < Dims::HIDDEN_STATES) {
        activations_out[tok0 * Dims::HIDDEN_STATES + k_idx] = __float2bfloat16(d0);
    }
    if (store_row1 && k_idx < Dims::HIDDEN_STATES) {
        activations_out[tok1 * Dims::HIDDEN_STATES + k_idx] = __float2bfloat16(d1);
    }
}
```

### Qwen3: FP32 Accumulation (top_k=8, expert-major)

```cuda
// From csrc/moe/moe_monokernel/src/moe_down_projection.cu
// Must accumulate contributions from 8 experts per token
template <typename Dims>
__device__ static void moe_down_accumulate_tc(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const std::uint16_t* tokens,
    std::uint32_t k_offset,
    std::uint32_t num_valid_tokens)
{
    auto* partial_result = shm->u.gemm2.partial_result;
    const uint32_t thread = get_thread<Dims>();
    const uint32_t warp = get_calc_warp<Dims>();

    if (warp == 0) {
        uint32_t token_row0 = (thread % 4) * 2;
        uint32_t token_row1 = (thread % 4) * 2 + 1;
        uint32_t k_local = thread / 4;

        // Get TOKEN index (not pair index) for accumulation
        uint16_t pair_idx0 = tokens[token_row0];
        uint16_t token_idx0 = pair_idx0 / Dims::TOP_K;  // Convert pair -> token

        for (uint32_t w_row = 0; w_row < W_DOWN_TILE; w_row += W_DOWN_MMA_TILE) {
            // Sum partial results from all calc warps
            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
            for (uint32_t i = 0; i < CALC_WARP_COUNT; ++i) {
                d0 += partial_result[i][thread + 0];
                d1 += partial_result[i][thread + 32];
                d2 += partial_result[i][thread + 64];
                d3 += partial_result[i][thread + 96];
            }

            uint32_t k_idx0 = k_offset + w_row + k_local;

            // ACCUMULATE to FP32 buffer (NOT direct write!)
            // Multiple experts contribute to same token output
            if (row0_valid && k_idx0 < Dims::HIDDEN_STATES) {
                // Note: Using += for accumulation across experts
                scratchpad->output_accum[token_idx0 * Dims::HIDDEN_STATES + k_idx0] += d0;
            }
        }
    }
}
// Token-major alternative: each block owns token + K-slice and writes directly (no atomics).

// Final conversion from FP32 accumulator to BF16 output
template <typename Dims>
__device__ void convert_output_fp32_to_bf16(
    MoEGemmSpec<Dims>* scratchpad,
    R_element* __restrict__ output,
    uint32_t num_tokens)
{
    // After all experts processed, convert accumulated FP32 to BF16
    for (uint32_t tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
        for (uint32_t k = threadIdx.x; k < Dims::HIDDEN_STATES; k += blockDim.x) {
            float val = scratchpad->output_accum[tok * Dims::HIDDEN_STATES + k];
            output[tok * Dims::HIDDEN_STATES + k] = __float2bfloat16(val);
        }
    }
}
```

**Key Difference**: when output overlaps (expert‑major or split‑H), use an FP32 accumulator because BF16 atomicAdd is unreliable. The `pair_idx / TOP_K` conversion maps expert‑pair indices back to token indices for correct accumulation.

---

## Pattern 3: Block Quantization Scale Loading

### Llama 4: Per-Tensor/Per-Channel Scales

```cuda
// From LLAMA4_MONOKERNEL_PATCH.md
// Scale shape: [E, K] - one scale per K row
template <typename Dims>
__device__ inline void moe_request_down_expert(
    const W_element* expert_weights_down,
    const S_element* expert_scales_down,  // Shape: [E, K]
    uint32_t expert_id,
    MoE_SHM<Dims>* shm,
    uint32_t w_index,
    cuda::pipeline<cuda::thread_scope_thread>& pipe)
{
    const uint32_t base_row = blockIdx.x * W_DOWN_TILE;
    const uint32_t lane = get_thread<Dims>();

    // Load W_DOWN_TILE=40 scales (one per K row)
    if (lane < W_DOWN_TILE) {
        uint32_t global_row = base_row + lane;
        // Simple 1D indexing: expert_id * K + row
        shm->scale[w_index][lane] = expert_scales_down[expert_id * Dims::K + global_row];
    }
}

// Apply scale during GEMM - one scale per row
template <typename Dims>
__device__ inline void moe_down_mult(...)
{
    // ... MMA computation ...

    // Apply per-row scale
    float s0 = scale[w_row + thread / 4];
    float s1 = scale[far_row + thread / 4];

    d0 *= s0;  // Same scale for all columns in this row
    d1 *= s0;
    d2 *= s1;
    d3 *= s1;
}
```

### Qwen3: 128x128 Block Quantization

```cuda
// From csrc/moe/moe_monokernel/src/moe_down_projection.cu
// Scale shape: [E, K_blocks, N_blocks] = [128, 16, 6] for Qwen3
template <typename Dims>
__device__ static void moe_request_down_expert(
    const W_element* expert_weights_down,
    const S_element* expert_scales_down,  // Shape: [E, K/128, N/128]
    MoE_SHM<Dims>* shm,
    cuda::pipeline<cuda::thread_scope_thread>& pipeline,
    uint32_t expert_id,
    uint32_t k_offset,
    uint32_t w_buffer_idx)
{
    auto* scale_dest = w_buffer_idx == 0 ? shm->u.gemm2.scale_g0 : shm->u.gemm2.scale_g1;

    if constexpr (Dims::USE_BLOCK_QUANT) {
        // Block quantization: load all N-block scales for current K-block
        // k_offset / BLOCK_SIZE_QUANT gives which K-block we're processing
        uint32_t k_block = k_offset / Dims::BLOCK_SIZE_QUANT;

        // Load DOWN_SCALE_N_BLOCKS=6 scales for this K-block
        if (warp == 0 && lane < Dims::DOWN_SCALE_N_BLOCKS) {
            // 3D indexing: [expert][k_block][n_block]
            scale_dest[lane] = expert_scales_down[
                expert_id * Dims::DOWN_SCALE_K_BLOCKS * Dims::DOWN_SCALE_N_BLOCKS
                + k_block * Dims::DOWN_SCALE_N_BLOCKS
                + lane
            ];
        }
    }
}

// Apply scale during GEMM - varies by N-block
template <typename Dims>
__device__ static void moe_down_gemm_tile_tc(MoE_SHM<Dims>* shm, uint32_t buffer_idx)
{
    auto& scale = shm->u.gemm2.scale(buffer_idx);

    for (uint32_t base_col = warp * K_TILE; base_col < Dims::N; base_col += BLOCK_STRIDE) {
        // Load weights and activations...

        if constexpr (Dims::USE_BLOCK_QUANT) {
            // Compute MMA into temporaries
            float t0, t1, t2, t3;
            mma_fp8_fp8(t0, t1, t2, t3, w0, w1, w2, w3, a02, a13, 0.f, 0.f, 0.f, 0.f);

            // Get scale for THIS N-block (varies across N dimension)
            uint32_t n_block = base_col / Dims::BLOCK_SIZE_QUANT;
            float block_scale = scale[n_block];

            // Apply N-block-specific scale
            d0 += t0 * block_scale;
            d1 += t1 * block_scale;
            d2 += t2 * block_scale;
            d3 += t3 * block_scale;
        }
    }
}
```

**Key Difference**: Block quantization requires 2D scale indexing `[k_block][n_block]` and scale application during MMA accumulation (not after), since different N-regions have different scales.

---

## Pattern 4: MMA Instruction Selection

### Llama 4: FP8 Weights, TF32 Activations

```cuda
// From LLAMA4_MONOKERNEL_PATCH.md - ptx_utils.h
// Down-projection uses TF32 for activations (intermediate is FP32)
__device__ static inline void mma_fp8_tf32(
    float& d0, float& d1, float& d2, float& d3,
    __nv_fp8x4_e4m3 const& a0, ...,  // FP8 weights
    float4 const& b0, float4 const& b1,  // TF32 activations (from up-proj)
    float const& c0, ...)
{
    asm volatile(
        // Convert FP8 weights to FP16, then to TF32
        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
        // ... conversion chain ...
        "cvt.f32.f16 w0, h0;\n"  // Final TF32

        // TF32 MMA: mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{t0, t1, t2, t3}, "
        "{w0, w4, w8, w12}, "  // TF32 weights (converted from FP8)
        "{%8, %12}, "          // TF32 activations
        "{%16, %17, %18, %19};\n"
        // ... 4 chained MMAs for K=32 ...
    );
}
```

### Qwen3: FP8 Weights, FP8 Activations (W8A8)

```cuda
// From csrc/moe/moe_monokernel/src/ptx_utils.h
// Both weights and activations are FP8 (true W8A8)
__device__ static inline void mma_fp8_fp8(
    float& d0, float& d1, float& d2, float& d3,
    __nv_fp8x4_e4m3 const& a0, ...,  // FP8 weights
    __nv_fp8x4_e4m3 const& b02, __nv_fp8x4_e4m3 const& b13,  // FP8 activations
    float const& c0, ...)
{
    asm volatile(
        // Convert both FP8 operands to FP16
        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"  // Weights FP8 -> FP16
        "cvt.rn.f16x2.e4m3x2 b0, bh0;\n"   // Activations FP8 -> FP16

        // FP16 MMA: mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{t0, t1, t2, t3}, "
        "{al0, al1, al2, al3}, "  // FP16 weights
        "{b0, b1}, "              // FP16 activations
        "{%10, %11, %12, %13};\n"
        // Two chained MMAs for K=32
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{ah0, ah1, ah2, ah3}, "
        "{b2, b3}, "
        "{t0, t1, t2, t3};\n"
    );
}
```

**Key Difference**:
- Llama 4 uses `mma.f32.tf32.tf32` (4 chained k8 MMAs) because intermediate activations are FP32
- Qwen3 uses `mma.f32.f16.f16` (2 chained k16 MMAs) with FP8->FP16 conversion for true W8A8

---

## Pattern 5: Split-H Optimization (Qwen3 Only)

For very small batch sizes (BS <= 4), a single block per K-slice leaves SMs underutilized. Split-H uses multiple blocks per (token, expert) pair.

```cuda
// From csrc/moe/moe_monokernel/src/moe_interface.h
struct KernelConfig {
    static constexpr uint32_t SM_COUNT = 142;  // L40S
    static constexpr uint32_t SPLIT_H_THRESHOLD = 4;
    static constexpr uint32_t MAX_SPLIT_FACTOR = 16;

    // Calculate split factor for BS <= 4
    __host__ __device__ static constexpr uint32_t get_split_factor(uint32_t batch_size) {
        if (batch_size > SPLIT_H_THRESHOLD) return 1;

        uint32_t total_pairs = batch_size * TOP_K;  // e.g., 4 * 8 = 32
        uint32_t target_blocks = (SM_COUNT * 8) / 10;  // 80% SM utilization
        uint32_t split = (target_blocks + total_pairs - 1) / total_pairs;
        return split < MAX_SPLIT_FACTOR ? split : MAX_SPLIT_FACTOR;
    }

    // Dynamic grid size
    __host__ __device__ static constexpr uint32_t get_grid_size(uint32_t batch_size) {
        if (batch_size <= SPLIT_H_THRESHOLD) {
            // Split-H: multiple blocks per (token, expert) pair
            return batch_size * TOP_K * get_split_factor(batch_size);
        } else {
            return STANDARD_GRID_SIZE;  // K / 16
        }
    }
};
```

**When to Use**: When `batch_size * top_k / SM_COUNT < 0.8` (underutilization detected).

---

## Future Model Patterns

### DeepSeek: Shared Experts (Preview)

DeepSeek has dedicated shared experts that bypass routing. This pattern will be documented when implemented.

```cuda
// Pattern preview - NOT YET IMPLEMENTED
// DeepSeek has N routed experts + M shared experts
// Shared experts process ALL tokens, routed experts process top_k
template <typename Dims>
__device__ void moe_with_shared_experts(...)
{
    // Phase 1: Process shared experts (all tokens)
    for (uint32_t shared_idx = 0; shared_idx < Dims::NUM_SHARED_EXPERTS; ++shared_idx) {
        process_expert_for_all_tokens(shared_experts[shared_idx], ...);
    }

    // Phase 2: Top-k routing for routed experts
    topk_route<Dims>(router_logits, num_tokens, shmem);

    // Phase 3: Process routed experts
    // ... standard monokernel flow ...
}
```

---

## When to Add a New Model Example

Add a new reference implementation to this document when the model has:

1. **New architectural pattern** not covered by existing examples:
   - Different expert arrangement (e.g., shared + routed)
   - Different activation function (e.g., GeGLU instead of SiLU)
   - Different routing mechanism (e.g., soft routing)

2. **New quantization scheme** requiring different scale handling:
   - Different block sizes (e.g., 64x64 vs 128x128)
   - Different scale precision (e.g., FP16 scales instead of FP32)
   - Per-group quantization

3. **New hardware constraints** affecting kernel design:
   - Different SM architecture (e.g., sm_100 Blackwell)
   - Significantly different SMEM budget
   - New instructions (e.g., TMA for Hopper)

**Do NOT add for**:
- Same architecture with different model size (e.g., Qwen3-14B vs Qwen3-30B)
- Same quantization with different TP configuration
- Minor hyperparameter differences (e.g., different hidden_size)
