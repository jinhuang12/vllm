# Router Design Reference

## Overview

In-kernel top-k selection using warp-level primitives, avoiding separate router kernel launch.
Use only when routing must be fused. For split-kernel paths, run a standalone router and pass top-k ids/weights into GEMM.

## Router Semantics Checklist (before reordering)

Confirm these from the model code:
- Are routing weights renormalized after top‑k? (`norm_topk_prob`)
- Are weights applied before or after activation?
- Are weights multiplied into activation scales?
- Is stable token ordering required for correctness?
- Are shared experts always active?

Do not infer any of these from `top_k` alone.

## Strategy Selection Matrix

| Batch Size | Expert Count | Strategy | Implementation |
|------------|--------------|----------|----------------|
| BS ≤ 8 | E ≤ 16 | `top1_BS8_E16` | 1 warp/token, scalar loop |
| BS ≤ 8 | E ≤ 128 | `top1_BS8_E128` | 1 warp/token, thread subsets |
| BS ≤ 64 | E ≤ 16 | `top1_BS64_E16` | Per-thread loops |
| BS ≤ 64 | E ≤ 128 | `top1_BS64_E128` | Per-thread loops + warp reduce |
| Any | E > 128 | Hierarchical | Tile into 128-chunks, reduce |

## Implementation Patterns

### Pattern 1: Scalar Loop (BS ≤ 8, E ≤ 16)

```cpp
template<typename Dims>
__device__ void top1_BS8_E16(
    __nv_bfloat16 const* router_logits,  // [BS, E]
    uint8_t* topk_ids,                    // [BS]
    float* topk_weights                   // [BS]
) {
    constexpr uint32_t E = Dims::NUM_EXPERTS;
    unsigned lane = threadIdx.x % 32;
    unsigned warp = threadIdx.x / 32;
    
    if (warp < Dims::BS) {
        unsigned tokidx = warp;
        float max_value = -INFINITY;
        uint32_t max_index = 0;
        
        // Branchless max selection
        for (uint32_t idx = 0; idx < E; idx++) {
            float value = (float)router_logits[tokidx * E + idx];
            int is_new = max_value < value;
            max_value = fmaxf(max_value, value);
            max_index = max_index * (1 - is_new) + idx * is_new;
        }
        
        if (lane == 0) {
            topk_ids[tokidx] = max_index;
            topk_weights[tokidx] = 1.0f / (1.0f + expf(-max_value));  // sigmoid
        }
    }
}
```

### Pattern 2: Thread-Parallel (BS ≤ 8, E ≤ 128)

```cpp
template<typename Dims>
__device__ void top1_BS8_E128(
    __nv_bfloat16 const* router_logits,
    uint8_t* topk_ids,
    float* topk_weights
) {
    constexpr uint32_t E = Dims::NUM_EXPERTS;
    unsigned lane = threadIdx.x % 32;
    unsigned warp = threadIdx.x / 32;
    
    if (warp < Dims::BS) {
        unsigned tokidx = warp;
        float max_value = -INFINITY;
        uint32_t max_index = 0;
        
        // Each thread handles E/32 experts
        constexpr uint32_t EXPERTS_PER_THREAD = (E + 31) / 32;
        
        for (uint32_t i = 0; i < EXPERTS_PER_THREAD; i++) {
            uint32_t idx = lane + i * 32;
            if (idx < E) {
                float value = (float)router_logits[tokidx * E + idx];
                if (value > max_value) {
                    max_value = value;
                    max_index = idx;
                }
            }
        }
        
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            float other_val = __shfl_xor_sync(0xffffffff, max_value, offset);
            uint32_t other_idx = __shfl_xor_sync(0xffffffff, max_index, offset);
            if (other_val > max_value) {
                max_value = other_val;
                max_index = other_idx;
            }
        }
        
        if (lane == 0) {
            topk_ids[tokidx] = max_index;
            topk_weights[tokidx] = 1.0f / (1.0f + expf(-max_value));
        }
    }
}
```

### Pattern 3: Multi-Token (BS ≤ 64)

```cpp
template<typename Dims>
__device__ void top1_BS64(
    __nv_bfloat16 const* router_logits,
    uint8_t* topk_ids,
    float* topk_weights
) {
    constexpr uint32_t BS = Dims::BS;
    constexpr uint32_t E = Dims::NUM_EXPERTS;
    
    unsigned tid = threadIdx.x;
    unsigned num_threads = blockDim.x;
    
    // Each thread handles multiple tokens
    for (unsigned tokidx = tid; tokidx < BS; tokidx += num_threads) {
        float max_value = -INFINITY;
        uint32_t max_index = 0;
        
        for (uint32_t idx = 0; idx < E; idx++) {
            float value = (float)router_logits[tokidx * E + idx];
            int is_new = max_value < value;
            max_value = fmaxf(max_value, value);
            max_index = max_index * (1 - is_new) + idx * is_new;
        }
        
        topk_ids[tokidx] = max_index;
        topk_weights[tokidx] = 1.0f / (1.0f + expf(-max_value));
    }
}
```

## Top-K Extension (k > 1)

For top-k with k=2 or k=4, maintain k best values per thread:

```cpp
template<typename Dims, int K>
__device__ void topk_selection(
    __nv_bfloat16 const* router_logits,
    uint8_t* topk_ids,      // [BS, K]
    float* topk_weights     // [BS, K]
) {
    // ... similar structure but track K best values
    float top_vals[K];
    uint32_t top_idxs[K];
    
    // Initialize to -inf
    for (int i = 0; i < K; i++) {
        top_vals[i] = -INFINITY;
        top_idxs[i] = 0;
    }
    
    // Insertion sort during scan
    for (uint32_t idx = 0; idx < E; idx++) {
        float value = (float)router_logits[tokidx * E + idx];
        // Insert if larger than smallest in top-k
        if (value > top_vals[K-1]) {
            int insert_pos = K - 1;
            while (insert_pos > 0 && value > top_vals[insert_pos-1]) {
                top_vals[insert_pos] = top_vals[insert_pos-1];
                top_idxs[insert_pos] = top_idxs[insert_pos-1];
                insert_pos--;
            }
            top_vals[insert_pos] = value;
            top_idxs[insert_pos] = idx;
        }
    }
    
    // Apply softmax over top-k
    float sum_exp = 0.0f;
    for (int i = 0; i < K; i++) {
        sum_exp += expf(top_vals[i]);
    }
    for (int i = 0; i < K; i++) {
        topk_ids[tokidx * K + i] = top_idxs[i];
        topk_weights[tokidx * K + i] = expf(top_vals[i]) / sum_exp;
    }
}
```

## Warp Reduction Helpers

```cpp
__device__ static inline float warp_reduce_max_float(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ static inline uint32_t warp_reduce_max_with_index(
    float& val, uint32_t idx
) {
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_val = __shfl_xor_sync(0xffffffff, val, offset);
        uint32_t other_idx = __shfl_xor_sync(0xffffffff, idx, offset);
        if (other_val > val) {
            val = other_val;
            idx = other_idx;
        }
    }
    return idx;
}
```

## Performance Tuning

**Target**: Router phase ≤ 5-10% of total kernel time.

**Profiling**: Check with Nsight Compute:
- Instruction count in router section
- Register pressure (watch for spills)
- Divergence metrics

**If Router Dominates**:
1. Consider partial fusion: compute `x @ W_router` in separate GEMM
2. Keep only top-k selection + combine inside monokernel
3. For k > 2 with large E: use hierarchical approach
