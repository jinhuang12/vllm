# Expert Grouping Reference

## Overview

After router picks experts for each token, group tokens by expert for efficient batched GEMMs.

## Strategy Selection

| Batch Size | Method | Data Structures |
|------------|--------|-----------------|
| BS ≤ 8 | Bitfield | `uint64_t expert_mask`, `uint64_t expert_ids` |
| BS ≤ 64 | Histogram | `counters[E][warps]`, `token_indexes[BS]` |
| BS > 128 | Multi-block Radix | Consider stock fused_moe |

## Method 1: Bitfield Packing (BS ≤ 8)

**Key Insight**: For tiny batches, pack all expert assignments into registers.

### Data Structures

```cpp
struct TinyExpertData {
    uint64_t expert_mask;   // Byte i = expert ID for token i (8 tokens × 8 bits)
    uint64_t expert_ids;    // Unique experts packed sequentially
    uint8_t expert_count;   // Number of unique experts
};
```

### Implementation

```cpp
template<typename Dims>
__device__ void prepare_moe_BS8(
    uint8_t const* topk_ids,     // [BS]
    TinyExpertData* data
) {
    static_assert(Dims::BS <= 8);
    
    // Pack expert IDs into 64-bit mask
    uint64_t mask = 0;
    uint32_t expert_bitset = 0;
    
    for (int i = 0; i < Dims::BS; i++) {
        uint8_t expert = topk_ids[i];
        mask |= (uint64_t)expert << (i * 8);
        expert_bitset |= (1u << expert);  // Mark expert as used
    }
    
    data->expert_mask = mask;
    data->expert_count = __popc(expert_bitset);
    
    // Extract unique experts
    uint64_t ids = 0;
    int pos = 0;
    while (expert_bitset) {
        uint32_t e = __ffs(expert_bitset) - 1;
        ids |= (uint64_t)e << (pos * 8);
        expert_bitset &= expert_bitset - 1;  // Clear lowest bit
        pos++;
    }
    data->expert_ids = ids;
}

// Filter tokens for a specific expert
__device__ uint8_t get_token_mask_for_expert(uint64_t expert_mask, uint8_t expert_id) {
    uint8_t token_mask = 0;
    for (int i = 0; i < 8; i++) {
        if (((expert_mask >> (i * 8)) & 0xff) == expert_id) {
            token_mask |= (1 << i);
        }
    }
    return token_mask;
}
```

### Usage in GEMM Loop

```cpp
// Iterate unique experts
for (int e_idx = 0; e_idx < data->expert_count; e_idx++) {
    uint8_t expert_id = (data->expert_ids >> (e_idx * 8)) & 0xff;
    uint8_t token_mask = get_token_mask_for_expert(data->expert_mask, expert_id);
    
    // Load weights for expert_id
    // Compute GEMM for tokens where token_mask bit is set
    // Write outputs only where mask matches
}
```

## Method 2: 128-Expert Bitfield Extension

For E ≤ 128, use `__uint128_t` split into two 64-bit halves:

```cpp
struct LargeExpertBitfield {
    uint64_t low;   // Experts 0-63
    uint64_t high;  // Experts 64-127
    
    __device__ void set(uint32_t expert) {
        if (expert < 64) {
            low |= (1ULL << expert);
        } else {
            high |= (1ULL << (expert - 64));
        }
    }
    
    __device__ uint32_t count() {
        return __popcll(low) + __popcll(high);
    }
    
    __device__ uint32_t pop_first() {
        if (low) {
            uint32_t e = __ffsll(low) - 1;
            low &= low - 1;
            return e;
        } else {
            uint32_t e = __ffsll(high) - 1 + 64;
            high &= high - 1;
            return e;
        }
    }
};
```

## Method 3: Histogram + Prefix Sum (BS ≤ 64)

**Key Insight**: Use shared memory histogram for larger batches, then prefix sum for offsets.

### Data Structures

```cpp
template<typename Dims>
struct ExpertRanges {
    uint16_t token_indexes[Dims::BS];      // Permutation: tokens sorted by expert
    uint8_t expert_counts[Dims::NUM_EXPERTS]; // Tokens per expert
    uint16_t expert_offsets[Dims::NUM_EXPERTS]; // Start index per expert
};
```

### Implementation

```cpp
template<typename Dims>
__device__ void prepare_moe_BS64(
    uint8_t const* topk_ids,
    ExpertRanges<Dims>* ranges
) {
    __shared__ uint16_t counters[Dims::NUM_EXPERTS];
    __shared__ uint16_t offsets[Dims::NUM_EXPERTS];
    
    unsigned tid = threadIdx.x;
    unsigned warp = tid / 32;
    unsigned lane = tid % 32;
    
    // Initialize counters
    if (tid < Dims::NUM_EXPERTS) {
        counters[tid] = 0;
    }
    __syncthreads();
    
    // Count tokens per expert (atomic within shared memory)
    for (unsigned i = tid; i < Dims::BS; i += blockDim.x) {
        uint8_t expert = topk_ids[i];
        atomicAdd(&counters[expert], 1);
    }
    __syncthreads();
    
    // Prefix sum for offsets
    if (tid < Dims::NUM_EXPERTS) {
        uint16_t sum = 0;
        for (unsigned e = 0; e < tid; e++) {
            sum += counters[e];
        }
        offsets[tid] = sum;
        ranges->expert_counts[tid] = counters[tid];
        ranges->expert_offsets[tid] = sum;
    }
    __syncthreads();
    
    // Scatter tokens to sorted positions
    __shared__ uint16_t write_pos[Dims::NUM_EXPERTS];
    if (tid < Dims::NUM_EXPERTS) {
        write_pos[tid] = offsets[tid];
    }
    __syncthreads();
    
    for (unsigned i = tid; i < Dims::BS; i += blockDim.x) {
        uint8_t expert = topk_ids[i];
        uint16_t pos = atomicAdd(&write_pos[expert], 1);
        ranges->token_indexes[pos] = i;
    }
    __syncthreads();
}
```

### Optimized Parallel Prefix Sum

For larger expert counts, use parallel prefix sum:

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

## Expert Reference Structure

```cpp
struct ExpertRef {
    uint8_t id;           // Expert index
    uint16_t first_token; // Start in token_indexes
    uint16_t last_token;  // End (exclusive) in token_indexes
    
    __device__ uint16_t count() const { return last_token - first_token; }
};

template<typename Dims>
struct MoESortedTokens {
    uint16_t token_indexes[Dims::BS];
    ExpertRef experts[Dims::NUM_EXPERTS];
    uint8_t active_expert_count;
};
```

## Usage Pattern

```cpp
// After sorting
MoESortedTokens<Dims> sorted;
prepare_moe<Dims>(topk_ids, &sorted);

// Process each active expert
for (uint8_t e = 0; e < sorted.active_expert_count; e++) {
    ExpertRef& expert = sorted.experts[e];
    
    // Load expert weights
    W_up = load_expert_weights(expert.id, UP);
    W_down = load_expert_weights(expert.id, DOWN);
    
    // Process tokens for this expert
    for (uint16_t t = expert.first_token; t < expert.last_token; t++) {
        uint16_t token_idx = sorted.token_indexes[t];
        // GEMM for token_idx with expert.id
    }
}
```

## Performance Considerations

1. **Memory Access Pattern**: Sorting enables coalesced access to expert weights
2. **Load Balance**: Skewed routing can cause idle threads - consider work stealing
3. **Atomic Overhead**: For BS > 64, atomics in histogram can serialize - use warp-local counting first

## Warp-Local Counting Optimization

```cpp
// Each warp counts locally, then merge
__shared__ uint16_t warp_counters[NUM_WARPS][Dims::NUM_EXPERTS];

// Warp-local counting (no atomics)
if (lane < Dims::NUM_EXPERTS) {
    warp_counters[warp][lane] = 0;
}
__syncwarp();

for (unsigned i = warp; i < Dims::BS; i += NUM_WARPS) {
    if (lane == 0) {
        uint8_t expert = topk_ids[i];
        warp_counters[warp][expert]++;
    }
}
__syncthreads();

// Merge across warps
if (tid < Dims::NUM_EXPERTS) {
    uint16_t total = 0;
    for (int w = 0; w < NUM_WARPS; w++) {
        total += warp_counters[w][tid];
    }
    counters[tid] = total;
}
```