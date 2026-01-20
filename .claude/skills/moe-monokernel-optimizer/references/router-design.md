# Router Design Reference

## Contents
- Overview
- Top‑K routing via k‑way merge
- Router Semantics Checklist
- Strategy Selection Matrix
- Warp‑Level Patterns
- Performance Tuning

## Search anchors
topk, routing, topk, k-way merge, sorting network, bitonic merge, SonicMoE, renormalize, norm_topk_prob, weight placement, stable ordering, shared experts.

## Overview

Replace existing top‑k selection with a small‑K GPU kernel that uses warp‑level primitives and a deterministic tie-break.

Use only when routing must be fused. For split-kernel paths, run a standalone router and pass top-k ids/weights into GEMM.

## Top‑K routing via k‑way merge (generic)

### Supported dispatch set (recommended)

To keep performance high, implement routing as runtime-dispatch to a small set of specializations (and fall back otherwise).

Recommended starter set (covers most practical vLLM MoE cases):
- `top_k ∈ {8, 16}`
- `E_local ∈ {64, 128, 160, 256}`
- `scoring_func ∈ {softmax, sigmoid}` with `renormalize ∈ {True, False}`

Notes:
- `E_local` is post-TP/EP sharding; do not dispatch on `E_global`.
- If your model uses grouped routing (e.g., expert groups), treat it as a distinct semantics variant and gate separately.

### Why k‑way merge

For `top_k=8` and `E≤256`, a common baseline pattern is:
- each lane holds 4 logits
- repeat 8 times: pick a lane-local best, warp-reduce to global best, then "delete" the selected expert (set to `-inf`) and repeat

This is simple but does **8 full warp reductions** plus repeated delete scans.

With k‑way merge:
- Each lane sorts its local candidate list once
- Then you do K selections where only the winning lane advances its cursor

This reduces repeated work and gives clean hooks for deterministic tie-break.

### Generic k‑way merge algorithm (E arbitrary)

Let:
- `E = E_local` experts on this rank
- `LANES = 32`
- `P = ceil(E / LANES)` candidates per lane

Each lane `l` owns candidate indices:
`expert = l + t*LANES` for `t in [0, P)`

Out-of-range experts (`expert >= E`) are treated as `-inf`.

**Steps per token**
1) Load `P` logits per lane
2) Locally sort lane's list descending by `(value, tie_key)`
3) Initialize `cursor[l] = 0`
4) Repeat `k=0..TOP_K-1`:
   - each lane proposes `cand = list[cursor[l]]` (or `-inf` if cursor exhausted)
   - warp-reduce to the best `(value, tie_key)` and broadcast
   - only the owning lane increments its cursor

Generalization notes:
- If `E` is not exactly 128, each lane holds `P = ceil(E/32)` candidates (padding with `-inf` for out-of-range).
- For `E=160`, `P=5`; for `E=256`, `P=8`. Use the same merge logic; only the per-lane local sort changes.

### Determinism (tie-break)

You must pick a deterministic total order when two experts have equal score (common when logits are quantized, clamped, or identical).

Recommended tie key:
- primary: value (descending)
- secondary: expert id (ascending)

If you need a "stable but value-aware" key, you can incorporate mantissa bits + expert id, but **expert id is sufficient** for determinism in most inference routing. A practical tie-key is `(mantissa_bits << bits) | expert_id` so that equal values choose lowest expert id.

### Compile-shaped snippet (k-way merge, deterministic)

This is the minimum "shape" to implement when swapping routing. Keep it in a separate TU and runtime-dispatch to a small `(E_local, top_k, semantics)` set.

```cuda
// Tie-break: higher score wins; if equal, lower expert id wins.
__device__ __forceinline__ bool better_pair(float s_a, int id_a,
                                            float s_b, int id_b) {
  return (s_a > s_b) || ((s_a == s_b) && (id_a < id_b));
}

__device__ __forceinline__ void sort2_desc(float &a, int &ia,
                                           float &b, int &ib) {
  if (!better_pair(a, ia, b, ib)) {
    float tv = a;
    a = b;
    b = tv;
    int ti = ia;
    ia = ib;
    ib = ti;
  }
}

// Example local sort for the common E_local=128 case where P=E/32=4 values/lane.
__device__ __forceinline__ void sort4_desc(float v[4], int idx[4]) {
  sort2_desc(v[0], idx[0], v[1], idx[1]);
  sort2_desc(v[2], idx[2], v[3], idx[3]);
  sort2_desc(v[0], idx[0], v[2], idx[2]);
  sort2_desc(v[1], idx[1], v[3], idx[3]);
  sort2_desc(v[1], idx[1], v[2], idx[2]);
}

// Warp-per-token, E_local=128, TOPK=8. Generalize by setting P=ceil(E/32) and
// providing a small sorting network for P ∈ {2,4,5,8,...}.
template <typename ScoreFn>
__device__ __forceinline__ void route_token_topk8_e128_kwaymerge(
    const float* __restrict__ logits_row, const ScoreFn& score_fn,
    int out_ids[8], float out_scores[8]) {
  constexpr int E = 128;
  constexpr int TOPK = 8;
  constexpr int LANES = 32;
  constexpr int P = E / LANES;  // 4

  const int lane = threadIdx.x & 31;

  float v[P];
  int idx[P];
#pragma unroll
  for (int i = 0; i < P; ++i) {
    const int expert = lane * P + i;
    v[i] = score_fn(logits_row[expert], expert);
    idx[i] = expert;
  }
  sort4_desc(v, idx);

  int cursor = 0;
#pragma unroll
  for (int k = 0; k < TOPK; ++k) {
    const float cand_s = (cursor < P) ? v[cursor] : -INFINITY;
    const int cand_id = (cursor < P) ? idx[cursor] : 0x7fffffff;

    float best_s = cand_s;
    int best_id = cand_id;
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      const float other_s = __shfl_down_sync(0xffffffff, best_s, offset);
      const int other_id = __shfl_down_sync(0xffffffff, best_id, offset);
      if (better_pair(other_s, other_id, best_s, best_id)) {
        best_s = other_s;
        best_id = other_id;
      }
    }

    best_s = __shfl_sync(0xffffffff, best_s, 0);
    best_id = __shfl_sync(0xffffffff, best_id, 0);

    if (lane == 0) {
      out_ids[k] = best_id;
      out_scores[k] = best_s;
    }
    if (cand_id == best_id) {
      cursor++;
    }
  }
}
```

### Output semantics checklist (must match model)

Before swapping routing, confirm:
- scoring function: softmax vs sigmoid (and whether bias is added pre-score)
- renormalize (`norm_topk_prob`)
- routed scaling factor (some models apply outside the fused MoE path)
- whether routing weights are applied pre/post activation (correctness critical)

### Scoring + renormalization (what "weights" actually mean)

The k‑way merge only selects the **top‑k indices**. You must still match the model's scoring semantics for **weights**.

Common cases:

- **softmax** (typical):
  - `w = softmax(topk_logits)`
- **sigmoid + renorm** (common in some MoE families):
  - `s = sigmoid(topk_logits + bias)`
  - if `renormalize=True`: `w = s / sum(s)`
  - else: `w = s` (rare; but must match model)

If the model applies a routed scaling factor (or other post-processing), ensure you apply it at the same place as baseline vLLM.

### Compile-shaped snippet (weights semantics)

Keep scoring and weight generation explicitly encoded so routing swaps cannot silently change semantics.

```cuda
// Example semantics: selection uses sigmoid(logit) + bias[expert].
// Weights use sigmoid(logit) (unbiased), optionally renormalized.
struct SigmoidBiasRenorm {
  const float* __restrict__ bias;  // [E_local]
  __device__ __forceinline__ float score(float logit, int expert) const {
    // Replace with the model's exact sigmoid approximation if it matters.
    const float s = 0.5f * tanhf(0.5f * logit) + 0.5f;
    return s + bias[expert];
  }
  __device__ __forceinline__ float weight_unorm(float logit) const {
    return 0.5f * tanhf(0.5f * logit) + 0.5f;
  }
};

__device__ __forceinline__ void renorm8(float w[8]) {
  float sum = 1.0e-20f;
#pragma unroll
  for (int i = 0; i < 8; ++i) sum += w[i];
  const float inv = 1.f / sum;
#pragma unroll
  for (int i = 0; i < 8; ++i) w[i] *= inv;
}
```

### Integration pattern in vLLM (generic)

To use a new routing kernel without destabilizing:
- Add a new custom op `ops.<model>_routing_kwaymerge(...) -> (topk_ids, topk_weights, optional prep outputs)`
- Gate it with an env var **and** with strict shape checks (E_local/top_k/scoring_func).
- Ensure CUDA graphs safety:
  - launch on `at::cuda::getCurrentCUDAStream()`
  - avoid allocations inside capture

Validation:
- Compare `topk_ids` exact match to baseline (if baseline deterministic)
- Compare weights within tolerance (rtol/atol), and validate end-to-end MoE output diff
- Include an "all equal logits" determinism test if feasible

```cpp
// Dispatch skeleton: shape-guard a small set, fall back otherwise.
// This is the "SonicMoE pattern": runtime inputs, compile-time specialization.
std::tuple<torch::Tensor, torch::Tensor> route_dispatch(
    torch::Tensor router_logits, torch::optional<torch::Tensor> bias,
    int64_t top_k, ScoringSemantics sem) {
  // Graph-safe requirements:
  // - no hidden allocations inside capture beyond stable-shape outputs
  // - launch on current stream
  const auto stream = at::cuda::getCurrentCUDAStream();

  const int64_t E_local = router_logits.size(1);
  if (top_k == 8 && E_local == 128 && sem == ScoringSemantics::SigmoidBiasRenorm) {
    return launch_route_topk8_e128_kwaymerge(router_logits, *bias, stream);
  }
  return route_baseline(router_logits, bias, top_k, sem, stream);
}
```

### Prepare metadata

If the baseline already has a fast prepare, keep it. If prepare is a bottleneck, start with a **single-block, stable** prepare for small `total_pairs = M * top_k` (common for decode buckets) before building a multi-block path.

Contract (recommended):
- `expert_offsets[E_local+1]`: exclusive prefix sum over per-expert counts
- `pair_indices[total_pairs]`: permutation of `[0..total_pairs-1]` grouped by expert, stable by pair index

```cuda
// Single-CTA stable prepare for E_local=128, top_k=8 with small total_pairs.
// Input: topk_ids is [total_pairs] as uint8/uint16 (pair-major: pair = tok*top_k + kslot).
// Output: expert_offsets [E+1] and pair_indices [total_pairs] grouped by expert.
template <int TOPK>
__device__ __forceinline__ void prepare_topk_e128_single_block(
    const uint8_t* __restrict__ topk_ids, uint32_t num_tokens,
    uint16_t* __restrict__ expert_offsets, uint16_t* __restrict__ pair_indices,
    uint32_t* __restrict__ smem_counts, uint32_t* __restrict__ smem_offsets) {
  constexpr uint32_t E = 128;
  const uint32_t total_pairs = num_tokens * TOPK;
  const uint32_t t = threadIdx.x;

  // 1) Count (no atomics): one thread per expert scans pairs (stable).
  if (t < E) {
    uint32_t count = 0;
#pragma unroll 4
    for (uint32_t i = 0; i < 512; ++i) {  // unroll hint; break early.
      if (i >= total_pairs) break;
      count += (static_cast<uint32_t>(topk_ids[i]) == t);
    }
    smem_counts[t] = count;
  }
  __syncthreads();

  // 2) Exclusive scan over 128 counts -> base offsets.
  if (t == 0) {
    uint32_t running = 0;
#pragma unroll
    for (uint32_t e = 0; e < E; ++e) {
      smem_offsets[e] = running;
      running += smem_counts[e];
    }
    smem_offsets[E] = running;
  }
  __syncthreads();

  // 3) Scatter stable: each expert thread writes pair indices in increasing order.
  if (t < E) {
    uint32_t pos = smem_offsets[t];
#pragma unroll 4
    for (uint32_t i = 0; i < 512; ++i) {
      if (i >= total_pairs) break;
      if (static_cast<uint32_t>(topk_ids[i]) == t) {
        pair_indices[pos++] = static_cast<uint16_t>(i);
      }
    }
  }
  __syncthreads();

  // 4) Write offsets for consumers (prefix sum; [E+1]).
  if (t <= E) {
    expert_offsets[t] = static_cast<uint16_t>(smem_offsets[t]);
  }
}
```

For a worked, model‑agnostic hybrid implementation pattern (including routing k‑way merge), see `examples/HYBRID_FUSION_KWAYMERGE_W1_EPILOGUE.md`.

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
