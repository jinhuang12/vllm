# Router Design Reference

## Contents
- Overview
- Top‑K routing via k‑way merge

## Search anchors
topk, routing, topk, k-way merge, sorting network, bitonic merge, SonicMoE

## Overview

Replace existing top‑k selection with a small‑K GPU kernel that uses warp‑level primitives and a deterministic tie-break.

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

The common “iterative delete” top‑k pattern does:
- repeat K times:
  - find lane-local best
  - warp-reduce to global best
  - mark selected as `-inf` and repeat

For `K=8`, that is **8 warp reductions** + repeated delete scans.

With k‑way merge:
- each lane sorts its local candidate list once
- then you do K selections where only the winning lane advances its cursor

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
2) Locally sort lane’s list descending by `(value, tie_key)`
3) Initialize `cursor[l] = 0`
4) Repeat `k=0..TOP_K-1`:
   - each lane proposes `cand = list[cursor[l]]` (or `-inf` if cursor exhausted)
   - warp-reduce to the best `(value, tie_key)` and broadcast
   - only the owning lane increments its cursor

### Determinism (tie-break)

You must pick a deterministic total order when two experts have equal score (common when logits are quantized, clamped, or identical).

Recommended tie key:
- primary: value (descending)
- secondary: expert id (ascending)

If you need a “stable but value-aware” key, you can incorporate mantissa bits + expert id, but **expert id is sufficient** for determinism in most inference routing.

### Compile-shaped snippet (k-way merge, deterministic)

This is the minimum “shape” Codex should implement (real code, not prose) when swapping routing. Keep it in a separate TU and runtime-dispatch to a small `(E_local, top_k, semantics)` set.

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

### Scoring + renormalization (what “weights” actually mean)

The k‑way merge only selects the **top‑k indices**. You must still match the model’s scoring semantics for **weights**.

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
- Include an “all equal logits” determinism test if feasible

```cpp
// Dispatch skeleton: shape-guard a small set, fall back otherwise.
// This is the “SonicMoE pattern”: runtime inputs, compile-time specialization.
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
