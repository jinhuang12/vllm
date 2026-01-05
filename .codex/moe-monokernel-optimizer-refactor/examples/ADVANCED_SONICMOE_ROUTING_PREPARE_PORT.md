# Advanced Example: SonicMoE-style routing+prepare port (runtime dispatch, specialized kernels)

Use this example **only** when your Phase 2 route decision is:
- **cooperative monokernel**, or
- you have proven (with CUDA-graph baseline + nsys) that routing+prepare are a dominant slice and the baseline routing op is not already optimal for your `(E_local, top_k, scoring_func)` semantics.

If your Phase 2 route decision is **hybrid large-grid fusion**, start with `examples/HYBRID_FUSION_KWAYMERGE_W1_EPILOGUE.md` and treat routing replacement as secondary.


## Contents
- Goal
- Supported dispatch set (recommended starter)
- Critical semantic contract
- Implementation outline
- CUDA graphs safety + validation notes

## Search anchors
SonicMoE, routing+prepare, top-k, k-way merge, softmax, sigmoid, renormalize, determinism, dispatch, E_local.

---

## Goal

Replace the “repeat max-reduction K times” top‑k routing + slow prepare with a SonicMoE-style selection:
- runtime dispatch on `(E_local, top_k, scoring_func, renormalize)`
- *specialized* kernels for a small supported set (fast)
- clean fallback to baseline when unsupported

This is a **routing/prepare optimization**. It is not a substitute for W1 epilogue fusion when activation/quant kernels dominate.

---

## Supported dispatch set (recommended starter)

Keep this small at first to avoid endless compilation and edge-case explosion.

- `top_k ∈ {8, 16}`
- `E_local ∈ {64, 128, 160, 256}`
- `scoring_func ∈ {softmax, sigmoid}`
- `renormalize ∈ {True, False}`

Everything else must:
- fall back to the existing baseline path, or
- hard error (only if you can guarantee it won’t break user workloads).

---

## Critical semantic contract (do not guess)

Routing “weights” must match the model semantics:

- **softmax**:
  - `w = softmax(topk_logits)`
- **sigmoid + renorm**:
  - `s = sigmoid(topk_logits + bias)` (bias placement is model-specific)
  - if `renormalize=True`: `w = s / sum(s)`
  - else: `w = s`

Also confirm:
- any routed scaling factor and where it applies (inside routing vs outside fused MoE)
- whether routing weights are applied pre/post activation (correctness-critical)

---

## Implementation outline (vLLM monokernel codebase)

### 1) Do not widen expert-id types unless required

If `E_local ≤ 256`, you can keep expert IDs in `uint8` internally.

Widening to `uint16` becomes necessary when you want `E_local > 256` or when existing code relies on sentinel values that collide with valid IDs.

### 2) Implement SonicMoE-style selection as a separate header-ish TU

Add a new internal include-style file:
- `csrc/moe/moe_monokernel/src/moe_routing_sonic.cuh`

It should provide:
- `template<int E, int TOPK, ScoringSemantics S> __device__ void route_token(...)`

Where `ScoringSemantics` encodes:
- scoring func (softmax vs sigmoid)
- whether to renormalize
- whether bias is applied before scoring

### 3) Prefer k-way merge over “iterative delete”

The simplest reliable Sonic-style improvement (and usually sufficient) is:
- per-lane local sort of `P = ceil(E/32)` candidates
- warp-level k-way merge to select top_k
- deterministic tie-break `(value desc, expert_id asc)`

Only add mantissa-bit packing if it is required by the specific sorting primitive you port.

### 4) Runtime dispatch wrapper

In `csrc/moe/moe_monokernel/src/moe_routing.cu`, keep the old path as fallback and add:

```cpp
// Pseudocode: select a specialization or fall back.
if (topk == 8 && num_experts == 128 && scoring == softmax && renorm) {
  return route_impl<128, 8, SoftmaxRenorm>(...);
}
...
return route_baseline(...);
```

Route dispatch must be **shape-guarded** and must not silently change semantics.

### 5) Prepare: do not implement “(threads+1)*E counters in shared memory”

Do not build a single-CTA prepare that allocates `O(threads * E)` shared arrays.
This will:
- blow dynamic SMEM,
- collapse occupancy,
- and often become slower than the baseline even if “fused”.

Prefer a small-memory prepare:
- for small `total_pairs = M * top_k` (e.g., ≤512), use count → scan → scatter patterns that keep scratch small and avoid atomics
- if you need expert alignment/padding, do it using compact per-expert arrays and a small scan

### 6) CUDA graphs safety (required)

- Launch on `at::cuda::getCurrentCUDAStream()`
- Avoid allocations in capture
- Keep shapes stable within a bucket

---

## Validation gates (minimum)

1) Routing-only correctness:
- ids match baseline (or match deterministic tie-break rules)
- weights match within tolerance (per scoring semantics)

2) Prepare correctness (if changed):
- permutation invariants + per-expert range invariants

3) CUDA graphs parity:
- re-run baseline truth snapshot under CUDA graphs
- demonstrate routing/prepare share moved in the intended direction (no regressions)
