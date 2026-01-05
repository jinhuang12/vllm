# Worked Example (Generic): Hybrid Large‑Grid Fusion via (1) Top‑K k‑way merge routing + (2) W1 epilogue fusion

This example is intentionally **model‑agnostic**. It captures a repeatable “hybrid large‑grid fusion” playbook that generalizes across vLLM MoE models that use:
- `top_k = 8` (or any small K where k‑way merge is worthwhile)
- Router logits shaped like `[M, E_local]`
- Large‑grid expert GEMM kernels (often Triton), with activation + quant kernels still materialized separately

**Goal**: keep the baseline large‑grid expert GEMM(s) (for concurrency) and fuse *around* them:
1) Faster deterministic routing selection (k‑way merge, no iterative delete loop)
2) Fuse W1 GEMM epilogue: activation (SiLU/SwiGLU) + W2‑input quantization (FP8) into the W1 kernel so you delete real kernel work and avoid large intermediate writes/reads.

This is the “don’t stop at tuning-only” hybrid that prevents the failure mode where the optimizer only generates Triton config files.


## Contents
- Hard anti-patterns
- When to choose this hybrid
- Part 1: Routing via k-way merge
- Part 2: W1 epilogue fusion
- Validation + integration notes

## Search anchors
hybrid, large-grid fusion, k-way merge, routing, W1 epilogue fusion, activation, SwiGLU/SiLU, FP8 quantization, CUDA graphs.

---

## Hard anti-patterns (avoid common “fusion made it slower” traps)

These are frequent ways a well-intentioned “fusion” loses badly. Avoid them unless you have profiler evidence they still win.

- **Single-CTA mega-kernel**: do not fuse routing+prepare into one CTA that scans all experts and uses `O(threads * E)` shared-memory counters (e.g., `(threads+1)*E` arrays). This typically inflates dynamic SMEM and collapses occupancy.
- **Claiming k-way merge without k-way merge**: “full scan + insertion topk” is not k‑way merge; it is still `O(E * top_k)` per token and can be slower than baseline fused routing ops.
- **Wrong scoring semantics**: do not assume softmax. If the model uses sigmoid + bias + renorm, implement that exact behavior.
- **Over-counting launch overhead under CUDA graphs**: if graphs are enabled, your win must come from GPU kernel time; removing “one kernel launch” is rarely the main win.

## When to choose this hybrid (route decision trigger)

From the **CUDA‑graph baseline** (Nsight Systems / per-kernel timings):
- `share_gemm` is high (W1+W2 dominate), and the GEMM kernel uses a large grid (grid >> SM count).
- But there are still **separate kernels** for activation and/or quantization for W2 input, and they move a large intermediate (typically `[M*top_k, N_local]` or `[M*top_k, 2*N_local]`).

If activation/quant kernels are already fused inside W1 in your baseline, skip epilogue fusion and pick a different hybrid target.

---

## Part 1: Top‑K routing via k‑way merge (generic)

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

---

## Part 2: W1 Triton epilogue fusion (generic)

### Target to fuse

Baseline often does:
1) W1 expert GEMM writes `[M*top_k, 2*N_local]` to global
2) activation kernel computes `[M*top_k, N_local]` (SiLU/SwiGLU)
3) quant kernel converts activations to FP8 for W2 input (+ scales)
4) W2 expert GEMM consumes FP8 activations

**Hybrid fusion goal**:
- Keep the baseline expert GEMM structure and grid shape
- In the W1 kernel, after producing the W1 output tile:
  - apply activation in registers
  - quantize to FP8 for W2 input
  - write only the FP8 activations (+ scale metadata)

This removes at least one large intermediate write/read and deletes 1–2 kernels.

### Preconditions / constraints

- You can compute activation from W1 outputs without needing a global sync.
- You can produce the same quantization metadata expected by W2:
  - per-tensor scales (simpler; common in FP8 MoE)
  - per-block / grouped scales (harder; requires matching baseline layout exactly)
- Register pressure must not explode; epilogue work should be “light” compared to GEMM.

### Triton implementation sketch (high-level)

1) Locate the Triton expert GEMM kernel used by the baseline (often called from `fused_experts`).
2) Add a kernel variant flag (or a separate kernel) that:
   - computes W1 tile
   - performs activation:
     - **SwiGLU**: `y = silu(gate) * up` (gate/up are halves of the `2*N_local` output)
     - **SiLU+mul**: same idea (depends on model)
   - quantizes `y` to FP8:
     - compute `amax` reduction for the quantization group
     - compute scale and apply FP8 cast
     - write FP8 and any needed scale tensor
3) Wire the new kernel into the fused MoE call path under an env var gate.

### Make it generic (don’t bake in one model)

To keep W1 epilogue fusion generic across MoE models, parameterize only what actually varies:
- activation type (SiLU / SwiGLU / GELU-family)
- quantization format (per-tensor vs per-block scales, dtype)
- where routing weights are applied (pre/post activation), and whether weights can be folded safely

Everything else (grid shape, expert grouping, weight layouts) should be inherited from the baseline expert GEMM call path for that model/config.

### What “success” must look like (profiling)

Under CUDA graphs (production parity):
- Nsight Systems kernel list should show activation/quant kernels removed or reduced.
- Kernel-level timing should improve **without** reducing GEMM concurrency (check achieved occupancy / spills).

### What can go wrong (and how to catch it)

- **Already fused baseline**: you implemented work that was already in the GEMM; no win.
  - Catch with nsys kernel list before coding.
- **Spills / lower occupancy**: epilogue increases registers, drops occupancy, kernel slows.
  - Catch with NCU (registers/spills, achieved occupancy, stall reasons).
- **Quantization mismatch**: scale semantics differ → output mismatch or silent accuracy loss.
  - Catch with dequant+compare vs baseline and end-to-end tests.

---

## Minimal “fusion deliverable” checklist (generic)

For hybrid large-grid fusion to count as implemented:

1) Baseline Truth Snapshot includes the **kernel names** (nsys) showing activation/quant kernels exist.
2) Route Decision selects hybrid and names at least one fusion target.
3) Implement at least one of:
   - k‑way merge routing (shape-gated + CUDA-graph safe), or
   - W1 epilogue fusion (activation+quant) into the expert GEMM kernel
4) Validation:
   - correctness: MoE output matches baseline within tolerance across BS buckets
   - kernel perf: speedup >= 1.0x in enabled buckets under CUDA graphs
   - nsys evidence: kernel list reflects the intended fusion
