# Route Selection Decision Tree (Generalizable)

Use this to choose **cooperative monokernel** vs **hybrid large‑grid fusion** vs **split kernels** for a *specific* (model, GPU, dtype, TP/EP, bucket set) target.

This file is intentionally **decision-focused**. Put detailed measurement math in:
- `references/fusion-feasibility-heuristics.md` (time-saved bounds, kill criteria)
- `references/gpu-configs.md` (hardware guardrails)

## Contents
- Inputs (from `constraints.md`)
- Phase 2 required artifact: Route Decision (template)
- Decision Tree (A/B/C)
- Kill criteria (pivot triggers)

## Search anchors
route selection, cooperative monokernel, hybrid large-grid fusion, split kernels, kill criteria, barrier tax, CUDA graphs parity

---

## Inputs (from `{artifact_dir}/constraints.md`)

Before using this decision tree, write `{artifact_dir}/constraints.md` using:
- `references/moe-parameters-template.md` (includes the **Baseline Truth Snapshot**)

From `constraints.md`, you should have:
- Geometry/topology: `E_local`, `top_k`, dims (`K`, `N`), dtype/quant, TP/EP
- Baseline Truth Snapshot (production parity):
  - per-bucket MoE kernel-time breakdown (GPU time)
  - baseline “already fused?” checklist (routing/activation/quant/reduce)
  - dominant GEMM concurrency facts (is the GPU already saturated?)

### Derived triage numbers (per bucket, per rank)

Compute (at least for the representative buckets):
- `P = BS * top_k`  (token–expert pairs on this rank)
- `M_avg ≈ BS * top_k / E_local` (uniform routing proxy; use real routing stats if you have them)

Interpretation:
- **Small `P` / low saturation** (decode buckets) is where fusion often helps (if you can keep GEMM efficiency).
- **Large `M_avg`** pushes you toward expert-major or split-kernel flows; full cooperative fusion often struggles.

---

## Phase 2 required artifact: Route Decision (paste into `{artifact_dir}/optimization_plan.md`)

```markdown
## Route decision
- Chosen route: cooperative monokernel / hybrid large-grid fusion / split kernels

## Why this route (tie to Baseline Truth Snapshot)
- 3–6 bullets referencing the snapshot (dominant kernels, what is/isn't already fused, saturation facts).

## Why not the other routes
- Cooperative monokernel:
- Hybrid large-grid fusion:
- Split kernels:

## Kill criteria (pivot trigger)
- If these conditions are met, pivot routes:
  - ...
```

### Hybrid deliverable (required when route == hybrid large‑grid fusion)

If you pick hybrid, define the **material fusion** you will deliver (one sentence each):
- Which global-memory hops are removed (e.g., fuse W1→act→W2-quant into a single write)
- Which baseline kernel(s) are replaced and by what (CUTLASS epilogue, Triton fusion, custom epilogue)
- Expected time-saved upper bound (cite `fusion-feasibility-heuristics.md`)

---

## Decision Tree (A/B/C)

### A) Baseline GEMMs dominate and baseline concurrency is strong → **Hybrid large‑grid fusion**

Choose **hybrid** when:
- The baseline W1/W2 GEMM kernels are already the dominant cost **and**
- They show high occupancy / strong concurrency (many waves), i.e., the GPU is already saturated on those kernels
- Your “already fused?” checklist shows **material** opportunities in the epilogue path (activation, W2-input quant, routed-weight multiply), not just kernel-count reduction

Why:
- Cooperative monokernels often lose to high-quality large-grid GEMMs once you pay barrier/SMEM/register tax.
- Hybrid keeps the strong GEMM kernels and fuses the expensive *hops* around them.

Next:
- Read `references/hybrid-large-grid-fusion.md`
- Use `references/fusion-feasibility-heuristics.md` to sanity-check required µs savings.

### B) Routing/prepare/quant dominates OR decode buckets underfill the GPU → **Cooperative monokernel**

Choose **cooperative monokernel** when:
- Representative decode buckets have low saturation (small `P`) and baseline kernels underfill the GPU
- The Baseline Truth Snapshot shows meaningful time in routing/prepare/quant/accumulate stages that you can collapse
- Your hardware supports cooperative patterns (see `references/gpu-configs.md`) and you can keep grid-wide barriers to a small count

Why:
- The point of cooperative fusion is to reclaim utilization and remove global-memory hops when `P` is too small for the baseline to saturate the device.

Next:
- Read `references/architecture-pattern.md` (controller-worker pattern)
- Then `references/algorithmic-branching.md` (post-route decisions)
- Keep `references/cudagraph-safety.md` open while designing.

### C) Graph safety, large `M_avg`, or EP/top_k constraints dominate → **Split kernels**

Choose **split kernels** when:
- `M_avg` is large (expert-major regime) or routing is highly imbalanced, and you need specialized scheduling/grouped GEMMs
- Cooperative barriers would be numerous or risky (multi-stage controller designs, heavy SMEM use)
- EP introduces constraints that are easier to satisfy with pre-dispatch + per-rank grouped kernels
- You need strict CUDA-graphs stability (shape buckets) and want to avoid hidden allocation/barrier failure modes

Why:
- Split kernels let you keep fast GEMMs, control memory traffic, and stay graph-safe without grid.sync overhead.

Next:
- Read `references/architecture-pattern.md` (split-kernel and token-major patterns)
- Use `references/expert-grouping.md` if you need expert reordering or load balancing.

---

## Kill criteria (pivot triggers)

Use these as “stop digging” signals; write them into the Phase 2 plan.

### Cooperative monokernel → pivot to hybrid/split if
- You need >1–2 grid-wide `grid.sync` points in the steady-state hot loop, or dynamic SMEM/regs prevent sufficient occupancy
- NCU shows spills / occupancy collapse relative to baseline GEMMs
- CUDA graphs capture is unstable (hidden allocations, shape instability)

### Hybrid → pivot if
- There is no material hop to remove (your “already fused?” checklist is already mostly “yes”)
- Fused epilogues measurably slow the GEMM (register explosion) and the hop removed is too small to pay for it

### Split kernels → pivot if
- Kernel count dominates even under CUDA graphs (rare; verify with `profiling-launch-vs-kernel.md`)
- Your split introduces extra DRAM hops that erase any scheduling benefit

For any pivot, recompute the expected time-saved bound using `references/fusion-feasibility-heuristics.md`.
