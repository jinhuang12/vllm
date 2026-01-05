# MoE Monokernel Task Guide (Single‑Agent)

This is the **canonical execution checklist** for this skill.
If any other orchestration doc conflicts with this one, treat this as the source of truth.

## Contents
- How to use this guide
- Directory convention
- state.json (recommended)
- Phase 1: Constraints + Baseline Truth Snapshot
- Phase 2: Route decision + Optimization plan
- Phase 3: Implementation
- Phase 4: Validation (gated)
- Phase 5: Integration

## Search anchors
Baseline Truth Snapshot, route selection, cooperative monokernel, hybrid large-grid fusion, split-kernel, CUDA graphs, torch.compile, nsys, ncu, state.json, validation_results.md.

## How to use this guide

1. Pick or create an `{artifact_dir}` for this target (model + hardware + dtype + TP/EP).
2. If `{artifact_dir}/state.json` exists, resume from its `phase` and `stage`.
3. For the phase/stage you are executing:
   - Write a 3–7 step **micro‑plan** (actions, not questions).
   - Execute the checklist items in order.
   - Update `state.json` with status + links to produced artifacts + a 1‑paragraph summary.
4. If any Phase 4 gate fails, stop thrashing:
   - Set status `needs_investigation`
   - Follow `orchestration/investigation-prompts.md`
5. Use `orchestration/llm-council.md` to decide whether to invoke the separate `llm-council` skill.

## Directory convention (recommended)

- `{artifact_dir}/constraints.md`
- `{artifact_dir}/optimization_plan.md`
- `{artifact_dir}/implementation_notes.md`
- `{artifact_dir}/validation_results.md`
- `{artifact_dir}/integration.md`
- `{artifact_dir}/state.json`
- `{artifact_dir}/investigation/...` (Phase 4 failures)

For CUDA/C++ work, prefer a sibling dir you can build from:

- `{cuda_dir}/` (e.g., `vllm/csrc/moe_monokernel/<target>/...`)

## state.json (recommended)

A minimal schema that supports resumability (extend as needed):

```json
{
  "target": {
    "model_id": "...",
    "hardware": "...",
    "dtype": "...",
    "tp": 1,
    "ep": 1,
    "notes": "optional"
  },
  "phase": "1_constraints | 2_planning | 3_implementation | 4_validation | 5_integration",
  "stage": "optional (e.g., 3.1_routing_and_prepare)",
  "status": "pending | in_progress | complete | needs_investigation | blocked | escalate_human",
  "artifacts": {
    "constraints": "constraints.md",
    "plan": "optimization_plan.md",
    "validation": "validation_results.md"
  },
  "last_update": "YYYY-MM-DD",
  "summary": "1 paragraph"
}
```

---

# Phase checklists

## Phase 1: Constraints + Baseline Truth Snapshot

**Goal:** capture production-parity baseline behavior and model semantics so Phase 2 planning is grounded.

**Primary output:** `{artifact_dir}/constraints.md` (and `state.json`).

Checklist:
- [ ] **Target envelope**: record model id, hardware, dtype/quant format, TP/EP, and the **decode buckets** you care about (e.g., BS ∈ {1,2,4,8,16,32,64}).
- [ ] **Read vLLM model code** to confirm MoE semantics:
  - router scoring (softmax vs sigmoid vs custom), renorm behavior, tie-break rules
  - when/where top-k weights are applied
  - activation function and gating structure (SiLU/SwiGLU/etc)
  - shared experts path (if any)
  - accumulation/reduction semantics (direct write vs reduce vs atomics)
  - Record using `references/moe-parameters-template.md`.
- [ ] **Normalize baseline configs** (if applicable):
  - If baseline MoE kernels fall back to a default/tuner-missing path, generate/choose a tuned config first.
  - Re-run the baseline snapshot after normalization.
- [ ] **Production-parity profiling** for the MoE subgraph (routing + prepare + experts):
  - Measure under the same CUDA graphs / torch.compile settings as production.
  - Separate **CUDA API/launch time vs GPU kernel time** (see `references/profiling-launch-vs-kernel.md`).
- [ ] **Kernel breakdown** (minimum):
  - Identify dominant kernel(s) and whether key hops are already fused (routing, prepare, activation, quant, reduce).
  - If you only have a wrapper timing (e.g., a single `fused_experts` node), run a trace to identify its internal kernels.
- [ ] **Write the Baseline Truth Snapshot** into `constraints.md`:
  - Copy the template from `references/moe-parameters-template.md` (Baseline Truth Snapshot section) and fill it for at least 1–2 representative buckets.
- [ ] **Stop-condition input**:
  - Record MoE share of end-to-end (`f = T_moe / T_total`) and use `references/e2e-delta-math.md` to bound expected system impact.

Exit criteria:
- `constraints.md` contains (1) semantics truth, (2) bucket timings under production parity, (3) baseline fusion facts.

---

## Phase 2: Route + Optimization Plan

**Goal:** choose the route (cooperative / hybrid / split), define a realistic win condition, and produce a plan that can be implemented and validated.

**Primary output:** `{artifact_dir}/optimization_plan.md`.

Checklist:
- [ ] **Pick route (A/B/C)** using `references/route-selection-decision-tree.md`:
  - A) cooperative monokernel
  - B) hybrid large-grid fusion (fuse around baseline GEMMs)
  - C) split-kernel graph-captured route
- [ ] **Quantify feasibility** before committing to heavy fusion work:
  - If proposing W1-epilogue or full monokernel fusion, compute `time_saved_max_us` using `references/fusion-feasibility-heuristics.md`.
  - Compare to the **required savings** from your baseline snapshot for the target buckets.
- [ ] **Choose algorithmic branches** (record decisions + rationale):
  - Ownership model (token-major / expert-major / hybrid)
  - Accumulation strategy (direct write / block reduce / atomics)
  - Sorter strategy and data layout
  - Weight placement (pre/post activation) and how to match baseline semantics
  - Use: `references/algorithmic-branching.md`.
- [ ] **Define kill criteria** (non-negotiable):
  - At least 2–3 crisp “stop” conditions tied to measurable signals (e.g., occupancy drop, barrier count, infeasible µs upper bound).
- [ ] **Plan validation**:
  - Specify exactly how you will validate correctness, kernel perf, and e2e (Phase 4), including what “production parity” means for your deployment.
- [ ] Optional: **Council review**:
  - If risk tier is high (semantics changes, top_k>1, major fusion boundary change), invoke `llm-council` per `orchestration/llm-council.md`.

Exit criteria:
- `optimization_plan.md` contains a route decision, feasibility argument, concrete implementation plan, and kill criteria.

---

## Phase 3: Implementation

**Goal:** implement the kernel(s) and bindings exactly as planned, with graph safety and a clean fallback story.

**Primary output:** `{artifact_dir}/implementation_notes.md` + code changes.

Checklist (all routes):
- [ ] Create `{cuda_dir}` and a build path that works locally (clean rebuild).
- [ ] Implement a minimal callable path first (even if slow) to unblock correctness testing.
- [ ] Ensure CUDA graphs safety from day 1 (see `references/cudagraph-safety.md`):
  - correct stream usage
  - no allocations in capture region
  - stable shapes per bucket
- [ ] Record all compile-time constants and specialization decisions (tile sizes, warp counts, SMEM size) in `implementation_notes.md`.

Route-specific checklist:
- **A) Cooperative monokernel**
  - [ ] Use `references/code-templates.md` and `references/tiling-config.md` for kernel structure and SRAM budgeting.
  - [ ] Minimize `grid.sync` barriers; treat each barrier as a tax that must be amortized.
  - [ ] Ensure accumulation semantics match baseline (especially for top_k>1).
- **B) Hybrid large-grid fusion**
  - [ ] Keep baseline large-grid GEMM(s); fuse *around* them (see `references/hybrid-large-grid-fusion.md`).
  - [ ] Target “material fusion” deliverables (e.g., W1 epilogue fusion, routing+prepare fusion).
  - [ ] Do not double-count: verify what baseline already fuses before claiming a win.
- **C) Split-kernel graph-captured route**
  - [ ] Prefer fewer, simpler kernels that each keep a large grid and high occupancy.
  - [ ] Make workspace allocation explicit and reusable; avoid per-iteration allocations.

Exit criteria:
- Kernel path compiles, runs, is graph-safe, and passes a smoke correctness test vs baseline.

---

## Phase 4: Validation (gated)

**Goal:** prove correctness first, then kernel perf, then end-to-end latency under production settings.

**Primary output:** `{artifact_dir}/validation_results.md` (+ traces / profiles).

**Do not reorder these gates.**

### 4.1 Correctness gate
- [ ] Compare outputs vs baseline for the target buckets and representative inputs.
- [ ] Use tolerances appropriate to dtype/quantization (prefer copying from the model’s vLLM tests if available); document the tolerance and why in `validation_results.md` (see `references/validation-defaults.md`).
- [ ] If top_k>1 or shared experts: add targeted tests for overlap/reduction edge cases.

If this fails: stop and use `orchestration/investigation-prompts.md`.

### 4.2 Kernel perf gate (production parity)
- [ ] Measure MoE GPU kernel time under CUDA graphs for the same buckets as the baseline snapshot (production parity).
- [ ] Gate (default): optimized MoE GPU time ≤ baseline MoE GPU time for every validated bucket in the fast-path envelope (see `references/validation-defaults.md`).
- [ ] Confirm there are no graph breaks and your kernels are actually captured.
- [ ] Use NCU to verify you did not regress occupancy / introduce spills vs baseline dominant GEMM(s).

If this fails: stop and use `orchestration/investigation-prompts.md`.

### 4.3 End-to-end gate
- [ ] Run the full vLLM latency benchmark under identical knobs (CUDA graphs / torch.compile / TP/EP / bucketing). Use `validation/E2E_LATENCY_GUIDE.md` for command templates + parity checklist.
- [ ] Report Δ vs baseline and relate it to MoE share using `references/e2e-delta-math.md`.

Exit criteria:
- `validation_results.md` shows correctness pass + kernel perf win (or justified “no win”) + e2e outcome.

---

## Phase 5: Integration (bounded fast-path + fallback)

**Goal:** land the change as a bounded, safe fast-path for the validated envelope.

**Primary output:** `{artifact_dir}/integration.md` + integrated code path.

Checklist:
- [ ] Add a fast-path dispatch guard for the **exact** validated envelope:
  - model id, dtype, TP/EP, hidden dims, top_k/E, bucket set
- [ ] Provide a correct fallback (baseline path) outside the envelope.
- [ ] Add a simple enable/disable switch (env var or config) for rollback.
- [ ] Add/extend tests so CI can catch correctness regressions.
- [ ] Document the validated envelope and how to reproduce the perf numbers.

Exit criteria:
- Integration is safe-by-default, bounded, testable, and easy to roll back.
