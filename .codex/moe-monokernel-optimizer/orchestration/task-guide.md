# MoE Monokernel Task Guide (Single‑Agent)

This is the **canonical execution checklist** for this skill.
If any other orchestration doc conflicts with this one, treat this as the source of truth.

## Contents
- How to use this guide
- Directory convention
- state.json 
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
   - Execute **ALL** the checklist items.
   - Update `state.json` with status + links to produced artifacts + a 1‑paragraph summary.
   - You **must** complete each phase/stage before moving forward. Checklist items must be complete before marking phase/stage as complete.
4. If any Phase 4 gate fails, stop thrashing:
   - Set status `needs_investigation`
   - Follow `orchestration/investigation-prompts.md`
5. Use `orchestration/llm-council.md` to decide whether to invoke the separate `llm-council` skill.

## Directory convention 

- `{artifact_dir}/constraints.md`
- `{artifact_dir}/optimization_plan.md`
- `{artifact_dir}/implementation_notes.md`
- `{artifact_dir}/validation_results.md`
- `{artifact_dir}/integration.md`
- `{artifact_dir}/state.json`
- `{artifact_dir}/investigation/...` (Phase 4 failures)

For CUDA/C++ work, prefer a sibling dir you can build from:

- `{cuda_dir}/` (e.g., `vllm/csrc/moe_monokernel/<target>/...`)

## state.json

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
- [ ] **Target envelope**: record model id, hardware, dtype/quant format, TP/EP, and the **decode buckets** you care about.
- [ ] **Fast Phase-1 bucket set (time-bounded default)**:
  - Unless you have a stronger reason, constrain Phase 1 to **two buckets** to keep iteration time low:
    - `batch_size ∈ {8, 64}`.
    - `input_len = 1024`, `output_len = 32`.
  - Expand to more buckets only after Phase 2 has a clear win hypothesis.
- [ ] **E2E baseline is required in Phase 1** (to compute MoE share `f` and to avoid optimizing the wrong bottleneck):
  - Run `vllm bench latency` for at least one representative bucket using production-parity knobs (TP/EP, CUDA graphs mode, torch.compile mode, bucketing).
  - If model weights are not present locally, **download them** (do not treat “weights missing” as a reason to skip Phase 1 E2E).
  - If download is blocked (gated repo / auth / terms / network / disk), set `state.json` status to `blocked` and ask the user explicitly whether they want to waive E2E.
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
  - Use `references/nsys-profiling-guide.md` for vLLM-specific `nsys` flags and command templates (offline benchmark preferred for Phase 1).
- [ ] **Kernel breakdown** (minimum):
  - Identify dominant kernel(s) and whether key hops are already fused (routing, prepare, activation, quant, reduce).
  - If you only have a wrapper timing (e.g., a single `fused_experts` node), run a trace to identify its internal kernels.
- [ ] **Hopper IO sanity-check (sm_90a only)**:
  - If targeting H100/H200, check whether the MoE critical path is limited by **output atomics / scatter-like stores / heavy epilogues** (vs pure GEMM math).
  - If yes, prefer designs that keep GEMM stores **regular/contiguous** and move irregularity into a separate **token-major aggregation** step (instead of fusing scatter/atomics into a GEMM epilogue). Treat as a hypothesis and gate by CUDA-graph kernel-time wins.
- [ ] **Write the Baseline Truth Snapshot** into `constraints.md`:
  - Copy the template from `references/moe-parameters-template.md` (Baseline Truth Snapshot section) and fill it for at least 1–2 representative buckets.
- [ ] **Stop-condition input**:
  - Record MoE share of end-to-end (`f = T_moe / T_total`) and use `references/e2e-delta-math.md` to bound expected system impact.

Exit criteria:
- `constraints.md` contains (1) semantics truth, (2) bucket timings under production parity, (3) baseline fusion facts.
  - and includes an explicit E2E baseline measurement (or an explicit user waiver / documented blocker).

---

## Phase 2: Route + Optimization Plan

**Goal:** evaluate all potenetial routes (cooperative / hybrid / split), define a realistic win condition, and produce a plan that can be implemented and validated.

**Primary output:** `{artifact_dir}/optimization_plan.md`.

Checklist:
- [ ] **Write a comprehensive detailed optimization plan** in `{artifact_dir}/optimization_plan.md`:
  - Must use `references/optimization-plan-template.md` (copy/paste).
  - Reject the plan if it contains placeholders or lacks concrete commands + acceptance criteria.
- [ ] Optional: **Council review**:
  - If risk tier is high (semantics changes, top_k>1, major fusion boundary change), invoke `llm-council` per `orchestration/llm-council.md`.

Exit criteria:
- `optimization_plan.md` contains: route decision, feasibility argument, **ranked top-10 opportunity list (3A)**, **2–3 selected hypotheses tied to OP IDs (3B)**, concrete implementation plan, and kill criteria.

---

## Phase 3: Implementation

**Goal:** implement the optimizations exactly as planned in `{artifact_dir}/optimization_plan.md`. Use `create-plan` skill to create an implementation plan.

**Primary output:** `{artifact_dir}/implementation_notes.md` + code changes.

Checklist (all routes):
- [ ] Create `{cuda_dir}` and a build path that works locally (clean rebuild).
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
