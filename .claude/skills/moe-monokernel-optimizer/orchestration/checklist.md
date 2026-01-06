# MoE Monokernel Checklist

Canonical phase gates for the optimization workflow.

## Directory Convention

- Artifacts: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/`
- CUDA: `csrc/moe/moe_monokernel_{model}_{hardware}_{dtype}_{tp}/`
- State: `{artifact_dir}/state.json`

## Phase 1: Constraints

**Goal**: Capture model semantics + baseline truth snapshot.

**Read first**:
- `references/moe-parameters-template.md`
- `references/profiling-launch-vs-kernel.md`

**Checklist**:
- [ ] Record target envelope (model, hardware, dtype, TP/EP, batch buckets)
- [ ] Read vLLM model code to confirm MoE semantics (routing, renorm, weight placement)
- [ ] Run production-parity profiling (CUDA graphs enabled)
- [ ] Identify dominant kernels and baseline fusion facts
- [ ] Write Baseline Truth Snapshot into `constraints.md`

**Exit**: `constraints.md` complete with semantics + bucket timings + fusion facts.

---

## Phase 2: Route + Optimization Plan

**Goal**: Choose route, define win condition, produce implementable plan.

**Read first**:
- `references/route-selection-decision-tree.md`
- `references/algorithmic-branching.md`
- `references/fusion-feasibility-heuristics.md`

**Checklist**:
- [ ] Pick route (A/B/C) using decision tree
- [ ] Compute feasibility math (`time_saved_max_us`)
- [ ] Record algorithmic decisions (ownership, accumulation, sorter, weight order)
- [ ] Define kill criteria (2-3 "stop" conditions)
- [ ] Plan validation approach

**Key Decisions**:
- Decision 0b: M_avg = BS × top_k / E_local
- Decision 0c: Ownership (token-major vs expert-major)
- Decision A: Atomics vs direct write
- Decision B: Sorter strategy (TOKENS_PER_WARP)
- Decision C: Weight placement (CORRECTNESS CRITICAL)

**Exit**: `optimization_plan.md` with route, feasibility, plan, kill criteria.

**Early exits**:
- Monokernel not applicable → document, stop
- Baseline delta implausible → re-plan or document limitation

---

## Phase 3: Implementation

**Goal**: Implement kernel(s) with graph safety and clean fallback.

**Read first**:
- `references/code-templates.md`
- `references/tiling-config.md`
- `references/cudagraph-safety.md`

**Stages** (spawn separate Task for each):

| Stage | Components |
|-------|------------|
| 3.1 routing_and_prepare | Router + token sorting |
| 3.2 activation_quantization | Scale inputs (FP8 only) |
| 3.3 gemm_implementation | Up + down projection |
| 3.4 kernel_assembly | Wire together + main entry |

**Checklist per stage**:
- [ ] Read relevant reference docs first
- [ ] Implement minimal callable path
- [ ] Ensure CUDA graphs safety from day 1
- [ ] Compile and fix errors before marking complete
- [ ] No TODOs in GEMM kernels (verify MMA calls present)

**Exit**: Kernel compiles, runs, is graph-safe, passes smoke test.

---

## Phase 4: Validation (gated)

**Goal**: Prove correctness → kernel perf → e2e latency.

**Read first**:
- `references/validation-defaults.md`
- `validation/E2E_LATENCY_GUIDE.md`

**Do not reorder gates.**

### 4.1 Correctness Gate
- [ ] Compare outputs vs baseline for target buckets
- [ ] Use dtype-appropriate tolerances
- [ ] If top_k>1: test overlap/reduction edge cases

**If fails**: Use `references/investigation-playbook.md` § Correctness.

### 4.2 Kernel Perf Gate
- [ ] Measure GPU kernel time under CUDA graphs
- [ ] Gate: optimized ≤ baseline for every validated bucket
- [ ] Confirm no graph breaks
- [ ] NCU: verify no occupancy/spill regression

**If fails**: Use `references/investigation-playbook.md` § Kernel Perf.

### 4.3 E2E Gate
- [ ] Run `vllm bench latency` under identical knobs
- [ ] Report Δ vs baseline
- [ ] Relate to MoE share using `references/e2e-delta-math.md`

**Exit**: `validation_results.md` with correctness + kernel + e2e results.

---

## Phase 5: Integration

**Goal**: Land as bounded, safe fast-path.

**Checklist**:
- [ ] Add fast-path dispatch guard for exact validated envelope
- [ ] Provide fallback outside envelope
- [ ] Add enable/disable switch (env var)
- [ ] Add/extend CI tests
- [ ] Document validated envelope

**Exit**: Integration is safe-by-default, bounded, testable, rollback-able.

---

## Investigation Protocol

When Phase 4 fails, follow `references/investigation-playbook.md`:

1. Characterize failure (bucket-specific? stage-specific?)
2. Collect evidence (traces, NCU profiles)
3. Form bounded hypothesis
4. Implement minimal fix
5. Re-measure

**Hard limits** (avoid thrash):
- Max 3 hypothesis cycles per failure mode
- Max 2 NCU deep dives
- If still blocked: invoke llm-council or escalate to human
