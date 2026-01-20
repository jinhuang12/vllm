# MoE Monokernel Checklist

Canonical phase gates for the optimization workflow.

**IMPORTANT**: Gates are STRICTLY SEQUENCED. Each gate has PRECONDITIONS that MUST be verified before proceeding.

## Directory Convention

- Artifacts: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/`
- CUDA: `csrc/moe/moe_monokernel_{model}_{hardware}_{dtype}_{tp}/`
- State: `{artifact_dir}/state.json`

---

## Phase 1: Constraints (BLOCKING for Phase 2)

**Goal**: Capture model semantics + baseline truth snapshot.

**Precondition**: None (first phase)

**Read first**:
- `references/moe-parameters-template.md`
- `references/profiling-launch-vs-kernel.md`

**BLOCKING Requirements**:
- [ ] **RUN** nsys profiling (files MUST exist in `{artifact_dir}/runs/`)
- [ ] **DOCUMENT** actual baseline kernel times (numbers, not commands)
- [ ] Baseline MUST be vLLM's `fused_moe`/`fused_experts` (NOT naive PyTorch)
- [ ] Production parity: CUDA graphs + torch.compile enabled

**Checklist**:
- [ ] Record target envelope (model, hardware, dtype, TP/EP, batch buckets)
- [ ] Read vLLM model code to confirm MoE semantics (routing, renorm, weight placement)
- [ ] Run production-parity profiling (CUDA graphs enabled)
- [ ] Identify dominant kernels and baseline fusion facts
- [ ] Write Baseline Truth Snapshot into `constraints.md` WITH NUMBERS

**Exit Verification** (MUST PASS):
```bash
python scripts/verify_phase1_baseline.py {artifact_dir}
# Exit code 0 = proceed to Phase 2
# Exit code 1 = BLOCKED, fix issues first
```

**Exit**: `constraints.md` complete with semantics + bucket timings + fusion facts.
**State update**: `"phase": "2_planning"` ONLY if verification passes.

---

## Phase 2: Route + Optimization Plan (BLOCKING for Phase 3)

**Goal**: Choose route, define win condition, produce implementable plan.

**Precondition**: Phase 1 verification PASSED
```bash
# Verify before starting Phase 2:
python scripts/verify_phase1_baseline.py {artifact_dir}
# MUST return exit code 0
```

**Read first**:
- `references/route-selection-decision-tree.md`
- `references/algorithmic-branching.md`
- `references/fusion-feasibility-heuristics.md`

**Checklist**:
- [ ] Pick route (A/B/C) using decision tree WITH profiling data
- [ ] Compute feasibility math (`time_saved_max_us`) using ACTUAL baseline numbers
- [ ] Record algorithmic decisions (ownership, accumulation, sorter, weight order)
- [ ] Define kill criteria (2-3 "stop" conditions) - ALL must be measurable
- [ ] Plan validation approach including baseline comparison

**Key Decisions**:
- Decision 0b: M_avg = BS × top_k / E_local
- Decision 0c: Ownership (token-major vs expert-major)
- Decision A: Atomics vs direct write
- Decision B: Sorter strategy (TOKENS_PER_WARP)
- Decision C: Weight placement (CORRECTNESS CRITICAL)

**Optional**:
- [ ] Council review if high-risk (semantics changes, top_k>1, major fusion boundary)

**Exit criteria**: `optimization_plan.md` contains:
- Route decision + feasibility argument
- **Ranked top-10 opportunity list (section 3A)** with evidence
- **2–3 selected hypotheses tied to OP IDs (section 3B)**
- Concrete implementation plan + kill criteria

**State update**: `"phase": "3_implementation"`

**Early exits**:
- Monokernel not applicable → document, set `"status": "not_applicable"`
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

## Phase 4: Validation (STRICTLY SEQUENCED, BLOCKING for Phase 5)

**Goal**: Prove correctness → kernel perf → e2e latency.

**Precondition**: Phase 3 complete (kernel compiles and runs)

**Read first**:
- `references/validation-defaults.md`
- `validation/E2E_LATENCY_GUIDE.md`

**BLOCKING Requirements**:
- [ ] Baseline MUST be vLLM's `fused_experts`/`fused_moe` (NOT naive PyTorch)
- [ ] Tests MUST include `torch.allclose()` numerical comparison
- [ ] Benchmarks MUST run with production parity (torch.compile enabled)
- [ ] **Kernel benchmarks MUST use CUDA graph capture** (not raw launches)
- [ ] ALL kill criteria MUST be evaluated (no "TODO" or "optional")
- [ ] `verify_phase4_gates.py` MUST return exit code 0

**Do NOT reorder gates. Each gate has preconditions.**

### Gate 4.1: Correctness (BLOCKING)
**Precondition**: None (first gate)
**Pass criteria**:
```python
assert torch.allclose(monokernel_out, vllm_baseline_out, atol=TOL, rtol=TOL)
```
- [ ] Import vLLM's `fused_experts` or `fused_moe` as baseline
- [ ] Compare outputs for ALL target buckets
- [ ] Use tolerances from validation-defaults.md
- [ ] If top_k>1: test overlap/reduction edge cases

**If fails**: STOP. Use `references/investigation-playbook.md` § Correctness.
**Cannot proceed to 4.2 until 4.1 PASSES.**

### Gate 4.2: Kernel Perf (BLOCKING)
**Precondition**: Gate 4.1 PASSED (check `state.json` gate_4_1 status)
**Pass criteria**:
```python
for bs in buckets:
    assert opt_time[bs] <= baseline_time[bs], f"Regression at BS={bs}"
```
- [ ] Baseline = vLLM's fused_moe kernel (NOT naive PyTorch)
- [ ] Measure GPU kernel time under CUDA graphs + torch.compile
- [ ] Gate: optimized ≤ baseline for EVERY validated bucket
- [ ] NCU: verify no occupancy/spill regression

**If fails**: STOP. Use `references/investigation-playbook.md` § Kernel Perf.
**Cannot proceed to 4.3 until 4.2 PASSES.**

### Gate 4.3: E2E Latency (BLOCKING)
**Precondition**: Gate 4.2 PASSED
**Pass criteria**: Documented improvement OR explicit infeasibility justification
- [ ] Run `vllm bench latency` with production settings (both baseline and opt)
- [ ] Report Δ vs baseline WITH ACTUAL NUMBERS
- [ ] Relate to MoE share using `references/e2e-delta-math.md`
- [ ] All kill criteria from Phase 2 MUST have results

**Exit Verification** (MUST PASS):
```bash
python scripts/verify_phase4_gates.py {artifact_dir}
# Exit code 0 = proceed to Phase 5
# Exit code 1 = BLOCKED, fix issues first, do NOT declare SHIP
```

### Phase 4 Completion Requirements (ENFORCEMENT)

**Before marking Phase 4 complete, ALL of these must be true:**

1. ✅ `verify_phase4_gates.py` returns exit code 0
2. ✅ state.json contains `verification_run.phase4.status = "PASS"`
3. ✅ Kernel benchmarks use CUDA graph capture (not raw launches)
4. ✅ No `TORCH_COMPILE_DISABLE=1` in any benchmark script
5. ✅ validation_results.md documents CUDA graph methodology

**If ANY check fails**: Do NOT mark phase complete. Do NOT proceed to Phase 5.

**State update MUST include**:
```json
{
  "verification_run": {
    "phase4": {
      "script": "verify_phase4_gates.py",
      "status": "PASS",
      "date": "YYYY-MM-DD"
    }
  }
}
```

**Exit**: `validation_results.md` with correctness + kernel + e2e results.
**State update**: `"phase": "5_integration"` ONLY if verification passes AND state.json contains verification_run.

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
