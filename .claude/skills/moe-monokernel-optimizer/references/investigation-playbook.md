# Phase 4 Investigation Playbook

Use when Phase 4 gates fail. Goal: collect evidence → form hypothesis → implement minimal fix → re-measure.

## Search Anchors

needs_investigation, correctness, NaNs, CUDA graphs, graph breaks, stream mismatch, occupancy, spills, barrier stalls, nsys, ncu

## Hard Limits (Avoid Thrash)

Per distinct failure mode:
- Max **3** hypothesis → implement → measure cycles
- Max **2** NCU deep dives
- If still blocked: invoke llm-council or exit with `escalate_human`

## Common Prerequisites

Before investigating:
1. **Reproduce**: Record exact bucket(s), dtype, TP/EP, CUDA graphs mode
2. **Confirm baseline parity**: Same model, weights, knobs, capture mode
3. **Collect artifacts**: constraints.md, optimization_plan.md, failure logs, traces

---

## Investigation 4.1: Correctness Failure

**Symptom**: Numerical mismatch, NaNs/INFs, wrong expert selection, crashes.

### Steps

1. **Characterize divergence**:
   - CUDA graphs only? (graph-safety issue)
   - Bucket-specific? (BS=1 only?)
   - Stage-specific? (routing vs quant vs GEMM vs reduce)

2. **Enable debug mode**:
   ```bash
   CUDA_LAUNCH_BLOCKING=1
   TORCH_SHOW_CPP_STACKTRACES=1
   ```

3. **Binary search the bad stage**:
   - Compare intermediate tensors vs reference
   - Routing outputs → prepared indices → quantized activations → partial results

4. **Check common culprits**:
   - Wrong routing math (softmax vs sigmoid, renorm, scaling)
   - Wrong expert indexing (E_local vs E_global under EP)
   - Stream/device mismatch
   - Hidden allocations during capture
   - Accumulation semantics (overlap requires atomics)

5. **Propose fix + success criteria**

### Output Template

Write `{artifact_dir}/investigation/correctness.md`:

```markdown
## Repro
- Buckets: ...
- Knobs: CUDA graphs=..., torch.compile=...
- Symptom: ...

## First Divergence Point
- Stage: ...
- Evidence: ...

## Hypothesis
...

## Fix
- Files/lines:
- Why this matches verified semantics:

## Success Criteria
- Correctness passes on buckets ...

## Next Step if Fails
- (next hypothesis) OR (invoke council)
```

---

## Investigation 4.2: Kernel Performance Regression

**Symptom**: Optimized kernels slower than baseline on GPU time.

### Steps

1. **Verify GPU kernel time measurement**:
   - Use CUDA events for microbench
   - nsys for attribution
   - Ignore "launch count" under CUDA graphs

2. **Identify slow stage**:
   - Per-stage timings vs baseline
   - Is regression in routing/prepare, act+quant, GEMM, reduce, or barriers?

3. **Run focused NCU profile** on slowest kernel

4. **Map symptoms → causes**:

| Symptom | Likely Cause | Typical Fix |
|---------|--------------|-------------|
| High barrier stalls | Too many grid.sync | Reduce barriers, change ownership |
| Low TC util | MMA not firing | Fix MMA path, tile sizes |
| Low occupancy | Register/SMEM blowup | Reduce unrolling, shrink SMEM |
| High DRAM BW | Excessive intermediates | Fuse to remove hop |
| Underfill at small BS | Too few CTAs | Split-H or token-major |

**Fast escape**: If barrier stalls high + TC util low + M_avg small → go back to Phase 2 to change fusion boundary.

### Output Template

Write `{artifact_dir}/investigation/kernel_perf.md`:

```markdown
## Repro
- Buckets: ...
- Baseline vs optimized: ... µs

## Attribution
- Slowest kernel/stage:
- nsys evidence:

## NCU Summary
- TC util:
- Occupancy / regs / spills:
- Top stall reasons:

## Hypothesis
...

## Fix
1. ...
2. ...

## Success Criteria
- Optimized GPU time ≤ baseline on buckets ...

## Next Step if Fails
- (next hypothesis) OR (back to Phase 2) OR (invoke council)
```

---

## Investigation 4.3: E2E Below Threshold

**Symptom**: Kernel microbench looks good, but vllm bench latency shows no win.

### Steps

1. **Verify optimized path is enabled**:
   - Confirm env var is set
   - Confirm dispatch hits intended envelope (add log)

2. **Compare kernel win vs expected e2e win**:
   - Use `references/e2e-delta-math.md`
   - Check if MoE share (f) can yield requested improvement

3. **Confirm baseline parity**:
   - Same CUDA graphs, torch.compile, bucketing, TP/EP

4. **Profile where time went**:
   - If MoE not dominant, kernel win may not move e2e
   - If non-MoE increased, you may have introduced graph breaks

### Output Template

Write `{artifact_dir}/investigation/e2e.md`:

```markdown
## Repro
- Command line: ...
- Buckets / input/output lens:
- Knobs: ...

## Enablement Check
- Evidence the optimized path ran:

## Kernel vs E2E
- Kernel delta:
- E2E delta:
- Expected bound (from e2e-delta-math):

## Hypothesis
...

## Fix / Next Move
- (tighten envelope, remove graph break) OR (stop: MoE not the bottleneck)

## Success Criteria
- E2E latency improves on target buckets
```

---

## Investigation Decisions

After investigation, pick one:

| Decision | Action |
|----------|--------|
| `retry_phase_3` | Back to Phase 3 with fix context |
| `retry_phase_2` | Back to Phase 2 (re-plan with new constraints) |
| `document_proceed` | Document limitation, proceed to Phase 5 |
| `rerun_validation` | Re-run validation (measurement issue) |
| `escalate_human` | Pause for human review |
