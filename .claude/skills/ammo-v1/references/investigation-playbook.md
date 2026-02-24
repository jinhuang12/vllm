# Stage 5 Investigation Playbook

Use when Stage 5 gates fail. Goal: collect evidence → form hypothesis → implement minimal fix → re-measure.

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

**Symptom**: Numerical mismatch, NaNs/INFs, wrong kernel output, crashes.

### Steps

1. **Characterize divergence**:
   - CUDA graphs only? (graph-safety issue)
   - Bucket-specific? (BS=1 only?)
   - Stage-specific? (input processing vs computation vs output)

2. **Enable debug mode**:
   ```bash
   CUDA_LAUNCH_BLOCKING=1
   TORCH_SHOW_CPP_STACKTRACES=1
   ```

3. **Binary search the bad stage**:
   - Compare intermediate tensors vs reference
   - Input processing → computation → output

4. **Check common culprits**:
   - Wrong component semantics (verify against vLLM source)
   - Component-specific indexing
   - Stream/device mismatch
   - Hidden allocations during capture
   - Output accumulation semantics

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
   - Identify per-stage attribution

3. **Run focused NCU profile** on slowest kernel

4. **Map symptoms → causes**:

| Symptom | Likely Cause | Typical Fix |
|---------|--------------|-------------|
| High barrier stalls | Too many grid.sync | Reduce barriers, change ownership |
| Low TC util | MMA not firing | Fix MMA path, tile sizes |
| Low occupancy | Register/SMEM blowup | Reduce unrolling, shrink SMEM |
| High DRAM BW | Excessive intermediates | Fuse to remove hop |
| Underfill at small BS | Too few CTAs | Split-H or token-major |

**Fast escape**: If barrier stalls high + TC util low + M_avg small → go back to Stage 3 to change fusion boundary.

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
   - Check if component share (f) can yield requested improvement

3. **Confirm baseline parity**:
   - Same CUDA graphs, torch.compile, bucketing, TP/EP

4. **Profile where time went**:
   - If target component not dominant, kernel win may not move e2e
   - If non-target kernels increased, you may have introduced graph breaks

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
- (tighten envelope, remove graph break) OR (stop: target component not the bottleneck)

## Success Criteria
- E2E latency improves on target buckets
```

---

## Investigation Decisions

After investigation, pick one:

| Decision | Action |
|----------|--------|
| `retry_stage_4` | Back to Stage 4 with fix context |
| `retry_stage_3` | Back to Stage 3 (re-plan with new constraints) |
| `document_proceed` | Document limitation, proceed to Stage 6 |
| `rerun_validation` | Re-run validation (measurement issue) |
| `escalate_human` | Pause for human review |

---

## Implementer-Verifier Loop

When the verifier rejects validation (T17/T18/T19 failure), the lead mediates a bounded fix cycle. The verifier does NOT communicate fix instructions directly to the implementer.

### Loop Flow

```
Verifier runs T17/T18/T19 → finds failure
  → Verifier sends rejection + evidence to Lead (see communication-patterns.md)
    → Lead reviews evidence, decides:
      a) retry_stage_4: creates fix task for Implementer (cycle N+1)
      b) retry_stage_3: creates re-plan task for Planner
      c) escalate_human: pauses for human review
      d) document_proceed: limitation acknowledged, restrict envelope
    → If (a): Implementer fixes → Lead runs T15 gate → Verifier re-validates
    → Loop bounded by max 3 cycles
```

### Termination Rules

1. **Hard limit**: Max 3 implementer fix cycles per distinct failure mode
2. **Convergence test**: If the last 2 cycles show <1% improvement in the failing metric, STOP — the approach is unlikely to work with incremental fixes
3. **Scope escalation**: If the fix requires changing the optimization approach (not just parameter tuning), go back to Stage 3 (`retry_stage_3`) — do not consume fix cycles on fundamentally wrong approaches

### Why the lead must mediate

- Prevents infinite loops (neither implementer nor verifier has incentive to declare "fundamentally wrong approach")
- Prevents convergence on local optima (small fixes that pass verifier but don't improve E2E)
- Maintains context tracking (lead tracks cycle count, can escalate at any point)
- Preserves deterministic gate enforcement (lead re-runs T15 between each cycle)

### Cycle Tracking

The lead tracks cycles in state.json:

```json
{
  "validation_cycles": {
    "gate_5_1_correctness": {"count": 1, "last_metric": 0.005},
    "gate_5_2_kernel_perf": {"count": 0},
    "gate_5_3_e2e": {"count": 2, "last_metric": -0.3}
  }
}
```
