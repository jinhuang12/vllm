# Validator Troubleshooting Guide

Extracted from the Stage 5 Investigation Playbook for use when validation gates fail.

## Search Anchors

needs_investigation, correctness, NaNs, CUDA graphs, graph breaks, stream mismatch, occupancy, spills, barrier stalls, nsys, ncu

## Hard Limits (Avoid Thrash)

Per distinct failure mode:
- Max **3** hypothesis → implement → measure cycles
- Max **2** NCU deep dives
- If still blocked: escalate to the main session with evidence

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
- (next hypothesis) OR (escalate to main session)
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

**Fast escape**: If barrier stalls high + TC util low + M_avg small → escalate to main session to change optimization approach.

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
- (next hypothesis) OR (escalate to main session)
```
