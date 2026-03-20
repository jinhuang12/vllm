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

---

## Investigation 4.3: Batch-Size-Dependent Performance (Crossover Pattern)

### Symptom
Optimization improves at small batch sizes but regresses at large batch sizes (or vice versa).

### Common Root Causes
1. **Tile occupancy**: Custom kernel's tile size (e.g., BLOCK_M=32) provides high occupancy at small M but wave quantization overhead at large M
2. **Different cuBLAS kernel selection**: cuBLAS selects different kernel variants at different M sizes; the optimization may only beat one variant
3. **L2 cache regime**: At small M, working set fits in L2 (fast); at large M, spills to DRAM (slow). Optimization's memory access pattern may be more sensitive to this
4. **Triton autotuning**: Triton may select different tile configs at different M, and the optimization interacts differently with each config

### Diagnostic Steps
1. Compare nsys kernel dispatch traces at BS=small vs BS=large -- check if the same kernel variant fires at both
2. Run NCU at both batch sizes -- compare occupancy, register usage, memory throughput
3. Check warm-cache vs cold-cache speedup ratio at each BS -- if > 1.5x, the optimization is L2-sensitive

### Resolution
This is a `GATING_REQUIRED` pattern, not a `FAIL`. See `references/crossover-probing.md` for the probing protocol and `references/code-templates.md` for dispatch mechanisms.

### Escalation
If the crossover BS is very small (< 4), the beneficial envelope may be too narrow to justify shipping. Discuss with orchestrator.
