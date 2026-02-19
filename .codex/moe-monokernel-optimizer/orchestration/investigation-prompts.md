# Phase 4 Investigation Playbook

Use this when Phase 4 gates fail (correctness, kernel perf, or end‑to‑end). The goal is to **collect evidence → form a bounded hypothesis → implement a minimal fix → re‑measure**.

## Contents
- Hard limits (avoid thrash)
- Common prerequisites (all investigations)
- Investigation 4.1: Correctness failure
- Investigation 4.2: Kernel performance regression
- Investigation 4.3: End‑to‑end below threshold

## Search anchors
needs_investigation, correctness, NaNs, CUDA graphs, graph breaks, stream mismatch, occupancy, spills, barrier stalls, nsys, ncu, enablement check, MoE share.

## Hard limits (avoid thrash)

Per distinct failure mode:
- Max **3** hypothesis → implement → measure cycles
- Max **2** Nsight Compute (NCU) deep dives
- If still blocked: invoke council (`orchestration/llm-council.md`) or exit with `escalate_human`

## Common prerequisites (all investigations)

1. **Reproduce**
   - Record the exact bucket(s): BS, seq lengths, dtype, TP/EP, CUDA graphs mode, torch.compile mode.
2. **Confirm baseline parity**
   - Same model, same weights/quantization, same knobs, same capture mode.
3. **Collect minimal artifacts**
   - `{artifact_dir}/constraints.md` baseline snapshot
   - `{artifact_dir}/optimization_plan.md`
   - `{artifact_dir}/validation_results.md` (or a prior run) and default gates from `references/validation-defaults.md`
   - logs for the failing run
   - (perf) one short **nsys** trace and (optionally) one focused **ncu** profile

## Investigation 4.1: Correctness failure

**Symptom**: numerical mismatch, NaNs/INFs, wrong expert selection, or crashes in the optimized path.

### Steps

1. Characterize the divergence
   - Does it happen only under CUDA graphs? (graph-safety suspicion)
   - Is it bucket-specific (e.g., BS=1 only)?
   - Is it tied to a specific stage (routing vs quant vs GEMM vs reduce)?

2. Turn on visibility (debug mode)
   - `CUDA_LAUNCH_BLOCKING=1`
   - `TORCH_SHOW_CPP_STACKTRACES=1`
   - If inside a custom op: add temporary asserts for shapes/dtypes/strides.

3. Binary search the first bad stage
   - Compare intermediate tensors (routing outputs, prepared indices, quantized activations, partial accumulators) vs reference.

4. Check the usual correctness culprits
   - Wrong routing math (softmax vs sigmoid, renorm, scaling, tie-breaks)
   - Wrong expert indexing (E_local vs E_global confusion under EP)
   - Stream/device mismatch (wrong stream or wrong device)
   - Hidden allocations during capture (graph breaks or invalid capture)
   - Accumulation/reduction semantics (overlap requires atomics or staged reduce)

5. Propose a minimal fix + success criteria
   - “Fix X in stage Y; verify by matching reference within tolerance on buckets {…} under CUDA graphs.”

### Output template

Write `{artifact_dir}/investigation/correctness.md`:

```markdown
# Correctness Investigation (4.1)

## Repro
- Buckets: …
- Knobs: CUDA graphs=…, torch.compile=…, TP/EP=…
- Symptom: …

## First divergence point
- Stage: …
- Evidence: (diff stats, tensor summaries, crash stack)

## Hypothesis
…

## Fix
- Files/lines:
- Why this matches verified semantics:

## Success criteria
- Correctness passes on buckets …
- CUDA graphs run does not break / crash

## Next step if fails
- (next hypothesis) OR (invoke council)
```

## Investigation 4.2: Kernel performance regression

**Symptom**: optimized kernels are slower than baseline on GPU time under CUDA graphs.

### Steps

1. Verify you are measuring **GPU kernel time**
   - Prefer CUDA events for microbench timing, and nsys for attribution.
   - Under CUDA graphs, ignore “launch count” arguments unless kernel time moved.

2. Identify the slow stage
   - Compare per-stage or per-kernel timings vs baseline.
   - Determine if the regression is in routing/prepare, act+quant, GEMM, reduce, or barriers.

3. Run one focused NCU profile on the slowest kernel
   - Look for: low Tensor Core utilization, low occupancy, high barrier stalls, high memory stalls, register spills.

4. Map symptoms → likely causes

| Symptom | Likely cause | Typical fix |
|---|---|---|
| High barrier stalls | too many `grid.sync()` / poor work per barrier | reduce barriers, change ownership/route |
| Low TC util | MMA not firing / bad tile | fix MMA path, adjust tile sizes, alignment |
| Low occupancy | register / SMEM blowup | reduce unrolling, shrink SMEM, split kernel |
| High DRAM BW | excessive intermediates | fuse to remove hop, improve layout |
| Underfill at small BS | too few CTAs | split‑H / token‑major / hybrid route |

**Fast escape hatch**: if barrier stalls are high, TC utilization is low, and `M_avg` is small → stop tuning and go back to **Phase 2** to change the fusion boundary/ownership.

### Output template

Write `{artifact_dir}/investigation/kernel_perf.md`:

```markdown
# Kernel Performance Investigation (4.2)

## Repro
- Buckets: …
- Baseline vs optimized: … ms (GPU time)

## Attribution
- Slowest kernel/stage:
- nsys evidence:

## NCU summary (slowest kernel)
- TC util:
- Occupancy / regs / spills:
- Top stall reasons:

## Hypothesis
…

## Fix
1. …
2. …

## Success criteria
- Optimized GPU time <= baseline GPU time on buckets …
- No regression in correctness

## Next step if fails
- (next hypothesis) OR (back to Phase 2 route decision) OR (invoke council)
```

## Investigation 4.3: End‑to‑end below threshold

**Symptom**: kernel microbench looks good, but `vllm bench latency` shows no win or regression.

### Steps

1. Verify the optimized path is actually enabled
   - Confirm the env var / guard is on.
   - Confirm dispatch hits the intended envelope (log once per run).

2. Compare kernel win vs expected end‑to‑end win
   - Use `references/e2e-delta-math.md` to sanity-check whether the MoE share (`f`) can yield the requested improvement.

3. Confirm baseline parity
   - Same CUDA graphs mode, torch.compile mode, batch bucketing, TP/EP, model length, etc.

4. Profile where time went
   - If MoE is not the dominant share, a large kernel win may not move e2e.
   - If non‑MoE increased, you may have introduced graph breaks or extra syncs.

### Output template

Write `{artifact_dir}/investigation/e2e.md`:

```markdown
# E2E Investigation (4.3)

## Repro
- Command line:
- Buckets / input/output lens:
- Knobs: …

## Enablement check
- Evidence the optimized path ran:

## Kernel vs E2E
- Kernel delta:
- E2E delta:
- Expected bound (from e2e-delta-math):

## Hypothesis
…

## Fix / next move
- (tighten envelope, remove graph break, re‑measure) OR (stop: MoE not the bottleneck)

## Success criteria
- E2E latency improves on target buckets without regressions
```
