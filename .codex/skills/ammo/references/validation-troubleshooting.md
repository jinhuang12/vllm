# Validation Troubleshooting Guide

Use this when an implementer track fails correctness, kernel, or E2E validation.

## Hard Limits

Per distinct failure mode:

- max 3 hypothesis -> fix -> measure loops
- max 2 deep NCU dives
- escalate to the lead if still blocked

## Common Prerequisites

Before debugging:

1. reproduce the failure on the same bucket, dtype, TP/EP, and graph mode
2. confirm Stage 1 baseline and production-parity settings
3. collect `constraints.md`, `optimization_plan.md`, `validation_results.md`, logs, and traces

## Investigation 5.1: Correctness Failure

Check:

- first divergence point in intermediate tensors
- bucket specificity
- graph-only versus non-graph behavior
- stream and device mismatches
- hidden allocations during capture
- reduction or routing semantics

Write findings to `{artifact_dir}/investigation/correctness.md`.

## Investigation 5.2: Kernel Performance Regression

Check:

- whether both paths were timed under CUDA graphs
- per-stage attribution against baseline
- occupancy, regs, spills, and stall reasons with NCU
- whether the optimization under-fills small buckets or increases synchronization

Write findings to `{artifact_dir}/investigation/kernel_perf.md`.

## Investigation 5.3: Kernel Faster but E2E Flat

Treat this as blocking until explained.

Check:

- optimized path activation in the end-to-end benchmark
- validated batch sizes versus benchmarked batch sizes
- graph breaks or unexpected fallbacks
- compile cache cold-start effects and insufficient warmup
- whether another component dominates the end-to-end path

Write findings to `{artifact_dir}/investigation/e2e_activation.md`.

## Investigation 5.4: Contamination or Baseline Reuse Failure

Check:

- worktree `.venv` isolation
- Stage 1 baseline citation in `validation_results.md`
- worktree baseline re-runs by mistake
- cross-track `.so` leakage when another track changed `csrc/`

Write findings to `{artifact_dir}/investigation/contamination.md`.
