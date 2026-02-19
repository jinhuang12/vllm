# Validation Defaults and Reporting (Phase 4)

Use this as the **default** guidance for Phase 4 validation artifacts (`{artifact_dir}/validation_results.md`).

Always prefer **model-specific** tolerances and invariants from:
- existing vLLM tests for that model (best), or
- a model-specific baseline doc (example: `validation/QWEN3_BASELINE.md`).

## Contents
- Default correctness tolerances (starting points)
- Default kernel perf gate (Phase 4.2)
- Default end-to-end gate (Phase 4.3)
- Required reporting checklist for `validation_results.md`

## Search anchors
tolerance, atol, rtol, speedup, CUDA graphs parity, torch.compile parity, kernel perf gate, E2E gate, enablement evidence.

## Default correctness tolerances (starting points)

These are *starting points*, not universal truths.

- **FP32**: `atol=1e-3`, `rtol=1e-3`
- **BF16/FP16**: `atol=1e-2`, `rtol=1e-2`
- **FP8 / block-quant**: **must be model-specific**.
  - As a placeholder, you may start with something like `atol=300`, `rtol=0.5` (see Qwen3 example),
  - but you should copy tolerances from the model’s actual tests whenever possible.

Also require:
- no NaNs/Infs
- shape/stride parity
- deterministic indexing for routing and pair ordering (when required by baseline)

## Default kernel perf gate (Phase 4.2)

Measure **GPU kernel time** under CUDA graphs for the same bucket set as Phase 1.

Default gate (safe, conservative):
- **No regressions allowed**: `T_opt_bucket_us ≤ T_base_bucket_us` for *every* validated bucket in the fast-path envelope.

Reporting requirements:
- baseline vs optimized per-bucket table (µs + speedup)
- if possible, per-stage breakdown (routing/prepare/W1/act/quant/W2/reduce)
- NCU sanity check for the dominant GEMM(s): occupancy, regs/thread, spills, SMEM/CTA

If you are intentionally trading a small regression in one bucket for a larger gain elsewhere, you must:
- explicitly document it,
- restrict the enablement envelope accordingly, or
- add a dispatch that picks the best variant per bucket.

## Default end-to-end gate (Phase 4.3)

Run E2E under identical knobs and capture/compile settings.

Default targets (adjust to business needs and MoE share `f`):
- `BS ∈ {1, 4, 8}`: **≥ 5%** improvement
- `BS ∈ {16, 32, 64}`: **> 0%** improvement

If MoE is a small fraction of end-to-end, these targets may be physically impossible. Use `references/e2e-delta-math.md` to set realistic expectations.

## Required reporting checklist for `{artifact_dir}/validation_results.md`

Include:

1) **Repro commands**
- exact commands for baseline and optimized runs
- env vars and flags that affect dispatch / CUDA graphs / torch.compile / quant

2) **Environment**
- GPU model + driver/CUDA
- vLLM commit or version
- model id + quant format
- TP/EP topology

3) **Correctness**
- tolerance used + rationale
- max/mean absolute error (and any outliers)
- special-case tests for top_k>1 (overlap / reduction)

4) **Kernel perf (production parity)**
- bucket set and capture mode
- baseline vs optimized per-bucket µs table
- profiler evidence that kernels were graph-captured and fast-path executed

5) **E2E latency**
- baseline vs optimized per-bucket table
- variance notes (iters, warmup, noise sources)
- connection to MoE share `f` (if improvement is small)

6) **Decision**
- ship / restrict envelope / pivot route / stop
- Phase 5 enablement guard proposal (what exactly will be enabled, where, and how to roll back)
