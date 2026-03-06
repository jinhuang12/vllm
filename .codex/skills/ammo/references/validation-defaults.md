# Validation Defaults and Reporting (Stage 5)

Use this as the default guidance for per-track validation artifacts at `{artifact_dir}/tracks/{op_id}/validation_results.md`.

## Contents

- Dual baseline requirement
- E2E baseline reuse requirement
- Production parity requirement
- Default correctness tolerances
- Default kernel performance gate
- Default end-to-end gate
- Required reporting checklist

## Dual Baseline Requirement (Non-Negotiable)

Validation must compare against vLLM production kernels, not naive PyTorch reimplementations.

### Correctness Baseline

```python
from vllm.model_executor.layers.fused_moe import fused_experts

baseline_out = fused_experts(x, w1, w2, topk_weights, topk_ids, ...)
```

### Performance Baseline

Measure the same production call path that vLLM actually uses for the target component.

### Invalid Baselines

```python
for expert_idx in range(num_experts):
    expert_out = torch.matmul(x_expert, weights[expert_idx])
    output.index_add_(0, indices, expert_out)
```

Run `python .codex/skills/ammo/scripts/verify_validation_gates.py {artifact_dir}` and require exit code 0.

## E2E Baseline Reuse Requirement (Non-Negotiable)

Implementers must use Stage 1 baseline numbers for all E2E latency comparisons. Never re-run a baseline from the worktree.

### Source of Truth

| Data | Location | Captured by |
|---|---|---|
| Per-batch-size E2E latency | `{artifact_dir}/runs/baseline_bs{N}.json` | Stage 1 profiler on clean main |
| Summary table | `{artifact_dir}/constraints.md` | Stage 1 profiler |
| Kernel breakdown | `{artifact_dir}/constraints.md` | Stage 1 profiler |

### Required Citation

Include this in `validation_results.md`:

`Baseline source: Stage 1 (not re-run)`

### Sweep Script Guidance

If you use `run_vllm_bench_latency_sweep.py`, run optimized-only from the worktree for pass/fail decisions. Do not use a worktree-generated baseline in final reporting.

## Production Parity Requirement (Non-Negotiable)

All measurements must use production-equivalent settings.

### Required Environment

```bash
export VLLM_TORCH_COMPILE_LEVEL=3
export VLLM_USE_V1=1
```

CUDA graphs remain enabled unless production explicitly disables them.

### Forbidden Settings

```bash
export TORCH_COMPILE_DISABLE=1
--enforce-eager
VLLM_TORCH_COMPILE_LEVEL=0
```

### GPU Isolation Requirement

Benchmark results are invalid under GPU contention.

- Use `run_vllm_bench_latency_sweep.py` for validation E2E runs.
- Re-run any result gathered under contention.
- Stop immediately if another process starts using the same GPU set.

### Kernel-Level Benchmark Requirement

Capture both baseline and optimized paths in CUDA graphs before timing. Raw event timing without graph capture is not a valid Stage 5 kernel benchmark.

## Default Correctness Tolerances

Starting points only:

- FP32: `atol=1e-3`, `rtol=1e-3`
- BF16 / FP16: `atol=1e-2`, `rtol=1e-2`
- FP8 / block quant: copy model-specific tolerances whenever possible

Also require:

- no NaNs or INFs
- shape and stride parity
- deterministic indexing when the baseline depends on it

## Default Kernel Performance Gate (Gate 5.2)

- Require measurable improvement on at least one target bucket.
- No regressions on the remaining validated buckets unless the enablement envelope is narrowed explicitly.
- Report per-bucket baseline and optimized times in microseconds, plus weighted average speedup.
- Include the kernel speedup used for Amdahl sanity.

## Default End-to-End Gate (Gate 5.3)

Default expectations:

- target batch sizes: at least 3 percent improvement
- non-target batch sizes: no regression
- regressions are acceptable only when the optimization is explicitly gated to the improved envelope

Use `references/e2e-delta-math.md` to set realistic expectations when the component share `f` is small.

## Required Reporting Checklist

Include all of the following in `{artifact_dir}/tracks/{op_id}/validation_results.md`:

1. Repro commands with exact env vars and flags
2. Environment: GPU, CUDA, vLLM revision, model, quant format, TP/EP
3. Correctness: tolerances, max or mean error, edge cases
4. Kernel performance: per-bucket table, CUDA graph capture mode, fast-path evidence
5. E2E latency: Stage 1 baseline citation, per-batch-size table, warmup and iteration counts
6. Amdahl sanity: component share `f`, kernel speedup `s`, expected E2E improvement, actual E2E improvement
7. Cross-track contamination note: `PASS`, `N/A`, or documented red flag
8. Final decision: ship, restrict, pivot, or stop
