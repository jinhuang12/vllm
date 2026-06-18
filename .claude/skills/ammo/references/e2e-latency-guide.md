# E2E Latency Benchmark Guide (vLLM) for Kernel Optimizations

Use this to validate that a kernel optimization improves *real* inference latency under **production parity**:
- CUDA graphs enabled (or the exact mode used in production)
- torch.compile enabled (or the exact mode used in production)
- same TP/EP topology and serving knobs

This file focuses on **how to run and interpret** E2E benchmarks. Default gates + required reporting live in `references/validation-defaults.md`.

## Tool Selection

| Stage | Tool | Why |
|-------|------|-----|
| Stage 1 (baseline) | `run_vllm_bench_latency_sweep.py --slot baseline` | Clean E2E timing — **no profiling flags allowed** (guard enforced). Authoritative numbers for speedup calculations. |
| Stage 1 (profiling) | `run_vllm_bench_latency_sweep.py --slot profiling --nsys-profile --nsys-mode node --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800` | Bounded selected-step nsys attribution capture. Add `--nsys-trace cuda-sw` on Blackwell (B200/B300). E2E numbers here are profiler-contaminated and not used for comparisons. See `nsys-profiling-guide.md`. |
| Stage 5 (validation) | `run_vllm_bench_latency_sweep.py` | GPU-locked A/B comparison with fastpath evidence |
| Stage 6 (integration) | `run_vllm_bench_latency_sweep.py --fresh-cache` | Gate-quality measurement with clean compile cache. Only runs when ≥2 tracks pass; single passer uses short-circuit (copies Stage 5 results). EXHAUSTED rounds skip entirely. Replaces the former T16 re-profile. |

For all measurements reported in `validation_results.md` or used for profiling, use the sweep script.

## Contents
- Quickstart (baseline vs optimized)
- Workload selection (decode-heavy vs prefill-heavy)
- Batch-size sweep
- Parity checklist (must match baseline)
- Interpreting output + speedup math
- Troubleshooting
- Recording results

## Quickstart

Run the sweep script from the artifact directory (which contains `target.json`):

```bash
# Stage 1a: Clean E2E baseline (no profiling — authoritative timing)
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round 1 --slot baseline --labels baseline \
  --capture-golden-refs

# Stage 1b: Profiling traces (separate invocation — overhead doesn't pollute baseline)
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round 1 --slot profiling --labels baseline \
  --nsys-profile --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% \
  --nsys-num-iters 1 --nsys-timeout-s 1800

# Stage 5: Per-track validation sweep (no nsys, no fresh-cache)
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir}

# Stage 6: Integration sweep (--fresh-cache for gate-quality measurement)
# This replaces the former T16 re-profile step.
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --fresh-cache

# Post-SHIP: Golden-refs capture (~15s, after env promotion)
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --labels baseline --capture-golden-refs --num-iters 1
```

The sweep script reads model, workload, and env config from `target.json` — no need to specify `--model`, `--dtype`, `--batch-size`, etc. on the command line.

### Fresh-cache isolation (`--fresh-cache`, v3.1)

Allocates `{out_root}/cache/{sweep_id}/` and injects `VLLM_CACHE_ROOT`
and `TRITON_CACHE_DIR` into the child env so the sweep does not inherit
warm compile caches from a previous sweep. Cache is removed on success.
First launch of N pays full compile (~5 min for large models); launches
2..N hit the warm in-sweep cache.

## Workload selection

### Decode-heavy (recommended for decode-bucket optimizations)

Many optimization fast-paths are tuned for **decode buckets** (small `M` per step). A decode-heavy benchmark makes that visible:

- `--input-len 64` (short prefill)
- `--output-len 512` (long decode)
- Sweep your decode bucket `--batch-size` set

### Prefill-heavy (optional)

If you claim a prefill win, run a second benchmark with a large input length (and usually smaller output length). Keep this separate from decode-heavy results.

## Batch-size sweep

Use the **same bucket set** you profiled in Stage 1 and plan to enable in Stage 6.

Use the sweep script for automated multi-bucket benchmarking (loads model once per label). Pass `--round {N} --slot {SLOT}` to write into the canonical round-scoped layout:

```bash
.venv/bin/python scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {CR} --slot baseline
```

`{SLOT}` is one of: `baseline` (Stage 1), `opt/{op_id}` (Stage 5 per-track), `integration` (Stage 6 combined sweep), `golden_capture` (post-SHIP golden-ref refresh).

The script reads `target.json` for workload config. Supports both the flat format
(`input_len`, `output_len`, `batch_sizes`) and `workload_matrix` for multi-dimensional
`(input_len x output_len x batch_size)` sweeps.

To also capture per-bucket nsys profiles, run a **separate short profiling invocation** after the clean baseline:

```bash
# Must use --slot profiling (--slot baseline + profiling flags = hard error)
.venv/bin/python scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {CR} --slot profiling --labels baseline \
  --nsys-profile --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% \
  --nsys-num-iters 1 --nsys-timeout-s 1800
```

This produces one `.nsys-rep` per bucket in `{artifact_dir}/rounds/{CR}/profiling/nsys/` (sibling to the sweep's results, NOT inside the sweep slot). The E2E numbers from this invocation are contaminated by nsys overhead and are NOT authoritative.
Append `--nsys-trace cuda-sw` on Blackwell (B200/B300); leave it omitted for Hopper/Ampere.
Use `--nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1` by default for profiling. The sweep shifts `input_len` and captures short selected-step windows with vLLM's CUDA profiler. Fall back to `--nsys-output-len 2 --nsys-num-iters 1` only if selected-step capture fails, and document the missing depth coverage.

## Parity checklist (must match baseline)

If this checklist is not satisfied, your numbers are not trustworthy.

- Same model weights + same revision/commit.
- Same dtype/quantization (FP8 formats and scale shapes matter).
- Same TP/EP topology and identical routing/dispatch mode.
- Same CUDA graphs mode and torch.compile mode.
- Same scheduler knobs that affect bucketing (e.g., max batched tokens / chunked prefill).
- Confirm **optimized path actually executed**:
  - enablement log line, or
  - instrumentation counter, or
  - an unmistakable kernel name in Nsight Systems.

### Debug-only: run eager

Use eager mode to debug correctness or functional issues (not for production-parity perf claims):

```bash
vllm bench latency --enforce-eager ...
```

## Interpreting output

`vllm bench latency` prints iteration latencies. The key value is typically **Avg latency**.

Compute speedup and improvement:

```python
baseline_s = 10.95
opt_s = 10.20

speedup = baseline_s / opt_s
improvement_pct = (baseline_s - opt_s) / baseline_s * 100
```

If your measured E2E improvement is small, sanity-check the component share using `references/e2e-delta-math.md`.

## Troubleshooting

### Optimized path not activating
- Verify the enable flag is set (env var / config).
- Verify a compiled specialization exists for your `(dtype, TP/EP, bucket set)`.
- Verify your bucket guard matches the validated envelope.
- Use Nsight Systems to confirm which kernels run under the captured graph.

### "No E2E win" even though microbench is faster
Common causes:
- Target component is a small fraction of end-to-end (`f` small) → expected E2E gain is bounded (see `references/e2e-delta-math.md`).
- Graph breaks or unexpected fallbacks (different kernels between baseline and optimized runs).
- Another bottleneck dominates (attention/KV/cache/scheduler).

### High variance / poor reproducibility
- Increase iterations.
- Ensure the GPU is isolated (no other jobs), not power/thermal limited.
- Use a consistent warmup protocol.

## Recording results

Write results to `{artifact_dir}/rounds/{CR}/tracks/{op_id}/validation_results.md` (per-track, Stage 5) or `{artifact_dir}/rounds/{CR}/integration_validation.md` (Stage 6 integration) with:
- full repro commands (baseline + optimized) and env vars
- the bucket set and capture/compile settings
- baseline vs optimized tables (speedup + improvement)
- evidence that the optimized path executed

Use `references/validation-defaults.md` for the minimum reporting template and default gates.
