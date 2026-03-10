# E2E Latency Benchmark Guide (vLLM) for Kernel Optimizations

Use this to validate that a kernel optimization improves *real* inference latency under **production parity**:
- CUDA graphs enabled (or the exact mode used in production)
- torch.compile enabled (or the exact mode used in production)
- same TP/EP topology and serving knobs

This file focuses on **how to run and interpret** E2E benchmarks. Default gates + required reporting live in `references/validation-defaults.md`.

## Tool Selection

| Stage | Tool | Why |
|-------|------|-----|
| Stage 1 (profiling) | `run_vllm_bench_latency_sweep.py --nsys-profile` | E2E baseline + per-bucket nsys traces in one pass |
| Stages 5-6 (validation) | `run_vllm_bench_latency_sweep.py` | GPU-locked A/B comparison with fastpath evidence |
| Development | `vllm bench latency` directly | Quick single-BS checks only (GPU must be idle, not for validation_results.md) |

For all measurements reported in `validation_results.md` or used for profiling, use the sweep script.
The examples below show raw `vllm bench latency` for reference only — do not use them directly.

## Contents
- Quickstart (baseline vs optimized)
- Workload selection (decode-heavy vs prefill-heavy)
- Batch-size sweep
- Parity checklist (must match baseline)
- Interpreting output + speedup math
- Troubleshooting
- Recording results

## Search anchors
vllm bench latency, CUDA graphs, torch.compile, enforce-eager, input-len, output-len, batch-size sweep, activation.

## Using Stage 1 Baselines (Validators)

**Validators MUST use Stage 1 baseline numbers for E2E comparisons.** Do NOT run a baseline from the worktree.

**Rationale**: Worktrees contain optimized code. Running `vllm bench latency` without the optimization flag from a worktree may still execute the optimized code path (e.g., if `pip install -e .` overwrote the global editable install, or if the optimization has no explicit enable flag). This contaminates the baseline, making both runs use the optimized path and hiding real improvements behind noise.

**Optimized-only run (validator workflow)**:

```bash
cd .claude/worktrees/ammo-track-{op_id}
source .venv/bin/activate
<ENABLE_FLAG>=1 vllm bench latency \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --max-model-len <MAX_LEN> \
  --input-len 64 \
  --output-len 512 \
  --batch-size 8 \
  --num-iters 5 \
  --output-json {artifact_dir}/tracks/{op_id}/opt_bs8.json
```

Then compare against the Stage 1 baseline:

```python
import json
baseline = json.load(open("{artifact_dir}/runs/baseline_bs8.json"))
optimized = json.load(open("{artifact_dir}/tracks/{op_id}/opt_bs8.json"))
speedup = baseline["avg_latency"] / optimized["avg_latency"]
```

Record in `validation_results.md`: "Baseline source: Stage 1 (not re-run)"

## Quickstart

Run the sweep script from the artifact directory (which contains `target.json`):

```bash
# Stage 1: E2E baseline + nsys profiling in one pass
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --nsys-profile

# Stages 5-6: Validation sweep (no nsys)
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir}
```

The sweep script reads model, workload, and env config from `target.json` — no need to specify `--model`, `--dtype`, `--batch-size`, etc. on the command line.

<details><summary>Raw vllm bench latency commands (development reference only)</summary>

```bash
# Baseline
vllm bench latency \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --max-model-len <MAX_LEN> \
  --input-len 64 \
  --output-len 512 \
  --batch-size 8 \
  --num-iters 5 \
  --output-json /tmp/baseline_bs8.json

# Optimized (replace <ENABLE_FLAG> with your optimization's flag)
<ENABLE_FLAG>=1 vllm bench latency \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --max-model-len <MAX_LEN> \
  --input-len 64 \
  --output-len 512 \
  --batch-size 8 \
  --num-iters 5 \
  --output-json /tmp/opt_bs8.json
```

</details>

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

Use the sweep script for automated multi-bucket benchmarking (loads model once per label):

```bash
python scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir}
```

The script reads `target.json` for workload config. Supports both the flat format
(`input_len`, `output_len`, `batch_sizes`) and `workload_matrix` for multi-dimensional
`(input_len x output_len x batch_size)` sweeps.

To also capture per-bucket nsys profiles during the sweep (avoiding model reload):

```bash
python scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir} --nsys-profile
```

This produces one `.nsys-rep` per bucket in `{artifact_dir}/e2e_latency/nsys/`.

<details><summary>Manual equivalent (development only, NOT for validation)</summary>

```bash
for BS in 8 32; do
  echo "=== batch_size=$BS ==="

  vllm bench latency \
    --model <MODEL_ID> \
    --tensor-parallel-size <TP> \
    --max-model-len <MAX_LEN> \
    --input-len 64 --output-len 512 \
    --batch-size $BS \
    --num-iters 5 \
    --output-json /tmp/baseline_bs${BS}.json

  <ENABLE_FLAG>=1 vllm bench latency \
    --model <MODEL_ID> \
    --tensor-parallel-size <TP> \
    --max-model-len <MAX_LEN> \
    --input-len 64 --output-len 512 \
    --batch-size $BS \
    --num-iters 5 \
    --output-json /tmp/opt_bs${BS}.json
done
```

</details>

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

### “No E2E win” even though microbench is faster
Common causes:
- Target component is a small fraction of end-to-end (`f` small) → expected E2E gain is bounded (see `references/e2e-delta-math.md`).
- Graph breaks or unexpected fallbacks (different kernels between baseline and optimized runs).
- Another bottleneck dominates (attention/KV/cache/scheduler).

### High variance / poor reproducibility
- Increase iterations.
- Ensure the GPU is isolated (no other jobs), not power/thermal limited.
- Use a consistent warmup protocol.

## Recording results

Write results to `{artifact_dir}/validation_results.md` with:
- full repro commands (baseline + optimized) and env vars
- the bucket set and capture/compile settings
- baseline vs optimized tables (speedup + improvement)
- evidence that the optimized path executed

Use `references/validation-defaults.md` for the minimum reporting template and default gates.
