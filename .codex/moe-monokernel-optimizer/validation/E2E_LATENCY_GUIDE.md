# E2E Latency Benchmark Guide (vLLM) for MoE Optimizations

Use this for **Phase 4.3** to validate that a MoE optimization (monokernel / hybrid fusion / split-kernel) improves *real* inference latency under **production parity**:
- CUDA graphs enabled (or the exact mode used in production)
- torch.compile enabled (or the exact mode used in production)
- same TP/EP topology and serving knobs

This file focuses on **how to run and interpret** E2E benchmarks. Default gates + required reporting live in `references/validation-defaults.md`.

## Contents
- Quickstart (baseline vs optimized)
- Workload selection (decode-heavy vs prefill-heavy)
- Batch-size sweep
- Parity checklist (must match baseline)
- Interpreting output + speedup math
- Troubleshooting
- Recording results

## Search anchors
vllm bench latency, VLLM_USE_MOE_MONOKERNEL, CUDA graphs, torch.compile, enforce-eager, input-len, output-len, batch-size sweep, activation.

## Quickstart

## Ensure model weights are present (do not skip)

If the model is not already cached locally, download the weights before benchmarking.

Common options:

```bash
# Fast download (if hf_transfer is available in your env)
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download into the default HF cache (usually ~/.cache/huggingface/hub)
huggingface-cli download <MODEL_ID> --local-dir-use-symlinks False
```

Notes:
- If the repo is gated and requires acceptance/auth, the download will fail until credentials/terms are satisfied.
- Do not mark E2E as “NOT RUN” just because the cache was empty; either download, or mark the run as blocked and ask the user for an explicit waiver.

### Baseline run

```bash
vllm bench latency \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --max-model-len <MAX_LEN> \
  --input-len 64 \
  --output-len 512 \
  --batch-size 8 \
  --num-iters 20 \
  --output-json /tmp/baseline_bs8.json
```

### Optimized run (example: monokernel fast-path)

> Use the correct enable flag for *your* optimization (env var / config).  
> For the monokernel path, the common flag is `VLLM_USE_MOE_MONOKERNEL=1`.

```bash
VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
  --model <MODEL_ID> \
  --tensor-parallel-size <TP> \
  --max-model-len <MAX_LEN> \
  --input-len 64 \
  --output-len 512 \
  --batch-size 8 \
  --num-iters 20 \
  --output-json /tmp/opt_bs8.json
```

## Workload selection

### Decode-heavy (recommended for decode-bucket optimizations)

Many MoE fast-paths are tuned for **decode buckets** (small `M` per step). A decode-heavy benchmark makes that visible:

- `--input-len 64` (short prefill)
- `--output-len 512` (long decode)
- Sweep your decode bucket `--batch-size` set

### Prefill-heavy (optional)

If you claim a prefill win, run a second benchmark with a large input length (and usually smaller output length). Keep this separate from decode-heavy results.

## Batch-size sweep

Use the **same bucket set** you profiled in Phase 1 and plan to enable in Phase 5.

```bash
for BS in 1 4 8 16 32 64; do
  echo "=== batch_size=$BS ==="

  vllm bench latency \
    --model <MODEL_ID> \
    --tensor-parallel-size <TP> \
    --max-model-len <MAX_LEN> \
    --input-len 64 --output-len 512 \
    --batch-size $BS \
    --num-iters 20 \
    --output-json /tmp/baseline_bs${BS}.json

  VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
    --model <MODEL_ID> \
    --tensor-parallel-size <TP> \
    --max-model-len <MAX_LEN> \
    --input-len 64 --output-len 512 \
    --batch-size $BS \
    --num-iters 20 \
    --output-json /tmp/opt_bs${BS}.json
done
```

## Parity checklist (must match baseline)

If this checklist is not satisfied, your numbers are not trustworthy.

- Same model weights + same revision/commit.
- Same dtype/quantization (FP8 formats and scale shapes matter).
- Same TP/EP topology and identical routing/dispatch mode.
- Same CUDA graphs mode and torch.compile mode.
- Same scheduler knobs that affect bucketing (e.g., max batched tokens / chunked prefill).
- Confirm **optimized fast-path actually executed**:
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

If your measured E2E improvement is small, sanity-check the MoE share using `references/e2e-delta-math.md`.

## Troubleshooting

### Optimized path not activating
- Verify the enable flag is set (env var / config).
- Verify a compiled specialization exists for your `(dtype, TP/EP, bucket set)`.
- Verify your bucket guard matches the validated envelope.
- Use Nsight Systems to confirm which kernels run under the captured graph.

### “No E2E win” even though microbench is faster
Common causes:
- MoE is a small fraction of end-to-end (`f` small) → expected E2E gain is bounded (see `references/e2e-delta-math.md`).
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
