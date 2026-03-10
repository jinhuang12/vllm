# E2E Latency Benchmark Guide (vLLM) for Kernel Optimizations

Use this to validate that a kernel optimization improves real inference latency under production parity: CUDA graphs enabled, `torch.compile` enabled, and the same TP/EP topology plus serving knobs as production.

Default gates and reporting rules live in `references/validation-defaults.md`.

## Tool Selection

| Stage | Tool | Why |
|---|---|---|
| Stage 1 profiling | `nsys profile -- vllm bench latency` | capture traces for kernel analysis |
| Stages 5-6 validation | `.codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` | GPU-locked benchmark workflow |
| Development only | `vllm bench latency` directly | quick checks, not final validation evidence |

## Using Stage 1 Baselines

Implementers must use Stage 1 baseline numbers for E2E comparisons. Do not run a baseline from the worktree.

### Optimized-Only Run From the Worktree

```bash
cd .codex/worktrees/ammo-track-{op_id}
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

Compare that result against `{artifact_dir}/runs/baseline_bs8.json`.

Record this citation in `validation_results.md`:

`Baseline source: Stage 1 (not re-run)`

## Workload Selection

### Decode-Heavy

Recommended for decode-bucket optimizations:

- `--input-len 64`
- `--output-len 512`
- sweep the decode bucket set validated in Stage 1

### Prefill-Heavy

Run separately when claiming a prefill win. Keep it separate from decode-heavy reporting.

## Batch-Size Sweep

Use the same bucket set profiled in Stage 1.

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
  <ENABLE_FLAG>=1 vllm bench latency \
    --model <MODEL_ID> \
    --tensor-parallel-size <TP> \
    --max-model-len <MAX_LEN> \
    --input-len 64 \
    --output-len 512 \
    --batch-size ${BS} \
    --num-iters 5 \
    --output-json /tmp/opt_bs${BS}.json
done
```

</details>

Use Stage 1 baseline JSONs for comparison. Do not regenerate baselines from the worktree.

## Parity Checklist

- same model weights and revision
- same dtype and quantization
- same TP/EP topology and routing mode
- same CUDA graph and compile settings
- same scheduler knobs affecting bucketing
- evidence that the optimized path actually executed

### Debug-Only Eager Runs

Use eager mode only for debugging correctness, never for production-parity performance claims.

## Interpreting Output

```python
baseline_s = 10.95
opt_s = 10.20
speedup = baseline_s / opt_s
improvement_pct = (baseline_s - opt_s) / baseline_s * 100
```

If measured E2E improvement is small, sanity-check the result against component share `f` using `references/e2e-delta-math.md`.

## Troubleshooting

### Optimized Path Not Activating

- verify the enable flag or dispatch condition
- confirm the specialization exists for the target envelope
- confirm the benchmark hits the validated bucket set
- use Nsight Systems if needed to confirm the kernel path

### Kernel Faster but E2E Flat

Investigate dispatch coverage, graph breaks, benchmark envelope mismatch, or a different dominant bottleneck. Treat this as a blocking validation issue until explained.

### High Variance

- increase iterations and warmup
- ensure the GPU is isolated
- keep power and thermal conditions stable

## Recording Results

Write results to `{artifact_dir}/tracks/{op_id}/validation_results.md` with:

- full optimized repro commands and env vars
- Stage 1 baseline citation
- baseline versus optimized tables
- evidence that the optimized path executed
- Amdahl sanity math and any contamination notes
