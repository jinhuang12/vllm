# E2E Latency Benchmark Guide for MoE Monokernel

This guide explains how to benchmark monokernel impact on real vLLM inference latency.

## Why E2E Latency Matters

Kernel-level benchmarks measure pure kernel execution time, but E2E latency benchmarks show **real-world impact** including:
- Model loading and initialization
- CUDA graphs and torch.compile overhead (amortized)
- KV cache management
- Scheduler overhead
- All model layers (not just MoE)

## Multi-GPU / EP Notes

- If EP is enabled, verify routing/dispatch happens consistently for baseline and monokernel.
- Use the same TP/EP topology and identical CUDA graph capture settings.
- Record per‑expert token distribution if possible; skew can dominate E2E variance.

## Understanding When Monokernel Activates

Monokernel activation depends on the dispatch rules for the model, hardware, and configuration (e.g., M_avg threshold, ownership, and batch size buckets).
Always confirm activation via logs or instrumentation in the dispatch path.

**General rule**:
- If dims match a compiled monokernel variant **and** M_avg is low, monokernel can activate.
- Otherwise, it should fall back to fused_moe or a split-kernel path.

## Benchmark Configuration

### Optimal Test Setup

To best showcase monokernel impact, use **decode-heavy workloads**:

```bash
--input-len 64    # Short prefill (may or may not use monokernel)
--output-len 512  # Long decode (monokernel used for all decode steps)
```

### Representative Batch Sizes

| Batch Size | Expected Improvement | Use Case |
|------------|---------------------|----------|
| 4 | Often best | Sweet spot for launch overhead savings |
| 8 | Often good | Typical low-concurrency serving |
| 32 | Sometimes marginal | Moderate load |
| 64 | Often minimal | Edge of monokernel range |

## Running Benchmarks

### Step 1: Baseline (Standard FusedMoE)

```bash
source /path/to/vllm/.venv/bin/activate

vllm bench latency \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --input-len 64 \
    --output-len 512 \
    --batch-size 8 \
    --num-iters 20 \
    --output-json /tmp/baseline_bs8.json
```

### Step 2: Monokernel

```bash
VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --input-len 64 \
    --output-len 512 \
    --batch-size 8 \
    --num-iters 20 \
    --output-json /tmp/monokernel_bs8.json
```

### Step 3: Batch Size Sweep

```bash
for BS in 4 8 32; do
    echo "=== Testing batch_size=$BS ==="

    # Baseline
    vllm bench latency \
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --input-len 64 --output-len 512 \
        --batch-size $BS \
        --num-iters 10 \
        --output-json /tmp/baseline_bs${BS}.json 2>&1 | grep "Avg latency"

    # Monokernel
    VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
        --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
        --tensor-parallel-size 1 \
        --max-model-len 4096 \
        --input-len 64 --output-len 512 \
        --batch-size $BS \
        --num-iters 10 \
        --output-json /tmp/monokernel_bs${BS}.json 2>&1 | grep "Avg latency"
done
```

## Understanding the Output

### Key Metrics

```
Avg latency: 10.945533825195161 seconds
10% percentile latency: 10.601871944294544 seconds
50% percentile latency: 10.906423719003215 seconds
90% percentile latency: 11.236299700191012 seconds
99% percentile latency: 11.650329866564133 seconds
```

- **Avg latency**: Total time for batch_size × output_len tokens
- **Percentile latencies**: Variance in iteration times

### Calculating Speedup

```python
baseline_latency = 10.95  # seconds
monokernel_latency = 10.20  # seconds

speedup = baseline_latency / monokernel_latency  # 1.07x
improvement = (baseline_latency - monokernel_latency) / baseline_latency * 100  # 6.9%
```

## Default Optimizations

`vllm bench latency` automatically enables performance optimizations:

| Optimization | Default | Effect |
|-------------|---------|--------|
| CUDA Graphs | `FULL_AND_PIECEWISE` | Eliminates kernel launch overhead |
| torch.compile | `VLLM_COMPILE` | Inductor-based model compilation |
| Prefix Caching | **Disabled** (for benchmark) | Clean latency measurements |
| Chunked Prefill | Enabled | max_num_batched_tokens=2048 |

### Baseline Parity Checklist
- Same TP/EP settings
- Same CUDA graph mode
- Same torch.compile mode
- Same batch size bucketing
- Same model weights and quantization
- Confirm CUDA graphs capture MoE kernels (no fallback paths)

To run in eager mode (without optimizations):
```bash
vllm bench latency --enforce-eager ...
```

## Expected Results Reference (Example Only)

### Qwen3-Coder-30B-A3B-FP8 on L40S (December 2024)

| Batch Size | Baseline (s) | Monokernel (s) | Speedup |
|------------|-------------|----------------|---------|
| 1 | 5.10 | 4.74 | 1.08x |
| 4 | 8.40 | 7.48 | **1.12x** |
| 8 | 10.95 | 10.20 | 1.07x |
| 16 | 14.76 | 13.65 | 1.08x |
| 32 | 18.79 | 17.95 | 1.05x |
| 64 | 22.82 | 22.41 | 1.02x |

## Troubleshooting

### Monokernel Not Activating

Check for a log message during model loading indicating monokernel activation.

If missing, verify:
1. `VLLM_USE_MOE_MONOKERNEL=1` is set
2. Model and dtype have a monokernel variant registered
3. TP/EP configuration matches the compiled specialization

### Similar Latency Between Baseline and Monokernel

Possible causes:
- Batch size too large (>64 tokens through MoE → fallback)
- Prefill-heavy workload (monokernel not used during prefill)
- Other bottlenecks dominating (attention, shared experts)

### Results Not Reproducible

Ensure:
1. Same number of warmup iterations
2. GPU is not thermal throttling
3. No other workloads on the GPU
4. Same vLLM version and configuration

## Integration with Phase 4 Validation

After running E2E latency benchmarks, document results in `{artifact_dir}/validation_results.md`:

```markdown
## E2E Latency Results

| Batch Size | Baseline (s) | Monokernel (s) | Speedup |
|------------|-------------|----------------|---------|
| 4 | X.XX | Y.YY | Z.ZZx |
| 8 | X.XX | Y.YY | Z.ZZx |
| 32 | X.XX | Y.YY | Z.ZZx |

### Recommendation
- Enable monokernel for batch_size ≤ {threshold}
- Expected improvement: {X}% for typical serving load
```
