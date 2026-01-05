# Qwen3-Coder-30B-A3B-Instruct-FP8 Validation Baseline

## Contents
- Model Configuration
- Correctness Tolerances
- Performance Targets
- Multi-expert verification checklist
- Block quantization verification
- Benchmark + test scripts
- Profiling commands
- E2E latency benchmark results (example)
- Known issues

## Search anchors
Qwen3, L40S, FP8, top_k=8, block scales, tolerances, speedup target, vllm bench latency, CUDA graphs, torch.compile.


This file is model-specific. Do not generalize thresholds here to other MoE models.

## Model Configuration

| Parameter | Value |
|-----------|-------|
| Model | `Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8` |
| Hardware | L40S (sm_89, 142 SMs) |
| Hidden Size (K) | 2048 |
| Intermediate Size (N) | 768 (per-TP) |
| Number of Experts | 128 |
| Top-K | 8 |
| Tensor Parallelism | 1 |
| Quantization | FP8 E4M3 with 128x128 block scales |
| Weight Scale Shape (up) | `[128, 12, 16]` |
| Weight Scale Shape (down) | `[128, 16, 6]` |

## Correctness Tolerances

FP8 block-quantized kernels have inherently higher error than BF16 due to quantization noise.

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| Absolute Tolerance (atol) | 300 | Higher due to FP8 quantization |
| Relative Tolerance (rtol) | 0.5 (50%) | Some relative error expected |
| Max Single-Element Diff | 500 | Outliers possible in large tensors |
| Mean Absolute Error | < 50 | Average should be reasonable |

**Reference**: From `tests/kernels/moe/test_moe_monokernel_qwen3.py`:
```python
torch.testing.assert_close(
    monokernel_output,
    fused_moe_output,
    atol=300,
    rtol=0.5
)
```

## Performance Targets

Benchmarked on L40S vs stock `fused_moe`:

| Batch Size | Baseline (ms) | Target (ms) | Speedup Target |
|------------|--------------|-------------|----------------|
| 1 | 1.5-2.0 | < 0.8 | 2x |
| 2 | 1.8-2.2 | < 1.0 | 2x |
| 4 | 2.0-2.5 | < 1.2 | 1.8x |
| 8 | 2.5-3.0 | < 1.5 | 1.7x |
| 16 | 3.5-4.0 | < 2.5 | 1.4x |
| 32 | 5.0-6.0 | < 4.0 | 1.3x |
| 64 | 8.0-10.0 | < 7.0 | 1.2x |

**Note**: Above BS=64, monokernel advantage diminishes. Consider using stock fused_moe.

## Multi-Expert Verification Checklist (top_k=8)

- [ ] FP32 accumulator used (not BF16 atomicAdd)
- [ ] Routing weights sum to 1.0 per token (softmax normalized)
- [ ] All 8 expert contributions accumulated correctly
- [ ] Pair-to-token index mapping verified (`token_idx = pair_idx / TOP_K`)
- [ ] Output shape matches input shape `[BS, K]`
- [ ] No NaN or Inf in output

## Block Quantization Verification

- [ ] Up-projection scales loaded correctly: `[E, 12, 16]`
- [ ] Down-projection scales loaded correctly: `[E, 16, 6]`
- [ ] Scale applied per N-block during down-projection MMA
- [ ] Scale indexing: `k_block = k_offset / 128`, `n_block = base_col / 128`

## Benchmark Script

Location: `benchmarks/kernels/benchmark_moe_monokernel_qwen3.py`

```bash
# Run benchmark
python benchmarks/kernels/benchmark_moe_monokernel_qwen3.py \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-iters 100 \
    --warmup-iters 10

# Expected output format:
# BS=1: monokernel=0.75ms, fused_moe=1.52ms, speedup=2.03x
# BS=8: monokernel=1.02ms, fused_moe=2.48ms, speedup=2.43x
# ...
```

## Test Script

Location: `tests/kernels/moe/test_moe_monokernel_qwen3.py`

```bash
# Run correctness tests
pytest tests/kernels/moe/test_moe_monokernel_qwen3.py -v

# Expected tests:
# test_moe_monokernel_qwen3_bs8_correctness
# test_moe_monokernel_qwen3_bs64_correctness
# test_moe_monokernel_qwen3_determinism
# test_moe_monokernel_qwen3_scratchpad_size
```

## Profiling Commands

```bash
# NCU profiling
ncu --set full \
    --target-processes all \
    --export profile_moe_qwen3 \
    python benchmarks/kernels/benchmark_moe_monokernel_qwen3.py --batch-sizes 8

# Analyze with:
ncu --import profile_moe_qwen3.ncu-rep --csv > profile_moe_qwen3.csv
```

## E2E Latency Benchmark Results (December 2024)

Benchmarked using `vllm bench latency` on L40S with CUDA graphs and torch.compile enabled.

**Config**: input_len=64, output_len=512 (decode-heavy workload)

| Batch Size | Baseline (s) | Monokernel (s) | Speedup | Improvement |
|------------|-------------|----------------|---------|-------------|
| 1 | 5.10 | 4.74 | **1.08x** | 7.1% |
| 4 | 8.40 | 7.48 | **1.12x** | 11.0% |
| 8 | 10.95 | 10.20 | **1.07x** | 6.9% |
| 16 | 14.76 | 13.65 | **1.08x** | 7.6% |
| 32 | 18.79 | 17.95 | **1.05x** | 4.5% |
| 64 | 22.82 | 22.41 | **1.02x** | 1.8% |

**Key Insights**:
- Monokernel is a **decode-phase optimization** (activates when batch_size ≤ 64 tokens through MoE)
- Best improvement at **batch_size=4** (11% faster)
- Consistent improvement across all supported batch sizes (1-64)
- Improvement decreases as batch size increases (GEMM compute dominates)

### Running E2E Latency Benchmarks

```bash
# Baseline
vllm bench latency \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --input-len 64 --output-len 512 \
    --batch-size 8 \
    --num-iters 20 \
    --output-json /tmp/baseline_bs8.json

# Monokernel
VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --input-len 64 --output-len 512 \
    --batch-size 8 \
    --num-iters 20 \
    --output-json /tmp/monokernel_bs8.json
```

### Why Decode-Heavy Workloads?

The monokernel only activates when **batch_size ≤ 64 tokens** pass through the MoE layer:
- During **prefill** with input_len=1024: 1024 tokens → monokernel NOT used (fallback)
- During **decode** with batch_size=8: 8 tokens → monokernel IS used

This is why we use `--input-len 64 --output-len 512` (decode-heavy) to showcase monokernel impact.

## Known Issues

1. **BS=1/2 Split-H**: Very small batches may show higher variance due to atomicAdd contention
2. **FP8 Outliers**: Some expert combinations produce larger errors than others
3. **Warmup Required**: First invocation may be slower due to CUDA lazy initialization
4. **Prefill Not Optimized**: Monokernel only helps decode phase when batch_size ≤ 64
