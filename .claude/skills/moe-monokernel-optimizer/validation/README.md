# MoE Monokernel Validation Scripts

Validation scripts are located in the vLLM repository, not in this skill directory.

## Script Locations

Validation scripts are located in:
```
Test scripts:      {vllm_repo}/tests/kernels/moe/
Benchmark scripts: {vllm_repo}/benchmarks/kernels/
```

## Required Scripts

For each model optimization, the following scripts should exist:

### 1. Correctness Test
`test_moe_monokernel_{model}.py`

```python
"""
Correctness test comparing monokernel output to reference fused_moe.
Expected tolerance: ~1% relative error for FP8.
"""
```

### 2. Monokernel Benchmark
`benchmark_moe_monokernel_{model}.py`

```python
"""
Benchmarks monokernel latency across batch sizes.
Outputs: mean, median, p90, p99 latencies.
"""
```

### 3. Baseline Benchmark
`benchmark_moe_baseline_{model}.py`

```python
"""
Benchmarks standard fused_moe for comparison.
Same output format as monokernel benchmark.
"""
```

### 4. NCU Analysis (shared)
`analyze_ncu_moe.py`

```python
"""
Parses Nsight Compute CSV output and identifies bottlenecks.
Model-agnostic - works for any monokernel.
"""
```

### 5. Per-Stage Profiling
`profile_moe_stages.py`

```python
"""
Breaks down latency by MoE stage (router, prepare, scale, gemm1, gemm2).
Requires clock64() instrumentation in kernel (optional).
"""
```

## Creating Scripts for New Model

When optimizing a new model, create test scripts by adapting from existing ones:

```bash
# Copy template
cp benchmarks/kernels/test_moe_monokernel_qwen3.py \
   benchmarks/kernels/test_moe_monokernel_{newmodel}.py

# Update dimensions
sed -i 's/HIDDEN_SIZE = 2048/HIDDEN_SIZE = {new_K}/' test_moe_monokernel_{newmodel}.py
sed -i 's/INTERMEDIATE_SIZE = 768/INTERMEDIATE_SIZE = {new_N}/' test_moe_monokernel_{newmodel}.py
sed -i 's/NUM_EXPERTS = 128/NUM_EXPERTS = {new_E}/' test_moe_monokernel_{newmodel}.py
sed -i 's/TOP_K = 8/TOP_K = {new_topk}/' test_moe_monokernel_{newmodel}.py
```

## Reference Scripts in Project Context

The following scripts are uploaded to the project as examples:

- `test_moe_monokernel_qwen3.py` - Qwen3-Coder-30B-A3B correctness test
- `benchmark_moe_monokernel_qwen3.py` - Qwen3 monokernel benchmark
- `benchmark_moe_baseline_qwen3.py` - Qwen3 baseline benchmark
- `analyze_ncu_moe.py` - NCU analysis tool
- `profile_moe_stages.py` - Per-stage profiling

Use these as templates when creating scripts for new models.

## Running Validation

### Quick Smoke Test (during development)
```bash
# Just check kernel launches without crash
python -c "
import torch
from vllm._custom_ops import moe_monokernel_{model}
# Minimal test
print('Kernel loaded successfully')
"
```

### Full Correctness Test
```bash
pytest tests/kernels/moe/test_moe_monokernel_{model}.py -v
```

### Performance Benchmark
```bash
python benchmarks/kernels/benchmark_moe_monokernel_{model}.py \
    --batch-sizes 1 2 4 8 16 32 64 \
    --num-warmup 10 \
    --num-iters 100 \
    --output /tmp/monokernel_benchmark
```

### Comparison Report
```bash
# Run both benchmarks
python benchmarks/kernels/benchmark_moe_baseline_{model}.py --output /tmp/baseline
python benchmarks/kernels/benchmark_moe_monokernel_{model}.py --output /tmp/monokernel

# Compare
python -c "
import json
baseline = json.load(open('/tmp/baseline/baseline_results.json'))
monokernel = json.load(open('/tmp/monokernel/monokernel_benchmark.json'))

print('BS | Baseline | Monokernel | Speedup')
for b, m in zip(baseline['results'], monokernel['results']):
    speedup = b['mean_ms'] / m['monokernel']['mean_ms']
    print(f\"{b['batch_size']:3} | {b['mean_ms']:8.3f} | {m['monokernel']['mean_ms']:10.3f} | {speedup:.2f}x\")
"
```

### NCU Profiling
```bash
# Capture profile
ncu --set full \
    --target-processes all \
    -o moe_monokernel_{model}_bs64 \
    python -c "
import torch
from vllm._custom_ops import moe_monokernel_{model}
# Run kernel multiple times for profiling
for _ in range(10):
    moe_monokernel_{model}(...)
"

# Export to CSV
ncu --import moe_monokernel_{model}_bs64.ncu-rep --csv > moe_{model}_bs64.csv

# Analyze
python benchmarks/kernels/analyze_ncu_moe.py moe_{model}_bs64.csv
```