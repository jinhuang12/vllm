# MoE Monokernel Validation Scripts

Validation scripts are located in the vLLM repository, not in this skill directory.

## Validation Workflow (3 Stages)

| Stage | Purpose | Tool |
|-------|---------|------|
| **4.1 Correctness** | Verify numerical accuracy | pytest / manual test |
| **4.2 Kernel-Level** | Pure kernel performance (CUDA graphs) | benchmark script |
| **4.3 E2E Latency** | Real inference impact | `vllm bench latency` |

**See also**: [E2E_LATENCY_GUIDE.md](E2E_LATENCY_GUIDE.md) for detailed E2E benchmark instructions.

**Important**: Validate the chosen ownership and fusion boundary. If a split-kernel path is available, benchmark both monokernel and split-kernel variants under CUDA graphs.
E2E latency under CUDA graphs/torch.compile is the final truth for production.

**Baseline source of truth**: Combined routing+experts CUDA‑graph profiling captured in Phase 1 constraints, plus NCU device metrics (occupancy, SM/TC utilization, memory traffic).

## Validation Failure Investigation

> **Codex CLI note:** If a stage exits with `needs_investigation`, run the matching investigation prompt in `orchestration/investigation-prompts.md` and update `{artifact_dir}/state.json` with the decision.

When a validation stage fails, the orchestrator spawns an investigation task 
to diagnose the root cause before deciding on next action. This replaces the 
generic "blocked" behavior used in implementation phases.

### Success Criteria Summary

**Stage 4.1 (Correctness)**:
| Data Type | Absolute Tolerance | Relative Tolerance |
|-----------|-------------------|-------------------|
| FP32 | 1e-3 | 1e-3 |
| BF16/FP16 | 1e-2 | 1e-2 |
| FP8 block-quant | 300 | 0.5 |

**Stage 4.2 (Kernel Performance)**:
- **STRICT**: `speedup >= 1.0x` at ALL tested batch sizes
- No regressions allowed under CUDA graphs
- Batch sizes tested: 1, 4, 8, 16, 32, 64
- Include token-major vs expert-major variants if both are implemented
- If regression: profile reference FusedMoE per‑kernel timings under parity **with combined routing+experts captured in a single CUDA graph**
- Produce a delta‑to‑baseline table (targets vs required µs savings) and feasibility call

## Minor Stage-Level Parity Checks (Phase 3)
Advisory only (non‑blocking). When routing/prepare/quant are newly implemented, validate they are within tolerance of the reference **under CUDA graphs**.
Rule of thumb: `<= 1.10x` or `<= +1–2 µs` (whichever is larger).  
If a stage is a thin wrapper around reference ops, mark as “parity assumed” and skip timing.

**Stage 4.3 (E2E Latency)**:
| Batch Size | Required Improvement |
|------------|---------------------|
| 1, 4, 8 | > 5% |
| 16, 32, 64 | > 0% |

### Investigation Triggers

| Stage | Exit Status | Trigger Condition |
|-------|-------------|-------------------|
| 4.1 | `needs_investigation` | `max_diff > tolerance` for any batch size |
| 4.2 | `needs_investigation` | `speedup < 1.0x` for any batch size |
| 4.3 | `needs_investigation` | Improvement below threshold |

### Investigation Types

| Type | Focus | Key Diagnostics |
|------|-------|-----------------|
| `correctness` | Logic bugs, scale errors | Binary search for divergence point, intermediate value comparison |
| `kernel_perf` | Performance regression | Per-stage profiling, NCU analysis, bottleneck identification |
| `e2e_perf` | E2E improvement below target | Monokernel activation check, MoE time fraction analysis |

### Investigation Task Prompts

See: `orchestration/investigation-prompts.md`

### Investigation Artifacts

Location: `{artifact_dir}/investigation/`

| File | Purpose | Created By |
|------|---------|------------|
| `correctness_analysis.md` | 4.1 failure root cause | correctness investigation |
| `stage_breakdown.json` | Per-stage latency comparison | kernel_perf investigation |
| `ncu_bs{N}.csv` | Nsight Compute metrics | kernel_perf investigation |
| `perf_analysis.md` | 4.2 failure root cause | kernel_perf investigation |
| `e2e_analysis.md` | 4.3 failure root cause | e2e_perf investigation |
| `fix_proposal.md` | Council-reviewed fix proposal | all investigations |

### Decision Matrix

After investigation completes with council approval:

| Decision | Meaning | Next Action |
|----------|---------|-------------|
| `phase_3` | Implementation bug found | Back to specific Phase 3 stage |
| `phase_2` | Algorithmic decision wrong | Re-plan with new constraints |
| `document_proceed` | Expected/acceptable behavior | Document limitation, proceed |
| `rerun_validation` | Measurement issue fixed | Re-run failing stage |
| `escalate_human` | Cannot determine cause | Pause for human review |

## Script Locations

Validation scripts are located in:
```
Test scripts:      {vllm_repo}/tests/kernels/moe/
Benchmark scripts: {vllm_repo}/benchmarks/kernels/
E2E benchmarks:    vllm bench latency (CLI tool)
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

### 6. E2E Latency Benchmark (CLI Tool)
Uses `vllm bench latency` - no script needed.

```bash
"""
Measures real inference latency with all optimizations enabled.
Tests monokernel impact on actual vLLM serving performance.
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

### E2E Latency Benchmark

See [E2E_LATENCY_GUIDE.md](E2E_LATENCY_GUIDE.md) for full details.

```bash
# Quick comparison (batch_size=8, decode-heavy workload)

# Baseline
vllm bench latency \
    --model {model_id} \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --input-len 64 --output-len 512 \
    --batch-size 8 \
    --num-iters 10 \
    --output-json /tmp/baseline.json

# Monokernel
VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
    --model {model_id} \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --input-len 64 --output-len 512 \
    --batch-size 8 \
    --num-iters 10 \
    --output-json /tmp/monokernel.json
```

**Key points:**
- Monokernel activation depends on dispatch thresholds (M_avg, batch buckets); confirm via logs
- Use `--input-len 64 --output-len 512` for decode-heavy workload
- Representative batch sizes: 4 (often best), 8 (typical), 32 (marginal)
- Expect 2-11% improvement depending on batch size
