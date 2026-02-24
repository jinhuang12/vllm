# Validation Defaults and Reporting (Validation Stage)

Use this as the **default** guidance for Validation Stage (Stage 5) validation artifacts (`{artifact_dir}/validation_results.md`).

Always prefer **model-specific** tolerances and invariants from:
- existing vLLM tests for that model (best), or
- a model-specific baseline doc (example: `validation/QWEN3_BASELINE.md`).

## Contents
- **Dual Baseline Requirement (NON-NEGOTIABLE)**
- **Production Parity Requirement (NON-NEGOTIABLE)**
- Default correctness tolerances (starting points)
- Default kernel perf gate (Stage 5.2)
- Default end-to-end gate (Stage 5.3)
- Required reporting checklist for `validation_results.md`

## Search anchors
tolerance, atol, rtol, speedup, CUDA graphs parity, torch.compile parity, kernel perf gate, E2E gate, enablement evidence, fused_experts, fused_moe, production parity.

---

## Dual Baseline Requirement (NON-NEGOTIABLE)

**BLOCKING**: Validation MUST compare against vLLM's actual production kernels, not naive PyTorch.

### Correctness Baseline
```python
# REQUIRED: Use vLLM's actual production kernel for the target component
# Adjust import to match your target component

# Example for MoE:
from vllm.model_executor.layers.fused_moe import fused_experts
# or
from vllm.model_executor.layers.fused_moe import fused_moe

baseline_out = fused_experts(x, w1, w2, topk_weights, topk_ids, ...)
```

### Performance Baseline
```python
# REQUIRED: Measure vLLM's actual kernel time
# NOT: naive PyTorch loops or reimplementations

# Use the same production call path that vLLM uses for the target component
```

### INVALID Baselines (DO NOT USE for any target component)
```python
# WRONG - naive PyTorch loops:
for expert_idx in range(num_experts):
    expert_out = torch.matmul(x_expert, weights[expert_idx])
    output.index_add_(0, indices, expert_out)

# WRONG - manual per-expert GEMM:
for e in range(E):
    mask = (expert_ids == e)
    out[mask] = F.linear(x[mask], w[e])
```

**Verification**: Run `python scripts/verify_validation_gates.py {artifact_dir}` and ensure it returns exit code 0.

---

## Production Parity Requirement (NON-NEGOTIABLE)

**BLOCKING**: All measurements MUST use production-equivalent settings.

### Required Environment
```bash
# MUST be set for BOTH baseline and optimized measurements:
export VLLM_TORCH_COMPILE_LEVEL=3  # Production default
export VLLM_USE_V1=1               # V1 engine

# CUDA graphs enabled by default (do NOT disable)
```

### Forbidden Settings
```bash
# DO NOT USE these in validation:
export TORCH_COMPILE_DISABLE=1     # FORBIDDEN
--enforce-eager                     # FORBIDDEN
VLLM_TORCH_COMPILE_LEVEL=0         # FORBIDDEN for validation
```

### Benchmark Requirements
```python
# Benchmark script MUST NOT contain:
os.environ["TORCH_COMPILE_DISABLE"] = "1"  # FORBIDDEN
enforce_eager=True                          # FORBIDDEN

# Benchmark script SHOULD contain:
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"  # Explicit production parity
```

### GPU Isolation Requirement (NON-NEGOTIABLE)

**BLOCKING**: Benchmark results are INVALID if collected under GPU contention.

- Only one GPU benchmark process may run at a time on a given set of GPUs
- Before starting any benchmark, verify GPU is idle: `nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader`
- Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements — it holds a
  system-wide GPU lock in `/tmp/ammo_gpu_locks/` to prevent concurrent runs
- Do NOT run `vllm bench latency` directly — this bypasses the GPU lock
- If contention is detected mid-benchmark: STOP, report to lead, and re-run after GPU is clear

**Why**: During the OLMo-3-7B verification run, concurrent GPU benchmarks inflated latencies
by ~80% (1.37s → 2.48s) and caused OOM errors on a 44 GiB L40S GPU.

### Kernel-Level Benchmark Requirements (NON-NEGOTIABLE)

For kernel-level (isolated) benchmarks comparing Triton vs CUDA C++:

**REQUIRED**: Capture kernel times under CUDA graphs
```python
# Option A: Use torch.cuda.make_graphed_callables
graphed_baseline = torch.cuda.make_graphed_callables(baseline_fn, (inputs,))
graphed_optimized = torch.cuda.make_graphed_callables(optimized_fn, (inputs,))

# Option B: Manual graph capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    baseline_out = baseline_fn(*inputs)
g.replay()  # Timed iterations
```

**WHY**: Launch overhead differences between Triton (many small ops) and CUDA C++
(single kernel) are ~100-200 µs. CUDA graphs eliminate this, enabling fair comparison.

**INVALID**: Timing with torch.cuda.Event alone without CUDA graph capture
```python
# WRONG - unfair comparison due to launch overhead:
start.record()
baseline_out = fused_experts(...)  # Triton launch overhead: ~50-100 µs
end.record()
```

**Verification**: The `verify_validation_gates.py` script checks for CUDA graph usage.
If absent, the `production_parity` gate will FAIL.

---

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

## Default kernel perf gate (Stage 5.2)

Measure **GPU kernel time** under CUDA graphs for the same bucket set as Stage 1.

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

## Default end-to-end gate (Stage 5.3)

Run E2E under identical knobs and capture/compile settings.

Default targets (adjust to business needs and component share `f`):
- `BS ∈ {1, 4, 8}`: **≥ 5%** improvement
- `BS ∈ {16, 32, 64}`: **> 0%** improvement

If the target component is a small fraction of end-to-end, these targets may be physically impossible. Use `references/e2e-delta-math.md` to set realistic expectations.

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
- connection to component share `f` (if improvement is small)

6) **Decision**
- ship / restrict envelope / pivot route / stop
- Stage 6 enablement guard proposal (what exactly will be enabled, where, and how to roll back)
