# Validation Defaults and Reporting (Validation Stage)

Use this as the **default** guidance for Validation Stage (Stage 5) validation artifacts (`{artifact_dir}/validation_results.md`).

## Contents
- **Dual Baseline Requirement (NON-NEGOTIABLE)**
- **Production Parity Requirement (NON-NEGOTIABLE)**
- Default correctness tolerances (starting points)
- Default kernel perf gate (Stage 5.2)
- Default end-to-end gate (Stage 5.3)
- **Minimum E2E improvement threshold** (campaign-wide viability gate)
- Required reporting checklist for `validation_results.md`

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

## E2E Baseline Reuse Requirement (NON-NEGOTIABLE)

**BLOCKING**: Validators MUST use Stage 1 baseline numbers for all E2E latency comparisons. NEVER re-run a baseline from the worktree.

### Source of Truth

| Data | Location | Captured by |
|------|----------|-------------|
| Per-BS E2E latency | `{artifact_dir}/runs/baseline_bs{N}.json` | Stage 1 profiler on clean main |
| Summary table | `{artifact_dir}/constraints.md` — "Baseline E2E latency" | Stage 1 profiler |
| Kernel breakdown | `{artifact_dir}/constraints.md` — "Baseline Truth Snapshot" | Stage 1 profiler |

### Rationale

Worktrees contain optimized code. Running `vllm bench latency` without the optimization flag from a worktree can execute the optimized code path (e.g., if `pip install -e .` overwrote the global editable install). This contaminates the baseline — both runs use the optimized path, hiding real improvements behind noise. This bug was observed in practice: a 5.5% real improvement was masked as 0.075% because both baseline and optimized ran FP8-optimized code.

### Procedure

1. Read Stage 1 baseline from `{artifact_dir}/runs/baseline_bs{N}.json`
2. Run ONLY the optimized benchmark from the worktree (with enable flag set)
3. Compare optimized `avg_latency` against Stage 1 `avg_latency`
4. In `validation_results.md`, cite: "Baseline source: Stage 1 (not re-run)"

### Sweep Script Guidance

If using `scripts/run_vllm_bench_latency_sweep.py`, configure it to run optimized-only. Do NOT use the sweep script's baseline output for pass/fail decisions — it runs from the worktree and may be contaminated.

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
- **Validation (Stages 5-6)**: Use `scripts/run_vllm_bench_latency_sweep.py` for all
  E2E measurements — it holds a system-wide GPU lock to prevent concurrent runs
- **Profiling (Stage 1)**: Use `run_vllm_bench_latency_sweep.py --nsys-profile`
  for combined E2E baseline + nsys traces (see `nsys-profiling-guide.md` §3.5)
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

### Cold-Cache Requirement for Bandwidth-Bound Kernels

For kernel benchmarks targeting bandwidth-bound kernels (arithmetic intensity < breakeven AI):

**REQUIRED**: Report both warm-cache and cold-cache kernel times.

- **Warm-cache**: Standard CUDA-graphed loop (100+ iterations on same tensors)
- **Cold-cache**: Use L2-busting methodology — chained distinct data totaling > 2.5x L2 cache size between measurements, or use distinct random tensors per iteration

**Rationale**: Tight CUDA graph loops on small tensors keep data in L2 cache, inflating speedups for BW-bound kernels. In production, the full model pipeline (N layers x per-layer state) typically exceeds L2, forcing DRAM access.

**Fusion kernels**: If the optimization fuses kernels, the cold-cache benchmark MUST use chained data totaling > 2.5x L2 cache size to simulate production L2 competition.

If warm/cold speedup ratio exceeds 1.5x, the E2E projection in `validation_results.md` MUST use the cold-cache speedup. Omitting cold-cache measurement for a BW-bound kernel is a Stage 5.2 FAIL.

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
- **Measurable speedup required**: `T_opt_bucket_us < T_base_bucket_us` with >1% improvement
  for at least one target bucket. For buckets where the optimized kernel is slower, apply the per-BS tiered verdict system (see Stage 5.3). A `REGRESSED` verdict at kernel level triggers the same gating workflow as at E2E level.

Reporting requirements:
- baseline vs optimized per-bucket table (µs + speedup)
- if possible, per-stage breakdown (routing/prepare/W1/act/quant/W2/reduce)
- NCU sanity check for the dominant GEMM(s): occupancy, regs/thread, spills, SMEM/CTA

## Default end-to-end gate (Stage 5.3)

### Gate 5.3a: Kernel Execution Proof (NON-NEGOTIABLE)

Before the E2E measurement sweep, the validator runs a minimal nsys-profiled run to confirm the optimized kernel actually dispatches under production conditions (CUDA graphs + torch.compile).

```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --labels opt \
  --nsys-profile --nsys-output-len 2 --nsys-num-iters 1 \
  --out-name kernel_proof
```

Then verifies via `nsys stats --report cuda_gpu_kern_sum` that the expected kernel name (provided by champion) appears in the GPU trace.

- **If kernel found**: PASS. Proceed to Gate 5.3b.
- **If kernel NOT found**: FAIL. Do NOT run Gate 5.3b. E2E results would be inadmissible — the optimization is not activating.
- **Latency numbers from this run are INVALID** (nsys overhead). Only the `.nsys-rep` trace matters.

Cost: ~85s (4B/L40S), ~4.5 min (70B/8xH100).

### Gate 5.3b: E2E Measurement Sweep

Run E2E under identical knobs and capture/compile settings. **Only runs after Gate 5.3a passes.**

Default iteration counts:
- **Profiling** (Stage 1): `--num-iters 1` (keep traces small)
- **Validation** (Stage 5): Use `num_iters` from `target.json` (default: 20 via `new_target.py`)

### Per-BS Tiered Verdict

Thresholds from `target.json` gating block (defaults: `noise_tolerance_pct: 0.5`, `catastrophic_regression_pct: 5.0`):

| Speedup | Verdict | Meaning |
|---------|---------|---------|
| >= 1.0 | `PASS` | Improvement at this batch size |
| >= (1.0 - noise_tolerance_pct/100) | `NOISE` | Within measurement noise, treated as neutral |
| >= (1.0 - catastrophic_regression_pct/100) | `REGRESSED` | Material regression, gating required |
| < (1.0 - catastrophic_regression_pct/100) | `CATASTROPHIC` | Too large to gate, track fails |

### Track-Level Verdict

| Per-BS Results | Track Verdict | Action |
|---------------|--------------|--------|
| All PASS or NOISE (at least one PASS) | `PASS` | Ship directly |
| Any CATASTROPHIC | `FAIL` | Kill track |
| Some PASS + some REGRESSED | `GATING_REQUIRED` | Champion implements gating (see below) |
| All REGRESSED/NOISE (no PASS) | `FAIL` | No batch size benefits — kill track |

### GATING_REQUIRED Workflow

When the track verdict is `GATING_REQUIRED`:
1. Validator reports per-BS verdict table to champion
2. Champion evaluates gating feasibility
3. Champion requests validator to run crossover probing (see `references/crossover-probing.md`)
4. Champion implements gating mechanism per `references/code-templates.md` dispatch decision tree
5. Validator re-validates gated version — all BS must be PASS or NOISE
6. If re-validation passes: track status = `GATED_PASS`
7. If re-validation fails or gating infeasible: track status = `FAIL` (one gating attempt per track)

Target batch sizes are defined in `target.json`. Use `references/e2e-delta-math.md` to set realistic expectations for E2E delta given component share `f`.

## Minimum E2E Improvement Threshold

All optimization candidates must meet a minimum expected E2E improvement to be worth pursuing. This single threshold replaces per-optimization ad-hoc criteria.

**Default**: `min_e2e_improvement_pct: 1.0` (configurable in `state.json` and `target.json`)

### Where It's Checked

| Decision Point | Check | Deflation Applied? | Rationale |
|---------------|-------|-------------------|-----------|
| **Pre-debate (campaign stop)** | `max(f_values) < threshold` | No | Amdahl's ceiling is a physical bound, not an estimate |
| **Post-debate (candidate gate)** | `max(deflated_e2e_projections) < threshold` | Yes | Champion speedup estimates are systematically optimistic |
| **Post-validation (GATED_PASS)** | At least one BS shows E2E improvement ≥ threshold | No | Per-BS verdict system handles regression classification |

**Pre-debate/campaign-stop math**: If the top bottleneck's share of decode latency (`f`) is less than `min_e2e_improvement_pct`, even complete elimination of that component cannot yield the minimum improvement (Amdahl's Law). No deflation — this is physics.

**Post-debate math**: `deflated_e2e = f × (1 - 1/s) / e2e_deflation_factor`. If no candidate's deflated E2E exceeds the threshold, none are worth implementing.

**GATED_PASS rule**: An optimization that benefits some batch sizes but regresses others is still worth pursuing if at least one BS shows E2E improvement ≥ `min_e2e_improvement_pct`.

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
- Gate 5.3a kernel execution proof (nsys trace confirming optimized kernel dispatched)

5) **E2E latency**
- baseline vs optimized per-bucket table
- variance notes (iters, warmup, noise sources)
- connection to component share `f` (if improvement is small)
- Per-BS verdict table (PASS/NOISE/REGRESSED/CATASTROPHIC) for each tested batch size

6) **Decision**
- ship / restrict envelope / pivot route / stop
- Stage 6 enablement guard proposal (what exactly will be enabled, where, and how to roll back)
- If GATED_PASS: dispatch mechanism type, env var name, dispatch condition, crossover_threshold_bs, pre-gating and post-gating per-BS E2E tables
