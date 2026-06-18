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

**Verification**: Review `validation_results.md` for each track — the champion documents baseline provenance there (Stage 1 vLLM production kernel, not re-run from worktree). Cross-reference against the baselines captured under `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json`.

---

## E2E Baseline Reuse Requirement (NON-NEGOTIABLE)

**BLOCKING**: The champion MUST use Stage 1 baseline numbers for all E2E latency comparisons. NEVER re-run a baseline from the worktree.

### Source of Truth

| Data | Location (`{CR}` = `campaign.current_round`) | Captured by |
|------|----------|-------------|
| Per-BS E2E latency | `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json` | Stage 1 profiler on session base branch |
| Summary table | `{artifact_dir}/rounds/{CR}/constraints.md` — "Baseline E2E latency" | Stage 1 profiler |
| Kernel breakdown | `{artifact_dir}/rounds/{CR}/constraints.md` — "Baseline Truth Snapshot" | Stage 1 profiler |

### Rationale

Worktrees contain optimized code. Running `vllm bench latency` without the optimization flag from a worktree can execute the optimized code path (e.g., if `pip install -e .` overwrote the global editable install). This contaminates the baseline — both runs use the optimized path, hiding real improvements behind noise. This bug was observed in practice: a 5.5% real improvement was masked as 0.075% because both baseline and optimized ran FP8-optimized code.

### Procedure

1. Read Stage 1 baseline from `{artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json`
2. Run ONLY the optimized benchmark from the worktree (with enable flag set)
3. Compare the optimized latency against the Stage 1 latency using the
   following field-resolution rule (multi-launch-aware):
   - Prefer `aggregate.mean_latency` from the per-bucket entry when present
     (emitted by optional `--num-launches N>=2` sweeps).
   - Fall back to the flat `avg_latency` / `avg_s` field when no aggregate
     block exists (single-launch sweeps, the default).
4. In `validation_results.md`, cite: "Baseline source: Stage 1 (not re-run)"
   and note whether comparison used `aggregate.mean_latency` or `avg_latency`.

### Latency Field Resolution

Gate sweeps use a single launch (the default `--num-launches 1`).
The T5 gate verifies `e2e_latency_results.json` exists (no multi-launch
requirement). Use this field-resolution helper to extract latency from
either schema:

```python
# Field-resolution helper — use in champion scripts and docs.
def _latency_seconds(entry: dict) -> float | None:
    agg = entry.get("aggregate")
    if isinstance(agg, dict) and isinstance(agg.get("mean_latency"), (int, float)):
        return float(agg["mean_latency"])
    for k in ("avg_latency", "avg_s"):
        v = entry.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return None
```

When multi-launch sweeps are used (optional), also consult `row["noise"]`:
when True, the |delta| is smaller than `2 * max(stddev_baseline, stddev_opt)`
— the result is indistinguishable from noise and should not be promoted on
latency alone. Under the default single-launch config, this flag is not
produced.

### Sweep Script Guidance

If using `scripts/run_vllm_bench_latency_sweep.py`, configure it to run optimized-only. Do NOT use the sweep script's baseline output for pass/fail decisions — it runs from the worktree and may be contaminated.

For gate sweeps, also pass:
- `--fresh-cache` (isolates vLLM/Triton compile caches under
  `{out_root}/cache/{sweep_id}` so the previous sweep's partially-warm
  cache does not skew measurements; cache is removed on success)

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
- **Profiling (Stage 1)**: Use a separate `run_vllm_bench_latency_sweep.py --slot profiling --nsys-profile --nsys-mode node` invocation. Add `--nsys-trace cuda-sw` on Blackwell (B200/B300). See `nsys-profiling-guide.md` for the architecture matrix.
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

**Verification**: The champion MUST confirm CUDA graph usage in their benchmark scripts before running Gate 5.2. Benchmarks without CUDA graph capture are a Stage 5.2 FAIL — the `production_parity` invariant is violated.

### Cold-Cache Requirement for Bandwidth-Bound Kernels

For kernel benchmarks targeting bandwidth-bound kernels (arithmetic intensity < breakeven AI):

**REQUIRED**: Report both warm-cache and cold-cache kernel times.

- **Warm-cache**: Standard CUDA-graphed loop (100+ iterations on same tensors)
- **Cold-cache**: Use L2-busting methodology — chained distinct data totaling > 2.5x L2 cache size *(hardware-anchored sizing; debate scoring only, NOT impl ship gate)* between measurements, or use distinct random tensors per iteration

**Rationale**: Tight CUDA graph loops on small tensors keep data in L2 cache, inflating speedups for BW-bound kernels. In production, the full model pipeline (N layers x per-layer state) typically exceeds L2, forcing DRAM access.

**Fusion kernels**: If the optimization fuses kernels, the cold-cache benchmark MUST use chained data totaling > 2.5x L2 cache size *(hardware-anchored)* to simulate production L2 competition.

If warm/cold speedup ratio exceeds 1.5x *(debate scoring only; NOT an impl ship gate)*, the E2E projection in `validation_results.md` MUST use the cold-cache speedup. Omitting cold-cache measurement for a BW-bound kernel is a Stage 5.2 FAIL.

---

## Default correctness tolerances (starting points)

These are *starting points*, not universal truths.

### Gate 5.1a: Synthetic kernel tests

- **FP32**: `atol=1e-3`, `rtol=1e-3`
- **BF16/FP16**: `atol=1e-2`, `rtol=1e-2`
- **FP8 / block-quant**: **must be model-specific**.
  - As a placeholder, you may start with something like `atol=300`, `rtol=0.5` (see Qwen3 example),
  - but you should copy tolerances from the model's actual tests whenever possible.

Also require:
- no NaNs/Infs
- shape/stride parity
- deterministic indexing for routing and pair ordering (when required by baseline)

### Gate 5.1b: E2E Greedy Decode Correctness (HARD GATE)

**Gate 5.1b is a hard gate.** Correctness failure blocks the track from proceeding to latency benchmarks.

**Owner**: Champion via sweep script (deterministic — no validator involvement).

**Mechanism**: The sweep script's Phase 1 runs GSM8K greedy decode with `logprobs=5`, comparing optimized outputs against golden refs captured in Stage 1.

**Invocation**:
- Stage 1 (capture): `--capture-golden-refs` → saves `json/golden_refs.json`
- Stage 5 (verify): `--verify-correctness --baseline-from $STAGE1_DIR` → writes `json/correctness_verdict.json`

**Gate logic**: `opt_gsm8k_accuracy >= baseline_gsm8k_accuracy - (tolerance_pct / 100)`

- **n = 1319** questions by default (configurable via `--correctness-num-questions`; bundled as `data/gsm8k_full.json` for offline/sandboxed AMMO sessions)
- **tolerance_pct = 1.0pp** by default (configurable via `--correctness-tolerance-pct`). Allows ~13 questions of noise at N=1319. Set to `0.0` for strict `opt >= baseline` comparison.
- **Percentage comparison** — allows question-level churn (different questions correct) as long as aggregate accuracy stays within tolerance of baseline
- Token-level data is computed and logged as diagnostics only
- Single gate for all tracks regardless of lossless/lossy classification
- **max_tokens = 1024**

**Metadata mismatch check**: If golden refs `num_questions` or `max_tokens` differs from the verification run, exit code 4 (infrastructure error). `tolerance_pct` is recorded in golden metadata for traceability but is NOT enforced on mismatch — agents may tune tolerance per campaign without re-capturing golden refs.

**Baseline accuracy floor**: If `baseline_correct_count == 0` and `num_questions > 0`, exit code 4 (infrastructure error — model cannot solve any GSM8K questions).

**Self-consistency check** (Stage 1 only): Golden ref capture runs prompts twice to verify greedy decode is deterministic. If non-deterministic, metadata records `deterministic: false`.

**Exit codes**: 3 = correctness FAIL (opt_accuracy < baseline_accuracy - tolerance), 4 = infrastructure error (retry).

**Classification scope**: Lossless/lossy classification does NOT affect Gate 5.1b. Classification determines Gate 5.1a tolerances (BF16 vs FP8).

## Default kernel perf gate (Stage 5.2)

Measure **GPU kernel time** under CUDA graphs for the same bucket set as Stage 1.

Default gate (safe, conservative):
- **Measurable speedup required**: `T_opt_bucket_us < T_base_bucket_us` with >1% improvement
  for at least one target bucket. For buckets where the optimized kernel is slower, apply the per-BS tiered verdict system (see Stage 5.3). A `REGRESSED` verdict at kernel level triggers the same gating workflow as at E2E level.

Reporting requirements:
- baseline vs optimized per-bucket table (µs + speedup)
- if possible, per-stage breakdown (routing/prepare/W1/act/quant/W2/reduce)
- NCU sanity check for the dominant GEMM(s): occupancy, regs/thread, spills, SMEM/CTA

### Inductor Baseline Caveat

Gate 5.2's isolated benchmark validates kernel correctness and raw speed, but does NOT reflect the compiled-graph baseline. vLLM's Inductor fusion passes (`RMSNormQuantFusionPass`, `ActQuantFusionPass`, etc.) may have already fused the target chain — meaning the "unfused baseline" in Gate 5.2 doesn't exist in production.

Before using Gate 5.2 speedup in Amdahl projections, verify the target chain is NOT already fused by checking the nsys decode trace from Stage 1. If it is fused, use the Inductor-fused kernel time as the baseline for projection.

See `references/e2e-delta-math.md` § Inductor Baseline Parity for the full check and corrected formula.

### Per-Category Gate 5.2 Routing (spec §5.1)

The default Gate 5.2 above applies to **decode-kernel-slice** categories. **Inter-kernel-slice** categories have a distinct routing. Routing is keyed on the declared projection-slice, NOT on a per-category row — the inter-kernel-slice default is byte-identical to the legacy `dispatch_optimization` carve-out, so a resumed paused track is never silently re-routed to Standard Gate 5.2.

| Slice / Category | Gate 5.2 form |
|----------|----------------|
| decode-kernel: `kernel_replacement`, `custom_kernel`, `attention_kv_layout` | Standard (above) |
| decode-kernel: `kernel_fusion`, `weight_layout_transform` | Chain-time vs fused-time (merged-time for weight-merge) |
| inter-kernel: `dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass` | **SKIPPED** by default. Runs only if a new kernel is introduced — binding metric is component wall-time drop (not kernel-vs-kernel speedup) |
| novel descriptor | route by declared Slice targeted (decode-kernel → Standard/chain; inter-kernel → SKIPPED default) |

Schema-version guard: per-category routing only fires for campaigns with `state.json.campaign.schema_version >= "4.1"`. Legacy campaigns run the default Gate 5.2 above regardless of any Category-block contents.

### Minimum E2E Improvement Threshold

Default: `campaign.config.min_e2e_improvement_pct` (applies uniformly to both categories).

## Default end-to-end gate (Stage 5.3)

### Gate 5.3a: Kernel Execution Proof (NON-NEGOTIABLE)

Gate 5.3a confirms the optimized kernel actually dispatches under production conditions (CUDA graphs + torch.compile). Two methods are available:

Run a dedicated nsys profiling invocation (separate from the E2E sweep):

```bash
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} --round {N} --slot opt/{op_id} --labels opt \
  --nsys-profile --nsys-mode node \
  --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1
```

Verify via `nsys stats --report cuda_gpu_kern_sum` that the expected kernel name appears in the GPU trace.

- **If kernel found**: PASS. Proceed to Gate 5.3b.
- **If kernel NOT found**: FAIL. Do NOT run Gate 5.3b. E2E results would be inadmissible — the optimization is not activating.
- **Latency numbers from profiled runs are INVALID** (profiler overhead). Only the trace data matters.

Cost: ~85s (4B/L40S), ~4.5 min (70B/8xH100).

### Gate 5.3b: E2E Measurement Sweep

Run E2E under identical knobs and capture/compile settings. **Only runs after Gate 5.3a passes.**

Default iteration counts:
- **Profiling** (Stage 1): `--num-iters 1` (keep traces small)
- **Validation** (Stage 5): Use `num_iters` from `target.json` (default: 10 via `new_target.py`)

### Per-BS Tiered Verdict

Thresholds from `target.json` gating block (defaults: `noise_tolerance_pct: 0.5`, `catastrophic_regression_pct: 5.0`):

| Speedup | Verdict | Meaning |
|---------|---------|---------|
| >= 1.0 | `PASS` | Improvement at this batch size |
| >= (1.0 - noise_tolerance_pct/100) | `NOISE` | Within measurement noise, treated as neutral |
| >= (1.0 - catastrophic_regression_pct/100) | `REGRESSED` | Material regression, gating required |
| < (1.0 - catastrophic_regression_pct/100) | `CATASTROPHIC` | Too large to gate, track fails |

### Per-BS Verdicts and Track-Level Fallback Ladder

| Per-BS Results | Track Verdict | Action |
|---------------|--------------|--------|
| All PASS or NOISE (at least one PASS) | `PASS` | Ship directly |
| Any CATASTROPHIC | candidate for `FAIL` | Walk `SKILL.md § Non-Negotiables` item #10 before authoring FAIL |
| Some PASS + some REGRESSED | `GATING_REQUIRED` | Champion implements gating (see below); on success: `GATED_PASS` |
| All REGRESSED/NOISE (no PASS) | candidate for `FAIL` | Walk `SKILL.md § Non-Negotiables` item #10 before authoring FAIL |

The canonical track-level fallback ladder (`PASS → GATED_PASS → GATING_REQUIRED → RETRY_WITH_CONTINGENCY → FAIL`) is defined lifecycle-neutrally at `SKILL.md § Non-Negotiables` item #10. This table names the per-BS inputs; the ladder names the verdict rungs. FAIL is only authored after all applicable rungs are exhausted — this table does not authorize terminal FAIL on its own.

### GATING_REQUIRED Workflow

When the track verdict is `GATING_REQUIRED`:
1. Sweep reports per-BS verdict table showing mixed results (some PASS + some REGRESSED)
2. Champion evaluates gating feasibility
3. Champion runs crossover probing (see `references/crossover-probing.md`)
4. Champion implements gating mechanism per `references/code-templates.md` dispatch decision tree
5. Champion re-runs the kernel correctness & speedup checks on the gated kernel (correctness test + CUDA-graph speedup bench)
6. Champion re-runs sweep on gated code (`--labels opt --verify-correctness --nsys-profile --baseline-from $STAGE1_DIR`) — all BS must be PASS or NOISE
7. If both kernel re-validation and sweep pass: track status = `GATED_PASS`
8. If either fails or gating infeasible: track status = `FAIL` (one gating attempt per track)

Target batch sizes are defined in `target.json`. Use `references/e2e-delta-math.md` to set realistic expectations for E2E delta given component share `f`.

## Minimum E2E Improvement Threshold

All optimization candidates must meet a minimum expected E2E improvement to be worth pursuing. This single threshold replaces per-optimization ad-hoc criteria.

**Default**: `campaign.config.min_e2e_improvement_pct: 0.25` in `state.json` (scaffolded by `new_target.py`). This is the single authoritative documentation of the default value — all other files reference this section rather than hardcoding a number.

### Where It's Checked

| Decision Point | Check | Basis |
|---------------|-------|-------|
| **Pre-debate (campaign stop)** | `max(f_values) < threshold` | Amdahl's ceiling is a physical bound, not an estimate |
| **Post-debate (candidate gate)** | `max(e2e_projections) < threshold` | Champion's projected E2E gain from the debate-scoring formula |
| **Post-validation (GATED_PASS)** | At least one BS shows E2E improvement ≥ threshold | Per-BS verdict system handles regression classification |

**Inter-kernel-slice note** (`dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass`): for these categories, `f` in the pre-debate stop check is `inter_kernel_share × host_fraction` (the maximum addressable slice). The same `min_e2e_improvement_pct` threshold applies uniformly across all categories.

**Pre-debate/campaign-stop math**: If the top bottleneck's share of decode latency (`f`) is less than `min_e2e_improvement_pct`, even complete elimination of that component cannot yield the minimum improvement (Amdahl's Law). This is physics.

**Post-debate math lives in `references/debate-scoring-rubric.md`**, not here. Do not restate the projection formula in validation-stage prose.

**GATED_PASS rule**: An optimization that benefits some batch sizes but regresses others is still worth pursuing if at least one BS shows E2E improvement ≥ `campaign.config.min_e2e_improvement_pct`.

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

## Ship precedents

Append-only registry of case-level ship outcomes. Each entry is a structured YAML block, **NOT prose policy**. The `not_a_threshold: true` flag on every entry is load-bearing: it declares that the band-level entry here is one campaign's specific outcome under a specific gating mechanism, **not** a threshold to generalize across future ops. Crystallizing a noise-band GATED_PASS precedent into "anything in the noise band ships" is exactly the crystallization failure mode this registry is designed to prevent.

Agents reading this section for precedent guidance: **do not copy numeric values into new policy**. Use the entries to see *what mechanisms have shipped*, not *what thresholds to invent*. The canonical ship/retract decision is the per-BS verdict table above + `SKILL.md § Non-Negotiables` item #10 (Track-Level Fallback Ladder).

Precedent entries deliberately use **symbolic bands** (`noise_band`, `above_noise`, `above_catastrophic`) instead of raw percentage values — the band is the mechanism-level fact; the exact percentage is a campaign-specific outcome that a 4.7-class reader can silently recrystallize into "any value ≥ N% ships." Raw numeric `e2e_pct` is out of band on purpose. The original per-BS numbers live in the source campaign's `tracks/{op_id}/validation_results.md` for human review.

```yaml
historical_precedents:   # past campaigns only — NOT a live registry, NOT guidance for current decisions
  - op_id: op-009
    campaign: 6327c5d6
    ship_verdict: GATED_PASS
    ship_trigger:
      env_var: VLLM_MOE_TRITON_GROUPED_TOPK
      value: 1
    passing_configs:
      - bs: 32
        e2e_band: noise_band   # e2e delta between noise_tolerance_pct and catastrophic_regression_pct
        tier: NOISE
    not_a_threshold: true
    campaign_artifact_path: tracks/op-009/validation_results.md  # historical campaign artifact; not part of any live decision path
```

Schema for new entries: `op_id`, `campaign`, `ship_verdict` (one of `PASS`, `GATED_PASS`, `GATING_REQUIRED`), `ship_trigger` (env_var + value for env-var gates, or `inline` for dispatch-condition gates), `passing_configs` (array of `{bs, e2e_band, tier}` — `e2e_band` ∈ `noise_band`, `above_noise`, `above_catastrophic`), `not_a_threshold: true` (always required), `campaign_artifact_path` (path to the historical campaign artifact — reviewers only; not part of any live decision path).

---

## Invalid Reasons to Stop

The campaign stop condition is purely mechanical: `f < min_e2e_improvement_pct`. The orchestrator has ZERO discretion. The following are NOT valid reasons to stop or ask the user:

- **"The round's chosen technology class (e.g., Triton) didn't work"** → let the next round's debate pick a different technology via the selection function in `references/technology-selection.md`
- **"The remaining bottleneck is near its physical ceiling"** → Amdahl fraction `f` caps maximum E2E improvement at `f × (1 - 1/s_max)`; `s_max` is component-class-dependent (see `references/debate-rules.md` § NCU Triggers and class caps for values). If `f >= min_e2e_improvement_pct`, the math says there's room
- **"A new round is unlikely to find better candidates"** → the orchestrator cannot predict debate outcomes
- **"The campaign has been running for many rounds"** → round count is not a stop criterion
- **"Implementation complexity is increasing"** → complexity is scored in debate, not a campaign-level gate

Non-Negotiable #9 (autonomous campaign loop) in SKILL.md codifies the same rule from the non-negotiables side. The Stop hook (`ammo-stop-guard.sh`) enforces this mechanically — it blocks session end while `campaign.status == "active"`.
