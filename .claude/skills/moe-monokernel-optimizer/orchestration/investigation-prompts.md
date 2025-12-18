# Phase 4 Investigation Task Prompts

When Phase 4 validation fails, the orchestrator spawns an investigation task to diagnose
the root cause before deciding next action. This replaces the generic "blocked" behavior
used in implementation phases.

## Investigation Behavioral Footer

**Use for**: All investigation tasks (correctness, kernel-perf, e2e-perf)

Append to all investigation task prompts:

```markdown
## MANDATORY: LLM Council Review (BLOCKING) - ONE CHECKPOINT

### Checkpoint: Fix Proposal Review
After completing investigation and drafting fix proposal, invoke `llm-council` to review.

### Council Acceptance Criteria
Council accepts if:
- [ ] Hypothesis is supported by profiling/diagnostic data
- [ ] Proposed fix is bounded (not "rewrite everything")
- [ ] Success criteria is measurable
- [ ] Fallback/escalation path is defined

### Investigation Bounds (HARD LIMITS)
These limits prevent infinite debugging loops:
- Max 2 NCU profiling runs
- Max 3 hypothesis-test cycles
- Max 2 council review rounds
- If bounds exceeded without resolution → Exit with `escalate_human`

### How to Invoke:
Use the Skill tool with skill name `llm-council`. The skill has its own context preparation instructions.

### Review Loop:
1. Complete investigation steps
2. Form hypothesis and draft fix proposal
3. Invoke `llm-council` to review proposal
4. If REJECTED → Revise (up to 2 rounds)
5. If ACCEPTED → Record decision, update state, exit

---

**Behavioral Expectations**:
1. **Read before investigating**: Load constraints.md, optimization_plan.md, validation failure details from state.json
2. **Systematic diagnosis**: Follow investigation steps in order, don't skip
3. **Document as you go**: Write findings to `{artifact_dir}/investigation/`
4. **Form hypothesis before fixing**: No speculative code changes during investigation
5. **Council reviews proposal**: Submit the fix proposal, not the raw investigation data

**State Management**:
Update `{artifact_dir}/state.json` with investigation progress:
```json
"phases": {
  "4_validation": {
    "status": "investigating",
    "investigation": {
      "type": "{correctness|kernel_perf|e2e_perf}",
      "started_at": "{timestamp}",
      "hypothesis_cycles": N,
      "ncu_runs": N,
      "council_reviews": N,
      "root_cause": "description or null",
      "proposed_fix": "description or null",
      "decision": "{phase_3|phase_2|document_proceed|rerun_validation|escalate_human}"
    }
  }
}
```

**Exit Conditions**:
- `decision: "phase_3"` → Include `target_stage` and `fix_context`
- `decision: "phase_2"` → Include `additional_constraints`
- `decision: "document_proceed"` → Include `limitation_doc`
- `decision: "escalate_human"` → Include `findings` and `recommendation`
```

---

## Investigation Task: Correctness Failure (4.1)

**Trigger**: Stage 4.1 reports `status: "needs_investigation"` with `max_diff > tolerance`

**Task Tool Parameters**:
- `description`: "Investigate 4.1 correctness failure for {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Investigate correctness failure for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Failure Context** (from state.json):
- Max absolute diff: {max_abs_diff}
- Tolerance: {tolerance}
- Failing batch sizes: {failing_batch_sizes}
- Failure details: {failure_details}

**Read First**:
- `{artifact_dir}/state.json` - full failure context
- `{artifact_dir}/constraints.md` - model dimensions and dtype
- `{artifact_dir}/optimization_plan.md` - algorithmic decisions made
- `{cuda_dir}/` - all implementation files

**Investigation Steps** (follow in order):

### Step 1: Characterize the Divergence
```python
import torch
from vllm._custom_ops import moe_monokernel_{model_short}
from vllm.model_executor.layers.fused_moe import fused_moe

# Test at failing batch size
BS = {first_failing_batch_size}
x = torch.randn(BS, {K}, dtype=torch.bfloat16, device='cuda') / 10
router = torch.randn(BS, {E}, dtype=torch.bfloat16, device='cuda')

mono_out = moe_monokernel_{model_short}(x, router, ...)
stock_out = fused_moe(x, router, ...)

diff = (mono_out - stock_out).abs()
print(f"Max diff: {diff.max().item()}")
print(f"Mean diff: {diff.mean().item()}")
print(f"Diff location (token, hidden): {torch.where(diff == diff.max())}")

# Check for patterns
print(f"Diff by token: {diff.max(dim=1).values}")  # Per-token max
print(f"Diff by hidden: {diff.max(dim=0).values}")  # Per-dimension max
```

Document: Is error uniform or localized? Growing with batch size?

### Step 2: Binary Search for Divergence Point
Isolate which stage introduces the error:

1. **Routing stage**: Compare topk_ids and topk_weights
2. **Quantization stage** (FP8): Compare scales and quantized values
3. **Up-projection**: Compare intermediate after first GEMM
4. **Activation**: Compare after SiLU/activation
5. **Down-projection**: Compare after second GEMM
6. **Accumulation**: Compare final output accumulation

For each stage, export intermediate values and compare against reference.

### Step 3: Check Common Correctness Issues

| Issue | Check | Fix |
|-------|-------|-----|
| Scale application order | FP8: scale before or after MMA? | Match reference pattern |
| Accumulator dtype | Using FP32 accumulator? | Change to FP32 if using BF16 |
| Token-expert mapping | pair_idx / TOP_K = token_idx? | Fix index calculation |
| Weight layout | [E, N, K] vs [E, K, N]? | Match vLLM convention |
| Renormalization | topk weights sum to 1? | Check renormalize flag |
| Shared experts | Handled separately? | Review architecture |

### Step 4: Search Reference Implementation
```bash
# Search Llama4 patch for analogous patterns
grep -n "{suspected_issue}" assets/LLAMA4_MONOKERNEL_PATCH.md
grep -A 20 "down_projection" assets/LLAMA4_MONOKERNEL_PATCH.md
```

Does reference handle this differently? Document findings.

### Step 5: Form Hypothesis and Propose Fix

Write to `{artifact_dir}/investigation/correctness_analysis.md`:
```markdown
# Correctness Investigation: {model_short}

## Failure Characterization
- Max diff: {value} (tolerance: {tolerance})
- Error pattern: {uniform/localized/growing}
- Affected tokens: {which}

## Divergence Point
Stage where error is introduced: {stage_name}
Evidence: {comparison data}

## Root Cause Analysis
{Why this is happening based on code review}

## Proposed Fix
1. {Specific change #1}
   - File: {cuda_dir}/{file}.cu
   - Lines: X-Y
   - Change: {description}

2. {Specific change #2 if needed}

## Success Criteria
- Max diff < {tolerance} across all batch sizes [1, 8, 64]

## Fallback
If fix doesn't work: {next hypothesis OR escalate}
```

### Step 6: Council Review
Submit `correctness_analysis.md` for review.

### Step 7: Record Decision
Update state.json with investigation result and decision.

**Decision Matrix**:
| Root Cause | Decision | Target |
|------------|----------|--------|
| Bug in routing stage | `phase_3` | `routing_and_prepare` |
| Bug in quantization | `phase_3` | `activation_quantization` |
| Bug in GEMM (up or down) | `phase_3` | `gemm_implementation` |
| Bug in assembly/output | `phase_3` | `kernel_assembly` |
| Wrong algorithmic choice | `phase_2` | re-plan |
| Cannot determine | `escalate_human` | report findings |

{investigation_behavioral_footer}
```

---

## Investigation Task: Kernel Performance Regression (4.2)

**Trigger**: Stage 4.2 reports `status: "needs_investigation"` with `speedup < 1.0x` at any batch size

**Task Tool Parameters**:
- `description`: "Investigate 4.2 kernel performance regression for {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Investigate kernel performance regression for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Failure Context** (from state.json):
- Failing batch sizes: {failing_batch_sizes}
- Worst speedup: {worst_speedup}x at BS={batch_size}
- Results: {kernel_benchmark_results}

**Read First**:
- `{artifact_dir}/state.json` - full benchmark results
- `{artifact_dir}/constraints.md` - model dimensions
- `{artifact_dir}/optimization_plan.md` - algorithmic decisions
- `benchmarks/kernels/benchmark_moe_monokernel_{model_short}.py` - benchmark script

**Investigation Steps** (follow in order):

### Step 1: Per-Stage Profiling
Break down latency by MoE stage to identify which component is slow:

```bash
python benchmarks/kernels/profile_moe_stages.py \
    --model {model_short} \
    --batch-sizes 1 8 64 \
    --compare-baseline \
    --output {artifact_dir}/investigation/stage_breakdown.json
```

If `profile_moe_stages.py` doesn't exist, create inline profiling:
```python
import torch
import time

def profile_stage(fn, name, iters=100):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1000  # ms

# Profile each stage separately
# Compare monokernel stages vs baseline components
```

**Expected Output**:
| Stage | Baseline (μs) | Monokernel (μs) | Delta |
|-------|--------------|-----------------|-------|
| routing | ... | ... | ... |
| prepare | ... | ... | ... |
| gemm_up | ... | ... | ... |
| gemm_down | ... | ... | ... |
| output | ... | ... | ... |
| **total** | ... | ... | ... |

Document: Which stage(s) are slower? By how much?

### Step 2: NCU Analysis (for slowest stage)
Run Nsight Compute profiling:

```bash
# Profile at batch size showing worst regression
ncu --replay-mode application --set full --clock-control base \
    --kernel-name regex:moe_kernel --launch-skip 3 --launch-count 1 \
    -o {artifact_dir}/investigation/ncu_bs{BS} \
    python benchmarks/kernels/ncu_profile_moe_monokernel.py --batch-size {BS}

# Export to CSV
ncu --import {artifact_dir}/investigation/ncu_bs{BS}.ncu-rep --csv \
    > {artifact_dir}/investigation/ncu_bs{BS}.csv

# Analyze
python benchmarks/kernels/analyze_ncu_moe.py {artifact_dir}/investigation/ncu_bs{BS}.csv
```

**Key Metrics to Examine**:
| Metric | Expected | Concern If |
|--------|----------|------------|
| Tensor Core util | 1-5% | < 0.5% (MMA not firing) |
| SM throughput | 5-15% | < 3% (compute stalls) |
| DRAM throughput | 20-50% | > 80% (memory bound) |
| Barrier stalls | 10-30% | > 50% (sync overhead) |
| Warp occupancy | 30-60% | < 20% (register pressure) |

### Step 3: Map Symptoms to Root Causes

| Symptom | Likely Cause | Investigation |
|---------|--------------|---------------|
| High barrier stalls | Too many grid.sync() | Count syncs in kernel |
| Low TC util + low mem BW | Instruction stalls | Check MMA issue rate |
| High memory stalls | Poor locality | Review data layout |
| Low occupancy | Register pressure | Check register count |
| Routing slow | Token sorting overhead | Review sorter choice |
| GEMM slow | Wrong tile size | Compare to plan |

### Step 4: Compare to Reference Implementation
```bash
# Search for optimization patterns in Llama4 patch
grep -n "grid.sync" assets/LLAMA4_MONOKERNEL_PATCH.md | wc -l
grep -A 10 "{identified_bottleneck}" assets/LLAMA4_MONOKERNEL_PATCH.md
```

### Step 5: Form Hypothesis and Propose Fix

Write to `{artifact_dir}/investigation/perf_analysis.md`:
```markdown
# Kernel Performance Investigation: {model_short}

## Performance Regression Summary
- Slowdown at BS={N}: {monokernel_ms}ms vs {baseline_ms}ms ({speedup}x)

## Per-Stage Breakdown
| Stage | Baseline (μs) | Monokernel (μs) | Delta |
|-------|--------------|-----------------|-------|
| ... | ... | ... | ... |

Slowest stage: {stage_name} (+{X}μs over baseline)

## NCU Analysis
- Tensor Core utilization: {X}%
- Memory throughput: {X}% of peak
- Primary stall reason: {barrier/memory/compute}
- Occupancy: {X}%

## Root Cause Analysis
{Why the kernel is slow based on profiling data}

## Proposed Fix
1. {Specific optimization #1}
   - Target: {stage/component}
   - Expected impact: {X}% improvement on {metric}
   - Code location: {file}:{lines}

2. {Specific optimization #2 if needed}

## Success Criteria
- Monokernel latency < baseline at all batch sizes (speedup >= 1.0x)

## Fallback
If fix doesn't achieve target: {next optimization OR escalate}
```

### Step 6: Council Review
Submit `perf_analysis.md` for review.

### Step 7: Record Decision
Update state.json with investigation result and decision.

**Decision Matrix**:
| Root Cause | Decision | Target |
|------------|----------|--------|
| Suboptimal GEMM tiling | `phase_3` | `gemm_implementation` |
| Too many grid syncs | `phase_3` | `kernel_assembly` |
| Wrong sorter algorithm | `phase_3` | `routing_and_prepare` |
| Fundamental design issue | `phase_2` | re-plan with new constraints |
| Hardware limitation | `document_proceed` | document expected behavior |
| Cannot determine | `escalate_human` | report findings |

{investigation_behavioral_footer}
```

---

## Investigation Task: E2E Performance Below Threshold (4.3)

**Trigger**: Stage 4.3 reports `status: "needs_investigation"` with improvement below threshold

**Task Tool Parameters**:
- `description`: "Investigate 4.3 E2E performance for {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Investigate E2E performance below threshold for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Failure Context** (from state.json):
- Failing batch sizes: {failing_batch_sizes}
- Improvement at BS=4: {improvement_bs4}% (required: >5%)
- Improvement at BS=8: {improvement_bs8}% (required: >5%)
- Results: {e2e_benchmark_results}

**Success Criteria Reminder**:
- Batch sizes ≤ 8: improvement must be > 5%
- Batch sizes > 8: improvement must be > 0%

**Read First**:
- `{artifact_dir}/state.json` - full E2E results
- `{artifact_dir}/validation_results.md` - kernel-level results (4.2)
- `validation/E2E_LATENCY_GUIDE.md` - benchmark methodology

**Investigation Steps** (follow in order):

### Step 1: Verify Monokernel Activation
Check that monokernel is actually being used:

```bash
# Look for activation log during benchmark
VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
    --model {model_id} \
    --tensor-parallel-size {tp} \
    --max-model-len 4096 \
    --input-len 64 --output-len 512 \
    --batch-size 8 \
    --num-iters 1 2>&1 | grep -i "monokernel\|MoE"
```

**Expected**: `Using MoE Monokernel backend (TP={tp})`

If NOT present:
- Check `VLLM_USE_MOE_MONOKERNEL=1` is set
- Check model matches expected (Qwen3-Coder-30B-A3B-FP8?)
- Check TP setting (monokernel may not support TP>1)

### Step 2: Compare Kernel vs E2E Improvement
Cross-reference with Stage 4.2 results:

| Batch Size | Kernel Speedup | E2E Improvement |
|------------|---------------|-----------------|
| 4 | {from 4.2} | {from 4.3} |
| 8 | {from 4.2} | {from 4.3} |
| 32 | {from 4.2} | {from 4.3} |

**Analysis**:
- If kernel speedup is good but E2E is poor → MoE not on critical path
- If both are poor → Kernel needs optimization (back to 4.2)
- If kernel is slower → Should have caught in 4.2

### Step 3: Profile MoE vs Non-MoE Time
Estimate how much time is spent in MoE vs other layers:

```python
# Rough calculation
# For Qwen3-30B-A3B: 48 layers, each has attention + MoE
# MoE time ≈ (num_moe_layers * moe_latency_per_layer)
# Total time ≈ prefill_time + decode_time

# If MoE is 20% of total time:
# 50% MoE speedup → 10% E2E speedup
# 10% MoE speedup → 2% E2E speedup

moe_fraction = estimate_moe_time_fraction()
expected_e2e_improvement = kernel_speedup * moe_fraction
```

### Step 4: Check for Measurement Issues
- **Warmup**: Were enough warmup iterations run?
- **Variance**: Is latency variance high between runs?
- **Thermal**: Is GPU throttling during benchmark?
- **Interference**: Other processes on GPU?

```bash
# Run multiple times to check consistency
for i in 1 2 3; do
    VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
        --model {model_id} \
        --tensor-parallel-size {tp} \
        --max-model-len 4096 \
        --input-len 64 --output-len 512 \
        --batch-size 8 \
        --num-iters 5 2>&1 | grep "Avg latency"
done
```

### Step 5: Determine if This is Expected Behavior
Some models may legitimately show smaller E2E improvements:

| Factor | Impact on E2E |
|--------|---------------|
| MoE fraction of compute | Lower fraction → smaller E2E impact |
| Attention dominance | Large KV cache → MoE less critical |
| Shared experts | Shared expert compute not optimized |
| TP>1 | Communication overhead |

If kernel speedup (4.2) is good but E2E improvement is small due to MoE
not being the bottleneck, this may be **expected behavior**.

### Step 6: Form Hypothesis and Propose Fix

Write to `{artifact_dir}/investigation/e2e_analysis.md`:
```markdown
# E2E Performance Investigation: {model_short}

## Performance Summary
| Batch Size | Kernel Speedup | E2E Improvement | Required |
|------------|---------------|-----------------|----------|
| 4 | {X}x | {Y}% | >5% |
| 8 | {X}x | {Y}% | >5% |
| 32 | {X}x | {Y}% | >0% |

## Monokernel Activation
- Confirmed active: {yes/no}
- Evidence: {log message}

## Analysis
- MoE fraction of total compute: ~{X}%
- Expected E2E improvement given kernel speedup: ~{Y}%
- Actual E2E improvement: {Z}%
- Gap explained by: {reason}

## Root Cause
{One of: kernel issue, MoE not bottleneck, measurement issue, expected behavior}

## Recommendation
{One of:}
- Fix kernel performance (back to Phase 3)
- Re-run with corrected measurement
- Document as expected behavior and proceed
- Escalate for human review

## If Proceeding Despite Low Improvement
Limitation documentation:
- Model {model_short} shows {X}% E2E improvement at BS={N}
- This is below the {Y}% target but acceptable because: {reason}
- Recommendation: Enable monokernel for BS ≤ {threshold}
```

### Step 7: Council Review
Submit `e2e_analysis.md` for review.

### Step 8: Record Decision
Update state.json with investigation result and decision.

**Decision Matrix**:
| Root Cause | Decision | Action |
|------------|----------|--------|
| Kernel actually slow | `phase_3` | Fix kernel (should have caught in 4.2) |
| Measurement error | `rerun_validation` | Re-run 4.3 with corrected setup |
| MoE not bottleneck | `document_proceed` | Document limitation, proceed |
| Model characteristic | `document_proceed` | Document expected behavior |
| Cannot determine | `escalate_human` | Report findings |

{investigation_behavioral_footer}
```

---

## Post-Investigation: Orchestrator Actions

After investigation task completes, orchestrator reads the decision and takes action:

### Decision: `phase_3`
```python
# Reset to Phase 3, specific stage
state.phases["3_implementation"].status = "in_progress"
state.phases["3_implementation"].stages[target_stage].status = "pending"
state.phases["4_validation"].status = "pending"

# Spawn task with fix context
spawn_stage_task(
    target_stage,
    additional_context=investigation.fix_proposal
)
```

### Decision: `phase_2`
```python
# Reset to Phase 2 for re-planning
state.phases["2_planning"].status = "in_progress"
state.phases["3_implementation"].status = "pending"
state.phases["4_validation"].status = "pending"

# Spawn planning task with additional constraints
spawn_phase_task(
    "2_planning",
    additional_constraints=investigation.findings
)
```

### Decision: `document_proceed`
```python
# Document limitation and proceed to Phase 5
state.phases["4_validation"].status = "complete"
state.phases["4_validation"].limitation = investigation.limitation_doc

# Add limitation to validation_results.md
append_to_file(
    f"{artifact_dir}/validation_results.md",
    f"\n## Known Limitations\n{investigation.limitation_doc}"
)

# Proceed to integration
advance_to_phase("5_integration")
```

### Decision: `rerun_validation`
```python
# Re-run the failing validation stage
failing_stage = state.phases["4_validation"].investigation.type
state.phases["4_validation"].stages[failing_stage].status = "pending"

spawn_validation_task(failing_stage)
```

### Decision: `escalate_human`
```python
# Cannot resolve automatically
state.phases["4_validation"].status = "escalated"

# Report to user
report = f"""
## MoE Monokernel: Human Review Required

Investigation could not determine root cause or fix.

### Findings
{investigation.findings}

### Attempted
- Hypothesis cycles: {investigation.hypothesis_cycles}
- NCU profiling runs: {investigation.ncu_runs}
- Council reviews: {investigation.council_reviews}

### Recommendation
{investigation.recommendation}

### Next Steps
Please review the investigation artifacts in:
{artifact_dir}/investigation/

Options:
1. Provide additional guidance and resume
2. Manually investigate and fix
3. Abandon monokernel for this model
"""

print(report)
# Workflow pauses here
```