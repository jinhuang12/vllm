# MoE Monokernel Workflow

## LLM Council Checkpoints

**IMPORTANT**: Invoke `llm-council` skill at these points to catch issues early:

| After | Council Topic | Why |
|-------|---------------|-----|
| Phase 1 (constraints) | "Review MoE constraints extraction for {model}" | Catch missed semantics before planning |
| Phase 2 (plan) | "Review optimization plan for {model} on {hardware}" | Validate algorithmic decisions before implementation |
| Stage 3 (GEMM) | "Review GEMM implementation for {model}" | Most complex stage - external review critical |
| 3 failed attempts | "Debug {stage} - failed 3 times" | Fresh perspective beats spinning |
| Phase 4 (validation) | "Interpret benchmark results for {model}" | Sanity check performance conclusions |

### Council Invocation Template

```python
# After completing a major phase
def invoke_council_checkpoint(phase, model, hardware):
    context = f"""
    ## MoE Monokernel Council Review

    **Phase Completed**: {phase}
    **Model**: {model}
    **Hardware**: {hardware}

    ## Artifacts
    - Constraints: $(cat {artifact_dir}/constraints.md)
    - Plan: $(cat {artifact_dir}/optimization_plan.md)

    ## Questions
    1. Are there any issues with our approach?
    2. What might we have missed?
    3. Any concerns before proceeding to next phase?
    """

    invoke_skill("llm-council", topic=f"MoE {phase} review", context=context)
```

### Council for Blocked Tasks

When a task exits with status "blocked" after 3 attempts, **automatically** invoke council:

```python
def on_task_blocked(stage, blocker_file):
    if attempts >= 3 and not council_invoked:
        council_context = build_council_context(stage, blocker_file)
        council_feedback = invoke_skill("llm-council",
            topic=f"MoE {stage} implementation blocked",
            context=council_context,
            mode="single-round"
        )
        save_feedback_and_retry(stage, council_feedback)
```

---

## Phase State Machine

```
┌─────────────────┐
│  1_constraints  │──────────────────────────────────────────┐
└────────┬────────┘                                          │
         │ complete                                          │
         ▼                                                   │
┌─────────────────┐                                          │
│   2_planning    │──────────────────────────────────────────┤
└────────┬────────┘                                          │
         │ complete                                          │
         ▼                                                   │
┌─────────────────────────────────────────────────────┐      │
│              3_implementation                        │      │
│  ┌──────────────────┐  ┌──────────────────────────┐ │      │
│  │routing_and_prepare│→ │activation_quantization  │ │      │
│  └──────────────────┘  └───────────┬──────────────┘ │      │
│                                    │ [CONDITIONAL]  │      │
│                                    ▼                │      │
│  ┌─────────────────────────────────────────────┐   │      │
│  │       gemm_implementation (CRITICAL)         │   │      │
│  │   up_projection + down_projection TOGETHER   │   │      │
│  └──────────────────────┬──────────────────────┘   │      │
│                         │ [TODO CHECK]             │      │
│                         ▼                          │      │
│  ┌─────────────────────────────────────────────┐   │      │
│  │           kernel_assembly                    │   │      │
│  └──────────────────────┬──────────────────────┘   │      │
│                         │                          │      │
└─────────────────────────┼──────────────────────────┘      │
         │ complete       │ blocked                         │
         │                ▼                                 │
         │       ┌─────────────────┐                        │
         │       │  llm-council    │────────────────────────┤
         │       └────────┬────────┘                        │
         │                │ feedback                        │
         │                ▼                                 │
         │       ┌─────────────────┐                        │
         │       │  retry stage    │                        │
         │       └─────────────────┘                        │
         ▼                                                   │
┌─────────────────┐                                          │
│  4_validation   │──────────────────────────────────────────┤
└────────┬────────┘                                          │
         │ complete                                          │
         ▼                                                   │
┌─────────────────┐                                          │
│  5_integration  │                                          │
└────────┬────────┘                                          │
         │ complete                                          │
         ▼                                                   │
      [DONE]                                                 │
                                                             │
◄────────────────────────────────────────────────────────────┘
                    (any phase can fail → report)
```

## Phase 3 Stage Structure (Revised)

Phase 3 has **4 stages** (not 6) to keep related GEMM work together:

| Stage | Components | Why Combined |
|-------|------------|--------------|
| routing_and_prepare | router + prepare | Tightly coupled, both non-GEMM |
| activation_quantization | scale_inputs | Conditional (FP8/INT8 only) |
| **gemm_implementation** | **up_proj + down_proj** | **Share 90% structure** |
| kernel_assembly | output + main kernel | Wire everything together |

**Critical insight**: `up_projection` and `down_projection` must be implemented together because:
1. They share the same MMA instruction patterns
2. They share warp specialization (8 calc, 4 prefetch)
3. They share double-buffering logic
4. Implementing separately loses context between tasks
5. Task can factor out common helpers and apply to both

## Stage Status Values

| Status | Meaning |
|--------|---------|
| `pending` | Not yet started |
| `in_progress` | Task spawned, running |
| `complete` | Successfully finished |
| `blocked` | Failed after max attempts, needs council |
| `failed` | Permanently failed (even after council + fallback) |

## Compile and Test Cadence

### After Each Stage: Quick Compile Check
```bash
cmake --build --preset release --target install 2>&1 | head -50
# Just verify no compile errors
# Don't run tests yet
```

### After Stage 2 (routing_and_prepare + activation_quantization): Input Pipeline Test
```bash
# Compile
cmake --build --preset release --target install

# Smoke test input pipeline
python -c "
import torch
# Test that routing and quantization work
from vllm._custom_ops import moe_monokernel_{model}_routing_test
result = moe_monokernel_{model}_routing_test(batch_size=4)
assert result is not None
print('Input pipeline smoke test: PASS')
"
```

### After Stage 3 (gemm_implementation): TODO CHECK + GEMM Test

**CRITICAL**: Before advancing, verify no TODOs in GEMM kernels:
```bash
# Check for incomplete implementations
if grep -n 'TODO\|FIXME\|XXX' csrc/moe/moe_monokernel_*/moe_*_projection.cu; then
    echo "ERROR: TODOs found in GEMM kernels - stage NOT complete"
    echo "Task should have exited BLOCKED, not COMPLETE"
    exit 1
fi

# Verify MMA calls exist
UP_MMA=$(grep -c 'mma_' csrc/moe/moe_monokernel_*/moe_up_projection.cu)
DOWN_MMA=$(grep -c 'mma_' csrc/moe/moe_monokernel_*/moe_down_projection.cu)
if [ "$UP_MMA" -eq 0 ] || [ "$DOWN_MMA" -eq 0 ]; then
    echo "ERROR: MMA loops not implemented (up=$UP_MMA, down=$DOWN_MMA)"
    exit 1
fi

echo "GEMM implementation verified: up=$UP_MMA mma calls, down=$DOWN_MMA mma calls"
```

### After Stage 4 (kernel_assembly): Full Smoke Test
```bash
# Compile
cmake --build --preset release --target install

# Smoke test full kernel
python -c "
import torch
from vllm._custom_ops import moe_monokernel_{model}
# Small tensor test
bs, k, n, e = 4, {K}, {N}, {E}
x = torch.randn(bs, k, dtype=torch.bfloat16, device='cuda')
router = torch.randn(bs, e, dtype=torch.bfloat16, device='cuda')
# ... setup weights ...
result = moe_monokernel_{model}(x, router, ...)
assert result.shape == (bs, k)
print('Full kernel smoke test: PASS')
"
```

## Orchestrator Decision Points

### Stage Order (4 stages)
```python
STAGE_ORDER = [
    "routing_and_prepare",
    "activation_quantization",  # May be skipped for BF16
    "gemm_implementation",       # CRITICAL - up + down together
    "kernel_assembly"
]
```

### On Task Completion
```python
def on_task_complete(state, stage, result):
    state.stages[stage].status = "complete"
    
    # Stage-specific verification
    if stage == "activation_quantization":
        run_compile_and_smoke_test("input_pipeline")
        
    elif stage == "gemm_implementation":
        # CRITICAL: Verify no TODOs before advancing
        if not verify_gemm_complete():
            state.stages[stage].status = "blocked"
            log_error("GEMM stage marked complete but has TODOs")
            spawn_blocker_task(stage, "TODO markers found in MMA loops")
            return
        run_compile_and_smoke_test("gemm")
        
    elif stage == "kernel_assembly":
        run_compile_and_smoke_test("full_kernel")
    else:
        run_quick_compile_check()
    
    # Advance to next stage
    next_stage = get_next_stage(stage)
    if next_stage:
        spawn_stage_task(next_stage)
    else:
        advance_to_phase("4_validation")

def verify_gemm_complete():
    """Check that GEMM implementation is actually complete."""
    import subprocess
    
    # Check for TODOs
    result = subprocess.run(
        ["grep", "-r", "TODO\\|FIXME\\|XXX", 
         "csrc/moe/moe_monokernel_*/moe_*_projection.cu"],
        capture_output=True, text=True
    )
    if result.returncode == 0:  # grep found matches
        return False
    
    # Check for MMA calls
    for proj in ["up", "down"]:
        result = subprocess.run(
            ["grep", "-c", "mma_", 
             f"csrc/moe/moe_monokernel_*/moe_{proj}_projection.cu"],
            capture_output=True, text=True
        )
        if int(result.stdout.strip()) == 0:
            return False
    
    return True
```

### On Task Blocked
```python
def on_task_blocked(state, stage, blocker_file):
    attempts = state.stages[stage].attempts
    
    if attempts < 3:
        # Retry without council
        state.stages[stage].attempts += 1
        spawn_stage_task(stage, retry=True)
    
    elif not state.stages[stage].llm_council_invoked:
        # Invoke council
        state.stages[stage].llm_council_invoked = True
        council_feedback = invoke_llm_council(blocker_file, state)
        save_council_feedback(stage, council_feedback)
        spawn_stage_task(stage, council_context=council_feedback)
    
    elif attempts < 6:
        # Retry with council feedback
        state.stages[stage].attempts += 1
        spawn_stage_task(stage, council_context=get_council_feedback(stage))
    
    else:
        # Fallback
        implement_fallback(stage)
        advance_to_next_stage(stage)
```

### On Phase Transition
```python
def advance_to_phase(new_phase):
    state.phases[current_phase].status = "complete"
    state.phases[new_phase].status = "in_progress"
    state.current_phase = new_phase
    save_state()
    
    # Update TodoWrite to reflect progress
    todo_update = f"""
    ## MoE Monokernel Progress
    - [x] Phase 1: Constraints gathered
    - [x] Phase 2: Optimization planned
    - [x] Phase 3: Implementation complete
    - [ ] Phase 4: Validation (CURRENT)
    - [ ] Phase 5: Integration
    """
    # Preserve existing todos, add/update monokernel section
```

## Resume Protocol

When user triggers resume:

```python
def resume_workflow():
    state = load_state("{artifact_dir}/state.json")
    
    if not state:
        return "No existing optimization found. Start fresh with model and hardware."
    
    # Report status
    report = f"""
    ## MoE Monokernel Status: {state.model} on {state.hardware}
    
    **Current Phase**: {state.current_phase}
    **Last Updated**: {state.updated}
    
    ### Phase Status:
    """
    for phase, info in state.phases.items():
        report += f"- {phase}: {info.status}\n"
    
    if state.current_phase == "3_implementation":
        report += f"\n### Stage Status:\n"
        for stage, info in state.stages.items():
            status = info.status
            if status == "blocked":
                status += f" (attempts: {info.attempts})"
            report += f"- {stage}: {status}\n"
    
    print(report)
    
    # Continue from current position
    if state.current_phase == "3_implementation":
        for stage in STAGE_ORDER:
            if state.stages[stage].status not in ["complete"]:
                spawn_stage_task(stage)
                break
    else:
        spawn_phase_task(state.current_phase)
```

## TodoWrite Integration

The orchestrator maintains a dedicated section in TodoWrite:

```markdown
## MoE Monokernel: {model} on {hardware}

### Current Focus
{current_stage_or_phase}: {brief_description}

### Progress
- [x] Constraints: {artifact_dir}/constraints.md
- [x] Plan: {artifact_dir}/optimization_plan.md
- [x] router: csrc/moe/moe_monokernel_{model}/moe_routing.cu
- [x] prepare: csrc/moe/moe_monokernel_{model}/moe_prepare.cu
- [ ] scale_inputs: IN PROGRESS
- [ ] up_projection
- [ ] down_projection
- [ ] output_conversion
- [ ] Validation
- [ ] Integration

### Blockers
{if any, with links to blocker files}

### Next Steps
1. {immediate_next_action}
2. {following_action}
```

**Important**: When updating todos after council feedback or retry:
- Never remove existing non-monokernel todos
- Update only the MoE Monokernel section
- Preserve the link to ultimate goal in each Task prompt
