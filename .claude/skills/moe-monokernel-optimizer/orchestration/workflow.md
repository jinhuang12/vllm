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

### How to Invoke Council

Use the Skill tool with `llm-council`. The llm-council skill has its own instructions for preparing context and running reviews.

**Example natural language trigger**:
"I've completed Phase 1 for {model}. Before proceeding to planning, let me invoke the llm-council skill to review the constraints."

### Council for Blocked Tasks

When a task exits with status "blocked" after 3 attempts, invoke council for external review:

1. Read the blocker file: `{artifact_dir}/blockers/{stage}_blocker.md`
2. State: "The {stage} has failed 3 times. I'll invoke llm-council for a fresh perspective."
3. Use the Skill tool with `llm-council`
4. Review feedback and spawn a new retry Task with council insights

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

## Phase 4 Validation Flow (with Investigation)

Phase 4 validation has a distinct failure handling path that triggers investigation
rather than the standard escalation ladder used in implementation phases.

### Validation Stage Flow
```
┌─────────────────────────────────────────────────────────────────┐
│                      Phase 4: Validation                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  4.1 Correctness ──┬── pass ──→ 4.2 Kernel Perf                 │
│                    │                    │                        │
│                    └── fail             ├── pass ──→ 4.3 E2E    │
│                         │               │                │       │
│                         ▼               └── fail         ├─ pass │
│               ┌─────────────────┐            │           │       │
│               │ investigate-    │            ▼           │       │
│               │ correctness     │   ┌─────────────────┐  │       │
│               └────────┬────────┘   │ investigate-    │  │       │
│                        │            │ kernel-perf     │  │       │
│                        ▼            └────────┬────────┘  │       │
│               ┌─────────────────┐            │           │       │
│               │ council-review  │            ▼           │       │
│               └────────┬────────┘   ┌─────────────────┐  │       │
│                        │            │ council-review  │  │       │
│                        │            └────────┬────────┘  │       │
│                        │                     │           │       │
│                        └─────────┬───────────┘           │       │
│                                  │                       │       │
│                                  ▼                       │       │
│                        ┌─────────────────┐               │       │
│                        │ Decision Matrix │               │       │
│                        └────────┬────────┘               │       │
│                                 │                        │       │
│     ┌───────────┬───────────────┼───────────┬───────────┤       │
│     ▼           ▼               ▼           ▼           ▼       │
│  Phase 3    Phase 2    Document &     Re-run      Escalate     │
│  (stage)   (re-plan)    Proceed        4.X       to human      │
│     │           │            │           │            │         │
│     └───────────┴────────────┴───────────┘            │         │
│                              │                        │         │
│                              ├── 4.3 fail ──┐         │         │
│                              │              ▼         │         │
│                              │     investigate-e2e    │         │
│                              │              │         │         │
│                              │              ▼         │         │
│                              │     council-review     │         │
│                              │              │         │         │
│                              │              ▼         │         │
│                              │     Decision Matrix ───┘         │
│                              │              │                   │
│                              ▼              │                   │
│                      Phase 5 Integration ◄──┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Investigation Triggers

| Stage | Failure Condition | Investigation Type |
|-------|------------------|-------------------|
| 4.1 Correctness | `max_diff > tolerance` | `correctness` |
| 4.2 Kernel Perf | `speedup < 1.0x` at any BS | `kernel_perf` |
| 4.3 E2E Latency | `improvement ≤ 5%` (BS≤8) or `≤ 0%` (BS>8) | `e2e_perf` |

### Decision Matrix

After investigation completes with council approval:

| Decision | When | Orchestrator Action |
|----------|------|---------------------|
| `phase_3` | Implementation bug identified | Reset target stage to `pending`, spawn task with fix context |
| `phase_2` | Algorithmic decision wrong | Reset Phase 2, spawn planning task with new constraints |
| `document_proceed` | Expected behavior, acceptable limitation | Document in validation_results.md, proceed to Phase 5 |
| `rerun_validation` | Measurement/environment issue fixed | Re-spawn the failing validation stage |
| `escalate_human` | Cannot determine root cause | Set status to `escalated`, report to user |

See `orchestration/investigation-prompts.md` for full investigation task prompts.
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
| `needs_investigation` | Validation failed, awaiting investigation task |
| `investigating` | Investigation task in progress |
| `escalated` | Cannot resolve automatically, needs human review |

## Compile and Test Cadence

**IMPORTANT**: All validations are **BLOCKING**. A stage cannot be marked complete unless its validation passes. See `orchestration/task-prompts.md` for full validation code.

### Stage Validation Summary

| Stage | Validation Type | Blocking? | Reference |
|-------|-----------------|-----------|-----------|
| routing_and_prepare | Compare vs fused_topk | **YES** | task-prompts.md Stage 1 |
| activation_quantization | Compare vs torch dynamic quant | **YES** (FP8 only) | task-prompts.md Stage 2 |
| gemm_implementation | Correctness + Performance sanity | **YES** | task-prompts.md Stage 3 |
| kernel_assembly | Full kernel vs fused_moe | **YES** | task-prompts.md Stage 4 |

### After Each Stage: Validation Must Pass

Each stage has inline validation code in its task prompt. The Task must:
1. Run the validation code
2. If validation fails → Exit with status "blocked"
3. If validation passes → May mark stage "complete"

### Stage 1 (routing_and_prepare): Routing Validation
- Compare topk_ids and topk_weights against `vllm.model_executor.layers.fused_moe.fused_topk`
- IDs must match exactly (same expert selection)
- Weights must be close (atol=1e-5, rtol=1e-5)

### Stage 2 (activation_quantization): Quantization Validation
- Skip for BF16/FP16 (no quantization)
- For FP8: Compare scales and dequantized values against torch reference
- Tolerance: atol=0.5, rtol=0.1 for dequantized values

### Stage 3 (gemm_implementation): CRITICAL Validation

**This is the most important validation** - catches "naive GEMM" implementations.

1. **Code completeness**: No TODO/FIXME/XXX in MMA loops
2. **MMA verification**: grep -c 'mma_' must be > 0 for both projections
3. **Correctness**: Compare against torch.matmul reference
4. **Performance sanity**: Must complete in < 10ms for BS=8 (catches naive loop implementations)

```bash
# Quick check (orchestrator can verify)
grep -c 'mma_' csrc/moe/moe_monokernel_*/moe_up_projection.cu    # Must be > 0
grep -c 'mma_' csrc/moe/moe_monokernel_*/moe_down_projection.cu  # Must be > 0
```

### Stage 4 (kernel_assembly): Integration Validation
- Test full kernel against `fused_moe` reference
- Test across batch sizes: 1, 4, 16, 32, 64
- Tolerance: atol=1e-2, rtol=1e-2

## Orchestrator Decision Points

**Note**: The Python code blocks below are **illustrative pseudo-code** showing the orchestrator's decision logic. They are not executable APIs. The orchestrator (Claude) implements this logic by reading state files, using the Task tool, and using the Skill tool.

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
        # Invoke council using Skill tool with "llm-council"
        state.stages[stage].llm_council_invoked = True
        # Claude uses Skill tool, reviews feedback, saves to artifact_dir
        council_feedback = get_council_feedback_from_skill_output()
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

When advancing to a new phase:
1. Update state.json with new phase status
2. Update TodoWrite using proper JSON format (see TodoWrite Integration section above)
3. Spawn Task for the new phase

### Orchestrator Logic for Phase 4 Validation

**Note**: The Python code blocks below are **illustrative pseudo-code** showing the 
orchestrator's decision logic. They are not executable APIs.
```python
def on_validation_stage_complete(state, stage, result):
    """Handle validation stage completion."""
    
    if result.status == "complete":
        # Validation passed - advance to next stage or Phase 5
        next_stage = get_next_validation_stage(stage)
        if next_stage:
            state.phases["4_validation"].stages[next_stage].status = "in_progress"
            spawn_validation_task(next_stage)
        else:
            # All validation passed
            state.phases["4_validation"].status = "complete"
            advance_to_phase("5_integration")
    
    elif result.status == "needs_investigation":
        # Validation failed - spawn investigation task
        investigation_type = {
            "4_1_correctness": "correctness",
            "4_2_kernel_perf": "kernel_perf",
            "4_3_e2e_latency": "e2e_perf"
        }[stage]
        
        state.phases["4_validation"].status = "investigating"
        state.phases["4_validation"].investigation = {
            "type": investigation_type,
            "started_at": now(),
            "hypothesis_cycles": 0,
            "ncu_runs": 0,
            "council_reviews": 0,
            "root_cause": None,
            "proposed_fix": None,
            "decision": None
        }
        
        # Spawn investigation task from investigation-prompts.md
        spawn_investigation_task(investigation_type, failure_context=result)


def on_investigation_complete(state, result):
    """Handle investigation task completion."""
    
    decision = result.decision
    investigation = state.phases["4_validation"].investigation
    
    # Update investigation record
    investigation.root_cause = result.root_cause
    investigation.proposed_fix = result.proposed_fix
    investigation.decision = decision
    
    if decision == "phase_3":
        # Implementation bug - go back to specific stage
        target_stage = result.target_stage
        
        state.phases["3_implementation"].status = "in_progress"
        state.phases["3_implementation"].stages[target_stage].status = "pending"
        state.phases["4_validation"].status = "pending"
        
        # Spawn task with fix context from investigation
        spawn_stage_task(
            target_stage,
            additional_context=f"""
## Fix Context from Investigation

{result.fix_proposal}

## Previous Failure
{result.root_cause}
"""
        )
    
    elif decision == "phase_2":
        # Algorithmic decision wrong - re-plan
        state.phases["2_planning"].status = "in_progress"
        state.phases["3_implementation"].status = "pending"
        state.phases["4_validation"].status = "pending"
        
        spawn_phase_task(
            "2_planning",
            additional_constraints=f"""
## Additional Constraints from Investigation

The previous optimization plan resulted in validation failure.

### Findings
{result.findings}

### Required Changes
{result.required_changes}
"""
        )
    
    elif decision == "document_proceed":
        # Expected behavior - document and continue
        state.phases["4_validation"].status = "complete"
        state.phases["4_validation"].limitation = result.limitation_doc
        
        # Append limitation to validation_results.md
        limitation_section = f"""

## Known Limitations

{result.limitation_doc}

### Recommendation
{result.recommendation}
"""
        append_to_file(f"{artifact_dir}/validation_results.md", limitation_section)
        
        advance_to_phase("5_integration")
    
    elif decision == "rerun_validation":
        # Measurement issue fixed - retry validation
        failing_stage = investigation.type.replace("_", "_")  # e.g., "correctness" -> "4_1_correctness"
        stage_map = {
            "correctness": "4_1_correctness",
            "kernel_perf": "4_2_kernel_perf",
            "e2e_perf": "4_3_e2e_latency"
        }
        target_stage = stage_map[investigation.type]
        
        state.phases["4_validation"].status = "in_progress"
        state.phases["4_validation"].stages[target_stage].status = "pending"
        state.phases["4_validation"].investigation = None  # Clear investigation
        
        spawn_validation_task(target_stage)
    
    elif decision == "escalate_human":
        # Cannot resolve - escalate to user
        state.phases["4_validation"].status = "escalated"
        
        report = f"""
## MoE Monokernel: Human Review Required

The automated investigation could not determine a fix.

### Investigation Summary
- Type: {investigation.type}
- Hypothesis cycles: {investigation.hypothesis_cycles}
- NCU profiling runs: {investigation.ncu_runs}
- Council reviews: {investigation.council_reviews}

### Findings
{result.findings}

### Recommendation
{result.recommendation}

### Investigation Artifacts
Location: {artifact_dir}/investigation/

### Next Steps
1. Review investigation artifacts
2. Provide guidance to resume, OR
3. Manually fix and re-run validation, OR
4. Abandon monokernel for this model
"""
        print(report)
        # Workflow pauses - user must intervene
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

    # Handle investigation status
    if state.phases.get("4_validation", {}).get("status") == "investigating":
        investigation = state.phases["4_validation"].get("investigation", {})
        report += f"\n### Investigation in Progress\n"
        report += f"- Type: {investigation.get('type')}\n"
        report += f"- Hypothesis cycles: {investigation.get('hypothesis_cycles', 0)}\n"
        report += f"- Council reviews: {investigation.get('council_reviews', 0)}\n"
        
        # Resume investigation task
        spawn_investigation_task(investigation.get("type"))
        return
    
    if state.phases.get("4_validation", {}).get("status") == "escalated":
        report += "\n### ⚠️ ESCALATED - Human Review Required\n"
        report += "See investigation artifacts in {artifact_dir}/investigation/\n"
        print(report)
        return  # Don't auto-resume, wait for user guidance    
```

## TodoWrite Integration

Use the TodoWrite tool to track progress. The tool requires a JSON array of todo objects.

**TodoWrite Format**:
```json
[
  {
    "content": "Phase 1: Gather constraints for {model}",
    "status": "completed",
    "activeForm": "Gathering constraints"
  },
  {
    "content": "Phase 2: Create optimization plan",
    "status": "completed",
    "activeForm": "Creating optimization plan"
  },
  {
    "content": "Phase 3 routing_and_prepare: Implement routing",
    "status": "in_progress",
    "activeForm": "Implementing routing and prepare"
  },
  {
    "content": "Phase 3 activation_quantization: Implement scale_inputs",
    "status": "pending",
    "activeForm": "Implementing activation quantization"
  },
  {
    "content": "Phase 3 gemm_implementation: Implement up/down projection",
    "status": "pending",
    "activeForm": "Implementing GEMM kernels"
  },
  {
    "content": "Phase 3 kernel_assembly: Assemble main kernel",
    "status": "pending",
    "activeForm": "Assembling kernel"
  },
  {
    "content": "Phase 4: Validate monokernel",
    "status": "pending",
    "activeForm": "Validating monokernel"
  },
  {
    "content": "Phase 5: Integrate into vLLM",
    "status": "pending",
    "activeForm": "Integrating into vLLM"
  }
]
```

**Status Values**:
- `"pending"` - Not yet started
- `"in_progress"` - Currently working on (limit to ONE at a time)
- `"completed"` - Successfully finished

**Update Pattern**: After completing a phase/stage, mark it `"completed"` and mark the next `"in_progress"`.
