# MoE Monokernel Failure Handling

## Failure Escalation Ladder

```
┌─────────────────────────────────────────────────────────────┐
│                    Task Encounters Error                     │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 1: Self-Fix (within Task)                             │
│ - Read error message                                        │
│ - Identify obvious fix (typo, missing include, etc.)        │
│ - Retry up to 3 times for SAME error type                   │
└─────────────────────────────┬───────────────────────────────┘
                              │ Still failing?
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 2: Document & Exit (Task → Orchestrator)              │
│ - Write blocker file with full context                      │
│ - Exit with status "blocked"                                │
│ - Orchestrator takes over                                   │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 3: Orchestrator Retry (without council)               │
│ - Review blocker file                                       │
│ - Try different approach heuristic                          │
│ - Spawn new Task with hints                                 │
│ - Up to 2 orchestrator retries                              │
└─────────────────────────────┬───────────────────────────────┘
                              │ Still failing?
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 4: LLM Council Consultation                           │
│ - Build rich context document                               │
│ - Invoke llm-council skill                                  │
│ - Synthesize recommendation                                 │
│ - Spawn new Task with council insight                       │
└─────────────────────────────┬───────────────────────────────┘
                              │ Still failing?
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 5: Post-Council Retries                               │
│ - 3 more attempts with council feedback                     │
│ - Try alternative interpretations                           │
└─────────────────────────────┬───────────────────────────────┘
                              │ Still failing?
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Level 6: Fallback Implementation                            │
│ - Implement simpler version                                 │
│ - Document limitations and performance impact               │
│ - Continue to next stage                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Level 1: Task Self-Fix

### Error Classification

| Error Type | Example | Fix Strategy |
|------------|---------|--------------|
| **Syntax** | Missing semicolon, brace | Direct fix |
| **Include** | Unknown type, undefined symbol | Add include |
| **Type mismatch** | Cannot convert X to Y | Cast or change type |
| **Dimension** | static_assert failed | Check constraints.md |
| **Undefined reference** | Linker error | Check declaration/definition |
| **CUDA** | Invalid device function | Check __device__ annotations |

### Self-Fix Protocol

```python
# Within Task execution
attempt = 0
max_self_fix = 3
error_types_seen = set()

while attempt < max_self_fix:
    result = compile()
    if result.success:
        break
    
    error_type = classify_error(result.error)
    
    if error_type in error_types_seen:
        # Same error type seen before - not making progress
        break
    
    error_types_seen.add(error_type)
    fix = generate_fix(error_type, result.error)
    apply_fix(fix)
    attempt += 1

if not result.success:
    write_blocker_file()
    exit(status="blocked")
```

---

## Level 2: Blocker Documentation

### Blocker File Format

Location: `{artifact_dir}/blockers/{stage_name}_blocker.md`

```markdown
# Blocker: {stage_name}

## Error
```
{full_error_message}
```

## Context
- Stage: {stage_name}
- Attempt: {N}
- File: {file_path}
- Line: {line_number if known}

## Code Snippet
```cpp
// Relevant code around error
{code_context}
```

## Attempts Made

### Attempt 1
- Hypothesis: {what_i_thought_was_wrong}
- Change: {what_i_changed}
- Result: {still_failed_because}

### Attempt 2
- Hypothesis: {next_theory}
- Change: {modification}
- Result: {outcome}

### Attempt 3
- Hypothesis: {final_theory}
- Change: {modification}
- Result: {outcome}

## Analysis
- Root cause hypothesis: {best_guess}
- Blocking constraint: {what_prevents_fix}
- Related patterns in Llama4 patch: {if_found}

## Questions for Escalation
1. {specific_question_1}
2. {specific_question_2}
```

---

## Level 3: Orchestrator Retry Heuristics

Before invoking council, orchestrator tries common fixes:

### Heuristic Matrix

| Error Pattern | Orchestrator Hint |
|---------------|-------------------|
| Bank conflict / shared memory | "Try adding padding to row stride" |
| Register pressure | "Reduce unrolling, split computation" |
| Occupancy low | "Reduce shared memory per block" |
| Dimension mismatch | "Verify K, N values match optimization_plan.md" |
| MMA error | "Check operand types match mma instruction" |
| Cooperative launch | "Verify grid size ≤ SM count" |

### Orchestrator Retry Task

```markdown
Task: "Retry {stage_name} with hint: {heuristic_hint}

**Previous Error**:
$(cat {artifact_dir}/blockers/{stage_name}_blocker.md | head -50)

**Hint from Orchestrator**:
{heuristic_hint}

**Suggested Change**:
{specific_suggestion}

Try this approach. If it doesn't work, document what happened and exit blocked.

{standard_behavioral_footer}
"
```

---

## Level 4: LLM Council Protocol

### Building Council Context

The orchestrator builds a comprehensive context document:

```markdown
# MoE Monokernel Council Request

## Ultimate Goal
Implement optimized MoE monokernel for {model} on {hardware} targeting decode BS ≤ 64.

## Current Status
- Phase: 3_implementation
- Stage: {blocked_stage}
- Attempts: {total_attempts}
- Completed stages: {list}

## Constraints
$(cat {artifact_dir}/constraints.md)

## Optimization Plan
$(cat {artifact_dir}/optimization_plan.md)

## Blocker Details
$(cat {artifact_dir}/blockers/{stage_name}_blocker.md)

## Current Implementation
```cpp
$(cat csrc/moe/moe_monokernel_{model}/{stage_file}.cu | head -200)
```

## Reference Implementation (Llama4)
```cpp
$(grep -A 100 "{relevant_function}" assets/LLAMA4_MONOKERNEL_PATCH.md)
```

## Specific Questions
1. {question_from_blocker_1}
2. {question_from_blocker_2}
3. What alternative approach would you suggest?
```

### Invoking Council

Use the Skill tool with `llm-council`. The llm-council skill has its own context preparation instructions.

**Steps**:
1. Prepare context: Write council context to `{artifact_dir}/council_context_{round}.md`
2. State intent: "I'll invoke llm-council to review this blocked stage."
3. Use Skill tool with `llm-council`
4. Review output from `.llm-council/tmp/critic_*.md`
5. Save synthesis to `{artifact_dir}/council_feedback_{round}.md`

### Synthesizing Recommendation

After council feedback, orchestrator synthesizes actionable recommendation:

```python
def synthesize_recommendation(council_feedback, blocker):
    """
    Extract actionable recommendation from council feedback.
    
    Priority:
    1. Consensus points (both Gemini and Codex agree)
    2. Specific code suggestions
    3. Alternative algorithm approaches
    4. Simplification suggestions (for fallback)
    """
    
    # Parse feedback sections
    gemini_points = extract_points(council_feedback, "Gemini")
    codex_points = extract_points(council_feedback, "Codex")
    
    # Find consensus
    consensus = find_overlap(gemini_points, codex_points)
    
    # Build recommendation
    if consensus:
        return f"""
        **Council Consensus**: {consensus}
        
        **Recommended Change**:
        {format_code_suggestion(consensus)}
        
        **Why This Should Work**:
        {extract_rationale(council_feedback)}
        """
    else:
        # No consensus - try Gemini's approach first (longer context window)
        return f"""
        **Primary Recommendation (Gemini)**: {gemini_points[0]}
        
        **Alternative (Codex)**: {codex_points[0]}
        
        Try primary first, then alternative if that fails.
        """
```

---

## Level 5: Post-Council Retries

### Council Retry Task Template

```markdown
Task: "Continue {stage_name} with LLM council guidance.

**Ultimate Goal**: {ultimate_goal}

**Council Feedback Summary**:
$(cat {artifact_dir}/council_feedback_{round}.md)

**Synthesized Recommendation**:
{recommendation}

**Instructions**:
1. Apply the recommended change
2. If it doesn't compile, try the alternative interpretation
3. If still stuck, document what the council suggestion didn't account for

**Progress Context**:
- Completed: {completed_stages}
- Current: {stage_name} (post-council attempt {N})
- Remaining: {remaining_stages}

{standard_behavioral_footer}
"
```

### Tracking Council Effectiveness

Update state after council retry:

```json
{
  "llm_council_history": [
    {
      "round": 1,
      "stage": "up_projection",
      "topic": "Bank conflict in shared memory swizzle",
      "recommendation": "Use XOR-based swizzle with 32-byte stride",
      "outcome": "resolved",  // or "partial", "ineffective"
      "attempts_after": 1
    }
  ]
}
```

---

## Level 6: Fallback Implementation

### When to Fallback

After Level 5 exhausted:
- 3 self-fix attempts (Level 1)
- 2 orchestrator retries (Level 3)
- 1 council round (Level 4)
- 3 post-council retries (Level 5)

Total: ~9 attempts on single issue

### Fallback Strategies by Stage

| Stage | Optimized | Fallback |
|-------|-----------|----------|
| router | Warp reduction | Naive argmax loop |
| prepare | Bitfield/histogram | Sequential scan |
| scale_inputs | Vectorized with swizzle | Element-by-element |
| up_projection | Triple-buffer + MMA | Single-buffer + MMA |
| down_projection | Swizzled + atomic | Simple accumulator |
| output | Fused conversion | Separate kernel |

### Fallback Documentation

```markdown
# Fallback: {stage_name}

## Original Approach
{description_of_optimized_implementation}

## Blocking Issue
{persistent_error_after_all_attempts}

## Council Feedback
{what_council_suggested}

## Why It Didn't Work
{analysis_of_failure}

## Fallback Implementation
{description_of_simpler_approach}

## Performance Impact
- Expected regression: {estimate}%
- Affected batch sizes: {which_BS}
- Mitigation: {if_any}

## Code Location
- Fallback: `csrc/moe/moe_monokernel_{model}/{stage}_fallback.cu`
- Original (commented): `csrc/moe/moe_monokernel_{model}/{stage}.cu.optimized`

## Future Work
To restore optimized version:
1. {what_needs_to_be_solved}
2. {prerequisite_knowledge}
3. {suggested_approach}
```

### Continuing After Fallback

Fallback doesn't block progress.

**Steps** (illustrative - orchestrator implements this logic using the Task tool):

1. **Spawn fallback Task**: Use Task tool with `subagent_type: "general-purpose"` to implement simplified version
2. **Document limitation**: Write fallback docs to `{artifact_dir}/fallback_{stage}.md`
3. **Update state**: Mark stage as `"complete"` with `is_fallback: true`
4. **Continue**: Spawn next stage Task

---

## Error Recovery for Orchestrator

### Orchestrator Crash Recovery

If orchestrator crashes mid-workflow:

1. State file `{artifact_dir}/state.json` persists
2. TodoWrite has checkpoint
3. Resume protocol restores position

### Partial Stage Output

If stage partially completed before failure:

```python
def recover_partial_stage(stage):
    stage_file = get_stage_file(stage)
    
    if file_exists(stage_file):
        # Check if compilable
        if quick_compile_check():
            # Usable partial - continue from here
            return "continue"
        else:
            # Broken partial - backup and restart
            backup_file(stage_file, f"{stage_file}.broken")
            return "restart"
    else:
        return "restart"
```

### State Corruption Recovery

If state file corrupted:

```python
def recover_state():
    # Try primary state file
    state = try_load("{artifact_dir}/state.json")
    if state:
        return state
    
    # Try backup
    state = try_load("{artifact_dir}/state.json.bak")
    if state:
        return state
    
    # Reconstruct from artifacts
    state = reconstruct_from_artifacts()
    return state

def reconstruct_from_artifacts():
    """Infer state from existing files."""
    state = new_state()
    
    # Check for constraints
    if file_exists("{artifact_dir}/constraints.md"):
        state.phases["1_constraints"].status = "complete"
    
    # Check for plan
    if file_exists("{artifact_dir}/optimization_plan.md"):
        state.phases["2_planning"].status = "complete"
    
    # Check for stage files
    for stage in STAGES:
        stage_file = f"csrc/moe/moe_monokernel_{model}/{stage}.cu"
        if file_exists(stage_file) and compiles(stage_file):
            state.stages[stage].status = "complete"
    
    return state
```

---

## Monitoring and Alerts

### Progress Stall Detection

If no progress for extended period:

```python
STALL_THRESHOLD = timedelta(minutes=30)

def check_stall():
    last_update = state.updated
    if datetime.now() - last_update > STALL_THRESHOLD:
        # Alert orchestrator
        report_stall(
            stage=state.current_stage,
            duration=datetime.now() - last_update,
            last_activity=get_last_activity()
        )
```

### Success Metrics

Track across optimization attempts:

```python
metrics = {
    "total_attempts": 0,
    "self_fixes": 0,
    "orchestrator_retries": 0,
    "council_invocations": 0,
    "fallbacks": 0,
    "stages_completed": 0,
    "wall_time": 0,
}
```

Report at end:

```markdown
## Optimization Summary

- Total attempts: {N}
- Self-fixed errors: {N}
- Orchestrator assists: {N}
- Council consultations: {N}
- Fallbacks used: {N} ({list_stages})
- Wall time: {duration}
- Final status: {COMPLETE | PARTIAL}
```
