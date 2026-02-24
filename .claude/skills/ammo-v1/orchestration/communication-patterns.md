# AMMO Communication Patterns

Standard message formats for team communication. All messages use `SendMessage`.

## Blocker Escalation (Teammate → Lead)

When a teammate encounters a blocker they cannot resolve:

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    BLOCKER [{severity}]: {brief description}
    Phase: {phase}, Gate/Task: {gate_or_task}
    Attempts: {number}
    Blocker file: {artifact_dir}/blockers/{filename}.md
    Root cause hypothesis: {hypothesis}
    Action needed: {what the lead should do}
  summary: "Blocker: {brief description}"
```

Always create the blocker file BEFORE sending the message so the lead can read it.

## Fix Instructions (Lead → Teammate)

When the lead has diagnosed a blocker or has guidance from llm-council:

```
SendMessage:
  type: "message"
  recipient: "{teammate_name}"
  content: |
    Fix for blocker in {task_subject}:

    Diagnosis: {what went wrong}

    Action required:
    1. {step 1}
    2. {step 2}

    References: {relevant files to re-read}

    After fixing, mark the task back to in_progress and re-run.
  summary: "Fix instructions for {blocker_brief}"
```

## Critical Stop (Lead → All)

When a critical issue requires all work to stop immediately:

```
SendMessage:
  type: "broadcast"
  content: |
    CRITICAL STOP: {reason}

    All work must pause. Do NOT mark any tasks as completed.
    Wait for further instructions.

    Details: {brief details}
  summary: "Critical stop: {reason}"
```

Use broadcast ONLY for true critical stops. For issues affecting a single teammate, use direct message.

## Gate Pass Notification (Lead → Affected Teammates)

When a gate task passes:

```
SendMessage:
  type: "message"
  recipient: "{teammate_name}"
  content: |
    Gate PASSED: {gate_name}
    Phase transition: {old_phase} → {new_phase}

    Your next tasks are now unblocked. Check TaskList for available work.
  summary: "Gate passed: {gate_name}"
```

Note: TaskList dependency resolution automatically unblocks tasks, so this message is a courtesy notification. Teammates should periodically check TaskList regardless.

## Gate Fail Notification (Lead → Responsible Teammate)

When a gate task fails:

```
SendMessage:
  type: "message"
  recipient: "{teammate_name}"
  content: |
    Gate FAILED: {gate_name}

    Failure details:
    {verification_script_output_or_review_notes}

    Action required: {what needs fixing}

    I'm creating an investigation task for you.
  summary: "Gate failed: {gate_name}"
```

## Task Completion (Teammate → Lead)

When a teammate finishes a task, they should:
1. Mark the task as `completed` via `TaskUpdate`
2. Send a brief status message to the lead

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    Completed: {task_subject}
    Key outputs: {list of files created/modified}
    Notes: {any issues or findings worth highlighting}
  summary: "Completed: {task_subject_brief}"
```

## Findings Handoff (Teammate → Teammate)

When one teammate's output feeds into another's work:

```
SendMessage:
  type: "message"
  recipient: "{receiving_teammate}"
  content: |
    Handoff from {task_subject}:

    Key findings for your work:
    {relevant findings}

    Files to read:
    - {file1}: {what's in it}
    - {file2}: {what's in it}
  summary: "Handoff: {topic_brief}"
```

Example: planner sends component semantics findings to verifier for constraints.md.

## GPU Benchmark Handoff (Verifier → Lead)

When the verifier completes a GPU benchmark task (T18 or T19):

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    GPU BENCHMARK COMPLETE: {task_subject}

    Results summary: {brief metrics}
    GPU state: {free_memory} GiB free on {gpu_model}
    Next GPU benchmark task unblocked.
  summary: "GPU benchmark complete: {task_brief}"
```

This helps the lead confirm GPU is free before the next benchmark task starts.

## GPU Contention Alert (Any Agent → Lead)

When any agent detects unexpected GPU processes during a benchmark:

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    GPU CONTENTION DETECTED:

    My task: {task_subject}
    Other processes on GPU:
    {nvidia_smi_output}

    Action taken: Stopped my benchmark to avoid unreliable results.
    Requesting: Ensure no other agents are running GPU workloads.
  summary: "GPU contention detected"
```

The lead should check TaskList for concurrent GPU tasks and ensure blockedBy
dependencies are correctly set.

## Shutdown Protocol

### Lead initiates shutdown:
```
SendMessage:
  type: "shutdown_request"
  recipient: "{teammate_name}"
  content: "All tasks complete. Run is finished."
```

Send to each teammate individually.

### Teammate approves shutdown:
```
SendMessage:
  type: "shutdown_response"
  request_id: "{request_id_from_shutdown_request}"
  approve: true
```

### Teammate rejects shutdown (still has work):
```
SendMessage:
  type: "shutdown_response"
  request_id: "{request_id_from_shutdown_request}"
  approve: false
  content: "Still working on {task}. Need {what_is_needed}."
```

## Validation Rejection (Verifier → Lead)

When the verifier finds a validation failure in T17/T18/T19:

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    VALIDATION REJECTED: {gate_name} (cycle {N}/3)

    Failure details:
    - Gate: {gate_5_1_correctness | gate_5_2_kernel_perf | gate_5_3_e2e}
    - Metric: {what was measured}
    - Expected: {threshold or baseline value}
    - Actual: {measured value}
    - Buckets affected: {list}

    Evidence:
    - {artifact_dir}/validation/{evidence_file}

    Recommendation: {retry_stage_4 | retry_stage_3 | escalate_human | document_proceed}
    Rationale: {why this recommendation}
  summary: "Validation rejected: {gate_name}"
```

The lead reviews the evidence and decides the next action. The verifier does NOT send fix instructions directly to the implementer.

## Implementer-Verifier Loop (Lead-Mediated)

When validation fails and the lead decides to retry Stage 4:

### Step 1: Lead creates fix task for implementer
```
SendMessage:
  type: "message"
  recipient: "implementer"
  content: |
    Validation cycle {N}/3: {gate_name} failed.

    Verifier's findings:
    {summary of verifier's rejection evidence}

    Action required:
    1. {specific fix instruction}
    2. {additional fix instruction}

    Constraints:
    - Do NOT change the optimization approach (that requires retry_stage_3)
    - Focus on the specific failure mode identified above

    After fixing: mark the fix task as completed. I will re-run the compilation gate.
  summary: "Fix request: {gate_name} failure (cycle {N}/3)"
```

### Step 2: After implementer fixes and lead runs T15 gate
```
SendMessage:
  type: "message"
  recipient: "verifier"
  content: |
    Implementer fix applied (cycle {N}/3). Compilation passed.

    Re-run validation for: {gate_name}
    Previous failure: {brief summary}

    Re-verify baseline measurements before running optimized benchmarks.
  summary: "Re-validate after fix (cycle {N}/3)"
```

### Loop termination
- **Hard limit**: Max 3 implementer fix cycles per failure mode
- **Convergence test**: If last 2 cycles show <1% improvement in the failing metric, stop
- **Scope escalation**: If fix requires changing the optimization approach, lead creates re-plan task for planner (retry_stage_3)

## Mid-Implementation Stop (Implementer → Lead)

When profiling during implementation shows the plan is wrong (Non-Negotiable: do not change plan mid-stage):

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    STOP: Plan appears invalid during implementation.

    Evidence: {what profiling showed}
    Expected: {what the plan predicted}
    Actual: {what was measured}

    Requesting re-plan with updated evidence.
    I have stopped implementation and will wait for instructions.
  summary: "Implementation stop: plan may be invalid"
```

## KILL → Pivot (Lead → Planner)

When T22 results in a KILL and the lead is pivoting to the next opportunity:

```
SendMessage:
  type: "message"
  recipient: "planner"
  content: |
    KILL → PIVOT: {killed_opportunity_id} failed (attempt {N}/{max_attempts}).

    Kill reason: {kill_reason}
    Kill criteria results:
    {kill_criteria_summary}

    Next opportunity: {new_opportunity_id} from ranked list.

    Action required:
    1. Read previous attempt failures in state.json opportunity_attempts
    2. Write updated optimization_plan.md for {new_opportunity_id}
    3. Include "0A) Previous Attempts" section documenting ALL prior KILLs
    4. Explicitly explain why {new_opportunity_id} avoids the failure mode of {killed_opportunity_id}

    Your task (T24) is now unblocked. Check TaskList.
  summary: "KILL→Pivot: {killed_opportunity_id} → {new_opportunity_id}"
```

## KILL → Pivot (Lead → Verifier)

Notify the verifier that a new iteration is starting:

```
SendMessage:
  type: "message"
  recipient: "verifier"
  content: |
    KILL → PIVOT: {killed_opportunity_id} failed (attempt {N}/{max_attempts}).

    A new optimization approach ({new_opportunity_id}) will be planned.
    You will receive new validation tasks (T16.5-T21) after the implementation stage.

    No action needed now. Check TaskList periodically for new assignments.
  summary: "KILL→Pivot: new iteration starting"
```

## Target Exhausted (Lead → All)

When all attempts are exhausted (len(opportunity_attempts) >= max_attempts):

```
SendMessage:
  type: "broadcast"
  content: |
    TARGET EXHAUSTED: All {max_attempts} optimization opportunities have been tried.

    Attempts:
    {for each attempt: "- Attempt {N}: {opportunity_id} → {status} ({kill_reason})"}

    No further optimization opportunities remain in the ranked list.
    The target ({model_id} on {hardware}) could not be improved within the explored opportunity set.

    Shutting down team. Shutdown requests incoming.
  summary: "Target exhausted after {max_attempts} attempts"
```

After broadcasting, the lead sends `shutdown_request` to each teammate individually.
