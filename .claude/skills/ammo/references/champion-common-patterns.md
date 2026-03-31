# Champion Common Patterns

Shared interaction patterns for AMMO champion agents (debate and implementation). Role-specific additions (what to delegate, common monitor flags, Self-Validation Gate) stay inline in each agent definition.

## Subagent Delegation

Use `ammo-delegate` subagents for parallelizable research tasks. Delegates are fire-and-forget — they have full AMMO domain context (references, scripts, GPU pool pattern, production parity rules) baked into their agent definition. You cannot send follow-up messages; give each delegate a complete, self-contained prompt.

**Your job is strategy, synthesis, and decision-making — NOT doing all the research yourself.** Delegates handle the investigation; you handle the interpretation and action.

### Spawn Pattern

```python
Agent(
  subagent_type="ammo-delegate",
  run_in_background=True,
  description="<short task description>",
  prompt="""
  <specific task>
  Artifact directory: {artifact_dir}
  Worktree: {worktree_path}  # if in implementation phase
  """
)
```

Spawn multiple delegates in parallel for independent tasks. Results return directly to your context; no SendMessage coordination needed.

### Model Override for Complex Tasks

For tasks requiring deeper reasoning (cross-system analysis, ambiguous root-cause investigation), override the delegate's default model:

```python
Agent(
  subagent_type="ammo-delegate",
  model="opus",
  run_in_background=True,
  description="Deep investigation of <issue>",
  prompt="..."
)
```

## Message Delivery & Responsiveness

### How Message Delivery Works

Teammate messages are delivered as new conversation turns. A new turn can only start when your current response ends — i.e., when you stop making tool calls. If you run blocking Bash commands without ending your response, queued messages from teammates are deferred until you pause. This applies even after a blocking command finishes: if you immediately start another tool call, the message is deferred again.

### Never Block With Sleep Loops

Never use sleep loops to wait for teammates or monitor processes. This blocks message delivery for the entire duration AND prevents delivery even afterward if you immediately chain more commands.

**Wrong:**
```bash
for i in 1 2 3 4 5; do sleep 30; check_status; done
```

**Right:** Check once, send a message, then end your turn:
```bash
check_status_once
```
Then use SendMessage to ask your teammate, and **stop making tool calls** so your turn ends and their response can be delivered.

### Long-Running Commands: Background + End Turn

For benchmarks, sweeps, ncu runs — anything >30 seconds where you don't need the result for your next immediate decision:

```json
{"command": "source .venv/bin/activate && python run_sweep.py ...",
 "run_in_background": true, "timeout": 1800000}
```

After starting the background command, **stop making tool calls** so your turn ends and queued messages can be delivered. You'll be notified when it completes.

### Foreground vs Background

| Use foreground | Use background |
|---------------|----------------|
| Need result before next step | Just monitoring progress |
| <30 seconds | >30 seconds |
| `cmake --build`, `pytest`, quick `nvidia-smi` | E2E sweeps, ncu profiling, model benchmarks |

### Status Checks While Waiting

One status check message per 10 minutes of silence. After sending, end your turn to receive the response. While waiting, do useful non-blocking work (review code, draft reports, pre-compute Amdahl's numbers, scaffold test files).

## Transcript Monitor

A transcript monitor agent reads your session log periodically and flags methodology errors via SendMessage. Messages arrive as `DA-MONITOR: [{SEVERITY}] ...` with evidence and recommended action.

### Severity Responses

| Severity | Action |
|----------|--------|
| **CRITICAL** | Stop current approach and address before continuing |
| **WARNING** | Investigate before committing to current approach |
| **INFO** | Note for later, continue current work |

### Enable Message Delivery

The monitor cannot interrupt mid-turn — messages arrive at turn boundaries. Background long-running commands to create more boundaries:

```
Bash(command="ncu --set full ...", run_in_background=True)
```

This ensures you receive monitor interjections promptly instead of discovering them after a 10-minute blocking command.

## Handling Incoming Messages (Tiered Assessment)

Messages from teammates (validator, monitor, orchestrator) are NOT automatically correct. Context pressure degrades reasoning quality — both yours and theirs. Before acting on any finding, triage it.

### Step 1: Read Without Acting

Read the full message. Do not start editing code, debugging, or responding yet. Just read.

### Step 2: Assess Correctness

Ask yourself: "Could this finding be wrong?" Consider:
- Could the sender's methodology be flawed (wrong tolerance, bad shapes, incorrect baseline)?
- Could they have misinterpreted in-progress work as a completed mistake?
- Does this finding conflict with profiling data or the debate/implementation plan?

### Step 3: Classify Assessment Complexity

Based on BOTH the finding's nature AND the required response, pick a tier:

| Tier | When | Action |
|------|------|--------|
| **Tier 1** (self-assess) | Simple finding + simple action. You can verify by reading a few lines of code or checking a single value. | Reason through it inline, document your assessment, proceed. |
| **Tier 2** (delegate to Sonnet) | Medium complexity — proper investigation would consume significant context. OR your first attempt to address this issue already failed. | Spawn `ammo-delegate` with the message + relevant context. |
| **Tier 3** (delegate to Opus) | High complexity — challenges core assumptions or involves cross-system reasoning. OR you've failed 2+ attempts at the same issue. | Spawn `ammo-delegate` with `model="opus"` and the full context package. |

**Auto-escalation**: If you've already attempted N responses to the same issue, auto-escalate to tier min(N, 3). Repeated surface-level responses indicate you're not grasping the root cause — a fresh-context agent will reason more clearly than you can at this point in the session.

### Assessment Delegation Template

```python
Agent(
    subagent_type="ammo-delegate",
    model="opus",  # or omit for Sonnet (Tier 2)
    run_in_background=True,
    description="Assess incoming finding",
    prompt=f"""
    Assess this finding for {op_id}:

    MESSAGE: {full_message}

    CONTEXT:
    - Plan: {artifact_dir}/debate/summary.md
    - Current work state: {brief description of where you are}
    - Previous attempts to address this issue: {count} — {brief description}

    TASKS:
    1. Is the finding CORRECT? Could the sender's methodology or
       observation be wrong?
    2. If correct: what is the ROOT CAUSE (not just the surface symptom)?
    3. What action addresses the root cause?
    4. What verification confirms the action works?

    Report your assessment. The champion will decide whether to act on it.
    Worktree: {worktree_path}
    Artifact dir: {artifact_dir}
    """
)
```
