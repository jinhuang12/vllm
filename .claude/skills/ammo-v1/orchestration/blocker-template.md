# Blocker Documentation Template

When a Task encounters a blocker it cannot resolve, create a blocker file at:
```
{artifact_dir}/blockers/{phase}_{stage}_{date}.md
```

Example: `moe_monokernel_artifacts/qwen3_l40s_fp8_tp1/blockers/phase4_gate41_2026-01-07.md`

---

## Template

```markdown
# Blocker: {Brief Description}

**Date**: YYYY-MM-DD
**Phase**: {phase number and name}
**Stage/Gate**: {specific stage or gate}
**Severity**: critical | major | minor

---

## Summary

{One paragraph describing what is blocked and why}

---

## Context

**Target**: {model_id}, {hardware}, {dtype}, TP={tp}
**Artifact dir**: {artifact_dir}
**State file**: {artifact_dir}/state.json

**What was attempted**:
1. {First attempt}
2. {Second attempt}
3. {Third attempt (if applicable)}

---

## Error Details

**Verification script output** (if applicable):
```
{paste verify_phase1_baseline.py or verify_phase4_gates.py output}
```

**Specific error message**:
```
{paste specific error or failure message}
```

**Relevant files**:
- {file1}: {why it's relevant}
- {file2}: {why it's relevant}

---

## Root Cause Analysis

**Hypothesis**: {What we believe is causing the issue}

**Evidence**:
- {Evidence point 1}
- {Evidence point 2}

**Confidence**: high | medium | low

---

## Attempted Fixes

### Attempt 1
- **Action**: {What was tried}
- **Result**: {What happened}
- **Why it didn't work**: {Analysis}

### Attempt 2
- **Action**: {What was tried}
- **Result**: {What happened}
- **Why it didn't work**: {Analysis}

### Attempt 3 (if applicable)
- **Action**: {What was tried}
- **Result**: {What happened}
- **Why it didn't work**: {Analysis}

---

## Proposed Resolution

**Option A**: {First option}
- Pros: {pros}
- Cons: {cons}
- Estimated effort: {effort}

**Option B**: {Second option}
- Pros: {pros}
- Cons: {cons}
- Estimated effort: {effort}

**Recommendation**: {Which option and why}

---

## Escalation Request

**Escalation needed**: yes | no

**If yes, reason**:
- [ ] Critical severity (blocks all progress)
- [ ] Exceeded 3 hypothesis cycles
- [ ] Requires expertise outside skill scope
- [ ] Resource constraint (hardware, time)

**Suggested escalation path**:
- [ ] Invoke llm-council for second opinion
- [ ] Escalate to human reviewer
- [ ] Adjust target envelope
- [ ] Pivot to different route (A/B/C)

---

## State.json Update

Update state.json with:
```json
{
  "status": "blocked",
  "blocker": {
    "description": "{brief description}",
    "severity": "{critical|major|minor}",
    "gate": "{gate name}",
    "attempts": {number of attempts},
    "escalation_needed": {true|false},
    "blocker_file": "blockers/{phase}_{stage}_{date}.md"
  }
}
```
```

---

## Severity Guidelines

| Severity | Definition | Orchestrator Action |
|----------|------------|---------------------|
| **critical** | Blocks ALL progress. Cannot proceed without resolution. | STOP immediately. Invoke llm-council. |
| **major** | Blocks current phase but alternatives exist. | Adjust constraints, try different approach. |
| **minor** | Degraded but not blocked. Can continue with caveats. | Document and continue, flag for review. |

---

## Examples

### Example 1: Critical - Wrong Baseline
```markdown
# Blocker: Phase 4 validation used naive PyTorch baseline

**Severity**: critical

**Summary**: Validation compared monokernel against naive PyTorch loops
instead of vLLM's fused_experts. All performance claims are invalid.

**Escalation needed**: yes
- [x] Critical severity (blocks all progress)

**Resolution**: Re-run Phase 4 with correct baseline imports.
```

### Example 2: Major - NCU Profiling Failed
```markdown
# Blocker: NCU profiling crashes on kernel

**Severity**: major

**Summary**: `ncu` command segfaults when profiling down-projection kernel.
Kill criterion #3 (occupancy) cannot be evaluated.

**Escalation needed**: no

**Resolution**: Use alternative occupancy measurement via nsys metrics.
Restrict envelope if occupancy cannot be verified.
```

### Example 3: Minor - Slight Regression at BS=64
```markdown
# Blocker: 2% regression at BS=64

**Severity**: minor

**Summary**: Monokernel shows 2% slowdown at BS=64 while showing
15% improvement at BS≤32. Kill criterion technically fails.

**Escalation needed**: no

**Resolution**: Restrict envelope to BS≤32, document limitation.
```

---

## Team Communication

When running as a teammate in an agent team, escalate blockers to the lead via `SendMessage` in addition to creating the blocker file.

### SendMessage Template

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    BLOCKER [{severity}]: {brief description}
    Phase: {phase}, Gate: {gate}
    Attempts: {number}
    Blocker file: {artifact_dir}/blockers/{phase}_{stage}_{date}.md
    Action needed: {what the lead should do}
  summary: "Blocker: {brief description}"
```

### Examples by Severity

**Critical** (requires immediate lead attention):
```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    BLOCKER [critical]: Phase 4 validation used naive PyTorch baseline instead of vLLM fused_experts.
    All performance claims are invalid. Blocker file: {artifact_dir}/blockers/phase4_gate41_2026-01-07.md
    Action needed: Invoke llm-council, do NOT proceed to Phase 5.
  summary: "Critical blocker: wrong baseline in Phase 4"
```

**Major** (lead can adjust and reassign):
```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    BLOCKER [major]: NCU profiling segfaults on down-projection kernel.
    Kill criterion #3 (occupancy) cannot be evaluated.
    Blocker file: {artifact_dir}/blockers/phase4_gate42_2026-01-07.md
    Action needed: Suggest alternative occupancy measurement or adjust constraints.
  summary: "Major blocker: NCU profiling crash"
```

**Minor** (informational, lead may continue):
```
SendMessage:
  type: "message"
  recipient: "lead"
  content: |
    BLOCKER [minor]: 2% regression at BS=64, 15% improvement at BS≤32.
    Kill criterion technically fails at BS=64 only.
    Blocker file: {artifact_dir}/blockers/phase4_gate42_2026-01-07.md
    Action needed: Consider restricting envelope to BS≤32.
  summary: "Minor blocker: slight BS=64 regression"
```
