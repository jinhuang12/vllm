# Context-Resilient Champion-Validator Interaction

**Date**: 2026-03-27
**Status**: Draft
**Scope**: `ammo-impl-champion.md`, `ammo-transcript-monitor.md`

## Problem

As the impl-champion's context window fills (from accumulated delegate results, build output, validator messages, and general interaction), two critical behaviors degrade:

1. **Self-validation bypass**: After receiving a bug report from the validator, the champion makes the fix and immediately sends for re-validation — no smoke test, no reasoning about root cause, no verification that the fix actually works. Early in context, the champion does both: reasons through the fix AND re-runs its own verification. Late in context, it skips both.

2. **Uncritical message acceptance**: The champion treats all validator and transcript-monitor messages as ground truth without reasoning about whether the message is correct. This has led to both unnecessary/harmful changes (when the validator was wrong) and sloppy fixes (when the champion addressed the surface symptom without understanding the root cause).

Both degradation patterns have been observed across multiple campaign tracks. The validator (Sonnet) also degrades but is lower priority — fix the champion first and observe results.

### Root Cause

Context pressure causes the champion to take shortcuts on reasoning-intensive steps. The current agent definition describes the expected behavior (Layer 2: Champion Review in `impl-track-rules.md`) but provides no mechanism to compensate when the champion's reasoning quality drops. The champion doesn't recognize its own degradation.

### Key Insight

A degraded champion can still perform one simple task: classify complexity. The fix delegates actual reasoning to fresh-context agents when assessment difficulty exceeds the champion's current capacity. This is structurally resilient because the delegate's context is never degraded.

## Design

### Part 1: Tiered Message Assessment Protocol (impl-champion)

When the champion receives a message from the validator or transcript-monitor, it follows a triage protocol instead of immediately acting:

#### Step 1: Read Without Acting

Read the full message. Do not start editing code, do not start debugging, do not start responding. Just read.

#### Step 2: Assess Correctness

Ask: "Could this finding be wrong? What would make it incorrect?" Consider:
- Could the validator's test methodology be flawed (wrong tolerance, bad tensor shapes, incorrect baseline)?
- Could the monitor have misinterpreted the transcript (in-progress work flagged as a completed mistake)?
- Does the finding conflict with profiling data or the debate plan?

#### Step 3: Classify Assessment Complexity

Based on BOTH the finding's nature AND the required response:

| Tier | Signal | Action | Example |
|------|--------|--------|---------|
| **Tier 1** (self-assess) | Simple finding + simple required action. Champion can verify by reading a few lines of code or checking a single value. | Champion reasons through it inline, documents assessment, proceeds. | "TypeError on line 42" — read the line, confirm the type mismatch, fix it. |
| **Tier 2** (delegate to Sonnet) | Medium complexity — would consume significant context to investigate properly. OR champion's first fix attempt for this issue already failed. | Spawn `ammo-delegate` (Sonnet) with the validator/monitor message + relevant code context. Delegate assesses validity and recommends action. | "Amdahl violation: actual E2E 3.2% but expected max 1.6%" — needs math cross-checking against constraints.md and Gate 5.2 raw data. |
| **Tier 3** (delegate to Opus) | High complexity — challenges core assumptions or involves cross-system reasoning. OR champion has failed 2+ fixes for the same issue. | Spawn `ammo-delegate` (Opus) with full context package. Delegate assesses, investigates root cause, recommends action. | "Cross-track .so contamination suspected", "kernel not dispatching under CUDA graphs", or third attempt at fixing the same Gate 5.1 failure. |

**Auto-escalation rule**: If the champion has already attempted N fixes for the same issue, auto-escalate to tier min(N, 3). This catches the thrashing pattern where repeated surface-level fixes indicate the champion isn't understanding the root cause.

#### Delegation Prompt Template

```
Assess this validation/monitor finding for {op_id}:

MESSAGE: {full validator or monitor message}

CONTEXT:
- Debate plan: {artifact_dir}/debate/summary.md
- Current implementation diff: {git diff output or file paths}
- Previous fix attempts for this issue: {count and brief description}

TASKS:
1. Is the finding CORRECT? Could the validator's test methodology or the monitor's observation be wrong?
2. If correct: what is the ROOT CAUSE (not just the surface symptom)?
3. What fix would address the root cause?
4. What verification confirms the fix works?

Report your assessment. The champion will decide whether to act on it.
```

### Part 2: Self-Validation Gate (impl-champion)

Before the champion can send a re-validation request to the validator (after fixing a reported issue), it must complete a self-validation checklist:

1. **Root cause reasoning**: Write 2-3 sentences explaining WHY this fix addresses the underlying issue, not just the surface symptom. If the champion cannot articulate the root cause, this is a signal to escalate to Tier 2+ assessment.

2. **Smoke test**: Re-run the champion's own correctness check (at minimum: `torch.allclose` on the optimized kernel output vs baseline for the smallest batch size). This takes <30 seconds and catches obvious regressions from the fix.

3. **Fix-attempt counter check**: If this is the 2nd+ fix attempt for the same issue, the champion MUST delegate assessment to a fresh-context agent (Tier 2+) before proceeding. No exceptions — repeated failures indicate the champion's context is too degraded to reason about this issue effectively.

4. **Commit the fix**: Only after steps 1-3 pass.

5. **Message the validator**: Include the root cause reasoning in the re-validation request so the validator has context for what changed and why.

### Part 3: Transcript Monitor — Degradation Detection (ammo-transcript-monitor)

Add a new section to the transcript monitor's "What to Watch For" covering champion quality degradation. These patterns indicate the champion is losing rigor due to context pressure:

#### Degradation Signals

| Signal | Detection Pattern | Severity | Recommended Message |
|--------|------------------|----------|-------------------|
| **Thrashing** | 8+ `Edit` calls to the same file within a 20-tool-call window, especially if edits repeatedly add/remove/re-add similar code. | WARNING | "You've edited {file} {N} times in quick succession. This pattern suggests you're fixing symptoms rather than root cause. Consider delegating the investigation to a fresh-context ammo-delegate." |
| **Blind fix-and-send** | Champion makes a code change → immediately calls `SendMessage` to validator with "ready for re-validation" and no verification step in between (no `Bash` running pytest/python, no reasoning about correctness). | WARNING | "You sent a re-validation request without running your own smoke test first. Your agent definition requires self-validation before re-requesting — run your correctness check." |
| **Shrinking reasoning** | Champion's response to validation results is <100 words when earlier responses (to similar validation results) were >300 words. Measured by comparing the champion's post-validation-message reasoning length across the session. | WARNING | "Your analysis of the latest validation results was significantly shorter than earlier analyses. This can indicate context pressure reducing reasoning depth. Consider delegating the assessment to a fresh-context ammo-delegate." |
| **Surface symptom fixing** | Champion reads an error message, then immediately edits the exact line mentioned in the error without investigating the broader context. Evidence: traceback → single `Edit` call → no `Read` of surrounding code or related files. | WARNING | "You addressed the error at {file}:{line} without investigating why that error occurred. The surface fix may not address the root cause — check the call site and data flow." |

#### Implementation Notes for Monitor

- These signals are checked on every poll with substantive activity, alongside the existing methodology/procedural checks.
- Degradation signals use WARNING severity (not CRITICAL) because they indicate risk, not proven invalidity. The champion may have a valid reason for brief responses or quick fixes.
- The auto-escalation rule in Part 1 (fix-attempt counter) is the structural enforcement; the monitor's degradation detection is an additional observational layer.
- When a degradation signal fires, the recommended message includes specific guidance to delegate to a fresh-context agent — this is the remedy, not just the diagnosis.

## What This Does NOT Change

- The validator's behavior (champion-first approach — validator improvements deferred)
- The gate structure (Gates 5.1/5.2/5.3 remain identical)
- The independence principle (validator still writes its own tests)
- The three-layer verification model (this strengthens Layer 2, doesn't replace Layers 1 or 3)
- The transcript monitor's existing methodology/procedural checks (degradation detection is additive)

## Integration Points

| Component | Change |
|-----------|--------|
| `ammo-impl-champion.md` | New section: "Handling Incoming Messages (Tiered Assessment)" after "Handling Validation Failures". New section: "Self-Validation Gate" integrated into the re-validation request flow. |
| `ammo-transcript-monitor.md` | New subsection under "What to Watch For": "Champion Quality Degradation". |
| `impl-track-rules.md` | No change — the existing "Three Layers of Verification" already describes Layer 2 behavior. The champion changes enforce what's already specified. |

## Risk Assessment

- **Token cost**: Tier 2/3 delegations add Sonnet/Opus token cost per assessment. Expected: 1-3 delegations per track (most issues are Tier 1). Net positive because it prevents wasted fix-revalidate cycles.
- **Latency**: Each delegation adds ~30-90 seconds. Acceptable because the alternative (3+ failed fix attempts) wastes 10-30 minutes each.
- **False positives from monitor**: The degradation signals (especially "shrinking reasoning") may flag legitimate brevity. WARNING severity ensures the champion considers it but isn't forced to act.
- **Over-delegation**: A champion that delegates everything to Tier 2+ wastes tokens on trivial assessments. The tier definitions and examples guide appropriate classification. The auto-escalation rule only kicks in on repeated failures.
