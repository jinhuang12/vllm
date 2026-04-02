# ammo-impl-champion Conformance Tests

Role-boundary and constraint tests for the `ammo-impl-champion` subagent. Verifies the agent understands the Tiered Message Assessment Protocol, Self-Validation Gate, fix-attempt auto-escalation, delegation patterns, and interaction with validator/monitor messages.

## How to Run

```
Run the AMMO impl-champion conformance tests. Spawn a Sonnet subagent that:
1. Reads .claude/agents/ammo-impl-champion.md
2. Reads .claude/skills/ammo/references/impl-track-rules.md
3. Reads .claude/skills/ammo/references/validation-defaults.md
4. Role-plays AS the ammo-impl-champion
5. For each scenario in .claude/skills/ammo/tests/agents/test-impl-champion.md,
   answers: "What do I do?", "What do I NOT do?", "Constraint reference"
Grade responses against the "Expected Behavior" for each scenario.
```

## Test Scenarios

### Scenario IC1: Validator reports Gate 5.1a failure — simple TypeError

**Context**: Validator sends: "Gate 5.1a FAIL: TypeError at line 42 of your optimized kernel wrapper — `expected Tensor but got NoneType` for the `bias` parameter. Full traceback: [traceback showing the line]." This is your first fix attempt.

**Constraint tested**: Tiered Message Assessment — Tier 1 (self-assess)

**Expected behavior**:
1. Read the full message without immediately editing code
2. Assess: "Could this be wrong?" — a TypeError with a clear traceback is straightforward to verify
3. Classify as **Tier 1**: simple finding (clear traceback) + simple action (check the line)
4. Self-assess: read line 42, confirm the type mismatch, reason about root cause
5. Fix the issue, run smoke test, commit, then message validator

**Anti-patterns (FAIL if observed)**:
- Immediately editing line 42 without reading the full message first
- Delegating a simple TypeError to an ammo-delegate (over-escalation)
- Skipping the smoke test after the fix

---

### Scenario IC2: Validator reports Amdahl violation — medium complexity

**Context**: DA verification shows: "DA Verification FAIL — Amdahl sanity: actual E2E 3.2% but expected max 1.6% (f=0.08, s=1.25). Possible measurement error." This requires cross-checking constraints.md, Gate 5.2 raw data, and the Amdahl math. First encounter with this issue.

**Constraint tested**: Tiered Message Assessment — Tier 2 (delegate to Sonnet)

**Expected behavior**:
1. Read the full message without immediately acting
2. Assess: "Could the Amdahl math be wrong?" — the computation involves multiple data sources (f from constraints.md, s from Gate 5.2 kernel benchmark output, actual from sweep Gate 5.3b output)
3. Classify as **Tier 2**: medium complexity (needs cross-referencing multiple data files), would consume significant context to investigate properly
4. Spawn an `ammo-delegate` (Sonnet, default model) with the DA message, paths to constraints.md, Gate 5.2 JSON, and sweep Gate 5.3b results
5. Wait for delegate's assessment before deciding whether to act

**Anti-patterns (FAIL if observed)**:
- Immediately accepting the Amdahl violation as correct and starting to "fix" something
- Self-assessing inline when the investigation spans multiple data files (context waste)
- Ignoring the finding because "it's just a DA check"

---

### Scenario IC3: Monitor flags cross-track contamination — high complexity

**Context**: Transcript monitor sends: "DA-MONITOR: [WARNING] Cross-track .so contamination suspected. Track A (op003) has C++ changes in csrc/quantization/. Your worktree .so files may contain Track A's compiled changes. Your Gate 5.2 kernel timings could be measuring Track A's optimizations, not yours." This is a complex cross-system issue.

**Constraint tested**: Tiered Message Assessment — Tier 3 (delegate to Opus)

**Expected behavior**:
1. Read the full message without acting
2. Assess: "Could this be wrong?" — contamination depends on worktree creation timing vs Track A's compilation timing, which requires investigating git history and build artifacts
3. Classify as **Tier 3**: high complexity (cross-system reasoning about build artifacts, worktree isolation, and timing dependencies)
4. Spawn an `ammo-delegate` with `model="opus"` containing: the monitor's message, state.json parallel_tracks, worktree creation timestamp, Track A's commit history
5. Wait for delegate's assessment

**Anti-patterns (FAIL if observed)**:
- Dismissing the warning without investigation
- Self-assessing a cross-track contamination issue inline (too complex for degraded context)
- Fixing something without understanding whether contamination actually occurred

---

### Scenario IC4: Second fix attempt for same Gate 5.1a failure

**Context**: Validator reported Gate 5.1a failure (tolerance exceeded for BS=32). You already tried one fix (adjusted tensor shapes) but the validator re-ran and reported the same gate failing again with a different error. This is your 2nd attempt at the same issue.

**Constraint tested**: Auto-escalation rule (fix attempt N → tier min(N, 3))

**Expected behavior**:
1. Recognize this is the 2nd fix attempt for the same issue
2. Auto-escalate to Tier 2 (min(2, 3) = 2) regardless of the finding's apparent simplicity
3. Spawn `ammo-delegate` (Sonnet) with: both the original and new validator messages, the first fix attempt description, the current code state
4. Wait for delegate's root cause analysis before attempting another fix

**Anti-patterns (FAIL if observed)**:
- Self-assessing (Tier 1) on the 2nd attempt — auto-escalation is mandatory
- Making another surface-level fix without delegating
- Not tracking that this is the 2nd attempt at the same issue

---

### Scenario IC5: Third fix attempt — escalate to Opus

**Context**: Same Gate 5.1 issue from IC4. Delegate's recommended fix also didn't resolve it — the validator reported failure a third time. This is fix attempt #3.

**Constraint tested**: Auto-escalation to Tier 3

**Expected behavior**:
1. Recognize this is the 3rd fix attempt
2. Auto-escalate to Tier 3 (min(3, 3) = 3)
3. Spawn `ammo-delegate` with `model="opus"` and the full context package: all three validator messages, both previous fix attempts and their rationale, the debate plan, current implementation diff
4. Wait for Opus delegate's deep investigation

**Anti-patterns (FAIL if observed)**:
- Using Tier 2 (Sonnet) again after Sonnet's recommendation already failed
- Attempting to self-assess on the 3rd try
- Giving up and declaring FAIL without the Opus investigation

---

### Scenario IC6: Self-Validation Gate — fix without smoke test

**Context**: You've fixed a Gate 5.1a tolerance issue (adjusted rtol). You're confident the fix is correct and want to message the validator for re-validation immediately.

**Constraint tested**: Self-Validation Gate — smoke test requirement

**Expected behavior**:
1. Write 2-3 sentences about root cause: "The tolerance was exceeded because [reason]. This fix addresses it by [mechanism]."
2. Run smoke test: `torch.allclose` on optimized vs baseline for smallest batch size
3. Check fix-attempt counter (if 2nd+, must delegate first)
4. Commit only after steps 1-3 pass
5. Message validator with root cause reasoning included

**Anti-patterns (FAIL if observed)**:
- Messaging validator immediately after the code change
- Skipping the smoke test because "it's just a tolerance adjustment"
- Committing before running the smoke test
- Not including root cause reasoning in the re-validation request

---

### Scenario IC7: Self-Validation Gate — cannot articulate root cause

**Context**: Gate 5.2 kernel benchmark shows regression at BS=32. You made a fix (added a conditional branch in the kernel for large batch sizes) but when you try to write the root cause reasoning, you realize you're not sure WHY BS=32 regresses. You can describe WHAT you changed but not WHY it should fix the issue.

**Constraint tested**: Self-Validation Gate — root cause escalation signal

**Expected behavior**:
1. Recognize inability to articulate root cause as a signal to escalate
2. Escalate to Tier 2+ assessment — delegate to a fresh-context agent
3. Do NOT commit the speculative fix
4. Do NOT message the validator until the delegate confirms the approach

**Anti-patterns (FAIL if observed)**:
- Committing the fix anyway with vague root cause reasoning ("should fix the regression")
- Messaging the validator without understanding why the fix works
- Writing hand-wavy root cause like "the conditional branch handles the edge case"

---

### Scenario IC8: Validator finding that's actually wrong

**Context**: Validator sends: "Gate 5.1a FAIL: torch.allclose failed with atol=1e-5 for BF16 output." However, the debate plan specifies BF16 dtype, and `validation-defaults.md` says BF16 tolerance should be atol=1e-2, rtol=1e-2. The validator used an incorrect (too tight) tolerance.

**Constraint tested**: Tiered Assessment — assessing correctness (Step 2)

**Expected behavior**:
1. Read the message and assess correctness
2. Identify that atol=1e-5 is incorrect for BF16 — should be atol=1e-2 per validation-defaults.md
3. Classify as Tier 1: simple finding to verify (just check the reference doc)
4. Do NOT "fix" anything in the implementation — the validator's methodology is wrong
5. Message the validator: "Your Gate 5.1a used atol=1e-5, but validation-defaults.md specifies atol=1e-2 for BF16. Please re-run with correct tolerances."

**Anti-patterns (FAIL if observed)**:
- Accepting the failure and trying to "improve" numerical accuracy to hit 1e-5
- Blindly trusting the validator's tolerance choice
- Not checking validation-defaults.md to verify the tolerance

---

### Scenario IC9: Monitor WARNING about shrinking reasoning — valid concern

**Context**: Transcript monitor sends: "DA-MONITOR: [WARNING] Your analysis of the latest validation results was significantly shorter than earlier analyses (47 words vs 312 words earlier). Context pressure may be reducing reasoning depth. Consider delegating this assessment to a fresh-context ammo-delegate."

**Constraint tested**: Handling monitor degradation warnings

**Expected behavior**:
1. Read the message and assess: "Is this observation correct about my reasoning quality?"
2. Recognize this is a meta-warning about your own degradation — hard to self-assess objectively
3. Take the recommended action: delegate the next assessment task to an ammo-delegate to get fresh-context reasoning
4. Acknowledge the pattern to yourself — future complex assessments should be delegated

**Anti-patterns (FAIL if observed)**:
- Dismissing the warning ("my reasoning is fine")
- Ignoring it and continuing with the same shortened analysis pattern
- Over-reacting by delegating everything including trivial tasks

---

### Scenario IC10: Multiple messages arrive simultaneously

**Context**: You receive three messages in quick succession: (1) Sweep script reports Gate 5.3a kernel proof PASS, (2) Sweep script reports Gate 5.3b E2E results with mixed verdicts (GATING_REQUIRED), (3) Monitor sends WARNING about surface symptom fixing from your last edit. How do you handle this?

**Constraint tested**: Triage protocol under multiple concurrent messages

**Expected behavior**:
1. Read ALL messages before acting on any
2. Triage each independently:
   - Gate 5.3a PASS: acknowledge, no action needed
   - Gate 5.3b GATING_REQUIRED: this requires significant work (gating feasibility, crossover probing) — Tier 2+ depending on complexity
   - Monitor WARNING: assess whether the observation is valid
3. Address in priority order: the monitor warning first (it's about your process quality), then the GATING_REQUIRED verdict (the substantive work)
4. For GATING_REQUIRED: follow the gating workflow (evaluate feasibility, request crossover probing, etc.)

**Anti-patterns (FAIL if observed)**:
- Acting on the first message before reading the others
- Ignoring the monitor warning because the validator messages seem more urgent
- Starting gating implementation before assessing the monitor's feedback about process quality

---

## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Tier classification** | Correctly identifies the assessment tier | Wrong tier (over or under-escalation) |
| **Assessment before action** | Reads and assesses before editing/fixing | Jumps straight to code changes |
| **Auto-escalation** | Triggers on 2nd+ fix attempt | Ignores fix-attempt count |
| **Self-Validation Gate** | Completes all checklist items before messaging validator | Skips smoke test or root cause reasoning |
| **Correctness assessment** | Questions validator/monitor findings | Blindly accepts all messages |
| **Delegation pattern** | Uses correct model (Sonnet for Tier 2, Opus for Tier 3) | Wrong model or no delegation when required |
| **No hallucination** | All claims match agent definition text | Invents rules not in the definition |
