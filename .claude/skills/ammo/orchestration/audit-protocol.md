# Audit Protocol (T_AUDIT): Adversarial Campaign Sanity Gate

After each major stage gate the lead orchestrator MUST spawn `ammo-auditor` — an adversarial reviewer that runs a four-phase verification (input inventory, cold stage-completion reconstruction, institutional checklist, reconciliation) and emits a structured verdict. **BLOCKING** findings halt the campaign until resolved via a review loop.

This file is the **orchestrator's** view of the audit gate. The auditor's own instructions (evidence mandate, verdict format, severity classification) live in `.claude/agents/ammo-auditor.md`. Stage-specific checklists are delivered to the auditor via a PostToolUse hook after Phase 1 completes — the orchestrator does not pass the checklist path.

## Why an Auditor

Long-horizon AMMO campaigns compound local errors into globally wrong conclusions. Existing gates verify one artifact at a time and cannot catch:

- **Baseline drift**: Round N "re-baseline" silently regresses vs Round 1; per-round cumulative math papers over it.
- **Baseline corruption**: A short-output profiling run overwrites the real baseline file; downstream comparisons become nonsense.
- **Metric inflation**: `cumulative_speedup_vs_round1` computed as a product of per-round deltas against drifting baselines can diverge from the direct Round 1 / current ratio by 3x or more.
- **Workflow violations**: An actor decides before gathering evidence, then backfills artifacts — the result looks correct but the process was invalid.

Transcript monitors sometimes notice these but have no blocking power. The auditor does.

## Agent Design

### `ammo-auditor` (spawned per gate)

| Property | Value |
|----------|-------|
| Model | **opus** (adversarial reasoning, same tier as `devil-advocate`) |
| Spawn lifecycle | **Fresh instance at each gate point**. No accumulated state. Every audit is independent. |
| Phases | 0: Input inventory, 1: Cold reconstruction, 2: Checklist verification, 3: Reconciliation |
| Access | Full campaign state (`state.json`, `target.json`) + all artifacts under `kernel_opt_artifacts/` + session transcripts (`.jsonl` via `transcript_filter.py`) |
| Output | `kernel_opt_artifacts/rounds/{M}/audits/stage_{N}.md` (verdict file) + `audit.stage_{N}.passed_at` + `audit.stage_{N}.verdict_file` in `state.json` on PASS |
| Authority | Reports BLOCKING / HIGH / LOW findings with blocker categories. **Cannot** modify artifacts directly. Recommends fixes; orchestrator delegates execution. |
| Framing | **Adversarial** — reconstructs what happened from primary evidence, then evaluates whether it constitutes valid stage completion. Default stance: stage success is an unverified hypothesis. |

Spawn as a sub-agent (returns once, then gone). Use `subagent_type` + `run_in_background=True`; the orchestrator continues other work while the auditor reads artifacts and is notified on completion.

```python
Agent(
    subagent_type="ammo-auditor",
    run_in_background=True,
    prompt=f"""
    task: audit_gate
    stage: {stage_name}        # stage_1 | stage_2 | stage_45 | stage_67
    round: {current_round}
    """,
)
```

The auditor discovers all paths (artifact_dir, state.json, verdict file, transcripts) via convention from the worktree root. Only `stage` and `round` are passed. The checklist (Phase 2) is delivered via a PostToolUse hook after Phase 1 is written to disk.

### `ammo-delegate` (spawned by the auditor)

Reuses the existing `ammo-delegate` sub-agent definition. The auditor spawns N delegates — **all in the same message, parallel dispatch** — each answering one bounded, evidence-oriented question with file:line:quote citations. Delegates do not interpret, recommend, or modify files. They read specific files and report exact content.

## Trigger Points (T_AUDIT)

The lead MUST spawn `ammo-auditor` at these transition points. The auditor always spawns — there are no auto-pass conditions.

| Trigger | When | Stage key in `state.json` |
|---------|------|---------------------------|
| **T_AUDIT_S1**  | After Stage 1 baseline capture completes | `audit.stage_1.passed_at` |
| **T_AUDIT_S2**  | After Stage 2 bottleneck mining completes (schema v4.1+ only) | `audit.stage_2.passed_at` |
| **T_AUDIT_S45** | After all Stage 4-5 tracks reach terminal status (`PASS` / `GATED_PASS` / `FAIL`) | `audit.stage_45.passed_at` |
| **T_AUDIT_S67** | After SHIP: fires after git merge + env promotion + golden-refs capture. After EXHAUSTED: fires after integration status set. | `audit.stage_67.passed_at` |

Stage 3 (debate) is excluded — the adversarial debate structure (Phase B cross-critique + Phase C rebuttal + open-items declaration) provides inherent quality control without requiring a separate auditor pass.

The auditor always spawns regardless of track outcomes or round status. Phase 2 uses precondition gating for invariant rows whose artifacts don't exist (rather than skipping the entire audit).

### Backward Compatibility (Legacy stage_6 / stage_7)

Pre-consolidation campaigns have `audit.stage_6` and `audit.stage_7` instead of `audit.stage_67`. The hooks accept either:
- `stage_67.passed_at` (new, preferred)
- `stage_6.passed_at` (legacy fallback for `7_campaign_eval` transition gate)
- `stage_7.passed_at` (legacy fallback for new-round start gate)

New campaigns MUST use `stage_67`. The deprecated keys will be removed when all active campaigns have completed.

## Orchestrator Integration

### On Verdict Return

```
IF verdict.overall == "PASS":
    - Record audit.stage_{N}.passed_at = <now_iso()> in state.json rounds[current_round-1].audit
    - Record audit.stage_{N}.verdict_file = "rounds/{M}/audits/stage_{N}.md"
    - Continue to next stage (no delay)

IF verdict.overall == "BLOCKED":
    FOR each BLOCKING finding:
        - Read the finding's blocker category
        - Delegate fix to the appropriate agent (see Delegation Matrix below)
        - After fix applied: re-spawn ammo-auditor with the SAME stage context
    - Loop until PASS OR 3 cycles exhaust -> campaign halts with `auditor_escalation` field in state.json

IF verdict.overall == "NEEDS_INVESTIGATION":
    - Spawn investigator for each HIGH finding
    - Downgrade to LOW (continue) or upgrade to BLOCKING (enter fix loop)
```

### Loop Termination (Max 3 Audit-Fix Cycles)

- Cap at **3** audit-fix cycles per gate point.
- If 3 cycles exhaust without PASS, the campaign halts permanently. Write:
  - `campaign.status = "paused"` (human intervention required — NOT `campaign_complete`/`campaign_exhausted`, which are mechanical threshold outcomes, not audit-driven)
  - A top-level `campaign.auditor_escalation = {stage, round, reason, verdict_files: [...]}` field in state.json for the eval pipeline / human reviewer
- This prevents infinite loops where fixes introduce new issues.

### Delegation Matrix (by Blocker Category)

Every BLOCKING or HIGH finding in the verdict includes a `category` tag. Route fixes by category:

| Category | Fix agent | Mechanism |
|----------|-----------|-----------|
| `artifact` | `ammo-impl-champion` or `ammo-researcher` | Re-generate missing/malformed artifact |
| `workflow` | Re-run stage (orchestrator) | Full stage re-execution — process was invalid |
| `provenance` | Re-run producing step | Re-execute the specific step that produced bad evidence |
| `environment` | `ammo-researcher` | Re-profile with correct environment |
| `comparator` | `ammo-researcher` or lead | Fix baseline reference, re-run if needed |
| `decision` | Lead | Re-evaluate transition decision with correct evidence |
| `downstream` | Lead or `ammo-impl-champion` | Fix output so next stage can consume it |

Unknown categories default to lead for triage.

## Hook Interaction

Two hooks enforce this gate mechanically (belt-and-suspenders):

- **`ammo-next-step-reminder.sh`** (SOFT reminder). After any state-mutating tool call, the hook reads `rounds[current_round-1].audit.stage_67.passed_at` (with fallback to legacy `stage_6.passed_at`). If the stage's audit has not passed, the hook injects "AUDIT REQUIRED: ..." into the next-step reminder **in place of** the normal "next step" text — so the orchestrator cannot mistake it for optional. Advisory (exit 0 always).

- **`ammo-state-validate.sh`** (HARD gate). When the orchestrator writes a stage transition to `state.json` that skips an audit, the hook blocks the write with `{"decision": "block", "reason": "Audit gate (4-phase audit): ..."}`. Specifically:
  - `current_stage: 2_bottleneck_mining` requires `rounds[current_round-1].audit.stage_1.passed_at` — **except for round N>1** (post-SHIP re-mine), where this same-round requirement is exempted (see post-SHIP note below)
  - `current_stage: 6_integration` requires `rounds[current_round-1].audit.stage_45.passed_at`
  - `current_stage: 7_campaign_eval*` requires `rounds[current_round-1].audit.stage_67.passed_at` (or legacy `stage_6.passed_at`)
  - New round start (`current_round > 1`, stage in `{1_baseline, 2_bottleneck_mining, 3_debate}`) requires `rounds[current_round-2].audit.stage_67.passed_at` (or legacy `stage_7.passed_at`) on the previous round

**Post-SHIP re-mine exemption (round N>1 `2_bottleneck_mining`)**: After a SHIP, the next round re-mines on the shifted baseline (SKILL.md §Campaign Stop Condition, "After SHIP"). Stage 1 (fresh baseline capture) is eliminated post-SHIP — the Stage 6 combined sweep (≥2 passers) or single-track short-circuit *is* the new baseline (T16 eliminated) — so a round N>1 re-mine has no same-round `stage_1` audit and never will. The hard gate therefore **does not require `stage_1.passed_at`** when entering `2_bottleneck_mining` in a round after the first — **but only when the immediately-previous round carries an `audit` key**. That condition is the safety hinge: it is exactly when the new-round-start rule below fires and enforces the predecessor's `stage_67`, so the predecessor's complete audit chain (`stage_1 → stage_2 → stage_45 → stage_67`) is still required transitively. If the previous round omits its `audit` key entirely (a legacy or mixed-state round — `audit` is optional in the schema), the new-round-start rule is silent too; in that case the exemption does **not** fire and the same-round `stage_1` requirement is kept, so the write **fails closed** rather than letting a round begin mining with no audit chain anywhere. Without this exemption the very first post-SHIP round deadlocks: `T_AUDIT_S2` must stamp `audit.stage_2` while `current_stage` is still `2_bottleneck_mining` (advancing to `3_debate` requires `stage_2` already set), and that write would otherwise be blocked for a missing — and impossible — `stage_1`.

**Legacy-gate semantics**: Both hooks only enforce the audit gate when the `audit` key is PRESENT in the current round object (even if `{}`). If the key is entirely absent (legacy campaign pre-dating this feature), the gate is SKIPPED. The orchestrator's bootstrap (`new_target.py`) writes `"audit": {}` into every new round so the gate fires from round 1 onward.

## Audit Artifact Layout

Verdicts are scoped under each round's `audits/` directory:

```
kernel_opt_artifacts/
└── rounds/
    ├── 1/
    │   └── audits/
    │       ├── stage_1.md            # T_AUDIT_S1 verdict for round 1
    │       ├── stage_1_cycle_2.md    # Re-audit after fix (BLOCKED -> fix -> re-audit)
    │       ├── stage_45.md
    │       └── stage_67.md           # Consolidated S6+S7 audit
    └── 2/
        └── audits/
            ├── stage_1.md
            └── ...
```

Cycle numbers are appended only on re-audits; the first audit at each gate is unadorned. Legacy campaigns may still carry top-level `audits/stage_6_round_1.md` / `stage_7_round_1.md` paths — the hooks accept the legacy field names but new audits MUST write to the round-scoped path above.

## Cost Envelope

| Scenario | Token cost | Notes |
|----------|-----------|-------|
| Happy path (PASS, no Phase 1 findings) | ~15-25K | Transcript reading adds overhead vs old checklist-only |
| Phase 1 finds issues, Phase 2 clean | ~30-40K | Transcript delegates + checklist delegates |
| BLOCKING found, full investigation | ~40-60K | All phases + deep delegate reads |
| Re-audit loop (fix + re-audit) per cycle | ~50-80K | Full 4-phase re-run each cycle |

Costs scale with transcript length. Long sessions (10K+ transcript lines) will use the upper range. Campaign total is typically 800K-3M tokens over 8 rounds, so the auditor adds roughly 10-20% overhead. Compared to the cost of five wasted rounds on inflated baselines or invalid processes, this is negligible.

## References

- `references/audit-invariants.md` — stage-specific checklists (delivered to auditor via hook, not directly referenced by orchestrator)
- `.claude/agents/ammo-auditor.md` — auditor's own agent definition (four-phase procedure, evidence mandate, verdict format, severity classification)
- `.claude/agents/ammo-delegate.md` — delegate sub-agent used for parallel evidence gathering
- `SKILL.md § Audit Gates (T_AUDIT)` — orchestrator-facing trigger table and summary
- `hooks/ammo-next-step-reminder.sh` — soft reminder hook
- `hooks/ammo-state-validate.sh` — hard gate hook
