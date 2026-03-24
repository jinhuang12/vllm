# Orchestrator (Lead) Conformance Tests

Workflow conformance tests for the AMMO lead orchestrator. Verifies the agent correctly follows the campaign workflow: overlapped debate pipeline, resume after interruption, campaign evaluation, integration decisions, role boundaries, and non-negotiable violation detection.

## How to Run

```
Run the AMMO orchestrator conformance tests. Spawn Sonnet subagents that:
1. Read these files first:
   - .claude/skills/ammo/SKILL.md
   - .claude/skills/ammo/orchestration/parallel-tracks.md
   - .claude/skills/ammo/orchestration/integration-logic.md
   - .claude/skills/ammo/orchestration/debate-protocol.md
2. Role-play AS the lead orchestrator (scaffolds, delegates, gates — never implements)
3. For each scenario, receive a state.json snapshot and context, then answer:
   - "Next actions (in order):"
   - "Must NOT do:"
   - "Skill reference:"

Run in 4 parallel batches:
- Batch A: Scenarios 1a-1e (Overlapped Debate)
- Batch B: Scenarios 2a-2c (Resume After Interruption)
- Batch C: Scenarios 3a-3c, 4a-4c (Campaign Eval + Integration)
- Batch D: Scenarios 5a-5b, 6a-6d (Role Boundaries + Violation Detection)

Grade each response against the "Expected Behavior" column.
```

## Test Scenarios

### Category 1: Overlapped Debate Pipeline

**Scenario 1a: Stage 4-5, Round 2, overlapped debate NOT started**

State:
```json
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "team_name": "ammo-round-2-llama70b-h100",
    "selected_winners": ["op001", "op002"],
    "next_round_overlap": { "active": false, "phase": null, "selected_winners": [], "profiling_basis": null, "f_values_at_proposal": {} }
  },
  "parallel_tracks": {
    "op001": { "status": "in_progress" },
    "op002": { "status": "in_progress" }
  }
}
```
Context: Two impl tracks are running (impl-champion + impl-validator pairs) in the round team. Existing bottleneck_analysis.md from round 1.

Expected behavior: Launch overlapped debate IMMEDIATELY. Spawn debate champions into the existing round team. Set `debate.next_round_overlap.active: true`. Monitor both impl tracks and debate. Do NOT stop until all complete.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Immediately launch the overlapped round 3 debate by spawning 2-4 ammo-champion agents (+ delegates if enabled) into the existing round team (`ammo-round-2-llama70b-h100`). Use the existing `bottleneck_analysis.md` from round 1. Set `debate.next_round_overlap.active: true` and `debate.next_round_overlap.phase: "phase_0"` in `state.json`.
2. While the overlapped debate runs, actively monitor the two impl tracks (op001 and op002).
3. Interleave debate moderation with impl monitoring: broadcast debate phase starts, then check for impl track completions, then wait for debate phase completions.
4. As each impl-champion completes (DA Stop hook passed), run its compilation gate (T9) and update `state.json` `parallel_tracks.{op_id}` (T10).
5. When debate finishes: score winners, shut down debate champions via `shutdown_request`. Record winners in `debate.next_round_overlap.selected_winners`. Set `debate.next_round_overlap.phase: "selection_complete"`.
6. Continue monitoring until all impl tracks have returned results AND the overlapped debate has completed.

**Must NOT do:**
- Skip or defer the overlapped debate. It is MANDATORY for round 2+ and must be launched immediately after impl agents are spawned.
- Create a separate team for the debate — debate champions join the existing round team.
- Implement anything yourself.
- Send overlapped debate winners to implementation now — they stay in `debate.next_round_overlap.selected_winners` until the next round (subject to lazy invalidation after re-profiling).
- Go idle/stop while either impl tracks or the debate are still running.
- Re-profile to generate new bottleneck data — the overlapped debate uses the EXISTING `bottleneck_analysis.md`.

**Skill reference:**
- `SKILL.md` § Stages 4-5, step 2: "Launch overlapped debate (round 2+ only): If `campaign.current_round >= 2`, spawn 2-4 ammo-champion agents (+ delegates if enabled) into the same round team."
- `SKILL.md` § Overlapped Debate: "When to launch: Immediately after spawning all implementation agents for round N."
- `SKILL.md` § Stages 4-5, step 3: "Do NOT stop or go idle until all implementation agents have returned results AND the overlapped debate (if launched) has completed."
</details>

---

**Scenario 1b: Stage 4-5, Round 1, no overlapped debate**

State:
```json
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 1 },
  "debate": {
    "team_name": "ammo-round-1-llama70b-h100",
    "selected_winners": ["op001"],
    "next_round_overlap": { "active": false, "phase": null, "selected_winners": [], "profiling_basis": null, "f_values_at_proposal": {} }
  },
  "parallel_tracks": { "op001": { "status": "in_progress" } }
}
```
Context: First round of the campaign. One impl track is running (impl-champion + impl-validator pair). No prior bottleneck data exists beyond the current round's.

Expected behavior: Do NOT launch overlapped debate (round 1). Monitor impl track only. Gate when it returns.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Monitor the running impl track (op001) — watch for the impl-champion's return.
2. When op001's impl-champion returns (DA Stop hook passed), run the compilation gate (T9) in its worktree.
3. Read `{artifact_dir}/tracks/op001/validation_results.md` and update `state.json` `parallel_tracks.op001` with structured status/metrics (T10).
4. Proceed to T11 (all tracks have results), then TeamDelete the round team (`ammo-round-1-llama70b-h100`), and advance to Stage 6 integration.

**Must NOT do:**
- Launch an overlapped debate. Round 1 has no prior bottleneck data to debate from — overlapped debate is explicitly skipped for round 1.
- Go idle while op001 is still running — continue actively monitoring.
- Implement anything yourself.

**Skill reference:**
- `SKILL.md` § Stages 4-5, step 2: "For round 1, skip this step."
- `SKILL.md` § Overlapped Debate: "If round N is round 1: Do NOT launch overlapped debate. Round 1 has no prior profiling data for the next round's debate to use."
</details>

---

**Scenario 1c: Stage 4-5, Round 3, overlapped debate already running**

State:
```json
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 3 },
  "debate": {
    "team_name": "ammo-round-3-llama70b-h100",
    "selected_winners": ["op003", "op004"],
    "next_round_overlap": { "active": true, "phase": "debating", "selected_winners": [], "profiling_basis": "bottleneck_analysis.md", "f_values_at_proposal": {} }
  },
  "parallel_tracks": {
    "op003": { "status": "in_progress" },
    "op004": { "status": "in_progress" }
  }
}
```
Context: Impl tracks for round 3 running (impl-champion + impl-validator pairs). Overlapped debate for round 4 already active in the round team, currently in debate rounds phase.

Expected behavior: Continue moderating debate. Also gate any completed impl tracks. Do NOT create a second debate. Wait for all to complete before Stage 6.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Continue actively monitoring both running impl tracks (op003 and op004).
2. As each impl-champion returns (DA Stop hook passed), run its compilation gate (T9) in its respective worktree.
3. Update `state.json` `parallel_tracks.{op_id}` after each T9 passes (T10).
4. Concurrently moderate the already-running overlapped debate — interleave debate phase broadcasts with impl completion checks.
5. When the debate concludes, score winners, shut down debate champions via `shutdown_request`, and record winners in `debate.next_round_overlap.selected_winners`. Set `debate.next_round_overlap.phase: "selection_complete"`.
6. Do NOT advance to Stage 6 until all impl tracks have returned AND the overlapped debate has completed.

**Must NOT do:**
- Create a second debate or re-launch the overlapped debate — one is already running (`debate.next_round_overlap.active: true`).
- Send the overlapped debate's winners to implementation immediately — they stay in `debate.next_round_overlap.selected_winners` for the next round.
- Stop or go idle while either impl tracks or the debate are still in-flight.
- Terminate the running debate early even if impl tracks finish first.

**Skill reference:**
- `SKILL.md` § Overlapped Debate: "If all implementation tracks complete before debate finishes: Wait for debate to complete before proceeding to Stage 6."
- `SKILL.md` § Stages 4-5, step 3: "Do NOT stop or go idle until all implementation agents have returned results AND the overlapped debate (if launched) has completed."
</details>

---

**Scenario 1d: All impl tracks done, overlapped debate still running**

State:
```json
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "team_name": "ammo-round-2-llama70b-h100",
    "next_round_overlap": { "active": true, "phase": "debating", "selected_winners": [], "profiling_basis": "bottleneck_analysis.md", "f_values_at_proposal": {} }
  },
  "parallel_tracks": {
    "op001": { "status": "PASSED", "result": { "e2e_speedup": 1.12 } },
    "op002": { "status": "FAILED", "result": { "reason": "correctness" } }
  }
}
```
Context: Both impl tracks returned. op001 passed, op002 failed. Overlapped debate for round 3 still in debate rounds phase within the round team.

Expected behavior: Wait for overlapped debate to complete. Do NOT advance to Stage 6 yet. Gate impl results while moderating debate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Run the compilation gate (T9) for op001 (passed) in op001's worktree. (op002 failed — mark it `FAILED` in state.json; no gate needed.)
2. Update `state.json` `parallel_tracks.op001` and `parallel_tracks.op002` with their structured statuses (T10).
3. Wait for the overlapped debate to finish all debate rounds. Continue moderating it — broadcast phases, receive champion messages.
4. Once the debate concludes, score winners, shut down debate champions via `shutdown_request`, record winners in `debate.next_round_overlap.selected_winners`, and set `debate.next_round_overlap.phase: "selection_complete"`.
5. Only after all impl tracks are fully recorded AND the overlapped debate is complete: satisfy T11, TeamDelete the round team (`ammo-round-2-llama70b-h100`), and advance to Stage 6 integration with op001 as the sole passing candidate.

**Must NOT do:**
- Advance to Stage 6 integration before the overlapped debate finishes.
- Terminate or abandon the in-progress overlapped debate because the impl tracks are already done.
- Send overlapped debate winners to implementation during this stage — they stay in `debate.next_round_overlap.selected_winners` for the next round.
- Skip the T9 compilation gate for op001 because it reported PASSED.

**Skill reference:**
- `SKILL.md` § Overlapped Debate: "If all implementation tracks complete before debate finishes: Wait for debate to complete before proceeding to Stage 6. Do not terminate the debate."
- `SKILL.md` § Stages 4-5, step 4: "TeamDelete after all tracks complete: Once all implementation tracks have finished... AND the overlapped debate (if launched) has completed, call TeamDelete on the round team."
</details>

---

**Scenario 1e: Overlapped debate complete, impl tracks still running**

State:
```json
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "team_name": "ammo-round-2-llama70b-h100",
    "next_round_overlap": { "active": true, "phase": "selection_complete", "selected_winners": ["op003", "op004"], "profiling_basis": "bottleneck_analysis.md", "f_values_at_proposal": { "op003": 0.12, "op004": 0.08 } }
  },
  "parallel_tracks": {
    "op001": { "status": "in_progress" },
    "op002": { "status": "PASSED", "result": { "e2e_speedup": 1.08 } }
  }
}
```
Context: Overlapped debate for round 3 finished — winners op003 and op004 selected and parked. op002 passed and was gated. op001 still running in the round team.

Expected behavior: Continue monitoring impl tracks. Winners are parked in `debate.next_round_overlap.selected_winners`. Do NOT advance to Stage 6 until op001 completes.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Continue actively monitoring impl track op001 — wait for the impl-champion to return.
2. When op001's impl-champion returns (DA Stop hook passed), run the compilation gate (T9) in its worktree.
3. Update `state.json` `parallel_tracks.op001` with structured status/metrics (T10).
4. Once all impl tracks have returned: satisfy T11, TeamDelete the round team (`ammo-round-2-llama70b-h100`), and advance to Stage 6 integration.
5. The overlapped debate winners (`op003`, `op004`) remain parked in `debate.next_round_overlap.selected_winners` — they will be consumed during the next round's campaign evaluation (subject to lazy invalidation after re-profiling).

**Must NOT do:**
- Stop or go idle while op001 is still running.
- TeamDelete the round team before op001 completes — the round team persists until ALL tracks are done.
- Move the overlapped debate winners to implementation now — they are for the next round.
- Re-launch or restart the overlapped debate — it is already complete (`phase: "selection_complete"`).

**Skill reference:**
- `SKILL.md` § Overlapped Debate: "If debate finishes before all implementation tracks complete: Record winners. Continue monitoring implementation tracks."
- `SKILL.md` § Stages 4-5, step 4: "TeamDelete after all tracks complete: Once all implementation tracks have finished (passed or failed) and results are collected, AND the overlapped debate (if launched) has completed, call TeamDelete on the round team."
</details>

---

### Category 2: Resume After Interruption

**Scenario 2a: Resume into Stage 4-5 with overlapped debate**

State:
```json
{
  "stage": "4_5_parallel_tracks",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "team_name": "ammo-round-2-llama70b-h100",
    "selected_winners": ["op001", "op002"],
    "next_round_overlap": { "active": true, "phase": "phase_0", "selected_winners": [], "profiling_basis": "bottleneck_analysis.md", "f_values_at_proposal": {} }
  },
  "parallel_tracks": {
    "op001": { "status": "in_progress", "worktree_path": "/tmp/worktree-op001" },
    "op002": { "status": "in_progress", "worktree_path": "/tmp/worktree-op002" }
  }
}
```
Context: Resuming after compaction. Session was interrupted while impl tracks were running and overlapped debate was in Phase 0.

Expected behavior: Read SKILL.md + state.json. Check debate artifacts on disk. Restart debate from Phase 0 (spawn champions). Resume monitoring impl tracks.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Read `state.json` to confirm the full state (worktree paths, op IDs, artifact directory, overlap state).
2. Check `debate.next_round_overlap.phase` — it is `"phase_0"`, meaning the debate was launched but no progress was recorded.
3. Check debate artifacts in `debate/campaign_round_3/` to determine if any Phase 0 proposals exist on disk.
4. Re-spawn debate champions into the existing round team (`ammo-round-2-llama70b-h100`) and start from Phase 0 (debate is restartable — champions are stateless, artifacts on disk capture progress).
5. Check whether the two impl tracks are actually still running by inspecting their worktrees.
6. **If impl tracks have already completed**: run compilation gate (T9) and update state.json.
7. **If impl tracks are still in-flight**: resume monitoring, do NOT re-spawn them.
8. Interleave debate moderation with impl track monitoring concurrently.

**Must NOT do:**
- Re-spawn impl agents without first verifying they are not already complete or still running.
- Skip or delay restarting the overlapped debate — `next_round_overlap.active: true` with `phase: "phase_0"` means it needs to be resumed.
- Create a new round team — use the existing `ammo-round-2-llama70b-h100`.
- Go idle waiting for impl tracks without also restarting the overlapped debate.

**Skill reference:**
- SKILL.md § Resume Protocol, step 4b: "If Stages 4-5 active AND `debate.next_round_overlap.active` is `true`: Check `debate.next_round_overlap.phase` to determine debate progress."
- SKILL.md § Resume Protocol, step 4b: "If `phase` is null but `active` is true: Debate was launched but no progress. Re-spawn debate champions and start from Phase 0."
</details>

---

**Scenario 2b: Resume into Stage 3, debate team gone**

State:
```json
{
  "stage": "3_debate",
  "campaign": { "status": "active", "current_round": 1 },
  "debate": { "team_name": "ammo-round-1-llama70b-h100", "candidates": [], "rounds_completed": 0, "selected_winners": [] },
  "parallel_tracks": {}
}
```
Context: Resuming after interruption. Round team was created but no proposals exist. Team may be lost.

Expected behavior: TeamDelete the stale team. Re-create from scratch. Do NOT skip debate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Read `state.json`. Also check `{artifact_dir}/debate/` for any proposal files.
2. Attempt to contact the round team via SendMessage to determine if agents are still alive.
3. Confirm the team is lost: candidates empty, rounds_completed 0, no debate files.
4. Issue `TeamDelete` for the stale team name.
5. Re-run Stage 3 from scratch: create a new round team (`ammo-round-1-llama70b-h100`), spawn champions, restart Phase 0.
6. Update `state.json` with the new `debate.team_name`.
7. Moderate the debate through completion.

**Must NOT do:**
- Fabricate proposals from `bottleneck_analysis.md` directly as the lead.
- Skip the debate and proceed to Stage 4.
- Assume the old team is still alive.

**Skill reference:**
- SKILL.md § Resume Protocol, step 4: "If Stage 3 debate active: check debate artifacts in `debate/` or `debate/campaign_round_N/`."
- SKILL.md § Stage 3: "Debate is always mandatory."
</details>

---

**Scenario 2c: Resume into Stage 7, SHIP decision made but no re-profile**

State:
```json
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "active", "current_round": 1, "shipped_optimizations": ["op001"], "cumulative_e2e_speedup": 1.12, "rounds": [{"round_id": 1, "shipped": ["op001"]}] },
  "integration": { "status": "combined", "final_decision": { "action": "ship_combined", "total_e2e_speedup": 1.12 } },
  "debate": {
    "next_round_overlap": { "active": false, "phase": null, "selected_winners": [], "profiling_basis": null, "f_values_at_proposal": {} }
  }
}
```
Context: Resuming. A candidate shipped in round 1 but re-profiling hasn't happened yet. No overlapped debate was launched (round 1 does not have overlapped debate).

Expected behavior: Trigger re-profiling on patched codebase, then bottleneck mining, then diminishing returns check. Do NOT use stale data.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Read `state.json` to confirm full campaign state.
2. Confirm ship decision is recorded in `campaign.rounds` and `campaign.shipped_optimizations`.
3. Execute T16: trigger re-profiling — invoke `ammo-researcher` subagent for baseline capture on the patched codebase.
4. After re-profile: execute T17 — bottleneck mining on the new baseline (updated `bottleneck_analysis.md`).
5. Execute T18 (diminishing returns check):
   - If below threshold: set `campaign.status = "campaign_complete"`, spawn report subagent, done.
   - If above threshold: increment round, enter Stage 3 for round 2.

**Must NOT do:**
- Skip re-profiling — SKILL.md explicitly requires it after SHIP.
- Check diminishing returns against old `bottleneck_analysis.md`.
- Spawn the report subagent before confirming `campaign_complete` or `campaign_exhausted`.

**Skill reference:**
- SKILL.md § Campaign Loop: "After SHIP: Re-profile first (bottleneck landscape shifted), then check the NEW top bottleneck."
</details>

---

### Category 3: Campaign Evaluation

**Scenario 3a: SHIP, top bottleneck below threshold after re-profile**

State:
```json
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "active", "current_round": 2, "min_e2e_improvement_pct": 3, "shipped_optimizations": ["op001", "op003"], "cumulative_e2e_speedup": 1.25 }
}
```
Context: Re-profiling done. New top bottleneck = 2.1% of decode latency (below 3% threshold).

Expected behavior: Set `campaign.status = "campaign_complete"`. Spawn report subagent in background. Do NOT start new round.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Record round 2 results in `campaign.rounds`.
2. Update `campaign.shipped_optimizations` and `campaign.cumulative_e2e_speedup`.
3. Confirm 2.1% < 3% threshold.
4. Set `campaign.status = "campaign_complete"`.
5. Run gate T19.
6. Spawn report generation subagent in background (T20).
7. Declare campaign done. Do not block on report subagent.

**Must NOT do:**
- Proceed to a new debate round.
- Wait for the report subagent to finish.

**Skill reference:**
- SKILL.md § Diminishing Returns: "If below threshold... stop."
- Campaign State Transitions: `active → (threshold met after ship) → campaign_complete`
</details>

---

**Scenario 3b: EXHAUSTED, top bottleneck above threshold**

State:
```json
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "active", "current_round": 2, "min_e2e_improvement_pct": 3, "rounds": [{"round_id": 1, "shipped": ["op001"]}, {"round_id": 2, "shipped": []}] }
}
```
Context: Round 2 had no passing candidates. EXISTING profiling shows top bottleneck at 8.5%.

Expected behavior: No re-profile (nothing shipped). 8.5% > 3% → campaign continues. New debate from existing data. Do NOT set `campaign_exhausted`.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Record the failed round 2 in `campaign.rounds`.
2. Check diminishing returns against EXISTING profiling data: 8.5% > 3% → campaign continues.
3. Run gate T19.
4. Increment `campaign.current_round` to 3.
5. Start new debate from existing bottleneck data (skip re-profiling, skip Stage 2).

**Must NOT do:**
- Trigger re-profiling — nothing shipped.
- Set `campaign.status` to `campaign_exhausted` — threshold not met.
- Skip debate for the new round.

**Skill reference:**
- SKILL.md § Diminishing Returns: "After EXHAUSTED: Check threshold against EXISTING profiling data (no re-profile needed)."
</details>

---

**Scenario 3c: SHIP with overlapped debate winners needing invalidation**

State:
```json
{
  "stage": "7_campaign_eval",
  "campaign": { "status": "active", "current_round": 2 },
  "debate": {
    "next_round_overlap": {
      "active": true, "phase": "selection_complete",
      "selected_winners": ["op003", "op004"],
      "profiling_basis": "bottleneck_analysis.md",
      "f_values_at_proposal": { "op003": 0.12, "op004": 0.05 }
    }
  }
}
```
Context: After re-profiling: op003's target kernel (flash_attn_fwd) dropped from f=12% to f=0.8% (shipped optimization targeted it). op004's target kernel (rms_norm) shifted from f=5% to f=4.5%. Top bottleneck 7% (above threshold).

Expected behavior: Run lazy invalidation. Discard op003 (|0.008 - 0.12| / 0.12 = 0.93 > 0.3 threshold). Retain op004 (|0.045 - 0.05| / 0.05 = 0.1 < 0.3 threshold). Clear `debate.next_round_overlap` to initial state. Skip debate for next round — move op004 directly to implementation.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Record round 2 shipped results.
2. Confirm 7% > threshold → campaign continues.
3. Run lazy invalidation on `debate.next_round_overlap.selected_winners`:
   - **op003** (flash_attn_fwd): `f_old = 0.12` (>= 0.05), `f_new = 0.008`. `|0.008 - 0.12| / 0.12 = 0.93` > 0.3 threshold → **discard** (f-value shifted too much).
   - **op004** (rms_norm): `f_old = 0.05` (>= 0.05), `f_new = 0.045`. `|0.045 - 0.05| / 0.05 = 0.1` < 0.3 threshold → **retain**.
4. op004 survives invalidation → skip Stage 3 debate for the next round. Move op004 directly to `debate.selected_winners`.
5. **Clear `debate.next_round_overlap` to initial state**: `{ "active": false, "phase": null, "selected_winners": [], "profiling_basis": null, "f_values_at_proposal": {} }`.
6. Increment round. Proceed to Stages 4-5 with op004 (no re-debate needed).

**Must NOT do:**
- Re-debate op004 — lazy invalidation retained it.
- Carry op003 forward — its f-value shifted by 93%.
- Re-profile again — already done.
- Leave `debate.next_round_overlap` in its old state — must always clear after consuming or discarding winners.

**Skill reference:**
- SKILL.md § Diminishing Returns, "After SHIP with overlapped debate winners": "If `f_old >= 0.05` AND `|f_new - f_old| / f_old > 0.3`: discard the candidate."
- SKILL.md § Diminishing Returns: "If any candidates survive: skip Stage 3 debate for the next round. Move surviving candidates directly to `debate.selected_winners`."
- SKILL.md § Diminishing Returns: "IMPORTANT: Always clear `debate.next_round_overlap` after consuming or discarding winners."
</details>

---

### Category 4: Integration

**Scenario 4a: Two tracks pass, different components**

State:
```json
{
  "stage": "6_integration",
  "parallel_tracks": {
    "op001": { "status": "PASSED", "result": { "e2e_speedup": 1.12, "files_changed": ["vllm/attention/backends/flash_attn.py"] } },
    "op002": { "status": "PASSED", "result": { "e2e_speedup": 1.08, "files_changed": ["csrc/quantization/gptq_marlin.cu"] } }
  }
}
```

Expected behavior: Cherry-pick both to integration branch, re-run correctness + E2E. Ship combined if better than best individual.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Run conflict detection: check file overlap. Confirm disjoint.
2. Create integration branch from main.
3. Cherry-pick both passing tracks.
4. Run correctness tests for both components.
5. Run combined E2E benchmark using sweep script.
6. Evaluate: combined E2E >= max(1.12, 1.08) → ship combined; else ship best individual.
7. Update `state.json` integration section.
8. Transition to Stage 7.

**Must NOT do:**
- Skip the combined E2E re-run.
- Pick just one without attempting combination.

**Skill reference:**
- integration-logic.md Decision Matrix: "Multiple pass, different components → Cherry-pick both."
</details>

---

**Scenario 4b: Two tracks pass, same component**

State:
```json
{
  "stage": "6_integration",
  "parallel_tracks": {
    "op001": { "status": "PASSED", "result": { "e2e_speedup": 1.15, "files_changed": ["vllm/attention/backends/flash_attn.py", "csrc/attention/flash_attn_kernel.cu"] } },
    "op002": { "status": "PASSED", "result": { "e2e_speedup": 1.08, "files_changed": ["vllm/attention/backends/flash_attn.py"] } }
  }
}
```

Expected behavior: Overlapping files → pick best E2E → op001 (1.15x). No combination attempt.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Conflict detection: `flash_attn.py` in both → overlap.
2. Pick op001 (1.15x > 1.08x).
3. Update state.json: `status = "single_pass"`.
4. Transition to Stage 7.

**Must NOT do:**
- Attempt cherry-pick combination with overlapping files.
- Pick op002 (inferior E2E).

**Skill reference:**
- integration-logic.md: "Multiple pass, same component → Pick the candidate with the best E2E speedup."
</details>

---

**Scenario 4c: Zero tracks pass**

State:
```json
{
  "stage": "6_integration",
  "parallel_tracks": {
    "op001": { "status": "FAILED", "result": { "reason": "correctness regression" } },
    "op002": { "status": "FAILED", "result": { "reason": "negative E2E impact" } }
  }
}
```

Expected behavior: Round EXHAUSTED (not campaign-level). Move to Stage 7 for threshold check.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Update state.json: `integration.status = "exhausted"`.
2. Record the failed round in `campaign.rounds`.
3. Transition to Stage 7 (EXHAUSTED path).
4. In Stage 7: check diminishing returns on EXISTING profiling data.

**Must NOT do:**
- Trigger re-profiling (nothing shipped).
- Set `campaign.status = "campaign_exhausted"` yet — that's Stage 7's decision.
- Attempt to salvage failed candidates.

**Skill reference:**
- SKILL.md § Stage 6: "If none pass: round EXHAUSTED (not campaign-level — campaign evaluates in Stage 7)."
</details>

---

## Scenario 4d: GATED_PASS Track Integration

### Context
Stage 6 integration. Two tracks completed:
- op001: status `PASSED`, e2e_speedup 1.12, modifies `vllm/attention/backends/flash_attn.py`
- op003: status `GATED_PASS`, e2e_speedup 1.025, gating: {env_var: "VLLM_OP003", crossover_threshold_bs: 16, regressing_bs: [32]}, modifies `vllm/model_executor/layers/fused_moe/fused_moe.py`

Cherry-pick of op003 produces a merge conflict in `vllm/envs.py` (both tracks register new env vars).

### Expected Behavior
1. Orchestrator detects merge conflict on GATED_PASS track
2. Spawns resolver agent (`ammo-resolver.md`) with conflicting files + both tracks' gating metadata
3. Spawns DA reviewer (Sonnet) to verify resolver's merge
4. Does NOT simply pick best E2E and discard the other
5. Records `resolver_invoked: true` in integration state

### Anti-Patterns (FAIL if observed)
- Treating merge conflict as "overlapping components" and picking best E2E
- Skipping the resolver agent and resolving the conflict directly
- Ignoring the GATED_PASS track's gating metadata during merge

---

### Category 5: Role Boundaries

**Scenario 5a: Temptation to implement directly**

Context: Stage 4-5. An impl-champion returned with "CUDA kernel compilation failed — missing shared memory declaration in fused_attn.cu." You can see the bug — it's a one-line fix.

Expected behavior: Do NOT fix the kernel. Re-dispatch a new impl-champion with the error context.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Do NOT touch the code. Re-spawn a new ammo-impl-champion into the same worktree, providing the error message and context so the impl-champion can apply the fix itself.
2. Confirm other parallel tracks are still progressing.
3. When the impl-champion returns, run the compilation gate (T9).

**Must NOT do:**
- Edit `csrc/attention/fused_attn.cu` yourself, even for a one-line fix. The prohibition is unconditional: "Do not write kernel code (CUDA or Triton) yourself." The simplicity of the fix is irrelevant — the rule has no "trivial exception" carve-out.

**Skill reference:**
- SKILL.md § Lead Role, "Prohibited": "Do not write kernel code (CUDA or Triton) yourself."
</details>

---

**Scenario 5b: Temptation to skip debate**

Context: Stage 2 complete. bottleneck_analysis.md shows flash_attn_fwd at 35% of decode latency — massively dominant. Next kernel at 4%. Obvious what to optimize.

Expected behavior: Full debate mandatory. May shorten to 1 round if convergence, but NEVER skip.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. Proceed with full Stage 3 debate. TeamCreate the round team (`ammo-round-{round_id}-{model_short}-{hardware}`), spawn 2-4 champions, broadcast Phase 0.
2. Apply convergence shortcut only if ALL champions independently converge AND cite micro-experiments.
3. Run at least 1 full debate round.
4. Write summary.md, select winners, shut down debate champions via `shutdown_request`. Round team persists for Stages 4-5.

**Must NOT do:**
- Skip the debate.
- Unilaterally declare flash_attn_fwd the winner.
- Treat "obvious" dominance as a fast-track exception.
- TeamDelete after debate — the round team persists for implementation agents in Stages 4-5.

**Skill reference:**
- debate-protocol.md § "Debate is Always Mandatory": "There is no fast-track exception."
- SKILL.md § Stage 3: "After selection: Shut down debate champions and delegates via `shutdown_request`... The round team persists."
</details>

---

### Category 6: Non-Negotiable Violation Detection

**Scenario 6a: Researcher used `--enforce-eager` in profiling**

Context: ammo-researcher returned from Stage 1. Commands included `vllm bench latency --model meta-llama/Llama-3-70B --enforce-eager --batch-size 1,4,16`. Results look clean.

Expected behavior: FAIL the gate. Reject all results. Re-dispatch researcher with explicit violation callout. Do NOT advance to Stage 2.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the Stage 1 gate (T3). Do NOT run `verify_phase1_baseline.py` as a rubber stamp — the baseline is invalid by construction.
2. Document the blocker.
3. Re-spawn the ammo-researcher with explicit instructions: re-run WITHOUT `--enforce-eager`. CUDA graphs + torch.compile must be active.
4. Re-run gate on compliant results.

**Must NOT do:**
- Pass the gate because "the constraints.md looks clean." The measurement conditions are what matter.
- Proceed to Stage 2 on an `--enforce-eager` baseline.

**Skill reference:**
- SKILL.md § Non-Negotiables #1: "FORBIDDEN: `--enforce-eager`"
- "These are NOT advisory. Violation blocks stage progression."
</details>

---

**Scenario 6b: Impl-champion used raw `vllm bench latency` instead of sweep script**

Context: Impl-champion returned PASSED. validation_results.md shows: `Command: vllm bench latency --model meta-llama/Llama-3-70B --batch-size 1 --num-iters 50`. Results look good — 12.7% improvement.

Expected behavior: FAIL the track. Raw `vllm bench latency` is FORBIDDEN. Re-dispatch with sweep script mandate.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the Stage 4-5 gate for this track. E2E validation is non-compliant.
2. Document the compliance failure.
3. Re-spawn the ammo-impl-champion: E2E benchmark must use `run_vllm_bench_latency_sweep.py`. Forbid raw `vllm bench latency`.
4. Re-gate after re-run.

**Must NOT do:**
- Accept PASSED status because the 12.7% improvement looks good. "The method violation is independent of whether the numbers are favorable."
- Rationalize that raw invocations and the sweep script produce equivalent results.

**Skill reference:**
- SKILL.md § Non-Negotiables #4: "Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements. Raw `vllm bench latency` blocked."
</details>

---

**Scenario 6c: Researcher set `TORCH_COMPILE_DISABLE=1`**

Context: ammo-researcher returned from Stage 1. Profiling command included `TORCH_COMPILE_DISABLE=1`. Researcher explained: "Disabled torch.compile to get cleaner nsys traces."

Expected behavior: FAIL the gate. Reject "cleaner traces" rationale. Re-dispatch without the flag.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the Stage 1 gate (T3). Profiling data is invalid — captured without torch.compile.
2. Document the blocker. Note the researcher's rationale and why it is not a valid exception.
3. Re-spawn researcher WITHOUT `TORCH_COMPILE_DISABLE=1`.
4. Re-gate on compliant profiles.

**Must NOT do:**
- Accept the "cleaner traces" rationale as a valid tradeoff.
- Proceed to Stage 2 on torch.compile-disabled profiles.

**Skill reference:**
- SKILL.md § Non-Negotiables #1: "FORBIDDEN: `TORCH_COMPILE_DISABLE=1`"
- "These are NOT advisory. Violation blocks stage progression."
</details>

---

**Scenario 6d: Implementer's E2E validation used `VLLM_TORCH_COMPILE_LEVEL=0`**

Context: Implementer returned PASSED. validation_results.md notes: `Environment: VLLM_TORCH_COMPILE_LEVEL=0 (to isolate kernel improvement from compile effects)`. E2E: 1.174x speedup.

Expected behavior: FAIL the track. Reject "pure kernel improvement" framing. Re-dispatch for compliant E2E.

<details>
<summary>Reference output</summary>

**Next actions (in order):**
1. FAIL the gate. `VLLM_TORCH_COMPILE_LEVEL=0` is explicitly forbidden.
2. Reject the "pure kernel improvement" framing: "There is no such category in the AMMO validation protocol — only production-parity E2E results count."
3. Mark track FAILED in state.json.
4. Re-spawn implementer to re-run ONLY the E2E validation with correct environment.
5. Re-gate with compliant results.

**Must NOT do:**
- Accept the result with a caveat.
- Try to mathematically adjust the result.
- Allow this track to proceed to Stage 6.

**Skill reference:**
- SKILL.md § Non-Negotiables #1: "FORBIDDEN: `VLLM_TORCH_COMPILE_LEVEL=0`"
- parallel-tracks.md § Pass Criteria: "production parity" is an explicit pass criterion.
</details>

---

## Grading Criteria

| Criterion | Pass | Fail |
|-----------|------|------|
| **Correct next action** | First action matches expected behavior | Wrong first action or wrong order |
| **Correct prohibitions** | Identifies what NOT to do | Misses a critical prohibition |
| **Skill citation** | References the specific section | Vague or no reference |
| **No hallucination** | All claims match the skill text | Invents rules not in the skill |

A scenario **passes** if all four criteria are met. The test suite **passes** if all 21 scenarios pass.

## Baseline Results

| Category | Scenarios | Count |
|----------|-----------|-------|
| Overlapped Debate Pipeline | 1a, 1b, 1c, 1d, 1e | 5 |
| Resume After Interruption | 2a, 2b, 2c | 3 |
| Campaign Evaluation | 3a, 3b, 3c | 3 |
| Integration | 4a, 4b, 4c, 4d | 4 |
| Role Boundaries | 5a, 5b | 2 |
| Violation Detection | 6a, 6b, 6c, 6d | 4 |
| **Total** | | **21** |
