# Stage 3: Adversarial Debate Protocol

Champions independently propose optimization candidates from grounded Stage 2 data, then debate them adversarially. The main session acts as moderator. Stage 2 provides ONLY measured facts and physical ceilings — champions generate candidates and feasibility estimates themselves.

## Team Structure

- **Team name**: `ammo-round-{round_id}-{model_short}-{hardware}` -- this is the **round team**, created once per round and reused for both debate (Stage 3) and implementation (Stages 4-5). During overlapped debate (round 2+ only), debate champions for round N+1 are spawned into the EXISTING round team from round N, alongside implementation agents. The debate uses a distinct `round_id` in its artifact paths (`debate/campaign_round_{N+1}/`) to avoid collisions.
  - Example: `ammo-round-1-llama70b-h100`
- **Champions**: 2-4 `ammo-champion` agents. Each reads the grounded bottleneck_analysis.md independently.
- Each champion is spawned with:
  - `name="champion-{N}"` (e.g., `champion-1`, `champion-2`)
  - `team_name` set to the round team name above
- **Transcript Monitors**: Spawn 1 `ammo-transcript-monitor` agent per champion agent. See below

## Transcript Monitoring

After spawning each champion, the orchestrator spawns a transcript monitor in background:

## Team Composition with Monitors

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
|
| ... debate champions for round N (Stage 3) ...
| +-- champion-1 (Opus)         [Stage 3 debate -- shut down after selection]
| |   +-- monitor-1 (Sonnet)    [Stage 3 debate -- shut down after selection]
| +-- champion-2 (Opus)         [Stage 3 debate -- shut down after selection]
| |   +-- monitor-2 (Sonnet)    [Stage 3 debate -- shut down after selection]
| +-- champion-3 (Opus)         [Stage 3 debate -- shut down after selection]
|     +-- monitor-3 (Sonnet)    [Stage 3 debate -- shut down after selection]
|
| ... after debate: shutdown round N champions ...
|
| ... implementation agents for round N (Stages 4-5) ...
| +-- impl-champion-{op_id_1} (Opus)     [Stages 4-5]
| |   +-- impl-monitor-1 (Sonnet)        [Stages 4-5]
| +-- impl-validator-{op_id_1} (Sonnet)  [Stages 4-5]
| +-- impl-champion-{op_id_2} (Opus)     [Stages 4-5]
| |   +-- impl-monitor-2 (Sonnet)        [Stages 4-5]
| +-- impl-validator-{op_id_2} (Sonnet)  [Stages 4-5]
| ... OVERLAPPED: debate champions for round N+1 (if round 2+) ...
| +-- champion-r{N+1}-1 (Opus)           [next-round debate -- shut down after selection]
| |   +-- monitor-r{N+1}-1 (Sonnet)      [next-round debate -- shut down after selection]
| +-- champion-r{N+1}-2 (Opus)           [next-round debate -- shut down after selection]
| |   +-- monitor-r{N+1}-2 (Sonnet)      [next-round debate -- shut down after selection]
```

Naming convention: `monitor-{champion_number}` or `impl-monitor-{impl_champion_number}`

### Monitor Spawn Pattern (REQUIRED — one per champion)

After spawning each champion, immediately spawn its monitor as a **team member** (not a subagent):

```python
# projects_dir = os.path.expanduser("~/.claude/projects/") + os.getcwd().replace("/", "-")

# Team member — NOT a subagent. Uses team_name for SendMessage access.
Agent(
    name=f"monitor-{champion_name}",
    subagent_type="ammo-transcript-monitor",
    model="sonnet",
    team_name=round_team_name,       # Same team as champion — enables SendMessage
    run_in_background=True,
    prompt=f"""Monitor {champion_name} via session transcript.

    ## Target
    - Agent name: {champion_name}
    - Team: {round_team_name}
    - Stage: debate
    - Artifact dir: {artifact_dir}
    - Projects dir: {projects_dir}

    Focus on DEBATE-STAGE concerns: proposal methodology, evidence tiers,
    Amdahl consistency, baseline provenance, framing biases, micro-experiment methodology.
    See your agent definition § Stage-Specific Focus for the full list."""
)
```

### Communication Flow

Monitors passively read the champion's `.jsonl` session transcript (polling every ~5 seconds) and interject via SendMessage only when they detect methodology errors, framing biases, or procedural violations. Key properties:
- **Zero champion overhead**: Champions don't need to report status or interact with monitors
- **Unmediated observation**: Monitors read raw transcripts (tool calls, thinking blocks, results) — champions cannot filter what monitors see
- **Escalation**: If a CRITICAL finding is ignored for 2+ minutes, the monitor escalates to the orchestrator

## Debate is Always Mandatory

There is no fast-track exception. Every run must go through at least Phase 0 (proposals) + 1 debate round.

**No Convergence Shortcut**: Even if all champions converge on the same candidate, the minimum 2 debate rounds are mandatory. Convergence reduces critique diversity, making additional scrutiny MORE important, not less.

## Phase 0: Independent Proposals

All champions execute **in parallel**. Each champion reads the grounded bottleneck_analysis.md (which contains ONLY measured facts and physical ceilings) and independently proposes 1-2 optimization candidates.

Each champion writes:

```
{artifact_dir}/debate/proposals/{champion_id}_proposal.md
```

Champions write proposals per `references/debate-rules.md` (see Evidence Tiers for claim-evidence requirements, Micro-Experiment Rules for allowed/forbidden experiments, Baseline Provenance Rule for API matching, and Micro-Experiment Artifact Requirements for proof-of-execution).

### Proposal Eligibility Gate (Lead)

After Phase 0 submissions, the lead checks each proposal against the **Custom Kernel Mandate** before any debate begins:

- **Pass**: Proposal involves writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code.
- **Reject**: Config-only changes, flag-flipping, parameter tuning, or no kernel code scope described.

**Rejection action**: Message the champion to revise with a kernel-based proposal. If no compliant revision is submitted, the candidate is eliminated.

**Non-compliant proposals MUST NOT advance to Round 1.**

### Diversity Check (Lead)

After the eligibility gate, the lead reviews proposal diversity:

1. **f-value source check**: For each proposal, check whether the champion used `f_decode` (from the per-decode-step breakdown) or `f_total` (from the full trace). If the target kernel isn't in the decode breakdown, note this — the champion may be targeting prefill latency intentionally, or may have used a misleading f-value.

2. **Dominant Component Coverage (MANDATORY)**: At least one proposal MUST target the component with the highest `f_decode` from bottleneck_analysis.md. If the top component exceeds 50% of decode time, at least two proposals must target it with distinct techniques. A component may only be excluded from all proposals if two independent micro-experiments (from different champions) demonstrate it is within 10% of its physical ceiling (BW or compute utilization). If no proposal targets the dominant component after Phase 0, the lead MUST reject the lowest-scoring proposal and require a revision targeting it.

## Round Structure

Normal minimum: **2 rounds**. Maximum: **5 rounds**. Each round has three sequential phases.

### Phase A: Evidence Presentation

All champions execute **in parallel**.

Each champion writes:

```
{artifact_dir}/debate/round_{N}/{op_id}_argument.md
```

Champions write arguments per `references/debate-rules.md` (see Evidence Tiers for required evidence levels). Champions **must** run micro-experiments during this phase (see `references/debate-rules.md` § Micro-Experiment Rules).

### Phase B: Critique

Round-robin assignment:

- Champion for OP-001 critiques OP-002
- Champion for OP-002 critiques OP-003
- ...
- Champion for OP-N critiques OP-001

Each champion writes:

```
{artifact_dir}/debate/round_{N}/{op_id}_critique_{target_id}.md
```

Critiques must address: feasibility math weaknesses, overlooked risks, incorrect assumptions, and hardware resource accounting (SMEM budget, register usage, occupancy, wave count). See `references/debate-rules.md` for evidence tier requirements.

### Phase C: Rebuttal

All champions execute **in parallel**.

Each champion responds to the critique they received and writes:

```
{artifact_dir}/debate/round_{N}/{op_id}_rebuttal.md
```

Rebuttals must provide counter-evidence, concede valid points explicitly, or propose concrete mitigation for acknowledged weaknesses. See `references/debate-rules.md` for evidence tier requirements.

## Communication Flow

The main session moderates the debate using `SendMessage`:

1. **Broadcast** phase-start message to all champions (includes round number and phase identifier).
2. Each champion **messages main** upon phase completion with a short status line.
3. Main **waits** for all champions to report before advancing to the next phase.
4. After each complete round, main evaluates convergence criteria.

## Convergence Criteria

Stop early (only after exceeding minimum rounds) if **either** condition is met:

1. **Clear winners**: The top 3-4 candidates have no unaddressed critiques remaining, and all other candidates have conceded material weaknesses.
2. **Stagnation**: Round N+1 arguments substantially repeat round N with no new evidence or counter-arguments introduced.

## Winner Selection

After the final round:

1. Main session reads **all** debate artifacts across all rounds.
2. Scores each candidate per `references/debate-scoring-rubric.md`.
3. Selects **3-4 winners** to advance to Stage 4 parallel tracks.
4. Writes the decision to:

```
{artifact_dir}/debate/summary.md
```

The summary includes: per-candidate scores, rationale for selection, and any conceded weaknesses that Stage 4 implementation must address.

Flag proposals with per-BS differentiated impact (e.g., M<=32 kernel specialization, decode-only path). These are candidates for `GATED_PASS` and should be noted in `summary.md` so Stage 4-5 implementers are prepared for crossover probing.

### Post-Selection Evidence Gate

After winner selection but BEFORE shutting down debate champions, the lead opens a **15-minute evidence window**:

1. Broadcast to all champions: "Winners selected. 15-minute window for late findings that could override selection. Submit to `{artifact_dir}/debate/late_findings/`."
2. If any champion submits a late finding citing a hard constraint violation (SMEM exceeded, API incompatibility, optimization already deployed):
   - The lead evaluates the finding against available evidence
   - If the finding is substantiated: the winner is eliminated and the next-highest-scoring candidate advances
   - If the finding is unsubstantiated: noted but selection stands
3. After 15 minutes (or all champions respond "no findings"): proceed to shutdown.

This gate addresses the structural gap where Champion-1 in c08b370fc identified both fatal flaws in "Late Findings" after selection was finalized, with no mechanism for action.

## Debate Rules Reference

Micro-experiment guidelines, artifact requirements, baseline provenance rules, and NCU triggers are defined in `references/debate-rules.md`. Champions must read this reference. The lead uses NCU Trigger 4 (Baseline BW Discrepancy) during the eligibility gate.

## Teardown (Post-Debate)

After winner selection:

1. Send `shutdown_request` to all champion agents.
2. Do NOT call TeamDelete. The round team persists for Stages 4-5 implementation.
3. TeamDelete is called only after all implementation tracks complete (see `parallel-tracks.md` and `SKILL.md` Stage 4-5 section).

## Artifact Structure

### Round 1 (or single-pass campaigns)

```
{artifact_dir}/debate/
  summary.md
  proposals/
    champion-1_proposal.md
    champion-2_proposal.md
    champion-3_proposal.md
  round_1/
    champion-1_argument.md
    champion-1_critique_champion-2.md
    champion-1_rebuttal.md
    champion-2_argument.md
    champion-2_critique_champion-3.md
    champion-2_rebuttal.md
    ...
  round_2/
    ...
  micro_experiments/
    champion-1_roofline.py
    champion-2_ncu_query.txt
    ...
```

### Campaign Round 2+ (scoped paths)

For campaign rounds beyond the first, debate artifacts must use campaign-round-scoped paths to avoid overwriting previous rounds' evidence:

```
{artifact_dir}/debate/campaign_round_{N}/
  summary.md
  proposals/
    champion-1_proposal.md
    ...
  round_1/          (debate round within this campaign round)
    ...
  round_2/
    ...
  micro_experiments/
    ...
```

The debate gate hook enforces this: round 2+ debates that use the legacy `debate/` path are blocked.

Note: "round_1" inside the campaign round directory refers to **debate rounds** (argument/critique/rebuttal cycles). The outer "campaign_round_{N}" refers to **campaign rounds** (profiling cycles).

## Overlapped Debate Considerations

When running as an overlapped debate during Stages 4-5:

1. **Debate champions MUST NOT message implementation agents.** The orchestrator does not provide implementation agent names. If a champion receives an unexpected message from an unknown agent, ignore it and notify the orchestrator.

2. **Artifact paths MUST use campaign-round scoping**: `debate/campaign_round_{N+1}/`. This is already enforced by the debate gate hook for round 2+, and remains enforced for overlapped debates.

3. **GPU access during overlap**: Debate champions may run GPU micro-benchmarks via the pool (`--num-gpus 1`). If the pool is exhausted, the reserve call blocks — keep experiments brief to minimize contention with implementation tracks. See `references/gpu-pool.md` for the reservation pattern.

4. **Phase timing may be longer**: The orchestrator interleaves debate moderation with implementation monitoring. Debate phases may take longer to start because the orchestrator is busy gating implementation results. Champions should NOT assume a phase will start within any specific time window.

5. **The orchestrator decides debate timing**: Debate champions should wait for phase-start broadcasts from the orchestrator. Do not self-advance to the next phase.

