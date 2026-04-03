# Stages 4-5: Parallel Worktree Track Management

Each winning candidate from Stage 3 gets its own git worktree, branch, and implementation champion. All agents -- across all tracks -- belong to the **same round team** created at the start of Stage 3. The orchestrator can only lead one team at a time, so a single round-scoped team is used for the entire round lifecycle (debate through implementation). Tracks run in parallel across GPUs. Within a track, the champion manages validation internally by spawning a kernel validation sub-agent (not a team member). During round 2+, debate champions for the next round may also be present in the team (see Overlapped Debate below). Implementation and debate agents share the team but operate as independent workstreams -- they do not communicate directly.

## Team Structure (Single Round Team)

All implementation agents join the existing round team. The team was created at Stage 3 start (`ammo-round-{round_id}-{model_short}-{hardware}`) and persists through Stages 4-5.

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
[Implementation Workstream]
+-- impl-champion-{op_id_1} (Opus)           -- implementation + validation orchestration
+-- monitor-impl-champion-{op_id_1} (Sonnet) -- transcript monitor (background, team member)
+-- impl-champion-{op_id_2} (Opus)           -- implementation + validation orchestration
+-- monitor-impl-champion-{op_id_2} (Sonnet) -- transcript monitor (background, team member)
    (kernel validation sub-agents spawned by champions as needed -- NOT team members)
[Overlapped Debate Workstream -- round 2+ only]
+-- champion-r{N+1}-1 (Opus)                 -- next-round debate champion [shut down after selection]
+-- monitor-champion-r{N+1}-1 (Sonnet)       -- transcript monitor (background) [shut down after selection]
+-- champion-r{N+1}-2 (Opus)                 -- next-round debate champion [shut down after selection]
+-- monitor-champion-r{N+1}-2 (Sonnet)       -- transcript monitor (background) [shut down after selection]
```

Each track uses an **adversarial validation model**: the champion implements, then spawns a kernel validation sub-agent that independently writes its own correctness tests and benchmarks. This separation prevents reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation). Transcript monitors provide continuous DA oversight of champion work.

**Why adversarial validation**: The AMMO pipeline has observed four types of reward hacking when the same agent both implements and validates. Separating implementation from validation into independent agents with differing goals catches issues in real-time.

**Why a single team**: The orchestrator can only lead ONE team at a time. Creating per-track teams (`ammo-impl-{op_id}`) is architecturally impossible — the orchestrator would lose contact with all but the last-created team. A single round-scoped team keeps all agents under one roof.

## Worktree Creation

Worktrees are created automatically by the Agent tool when spawning `ammo-impl-champion` subagents (which have `isolation: worktree` in their definition). The `WorktreeCreate` hook (`worktree-create-with-build.sh`) pre-configures Python isolation, copies `.so` files, and creates a per-worktree `.venv`.

The champion spawns the kernel validation sub-agent with the worktree path in the spawn prompt. The sub-agent runs sequentially (Gates 5.1a + 5.2 first, then champion runs the E2E sweep).

### GPU Pool

All agents share a machine-wide GPU pool. See `references/gpu-pool.md` for the reservation pattern and contention handling.

## Worktree Build Rules (CRITICAL)

See `references/impl-track-rules.md` § Build Rules for the change-type/action matrix. Only the champion compiles — the validator never runs cmake.

## Per-Track Execution Pipeline

### Orchestrator Spawns Implementation Agents

The round team already exists from Stage 3 (`ammo-round-{round_id}-{model_short}-{hardware}`). No new TeamCreate is needed. All implementation agents join the existing team.

```python
# existing_team_name = state.json -> debate.team_name
# e.g., "ammo-round-1-llama70b-h100"
# projects_dir = os.path.expanduser("~/.claude/projects/") + os.getcwd().replace("/", "-")

# Per winning candidate — spawn impl-champion + monitor into the EXISTING round team:
Agent(
    name=f"impl-champion-{op_id}",
    subagent_type="ammo-impl-champion",
    team_name=existing_team_name,    # Reuse round team, NOT a per-track team
    prompt="""
    You are implementing optimization {op_id} for the AMMO pipeline.

    Artifact dir: {artifact_dir}
    Optimization plan: {artifact_dir}/debate/summary.md (section for {op_id})
    Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
    GPU pool: {gpu_count} GPUs available (TP={tp}). Acquire at runtime:
      CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=$CVD <cmd>
      Kernel work: --num-gpus 1  |  E2E sweep: --num-gpus {tp}

    ## Stage 1 Baseline (DO NOT RE-RUN)
    Baseline E2E latency files (captured from the session base branch in Stage 1):
    - Per-batch-size JSON: {artifact_dir}/runs/baseline_bs{N}.json
    - Summary table: {artifact_dir}/constraints.md ("Baseline E2E latency" section)
    - Kernel breakdown: {artifact_dir}/constraints.md ("Baseline Truth Snapshot" section)

    ## Precision Classification (from debate summary)
    Classification: {classification}  # "lossless" or "lossy" — determines Gate 5.1a tolerances and debate scoring deflation

    ## E2E Threshold
    E2E threshold: min_e2e_improvement_pct (from state.json)

    ## Regression Thresholds (from campaign config)
    - noise_tolerance_pct: {noise_tolerance_pct} (default: 0.5%)
    - catastrophic_regression_pct: {catastrophic_regression_pct} (default: 5.0%)
    - Per-BS verdicts: PASS / NOISE / REGRESSED / CATASTROPHIC
    - Track verdicts: PASS / GATING_REQUIRED / GATED_PASS / FAIL
    - See references/validation-defaults.md and references/crossover-probing.md

    Workflow:
    1. Read debate artifacts, spawn ammo-delegate subagents for research tasks
    2. Implement the kernel optimization
    3. Commit implementation
    4. Spawn kernel validation sub-agent for Gates 5.1a + 5.2 (see your agent definition § Kernel Validation)
    5. Run E2E sweep per your agent definition § E2E Validation (ONE command handles 5.1b + 5.3a + 5.3b)
    6. Evaluate E2E results against min_e2e_improvement_pct threshold, write validation_results.md
    """
)

# Transcript monitor — team member (NOT a subagent). Needs SendMessage for DA interjections.
Agent(
    name=f"monitor-impl-champion-{op_id}",
    subagent_type="ammo-transcript-monitor",
    model="sonnet",
    team_name=existing_team_name,    # Same team — enables SendMessage to champion and team-lead
    run_in_background=True,
    prompt=f"""Monitor impl-champion-{op_id} via session transcript.

    ## Target
    - Agent name: impl-champion-{op_id}
    - Team: {existing_team_name}
    - Stage: implementation
    - Artifact dir: {artifact_dir}
    - Projects dir: {projects_dir}
    - Classification: {classification}  # "lossless" or "lossy" — for undisclosed precision reduction CRITICAL check

    Focus on IMPLEMENTATION-STAGE concerns: production parity, worktree discipline,
    validation integrity, gate completeness, baseline reuse, reasoning gaps.
    See your agent definition § Stage-Specific Focus for the full list."""
)

# NOTE: The champion spawns ammo-impl-validator as a sub-agent internally.
# The orchestrator does NOT spawn or manage the validator.
```

### Champion-Managed Validation

The champion manages kernel validation internally:
1. Champion spawns `ammo-impl-validator` as a sub-agent via `Agent()` (not a team member)
2. Sub-agent runs Gates 5.1a (kernel correctness) + 5.2 (kernel speedup), returns results
3. If 5.1a FAIL: champion fixes, re-spawns sub-agent (no wasted E2E sweep)
4. If 5.1a PASS: champion runs sweep (5.1b + 5.3a + 5.3b)
5. Champion combines all gate results into `validation_results.md`
6. Champion reports `TRACK_COMPLETE` to orchestrator via SendMessage

The orchestrator reads `validation_results.md` for gate decisions but does not participate in the validation loop.

### Collaboration Timeline and Key Rules

The champion manages kernel validation by spawning the sub-agent sequentially before running the E2E sweep. Key rules: only the champion modifies source, independent validation is non-negotiable (sub-agent writes own tests from debate plan). See `references/impl-track-rules.md` for full constraints.

### Champion-Owned Validation with Kernel Sub-Agent

```
Kernel-Level (Sub-Agent, Sonnet):
  Gate 5.1a: Writes OWN kernel correctness tests, returns structured results
  Gate 5.2: Runs kernel speedup benchmark under CUDA graphs

E2E-Level (Champion, Opus):
  Gate 5.1b: Sweep --verify-correctness (GSM8K greedy decode)
  Gate 5.3a: Sweep --nsys-profile (kernel execution proof)
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Cross-checks Gate 5.1a against correctness_verdict.json
  Writes final validation_results.md with evidence chain
```

The sub-agent provides adversarial kernel-level verification. The champion provides E2E-level verification. The transcript monitor provides continuous DA oversight.

### Handling Validation Failures and GATING_REQUIRED

When validation fails or a GATING_REQUIRED verdict is reported, follow the workflows in `references/impl-track-rules.md` § Validation Failures and § GATING_REQUIRED. Key principle: the validator re-runs ALL gates from scratch with fresh independent tests each cycle. For GATING_REQUIRED, one gating attempt per track — no nested gating.

## Result Collection

After all tracks complete, main reads each track's outputs:

1. `{artifact_dir}/tracks/{op_id}/validation_results.md` — champion's final report (includes sub-agent results)
2. `{artifact_dir}/tracks/{op_id}/validator_tests/` — sub-agent's independent scripts and results
3. `state.json` field `parallel_tracks.{op_id}.result` — structured summary

Main aggregates results to determine which candidates pass to Stage 6 integration.

For `GATED_PASS` tracks, the `state.json` `parallel_tracks.{op_id}` entry includes additional fields:
- `verdict`: `"GATED_PASS"`
- `per_bs_verdict`: per-BS verdict map (e.g., `{"1": "PASS", "8": "PASS", "32": "REGRESSED"}`)
- `gating`: gating metadata object (mechanism, env_var, dispatch_condition, crossover_threshold_bs, crossover_probing sub-object, pre_gating_results, post_gating_results)

### Pass Criteria (Tiered Verdict System)

See `references/validation-defaults.md` § Gate 5.3b for threshold values and per-BS verdict computation logic.

A track **ships** if its final status is `PASS` or `GATED_PASS` (after successful gating).

Additional requirements unchanged:
- Gate 5.1: Correctness — both sub-gates must pass:
  - 5.1a: Kernel validation sub-agent's independent correctness tests pass
  - 5.1b: Sweep script `--verify-correctness` verdict is PASS in `correctness_verdict.json` (deterministic — no N/A escape)

### Track Status Machine

```
IN_PROGRESS → PASS       (all BS PASS/NOISE, at least one PASS)
IN_PROGRESS → GATING_REQUIRED (some PASS + some REGRESSED)
GATING_REQUIRED → GATED_PASS   (crossover probing + gating + re-validation succeeded)
GATING_REQUIRED → FAIL         (gating infeasible, re-validation failed, or probing timed out)
IN_PROGRESS → FAIL       (CATASTROPHIC, all REGRESSED, correctness failure)
```

## Team and Worktree Cleanup

After all implementation tracks have completed and results are collected:

1. **TeamDelete** the round team (`ammo-round-{round_id}-{model_short}-{hardware}`). This is the only TeamDelete in the round lifecycle -- it shuts down any remaining implementation agents. TeamDelete is called only after all implementation tracks complete AND the overlapped debate (if launched) has completed.

2. **Remove worktrees** for all tracks (after Stage 6 integration is complete or a track is abandoned):

```bash
git worktree remove {worktree_path} --force
```

Run cleanup for all tracks, including failed ones.

## In-Flight Tracks During Campaign Re-profiling

When a candidate ships and triggers re-profiling, other tracks from the same round may still be running:

1. Let all in-flight implementations complete against the ORIGINAL round's baseline
2. Validate using Stage 1 baseline from the current round (not re-profiled baseline)
3. If they pass: they also ship as additional cumulative gain
4. Record all track results in the current round's `campaign.rounds` entry
5. Next campaign round starts only after all current-round tracks complete

## Overlapped Debate Within the Round Team (Round 2+ Only)

During round 2+, the orchestrator spawns debate champions for the next campaign round into the same round team alongside implementation agents. This overlaps debate with implementation, saving 35-70 minutes per extra round.

### Communication Isolation

Debate champions and implementation agents MUST NOT communicate:
- The orchestrator does NOT provide implementation agent names in debate champion spawn prompts (and vice versa).
- Debate champions write to `debate/campaign_round_{N+1}/` (scoped paths).
- Implementation agents write to `tracks/{op_id}/`.
- The orchestrator is the only agent that communicates with both workstreams.

### Orchestrator Interleaving Pattern

The orchestrator manages both workstreams through the same team inbox:

1. Broadcast debate Phase 0 start to debate champions.
2. Check for any impl track completions.
3. If an impl track completed: run T9 gate, update state.json.
4. Wait for debate Phase 0 completions from all champions.
5. Run eligibility gate on proposals.
6. Broadcast debate Round 1 Phase A.
7. Check for impl track completions again.
8. Continue alternating until both workstreams complete.

The orchestrator MUST NOT advance to Stage 6 integration until:
- ALL implementation tracks have completed (PASSED or FAILED)
- The overlapped debate has completed (winners selected, champions shut down) OR has exceeded the 90-minute timeout (in which case: shut down debate champions, discard partial results, proceed without overlapped debate winners)

### GPU Allocation During Overlap

| Agent Type | GPU Access |
|-----------|-----------|
| Implementation champions | Pool access — kernel work (--num-gpus 1) + E2E sweep (--num-gpus {tp}) |
| Kernel validation sub-agents | Pool access — kernel work only (--num-gpus 1) |
| Debate champions | Pool access — micro-benchmarks OK (--num-gpus 1) |
| Transcript monitors | Read-only — no GPU access (transcript parsing + analysis only) |

### Debate Results Handling

When the overlapped debate completes:
1. Orchestrator scores candidates using the debate-scoring-rubric.md.
2. Orchestrator writes `debate/campaign_round_{N+1}/summary.md`.
3. Orchestrator records winners in `state.json` at `debate.next_round_overlap.selected_winners`.
4. Orchestrator records each winner's f-value at proposal time in `debate.next_round_overlap.f_values_at_proposal`.
5. Orchestrator sends `shutdown_request` to all debate champions and their monitors.
6. Winners are used in round N+1 after lazy invalidation (see SKILL.md, Campaign Loop Transition).
