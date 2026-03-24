# Stages 4-5: Parallel Worktree Track Management

Each winning candidate from Stage 3 gets its own git worktree, branch, and implementation agent pair. All agents -- across all tracks -- belong to the **same round team** created at the start of Stage 3. The orchestrator can only lead one team at a time, so a single round-scoped team is used for the entire round lifecycle (debate through implementation). Tracks run in parallel across GPUs. Within a track, the champion and validator collaborate fluidly -- both may be active simultaneously, with GPU access coordinated via SendMessage. During round 2+, debate champions for the next round may also be present in the team (see Overlapped Debate below). Implementation and debate agents share the team but operate as independent workstreams -- they do not communicate directly.

## Team Structure (Single Round Team)

All implementation agents join the existing round team. The team was created at Stage 3 start (`ammo-round-{round_id}-{model_short}-{hardware}`) and persists through Stages 4-5.

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
[Implementation Workstream]
+-- impl-champion-{op_id_1} (Opus)    -- implementation, E2E threshold evaluation
+-- impl-validator-{op_id_1} (Sonnet)  -- independent correctness tests, benchmarks, E2E sweep
+-- impl-champion-{op_id_2} (Opus)    -- implementation, E2E threshold evaluation
+-- impl-validator-{op_id_2} (Sonnet)  -- independent correctness tests, benchmarks, E2E sweep
[Overlapped Debate Workstream -- round 2+ only]
+-- champion-r{N+1}-1 (Opus)          -- next-round debate champion [shut down after selection]
+-- champion-r{N+1}-2 (Opus)          -- next-round debate champion [shut down after selection]
+-- delegate-r{N+1}-1a (Sonnet)       -- debate delegate [shut down after selection]
```

Each track uses an **adversarial validation model**: the champion implements, an independent validator validates. This separation prevents reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation).

**Why adversarial validation**: The AMMO pipeline has observed four types of reward hacking when the same agent both implements and validates. Separating implementation from validation into independent agents with differing goals catches issues in real-time. The existing DA Stop hook remains as a third verification layer.

**Why a single team**: The orchestrator can only lead ONE team at a time. Creating per-track teams (`ammo-impl-{op_id}`) is architecturally impossible — the orchestrator would lose contact with all but the last-created team. A single round-scoped team keeps all agents under one roof.

## Worktree Creation

Worktrees are created automatically by the Agent tool when spawning `ammo-impl-champion` subagents (which have `isolation: worktree` in their definition). The `WorktreeCreate` hook (`worktree-create-with-build.sh`) pre-configures Python isolation, copies `.so` files, and creates a per-worktree `.venv`.

The validator agent shares the champion's worktree (same track team). Both agents may be active simultaneously — GPU access is coordinated via SendMessage, with the champion having priority.

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

# Per winning candidate — spawn into the EXISTING round team (no TeamCreate):
Agent(
    name=f"impl-champion-{op_id}",
    subagent_type="ammo-impl-champion",
    team_name=existing_team_name,    # Reuse round team, NOT a per-track team
    prompt="""
    You are implementing optimization {op_id} for the AMMO pipeline.
    Your validator is impl-validator-{op_id}.

    Artifact dir: {artifact_dir}
    Optimization plan: {artifact_dir}/debate/summary.md (section for {op_id})
    Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
    GPU pool: {gpu_count} GPUs available (TP={tp}). Acquire at runtime:
      CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=$CVD <cmd>
      Kernel work: --num-gpus 1  |  E2E sweep: --num-gpus {tp}

    ## Stage 1 Baseline (DO NOT RE-RUN)
    Baseline E2E latency files (captured from clean main in Stage 1):
    - Per-batch-size JSON: {artifact_dir}/runs/baseline_bs{N}.json
    - Summary table: {artifact_dir}/constraints.md ("Baseline E2E latency" section)
    - Kernel breakdown: {artifact_dir}/constraints.md ("Baseline Truth Snapshot" section)

    ## E2E Threshold
    E2E threshold: min_e2e_improvement_pct (from state.json)

    ## Regression Thresholds (from campaign config)
    - noise_tolerance_pct: {noise_tolerance_pct} (default: 0.5%)
    - catastrophic_regression_pct: {catastrophic_regression_pct} (default: 5.0%)
    - Per-BS verdicts: PASS / NOISE / REGRESSED / CATASTROPHIC
    - Track verdicts: PASS / GATING_REQUIRED / GATED_PASS / FAIL
    - See references/validation-defaults.md and references/crossover-probing.md

    Workflow:
    1. Delegate research to your validator while you read debate artifacts
    2. Implement the kernel optimization (use validator for codebase lookups as needed)
    3. Commit implementation, then delegate validation (all 3 gates) to your validator
    4. Evaluate E2E results against min_e2e_improvement_pct threshold, write validation_results.md
    """
)

# Validator in the same round team (shared worktree via champion)
Agent(
    name=f"impl-validator-{op_id}",
    subagent_type="ammo-impl-validator",
    model="sonnet",
    team_name=existing_team_name,    # Same round team as champion
    prompt="""
    You are the independent validator for optimization {op_id}.
    Your champion is impl-champion-{op_id}.

    Artifact dir: {artifact_dir}
    GPU pool: {gpu_count} GPUs available (TP={tp}). Acquire at runtime:
      CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=$CVD <cmd>
      Kernel work: --num-gpus 1  |  E2E sweep: --num-gpus {tp}

    Wait for initial tasks from your champion via SendMessage.
    You support the champion throughout: research, profiling, codebase lookups, proactive advisories.
    When the champion requests validation, write your OWN independent tests and benchmarks (Gates 5.1, 5.2, 5.3).

    Key references:
    - Benchmark template: .claude/skills/ammo/references/kernel-benchmark-template.py
    - Validation defaults: .claude/skills/ammo/references/validation-defaults.md
    - CUDA graph safety: .claude/skills/ammo/references/cudagraph-safety.md
    """
)
```

### Collaboration Timeline and Key Rules

Champion-validator collaboration follows the fluid timeline in `references/impl-track-rules.md` § Collaboration. Key rules: only champion modifies source, champion has GPU priority, independent validation is non-negotiable, champion idle during validation. See `references/impl-track-rules.md` § Track Rules for full details.

### Three Layers of Verification

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN correctness tests, OWN benchmark scripts, runs E2E sweep
  Reports raw structured results — no interpretation

Layer 2: Champion Review (Opus)
  Evaluates E2E results against min_e2e_improvement_pct threshold
  Cross-checks Gate 5.2 numbers against own smoke-test
  Writes final validation_results.md with evidence chain

Layer 3: DA Stop Hook (Sonnet, frontmatter on champion)
  Fires when champion attempts to stop
  Audits validation_results.md: completeness, Amdahl's consistency,
  production parity, independent validation existence, benchmark cross-check
```

### Handling Validation Failures and GATING_REQUIRED

When validation fails or a GATING_REQUIRED verdict is reported, follow the workflows in `references/impl-track-rules.md` § Validation Failures and § GATING_REQUIRED. Key principle: the validator re-runs ALL gates from scratch with fresh independent tests each cycle. For GATING_REQUIRED, one gating attempt per track — no nested gating.

## Result Collection

After all tracks complete, main reads each track's outputs:

1. `{artifact_dir}/tracks/{op_id}/validation_results.md` — champion's final report
2. `{artifact_dir}/tracks/{op_id}/validator_tests/` — validator's independent scripts and results
3. `state.json` field `parallel_tracks.{op_id}.result` — structured summary

Main aggregates results to determine which candidates pass to Stage 6 integration.

For `GATED_PASS` tracks, the `state.json` `parallel_tracks.{op_id}` entry includes additional fields:
- `verdict`: `"GATED_PASS"`
- `per_bs_verdict`: per-BS verdict map (e.g., `{"1": "PASS", "8": "PASS", "32": "REGRESSED"}`)
- `gating`: gating metadata object (mechanism, env_var, dispatch_condition, crossover_threshold_bs, crossover_probing sub-object, pre_gating_results, post_gating_results)

### Pass Criteria (Tiered Verdict System)

Track verdicts are determined by the tiered verdict system. See `references/validation-defaults.md` §5.3 for threshold values and computation logic. Summary:

| Track Verdict | Condition |
|--------------|-----------|
| `PASS` | All BS are PASS or NOISE, with at least one PASS |
| `GATING_REQUIRED` | Some BS are PASS + some are REGRESSED |
| `FAIL` | Any CATASTROPHIC, or all REGRESSED/NOISE (no PASS), or gating failed |

A track **ships** if its final status is `PASS` or `GATED_PASS` (after successful gating).

Additional requirements unchanged:
- Gate 5.1: Validator's independent correctness tests pass
- Gate 5.2: Validator's independent kernel benchmark shows measurable speedup (>1% over baseline) for at least one target bucket
- Champion's DA Stop hook passed (Amdahl's check, baseline citation, parity, independent validation exists)
- No unresolved benchmark divergence between champion and validator

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
2. Check for any impl track completions (DA Stop hook notifications).
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
| Implementation validators | Pool access — kernel work (--num-gpus 1) + E2E sweep (--num-gpus {tp}) |
| Debate champions | Pool access — micro-benchmarks OK (--num-gpus 1) |
| Debate delegates | Static analysis only (ncu --query-metrics). No kernel benchmarks. |

### Debate Results Handling

When the overlapped debate completes:
1. Orchestrator scores candidates using the debate-scoring-rubric.md.
2. Orchestrator writes `debate/campaign_round_{N+1}/summary.md`.
3. Orchestrator records winners in `state.json` at `debate.next_round_overlap.selected_winners`.
4. Orchestrator records each winner's f-value at proposal time in `debate.next_round_overlap.f_values_at_proposal`.
5. Orchestrator sends `shutdown_request` to all debate champions and delegates.
6. Winners are used in round N+1 after lazy invalidation (see SKILL.md, Campaign Loop Transition).
