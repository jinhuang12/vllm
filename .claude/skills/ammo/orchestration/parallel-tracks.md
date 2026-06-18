# Stages 4-5: Parallel Worktree Track Management

Each winning candidate from Stage 3 gets its own git worktree, branch, and implementation champion. All agents -- across all tracks -- belong to the **same round team** created at the start of Stage 3. The orchestrator can only lead one team at a time, so a single round-scoped team is used for the entire round lifecycle (debate through implementation). Tracks run in parallel across GPUs. Within a track, the champion runs kernel correctness & speedup before the E2E sweep.

## Team Structure (Single Round Team)

All implementation agents join the existing round team. The team was created at Stage 3 start (`ammo-round-{round_id}-{model_short}-{hardware}`) and persists through Stages 4-5.

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
[Implementation Workstream]
+-- impl-champion-{op_id_1}           -- implementation + kernel validation
+-- monitor-impl-champion-{op_id_1}   -- transcript monitor (team member)
+-- impl-champion-{op_id_2}           -- implementation + kernel validation
+-- monitor-impl-champion-{op_id_2}   -- transcript monitor (background, team member)
```

In each track the champion implements, then writes and runs its own kernel correctness tests and CUDA-graph speedup benchmark before the E2E sweep. Transcript monitors provide continuous DA oversight of champion work.

**Why a single team**: The orchestrator can only lead ONE team at a time. Creating per-track teams (`ammo-impl-{op_id}`) is architecturally impossible — the orchestrator would lose contact with all but the last-created team. A single round-scoped team keeps all agents under one roof.

## Worktree Creation

Worktrees are created automatically by the Agent tool when spawning `ammo-impl-champion` subagents (which have `isolation: worktree` in their definition). The `WorktreeCreate` hook (`worktree-create-with-build.sh`) pre-configures Python isolation, copies `.so` files, and creates a per-worktree `.venv`.

The champion runs kernel validation in its own worktree: it writes and runs the kernel correctness & speedup tests first, then runs the E2E sweep.

### GPU Pool

All agents share a machine-wide GPU pool. See `references/gpu-pool.md` for the reservation pattern and contention handling.

## Worktree Build Rules (CRITICAL)

See `references/impl-track-rules.md` § Build Rules for the change-type/action matrix. The champion is the only agent that compiles in the track worktree.

## Per-Track Execution Pipeline

### Orchestrator Spawns Implementation Agents

The round team already exists from Stage 3 (`ammo-round-{round_id}-{model_short}-{hardware}`). No new TeamCreate is needed. All implementation agents join the existing team.

**CWD before spawn.** Spawn subagents from the session worktree root (`$CLAUDE_PROJECT_DIR`). If you `cd` into `.claude/worktrees/<op_id>/` to inspect a track, `cd "$CLAUDE_PROJECT_DIR"` before the next `Agent(...)` — subagents inherit cwd, and relative paths in the prompt resolve against it.

```python
# existing_team_name = state.json -> campaign.rounds[campaign.current_round - 1].team_name
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
    Optimization plan: state.json.campaign.rounds[-1].debate.selected_candidates (filter by op_id=={op_id}) — authoritative typed contract. rounds/{CR}/debate/summary.md is a rendered view; rounds/{CR}/debate/proposals/{op_id}_proposal.md has proposal-level depth.
    Bottleneck analysis: {artifact_dir}/rounds/{CR}/mining/bottleneck_analysis.md
    GPU pool: {gpu_count} GPUs available (TP={tp}, DP={dp}, model replica={tp*dp}). Acquire at runtime:
      CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve \
        --num-gpus N --session-id {op_id} --no-auto-release) && \
        CUDA_VISIBLE_DEVICES=$CVD <cmd>
      E2E sweep: --num-gpus {tp*dp}  (one full model replica / vLLM world size)
      Parallel kernel eval: --num-gpus 1  (can run up to {gpu_count - tp*dp} concurrent jobs on remaining pool)
      If reserve fails (pool exhausted), retry with 30s backoff — see references/gpu-pool.md § Contention Handling.

    ## Stage 1 Baseline (DO NOT RE-RUN)
    Baseline E2E latency files (captured from the session base branch in Stage 1):
    - Per-batch-size JSON: {artifact_dir}/rounds/{CR}/sweeps/baseline/json/baseline_bs{N}.json
    - Summary table: {artifact_dir}/rounds/{CR}/constraints.md ("Baseline E2E latency" section)
    - Kernel breakdown: {artifact_dir}/rounds/{CR}/constraints.md ("Baseline Truth Snapshot" section)

    ## Precision Classification (from debate summary)
    Classification: {classification}  # "lossless" or "lossy" — determines Gate 5.1a tolerances

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
    4. Write and run the kernel correctness & speedup checks (see your agent definition § Kernel Validation): write your own kernel correctness test + CUDA-graph speedup bench, run them, write the two gate JSONs to validator_tests/. If 5.1a FAILS, fix and re-run before any sweep.
    5. Run E2E sweep per your agent definition § E2E Validation (ONE command handles 5.1b + 5.3a + 5.3b)
    6. If Gate 5.1b FAILS: classify failure (fixable vs fundamental), investigate root cause, try fixes.
       See your agent definition § Accuracy Failure Persistence. Do NOT report FAIL without exhausting options.
    7. Evaluate E2E results against min_e2e_improvement_pct threshold, write validation_results.md
    """
)

# Transcript monitor — team member (NOT a subagent). Needs SendMessage for DA interjections.
Agent(
    name=f"monitor-impl-champion-{op_id}",
    subagent_type="ammo-transcript-monitor",
    team_name=existing_team_name,    # Same team — enables SendMessage to champion and team-lead
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
    ALSO: enforce accuracy failure persistence — if the champion hits a Gate 5.1b
    failure, ensure they classify it, investigate root cause, and try fixes before
    reporting FAIL. See your agent definition § Accuracy Failure Persistence.
    See your agent definition § Stage-Specific Focus for the full list."""
)

# NOTE: The champion writes and runs the kernel correctness & speedup checks.
# The orchestrator does NOT participate in kernel validation.
```

### Champion-Run Validation

The champion runs all kernel validation:
1. Champion writes its own kernel correctness test + CUDA-graph speedup bench in the track worktree
2. Champion runs Gates 5.1a (kernel correctness) + 5.2 (kernel speedup), writing both gate JSONs to `validator_tests/`
3. If 5.1a FAIL: champion fixes and re-runs (no wasted E2E sweep)
4. If 5.1a PASS: champion runs sweep (5.1b + 5.3a + 5.3b)
5. Champion combines all gate results into `validation_results.md`
6. Champion reports `TRACK_COMPLETE` to orchestrator via SendMessage

The orchestrator reads `validation_results.md` for gate decisions but does not participate in the validation loop.

### Collaboration Timeline and Key Rules

The champion runs kernel validation before the E2E sweep. Key rules: the champion is the only agent that modifies source. See `references/impl-track-rules.md` for full constraints.

### Champion-Owned Validation (Kernel + E2E)

```
Kernel-Level (Champion):
  Gate 5.1a: Champion writes its OWN kernel correctness tests, records structured results
  Gate 5.2: Champion runs kernel speedup benchmark under CUDA graphs

E2E-Level (Champion, Opus):
  Gate 5.1b: Sweep --verify-correctness (GSM8K greedy decode)
  Gate 5.3a: Sweep --nsys-profile (kernel execution proof)
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Cross-checks Gate 5.1a against correctness_verdict.json
  Writes final validation_results.md with evidence chain
```

The champion performs both kernel-level and E2E-level verification.

### Handling Validation Failures and GATING_REQUIRED

When validation fails or a GATING_REQUIRED verdict is reported, follow the workflows in `references/impl-track-rules.md` § Validation Failures and § GATING_REQUIRED. Key principle: the champion re-runs ALL gates from scratch with fresh tests each cycle. For GATING_REQUIRED, one gating attempt per track — no nested gating.

## Result Collection

After all tracks complete, main reads each track's outputs:

1. `{artifact_dir}/rounds/{CR}/tracks/{op_id}/validation_results.md` — champion's final report (includes the kernel correctness & speedup results)
2. `{artifact_dir}/rounds/{CR}/tracks/{op_id}/validator_tests/` — the champion's kernel test scripts and gate results (dirname kept as a historical label)
3. `{artifact_dir}/rounds/{CR}/sweeps/opt/{op_id}/e2e_latency_results.json` — E2E sweep output (Gate 5.3b)
4. `state.json` field `campaign.rounds[$IDX].parallel_tracks.tracks[op_id]` — structured summary (`$IDX = campaign.current_round - 1`)

Main aggregates results to determine which candidates pass to Stage 6 integration.

### Incremental track state updates (MANDATORY)

`campaign.rounds[$IDX].parallel_tracks.tracks[op_id]` is the authoritative per-track status record (where `$IDX = campaign.current_round - 1`). The lead MUST atomically update it at each of these checkpoints — not only at end-of-stage:

1. **Track spawned** (worktree created, champion dispatched): write `{status: "IN_PROGRESS", classification, worktree_branch, description}`.
2. **Gate 5.1a result lands** in `rounds/{CR}/tracks/{op_id}/validator_tests/gate_5_1a_results.json`: merge `{correctness: bool, gate_5_1a: "PASS"|"FAIL"}`.
3. **Gate 5.2 result lands** in `rounds/{CR}/tracks/{op_id}/validator_tests/gate_5_2_results.json`: merge `{kernel_speedup, kernel_speedup_warm, kernel_speedup_cold, gate_5_2: "PASS"|"RETRY_WITH_CONTINGENCY"|"FAIL"}`.
4. **Gate 5.3b (E2E sweep) result lands** in `rounds/{CR}/sweeps/opt/{op_id}/e2e_latency_results.json`: merge `{e2e_speedup, per_bs_verdict, e2e_latency_opt}`. Extract `e2e_latency_opt` as a map keyed by batch_size string → `{avg, p50, p10, p25, p75, p90, p99}` from the opt label's per-row metrics (strip `_s` suffix: `avg_s`→`avg`, `p50_s`→`p50`, etc.).
5. **Track final verdict** (after champion's `validation_results.md`): set `{status: "PASS"|"FAIL"|"GATED_PASS"|"GATING_REQUIRED", verdict, fail_reason?, gating?}`.

Writes are idempotent — re-reading the gate file and merging is safe. Use atomic `.tmp` + `os.replace` to avoid torn reads. Keys missing in earlier checkpoints (e.g. `e2e_speedup` at step 2) MUST remain absent or null — do NOT fabricate placeholder values.

A track that reached gate 5.2 but has not yet run E2E sweep should have `status: "IN_PROGRESS"`, populated `correctness` + `kernel_speedup`, and `e2e_speedup: null`.

For `GATED_PASS` tracks, the `campaign.rounds[$IDX].parallel_tracks.tracks[op_id]` entry includes additional fields:
- `verdict`: `"GATED_PASS"`
- `per_bs_verdict`: per-BS verdict map (e.g., `{"1": "PASS", "8": "PASS", "32": "REGRESSED"}`)
- `gating`: gating metadata object (mechanism, env_var, dispatch_condition, crossover_threshold_bs, crossover_probing sub-object, pre_gating_results, post_gating_results)

### Pass Criteria (Tiered Verdict System)

See `references/validation-defaults.md` § Gate 5.3b for threshold values and per-BS verdict computation logic.

A track **ships** if its final status is `PASS` or `GATED_PASS` (after successful gating).

Additional requirements unchanged:
- Gate 5.1: Correctness — both sub-gates must pass:
  - 5.1a: The champion's kernel correctness tests pass
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

1. **Shut down remaining agents, then TeamDelete** the round team (`ammo-round-{round_id}-{model_short}-{hardware}`). First send `shutdown_request` to each implementation agent and monitor still on the roster and confirm each `shutdown_approved` — TeamDelete succeeds only once all members have shut down. This is the only TeamDelete in the round lifecycle, called after all implementation tracks complete.

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

## GPU Allocation

| Agent Type | GPU Access |
|-----------|-----------|
| Implementation champions | Pool access — kernel validation (--num-gpus 1) + E2E sweep (--num-gpus {tp*dp}, one model replica) + parallel kernel eval (--num-gpus 1, up to {gpu_count - tp*dp} concurrent jobs) |
| Transcript monitors | Read-only — no GPU access (transcript parsing + analysis only) |
