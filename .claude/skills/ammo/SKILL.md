---
name: ammo
description: Profile and optimize GPU kernels for vLLM inference on NVIDIA GPUs. Use when targeting specific (model, hardware, dtype, TP) deployments to improve latency. Triggers on requests to speed up any vLLM kernel.
---

# AMMO - Automated Model Micro-Optimizer

Profile and optimize **GPU kernels** for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

## Lead Role

You are the **lead orchestrator**. You scaffold, delegate, and gate — you never implement.

**Responsibilities**:
- Spawn subagents and assign work — never implement stages directly
- Manage state.json — read before each action, update at stage transitions
- Own all gate tasks (T3, T5, T7, T9, T11, T13, T19) — run verification scripts yourself
- Use SendMessage to communicate with teammates — text output is NOT visible to them

**Prohibited**:
- Do not write kernel code (CUDA or Triton) yourself
- Do not skip team creation "for efficiency"
- Do not implement directly — always delegate to subagents

## Invocation

User provides: model_id, hardware, dtype, tp.

Lead (you) scaffolds artifact directory, orchestrates the **campaign loop** — an iterative pipeline of 7 stages that repeats until diminishing returns. Each iteration (round) discovers, debates, and implements optimizations against the current bottleneck landscape.

Before calling `new_target.py`, determine target batch sizes. If the user specified batch sizes, pass them via `--batch-sizes`. If not, use the default `[1, 8, 32]`. Batch sizes define the decode buckets for all profiling and validation throughout the campaign.

```bash
python .claude/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP> \
  [--batch-sizes 1 8 32]
```

## Campaign Workflow

```
Stage 1: Baseline Capture          [main session + ammo-researcher subagent]   → constraints.md
Stage 2: Bottleneck Mining          [main session + ammo-researcher subagent]   → bottleneck_analysis.md (grounded data only)
Stage 3: Candidate Proposal + Debate [ephemeral agent team: N ammo-champion agents] → debate/summary.md
Stage 4+5: Parallel Tracks          [2-3 worktrees, each: ammo-implementer (implements + validates) + DA audit subagent]
Stage 6: Integration Validation     [main session direct]                       → SHIP or round-fail
Stage 7: Campaign Evaluation        [main session direct]                       → next round, campaign_complete, or campaign_exhausted
Stage 7b: Report Generation          [general-purpose subagent, background]       → REPORT.md (on campaign_complete or campaign_exhausted)
```

Stages 1-6 form the **inner loop** of a round. Stage 7 decides whether to iterate. No persistent team across stages. Agents spawn when needed and terminate when done.

The campaign continues until the **diminishing returns threshold** is met (see Campaign Loop below).

## Campaign Loop

Each round of the campaign follows this structure:

```
Round N:
  1. Profile (Stages 1-2) — re-profile against patched baseline (skip for round 1)
  2. Debate (Stage 3) — full adversarial debate; may overlap with round N-1 implementation
  3. Implement + Validate (Stages 4-5) — parallel worktree tracks
  4. Integrate (Stage 6) — ship or round-fail
  5. Campaign Evaluation (Stage 7) — diminishing returns check → next round or stop
```

### Async Pipeline: Debate Overlaps Implementation

While round N implementers work on debate winners:

1. Orchestrator starts round N+1 debate from existing bottleneck data.
2. New debate follows the full adversarial protocol — no lighter screening.
3. Debate winners go to `campaign.pending_queue`, NOT to implementation yet.

**If a round N candidate ships** (triggers re-profile): Let the in-progress debate finish. Re-validate queued winners against new profiling data (see below). Discard stale candidates.

**If round N completes without any ship**: Queued winners proceed to implementation immediately.

### Re-validation After Re-profiling

When the bottleneck landscape shifts (because a candidate shipped), queued candidates may be stale. For each candidate in `campaign.pending_queue`:

1. Check if the target kernel still appears in the updated `bottleneck_analysis.md`.
2. Recalculate expected E2E impact using the new f-values.
3. If `new_f × kernel_speedup < 1%` E2E improvement: discard.
4. If still viable: proceed to implementation in the next available slot.

This is a feasibility recheck, NOT a full re-debate — only the f-value has changed.

### Diminishing Returns

After each round's integration:

1. Read the top bottleneck's share of total decode latency from profiling data.
2. Compare against `campaign.diminishing_returns_threshold_pct` (default: 0.5%).
3. If below threshold: no single kernel optimization can yield meaningful E2E gains → **stop**.

**After SHIP**: Re-profile first (bottleneck landscape shifted), then check the NEW top bottleneck.
**After EXHAUSTED**: Check threshold against EXISTING profiling data (no re-profile needed).

### Campaign State Transitions

```
active → (threshold not met) → active (next round)
active → (threshold met after ship) → campaign_complete
active → (threshold met after exhaust) → campaign_exhausted
active → (user requests pause) → paused
paused → (user requests resume) → active
```

### In-Flight Tracks During Re-profiling

When a candidate ships and triggers re-profiling, other tracks from the same round may still be running:

1. Let all in-flight implementations complete (do NOT terminate).
2. Validate against the ORIGINAL round's baseline (not the new one).
3. If they pass: they also ship (additional cumulative gain).
4. Record all results in the current round's `campaign.rounds` entry.
5. Next round starts only after all current-round tracks complete.

## Orchestration Model

### Stages 1-2: Main Session + Subagents

- Lead (you) invokes ammo-researcher as a subagent via Task tool for profiling, source analysis, bottleneck mining.
- Lead delegates profiling and E2E baseline capture to ammo-researcher (who uses the sweep script with `--nsys-profile --labels baseline`). Use `--labels baseline` to avoid running a meaningless opt sweep (opt_env is still a placeholder at this stage). The lead does NOT duplicate work assigned to the ammo-researcher (i.e. run benchmarks, extract profiling results).
- No TeamCreate. No persistent agents. Subagent returns results directly.
- Lead runs gates (`verify_phase1_baseline.py`, Stage 2 review) between stages.
      
**Profiling strategy selection (lead decides BEFORE dispatching researcher)**:
For TP > 1 or models > 10B params, the lead should instruct the researcher to use two-step delimited capture. The researcher handles this automatically when using the sweep script with `--nsys-profile` (it sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `--trace-fork-before-exec=true`, and `--capture-range=cudaProfilerApi`). See `references/nsys-profiling-guide.md` §3.1B for background on why full-run capture hangs on multi-GPU models.

When `--nsys-profile` is used, the sweep script automatically restricts `cudagraph_capture_sizes` to match `workload.batch_sizes` from target.json. This reduces the CUDA graph capture surface from ~50 default sizes to only the profiled batch sizes, mitigating `--cuda-graph-trace=node` replay hangs. This is NOT a parity violation — the profiled sizes are exact matches in vLLM's default capture list, so the graphs are identical to production.

### Stage 3: Candidate Proposal + Adversarial Debate

- **TeamCreate**: `ammo-debate-{model_short}-{hardware}`
- Spawn 2-4 ammo-champion agents. Each reads the grounded bottleneck_analysis.md independently.
- **Delegation**: If `state.json` has `debate.delegation.enabled: true`, also spawn Sonnet delegate agents alongside champions (1-N per champion, configurable via `delegates_per_champion`). Champions direct delegates via SendMessage for research and micro-experiments. See `orchestration/debate-protocol.md` § Delegation.
- **Phase 0 (Proposals)**: Each champion independently proposes 1-2 optimization candidates with micro-experiment-backed feasibility math. Champions derive candidates from the profiling data — NOT from pre-scored candidate lists. With delegation, champions may direct delegates to extract profiling data and run roofline calculations.
- **Debate rounds**: Champions argue for their proposals, critique others, rebut. See `orchestration/debate-protocol.md`.
- Main session selects 2-3 winners using scoring rubric (`references/debate-scoring-rubric.md`).
- **TeamDelete** after selection (shuts down all champions AND delegates).
- **Debate is always mandatory.** If all champions independently converge on the same candidate in Phase 0 with micro-experiment evidence, the lead may shorten to 1 debate round instead of 2.

### Stages 4-5: Parallel Worktree Tracks

Execute these steps **in order**:

1. **Spawn implementers**: Per track, spawn ammo-implementer as a **subagent** with `isolation: worktree` (NOT as a teammate in a team). This ensures the Stop hook (DA) fires on completion. Do NOT use `team_name` when spawning implementers. The implementer handles BOTH implementation AND validation (correctness, kernel benchmarks, E2E). No separate validator agent. GPU assignment: kernel benchmarks parallel on separate GPUs, E2E sequential via lock.

2. **Launch async debate for round N+1** (MANDATORY for round 2+, skip for round 1): Immediately after spawning implementers, create a new debate team from existing bottleneck data. Follow the full adversarial protocol (Stage 3). Winners go to `campaign.pending_queue` — do NOT send them to implementation yet. Set `debate.async_round_started: true` in state.json when you launch this.

3. **Monitor and gate**: While implementers and async debate run, actively monitor progress. As each implementer completes, run its compilation gate (T9) and update state.json. Do NOT stop or go idle until all implementers have returned results AND the async debate (if launched) has completed.

See `orchestration/parallel-tracks.md`.

### Stage 6: Integration Validation

- If multiple candidates pass and target different components: cherry-pick both, re-run E2E.
- If same component: pick best E2E.
- If none pass: round EXHAUSTED (not campaign-level — campaign evaluates in Stage 7).
- See `orchestration/integration-logic.md`.

### Stage 7: Campaign Evaluation

- After integration: record round results, check diminishing returns, decide next action.
- See Campaign Loop section above for the full evaluation protocol.
- **Hook-enforced**: Stop hook blocks session end while campaign is active.

### Stage 7b: Report Generation

When the campaign ends (`campaign_complete` or `campaign_exhausted`), spawn a general-purpose subagent in the background to generate the optimization report. The subagent reads `.claude/skills/ammo/report/SKILL.md` and follows its instructions, passing the artifact directory path. This produces `{artifact_dir}/REPORT.md` with supporting charts in `{artifact_dir}/report_assets/`.

The report subagent runs as a background task — the orchestrator does not wait for it before declaring the campaign done. No GPU access is needed; the subagent reads existing campaign artifacts (state.json, constraints.md, bottleneck_analysis.md, debate artifacts, validation results, benchmark JSONs) and writes the report.

## Task Graph

```
=== Round N Inner Loop (Stages 1-6) ===

T1:  Scaffold artifact directory                          [main]
T2:  Baseline capture + constraints.md                    [ammo-researcher subagent]    <- T1
T3:  GATE: verify_phase1_baseline.py                      [main]                        <- T2
T4:  Bottleneck mining (grounded data only)                [ammo-researcher subagent]    <- T3
T5:  GATE: Stage 2 review (no ungrounded estimates)       [main]                        <- T4
T6:  Champion proposals + debate (TeamCreate -> Phase 0 [+delegates if enabled] -> rounds -> selection) [main + debate team] <- T5
T7:  GATE: Debate winner selection (proposals + summary.md exist) [main]                <- T6

  +- Per winning candidate (parallel) -----------------------------------------+
  | T8_{id}: Implement + validate (correctness+kernel+E2E) [ammo-implementer] <- T7   |
  |          (frontmatter Stop hook = DA: Amdahl check, baseline, parity, cross-track) |
  | T9_{id}: GATE: compilation check                       [main]              <- T8   |
  | T10_{id}: State update                                 [main]              <- T9   |
  +-----------------------------------------------------------------------------+

T11: GATE: All tracks have results                        [main]               <- all T10
T12: Integration validation (if multiple pass)            [main]               <- T11
T13: Round decision (SHIP / round-EXHAUSTED)              [main]               <- T12

=== Campaign Loop (Stage 7) ===

T14: Record round in campaign.rounds                      [main]               <- T13
T15: Campaign evaluation                                  [main]               <- T14
  IF SHIP:
    T16: Re-profile (baseline capture on patched code)    [ammo-researcher]    <- T15
    T17: Bottleneck mining on new baseline                [ammo-researcher]    <- T16
    T18: Diminishing returns check                        [main]               <- T17
      IF below threshold: CAMPAIGN COMPLETE
      ELSE: Invalidate stale queue → new Round (T6 debate → ...)
  IF round-EXHAUSTED:
    T16b: Diminishing returns check (on existing profile) [main]               <- T15
      IF below threshold: CAMPAIGN EXHAUSTED
      ELSE: new debate round from existing data (→ T6)
T19: GATE: campaign evaluation                            [main]               <- T15..T18
T20: Generate optimization report                         [general-purpose subagent, background] <- T19 (campaign_complete or campaign_exhausted)

=== Async Pipeline (during Stages 4-5) ===

T_async: Next-round debate                                [main + debate team]  <- T7
         (MANDATORY after T7 completes; overlaps T8-T10)
```

## Non-Negotiables (BLOCKING)

These are NOT advisory. Violation blocks stage progression.

1. **Production parity**: CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `TORCH_COMPILE_DISABLE=1`, `--enforce-eager`, `VLLM_TORCH_COMPILE_LEVEL=0`. *(Reminded by `ammo-pretool-guard.sh` PreToolUse hook — warns but does not block)* Note: restricting `cudagraph_capture_sizes` to match profiled batch sizes during nsys capture is acceptable and does not violate production parity — these sizes are exact matches in the default capture list.
2. **vLLM baseline**: Compare against production kernel, NOT naive PyTorch.
3. **Numerical correctness**: `torch.allclose()` is mandatory in every correctness test.
4. **GPU sequencing**: E2E benchmarks sequential via GPU lock. Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements. *(Reminded by `ammo-pretool-guard.sh` — warns on raw `vllm bench latency`)*
5. **Full-model E2E**: Do not skip because "weights aren't available" — download them.
6. **E2E delta math**: `E2E_improvement ~ f x kernel_speedup`, where `f` = component share of total latency. If `f` is small, large kernel wins yield small E2E gains — this is expected, not a bug.
7. **Custom kernel mandate**: Stage 3 proposals MUST involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code. Config-only, flag-flipping, and parameter-tuning proposals are rejected outright in the Phase 0 eligibility gate.

## Hook Enforcement

Hooks in `.claude/settings.local.json` enforce the campaign protocol mechanically:

| Hook Event | Script | Purpose |
|------------|--------|---------|
| **Stop** | `ammo-stop-guard.sh` | Blocks session end if campaign is active (one-shot: blocks once, then allows) |
| **PreToolUse** (Bash) | `ammo-pretool-guard.sh` | Warns on `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, raw `vllm bench latency` (does not block) |
| **PreCompact** | `ammo-precompact.sh` | Saves campaign state checkpoint before compaction |
| **SessionStart** | `ammo-postcompact.sh` | Injects resume context after compaction |
| **WorktreeCreate** | `worktree-create-with-build.sh` | Sets up build environment in new worktrees |
| **WorktreeRemove** | `worktree-remove-cleanup.sh` | Cleans up worktree resources |

Subagent-level hooks (frontmatter in agent definitions):
- **ammo-researcher** Stop → DA checks for ungrounded claims
- **ammo-implementer** Stop → DA checks validation completeness, Amdahl's Law, baseline citation
- **ammo-champion** TeammateIdle → DA checks custom kernel mandate, micro-experiment evidence

## State Management

`state.json` in artifact directory tracks stage, status, debate, parallel tracks, and integration.

**Session ID**: The lead SHOULD record the session ID in state.json at campaign start: `"session_id": "<uuid>"`. This enables the eval pipeline to extract ground-truth timing and agent cost data from session logs automatically.

**Stage timestamps and agent costs**: Auto-extracted from session logs by `parse_session_logs.py` during eval. No manual recording needed — the `stage_timestamps` and `agent_costs` fields in state.json are populated by the eval pipeline, not by the lead.

```json
{
  "target": {"model_id": "...", "hardware": "...", "dtype": "...", "tp": 1},
  "stage": "1_baseline",
  "summary": "Initialized.",
  "gpu_resources": {"gpu_count": 1, "gpu_model": "...", "memory_total_gib": 0, "cuda_visible_devices": "0"},
  "debate": {
    "team_name": null,
    "candidates": [],
    "rounds_completed": 0,
    "max_rounds": 4,
    "selected_winners": [],
    "selection_rationale": null,
    "delegation": {
      "enabled": false,
      "delegates_per_champion": 1,
      "champion_delegate_mapping": {},
      "delegate_results": {}
    },
    "async_round_started": false
  },
  "stage_timestamps": {   /* lead records ISO timestamps at stage transitions */
    "1_baseline": {"started_at": null, "completed_at": null},
    "2_bottleneck_mining": {"started_at": null, "completed_at": null},
    "3_debate": {"started_at": null, "completed_at": null},
    "4_5_parallel_tracks": {"started_at": null, "completed_at": null},
    "6_integration": {"started_at": null, "completed_at": null},
    "7_campaign_eval": {"started_at": null, "completed_at": null}
  },
  "session_id": null,     /* lead records session UUID at campaign start */
  "agent_costs": [],      /* auto-populated by eval pipeline from session logs */
  "parallel_tracks": {},  /* per-track: { status, correctness, kernel_speedup, e2e_speedup, validation_results_path } */
  "integration": {
    "status": "pending",
    "passing_candidates": [],
    "conflict_analysis": null,
    "combined_patch_branch": null,
    "combined_e2e_result": null,
    "final_decision": null
  },
  "campaign": {
    "status": "active",                          /* active | paused | campaign_complete | campaign_exhausted */
    "current_round": 1,
    "diminishing_returns_threshold_pct": 0.5,    /* stop when top bottleneck < this % */
    "cumulative_e2e_speedup": 1.0,               /* multiplicative across shipped rounds */
    "rounds": [],                                /* per-round history (see schema below) */
    "shipped_optimizations": [],                 /* op_ids that shipped across all rounds */
    "pending_queue": []                          /* candidates from async debate awaiting implementation */
  }
}
```

Each entry in `campaign.rounds`:
```json
{
  "round_id": 1,
  "profiling_baseline_path": "runs/baseline_round_1/",
  "top_bottleneck_share_pct": 15.2,
  "debate_team_name": "ammo-debate-...",
  "selected_candidates": ["op001", "op002"],
  "implementation_results": {
    "op001": {"status": "PASSED", "e2e_speedup": 1.12},
    "op002": {"status": "FAILED", "reason": "correctness"}
  },
  "shipped": ["op001"],
  "cumulative_speedup_after": 1.12
}
```

### Stage Values

`1_baseline` -> `2_bottleneck_mining` -> `3_debate` -> `4_5_parallel_tracks` -> `6_integration` -> `7_campaign_eval` -> {next round | `campaign_complete` | `campaign_exhausted`}

On `campaign_complete` or `campaign_exhausted`, report generation (T20) is spawned in the background before the session ends.

## Communication Patterns

- **Blocker escalation**: Subagent returns error -> lead investigates.
- **Debate moderation**: Lead broadcasts phase starts, champions message back on completion.
- **Critical stop**: Lead broadcasts to halt debate team if needed.
- **Shutdown**: Clean debate team termination via `shutdown_request` -> TeamDelete.

## Helper Scripts

Run, don't modify:

- `scripts/new_target.py` — Scaffold artifact directory
- `scripts/collect_env.py` — Capture environment
- `scripts/verify_phase1_baseline.py` — Stage 1->2 gate
- `scripts/verify_validation_gates.py` — Stage 5 gate (supports `--track` for per-track validation)
- `scripts/run_vllm_bench_latency_sweep.py` — Batch E2E benchmark runner (GPU-locked). Supports `--labels baseline` for baseline-only sweeps (Stage 1), `--labels opt` for opt-only (Stage 5), or `--labels baseline,opt` (default, A/B comparison). Also supports `workload_matrix` for multi-dimensional sweeps and `--nsys-profile` for per-bucket nsys traces without model reload. **WARNING**: If `opt_env` in target.json still contains placeholder keys (e.g., `<ENABLE_FLAG>`), the script will fail fast — update `opt_env` before running with `--labels opt`.
- `scripts/generate_validation_report.py` — Structured reporting

## References

| Topic | File |
|-------|------|
| Nsys profiling | `references/nsys-profiling-guide.md` |
| Validation gates | `references/validation-defaults.md` |
| CUDA graph safety | `references/cudagraph-safety.md` |
| E2E latency | `references/e2e-latency-guide.md` |
| E2E delta math | `references/e2e-delta-math.md` |
| GPU hardware specs | `references/gpu-configs.md` |
| Optimization techniques | `references/optimization-techniques.md` |
| Fusion feasibility | `references/fusion-feasibility-heuristics.md` |
| Code templates | `references/code-templates.md` |
| Debate scoring | `references/debate-scoring-rubric.md` |
| Validator troubleshooting | `references/validator-troubleshooting.md` |
| DA audit checklist | `references/da-audit-checklist.md` |

## Orchestration Docs

| Topic | File |
|-------|------|
| Debate protocol | `orchestration/debate-protocol.md` |
| Parallel tracks | `orchestration/parallel-tracks.md` |
| Integration logic | `orchestration/integration-logic.md` |

## Escalation Protocol

When a subagent returns an error or a gate fails:

| Severity | Action |
|----------|--------|
| **critical** | STOP. Broadcast halt if debate team active. |
| **major** | Investigate, adjust constraints, re-run subagent. |
| **minor** | Document and continue. |

Save blocker details to `{artifact_dir}/blockers/{stage}_{date}.md`.

## Resume Protocol

After interruption or compaction:

1. Read this skill file (you are the LEAD — delegate, don't implement).
2. Read `state.json` from artifact directory.
3. Check `campaign.status` and `stage` to determine where you are.
4. If Stage 3 debate active: check debate artifacts in `debate/` or `debate/campaign_round_N/`.
5. If Stages 4-5 active: check `parallel_tracks` in `state.json` for worktree paths and status. Also check `debate.async_round_started` — if `false` and round > 1, launch the async debate before resuming monitor/gate duties.
6. Resume from last completed gate.
7. Read `campaign.current_round`, `campaign.pending_queue`. Check if an async debate was in progress.
8. If `campaign` key is missing (legacy state.json): treat as round 1 — initialize the campaign object.
9. The Stop hook will block session end while campaign is active — either complete the current stage or set `campaign.status` to `"paused"`.
10. If `campaign.status` is `campaign_complete` or `campaign_exhausted` but no `REPORT.md` exists, spawn the report generation subagent (T20).

## Quick Start Examples

**Example 1**: `User: "Use ammo for Qwen3-30B-A3B on L40S TP=1"` ->

1. Scaffold artifact directory (campaign initialized).
2. Round 1: invoke ammo-researcher subagent for baseline + bottleneck mining.
3. Run gates, spawn debate team for top candidates.
4. Select winners, create parallel worktree tracks.
5. Implement + validate in parallel. Start async debate for round 2.
6. Integration if multiple pass → SHIP or round-EXHAUSTED.
7. Campaign evaluation: record round, check diminishing returns.
8. If above threshold: re-profile → new round. Repeat until threshold met.

**Example 2**: Resume -> Read `state.json`, check `campaign.current_round` and active stage, resume from last gate.

**Example 3**: Mid-campaign resume after compaction -> SessionStart hook injects campaign context (round, status, cumulative speedup). Read `state.json`, resume current round.
