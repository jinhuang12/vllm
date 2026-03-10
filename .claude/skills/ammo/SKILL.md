---
name: ammo
description: Profile and optimize GPU kernels for vLLM inference on NVIDIA GPUs. Use when targeting specific (model, hardware, dtype, TP) deployments to improve latency. Triggers on requests to speed up any vLLM kernel.
---

# AMMO - Automated Model Micro-Optimizer

Profile and optimize **GPU kernels** for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

## Invocation

User provides: model_id, hardware, dtype, tp, component (or "auto").

Lead (you) scaffolds artifact directory, orchestrates the **campaign loop** — an iterative pipeline of 7 stages that repeats until diminishing returns. Each iteration (round) discovers, debates, and implements optimizations against the current bottleneck landscape.

```bash
python .claude/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP>
```

## Campaign Workflow

```
Stage 1: Baseline Capture          [main session + ammo-researcher subagent]   → constraints.md
Stage 2: Bottleneck Mining          [main session + ammo-researcher subagent]   → bottleneck_analysis.md (grounded data only)
Stage 3: Candidate Proposal + Debate [ephemeral agent team: N ammo-champion agents] → debate/summary.md
Stage 4+5: Parallel Tracks          [2-3 worktrees, each: ammo-implementer (implements + validates) + DA audit subagent]
Stage 6: Integration Validation     [main session direct]                       → SHIP or round-fail
Stage 7: Campaign Evaluation        [main session direct]                       → next round, campaign_complete, or campaign_exhausted
```

Stages 1-6 form the **inner loop** of a round. Stage 7 decides whether to iterate. No persistent team across stages. Agents spawn when needed and terminate when done.

The campaign continues until the **diminishing returns threshold** is met: the top remaining bottleneck is less than `campaign.diminishing_returns_threshold_pct` (default 3%) of total decode latency. See `orchestration/campaign-loop.md` for the full protocol.

## Orchestration Model

### Stages 1-2: Main Session + Subagents

- Lead (you) invokes ammo-researcher as a subagent via Task tool for profiling, source analysis, bottleneck mining.
- For multi-bucket nsys profiling, use `--nsys-profile` on the sweep script instead of manual per-batch-size nsys invocations (see `references/nsys-profiling-guide.md` §3.5).
- No TeamCreate. No persistent agents. Subagent returns results directly.
- Lead runs gates (`verify_phase1_baseline.py`, Stage 2 review) between stages.
      
**Profiling strategy selection (lead decides BEFORE dispatching researcher)**:                                                                                                                                                                                                                                      
For TP > 1 or models > 10B params, the lead should instruct the researcher to use two-step delimited nsys capture (pre-warm + `--capture-range=cudaProfilerApi`). Full-run nsys with `--cuda-graph-trace=node` will hang on multi-GPU models because it traces torch.compile and CUDA graph capture across all worker processes. See `references/nsys-profiling-guide.md` §3.1B and §3.3 for the exact commands. The lead may also run the E2E baseline benchmark and nsys pre-warm step itself (in parallel with source analysis by the researcher) to save time.

### Stage 3: Candidate Proposal + Adversarial Debate

- **TeamCreate**: `ammo-debate-{component}-{model_short}-{hardware}`
- Spawn 2-4 ammo-champion agents. Each reads the grounded bottleneck_analysis.md independently.
- **Phase 0 (Proposals)**: Each champion independently proposes 1-2 optimization candidates with micro-experiment-backed feasibility math. Champions derive candidates from the profiling data — NOT from pre-scored candidate lists.
- **Debate rounds**: Champions argue for their proposals, critique others, rebut. See `orchestration/debate-protocol.md`.
- Main session selects 2-3 winners using scoring rubric (`references/debate-scoring-rubric.md`).
- **TeamDelete** after selection.
- **Debate is always mandatory.** If all champions independently converge on the same candidate in Phase 0 with micro-experiment evidence, the lead may shorten to 1 debate round instead of 2.

### Stages 4-5: Parallel Worktree Tracks

- Per track: spawn ammo-implementer as a **subagent** with `isolation: worktree` (NOT as a teammate in a team). This ensures the Stop hook (DA) fires on completion. Do NOT use `team_name` when spawning implementers.
- The implementer handles BOTH implementation AND validation (correctness, kernel benchmarks, E2E). No separate validator agent.
- GPU assignment: kernel benchmarks parallel on separate GPUs, E2E sequential via lock.
- See `orchestration/parallel-tracks.md`.

### Stage 6: Integration Validation

- If multiple candidates pass and target different components: cherry-pick both, re-run E2E.
- If same component: pick best E2E.
- If none pass: round EXHAUSTED (not campaign-level — campaign evaluates in Stage 7).
- See `orchestration/integration-logic.md`.

### Stage 7: Campaign Evaluation

- After integration: record round results in `campaign.rounds`.
- If SHIP: update `campaign.cumulative_e2e_speedup`, trigger re-profiling (Stages 1-2 on patched code).
- Check diminishing returns: top bottleneck < `campaign.diminishing_returns_threshold_pct`?
  - Yes → `campaign.status = "campaign_complete"`. Done.
  - No → increment `campaign.current_round`, invalidate stale queue, enter next round.
- If round EXHAUSTED: check threshold against existing profile. Above → new debate. Below → `campaign_exhausted`.
- **Hook-enforced**: `GATE: campaign evaluation` blocked until round results and diminishing returns check are recorded.
- See `orchestration/campaign-loop.md` and `orchestration/integration-logic.md` § Campaign Loop Transition.

### Async Pipeline: Debate Overlaps Implementation

While Stages 4-5 implementers work on round N winners, the orchestrator may start round N+1 debate from existing bottleneck data to build a candidate queue:

1. New debate follows the full adversarial protocol (no lighter screening).
2. Winners are placed in `campaign.pending_queue`, NOT sent to implementation yet.
3. If a round N candidate ships (triggers re-profile): let the debate finish, then re-validate winners against the new profile. Discard stale candidates.
4. If round N completes without any ship: queued winners proceed to implementation immediately.

See `orchestration/campaign-loop.md` § Async Pipeline for details.

## Task Graph

```
=== Round N Inner Loop (Stages 1-6) ===

T1:  Scaffold artifact directory                          [main]
T2:  Baseline capture + constraints.md                    [ammo-researcher subagent]    <- T1
T3:  GATE: verify_phase1_baseline.py                      [main]                        <- T2
T4:  Bottleneck mining (grounded data only)                [ammo-researcher subagent]    <- T3
T5:  GATE: Stage 2 review (no ungrounded estimates)       [main]                        <- T4
T6:  Champion proposals + debate (TeamCreate -> Phase 0 -> rounds -> selection) [main + debate team] <- T5
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

=== Async Pipeline (during Stages 4-5) ===

T_async: Next-round debate                                [main + debate team]
         (overlaps T8-T10; winners queued in campaign.pending_queue)
```

## Non-Negotiables (BLOCKING)

These are NOT advisory. Violation blocks stage progression.

1. **Production parity**: CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `TORCH_COMPILE_DISABLE=1`, `--enforce-eager`, `VLLM_TORCH_COMPILE_LEVEL=0`.
2. **vLLM baseline**: Compare against production kernel, NOT naive PyTorch.
3. **Numerical correctness**: `torch.allclose()` is mandatory in every correctness test.
4. **GPU sequencing**: E2E benchmarks sequential via GPU lock. Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements.
5. **Full-model E2E**: Do not skip because "weights aren't available" — download them.
6. **E2E delta math**: `E2E_improvement ~ f x kernel_speedup`, where `f` = component share of total latency. If `f` is small, large kernel wins yield small E2E gains — this is expected, not a bug.
7. **Custom kernel mandate**: Stage 3 proposals MUST involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code. Config-only, flag-flipping, and parameter-tuning proposals are rejected outright in the Phase 0 eligibility gate.

## State Management

`state.json` in artifact directory tracks stage, status, debate, parallel tracks, and integration:

```json
{
  "target": {"model_id": "...", "hardware": "...", "dtype": "...", "tp": 1, "component": "auto"},
  "stage": "1_baseline",
  "status": "in_progress",
  "current_opportunity_id": null,
  "max_attempts": 3,
  "opportunity_attempts": [],
  "route_decision": {},
  "verification_run": {"stage1": null, "validation": null},
  "last_update": "2026-02-24",
  "summary": "Initialized.",
  "team": {"name": null, "members": []},
  "gpu_resources": {"gpu_count": 1, "gpu_model": "...", "memory_total_gib": 0, "cuda_visible_devices": "0"},
  "debate": {
    "team_name": null,
    "candidates": [],
    "rounds_completed": 0,
    "max_rounds": 4,
    "selected_winners": [],
    "selection_rationale": null
  },
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
    "diminishing_returns_threshold_pct": 3,      /* stop when top bottleneck < this % */
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
- `scripts/run_vllm_bench_latency_sweep.py` — Batch E2E benchmark runner (GPU-locked). Supports `workload_matrix` for multi-dimensional (input_len x output_len x batch_size) sweeps and `--nsys-profile` for per-bucket nsys traces without model reload.
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
| Campaign loop | `orchestration/campaign-loop.md` |
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

1. Read this skill file.
2. Read `state.json` from artifact directory.
3. Check which stage is active.
4. If Stage 3 debate active: read debate team config, check debate artifacts.
5. If Stages 4-5 active: check `parallel_tracks` in `state.json` for worktree paths and status. DA audit is embedded in each agent's frontmatter Stop hook — no separate spawn needed.
6. Resume from last completed gate.
7. If campaign is active: read `campaign.current_round`, `campaign.status`, and `campaign.pending_queue`. Check if an async debate was in progress.
8. If `campaign` key is missing (legacy state.json): treat as round 1 of a new campaign — initialize the campaign object from existing state.

## Quick Start Examples

**Example 1**: `User: "Use ammo for Qwen3-30B-A3B on L40S TP=1"` ->

1. Scaffold artifact directory (campaign initialized).
2. Round 1: invoke ammo-researcher subagent for baseline + bottleneck mining.
3. Run gates, spawn debate team for top candidates.
4. Select winners, create parallel worktree tracks.
5. Implement + validate in parallel. Optionally start async debate for round 2.
6. Integration if multiple pass → SHIP or round-EXHAUSTED.
7. Campaign evaluation: record round, check diminishing returns.
8. If above threshold: re-profile → new round. Repeat until threshold met.

**Example 2**: Resume -> Read `state.json`, check `campaign.current_round` and active stage, resume from last gate.

**Example 3**: Mid-campaign resume after compaction -> SessionStart hook injects campaign context (round, status, cumulative speedup). Read `state.json`, resume current round.
