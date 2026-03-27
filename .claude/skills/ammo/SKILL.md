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
- Do not override state.json configuration (thresholds, etc.) without user approval

**Configuration Fidelity**: The orchestrator MUST respect all `state.json` configuration flags as written. If the orchestrator believes a configuration should change (e.g., adjusting thresholds for a round), it MUST propose the change to the user and wait for approval before acting. The orchestrator does NOT have discretion to override configuration settings — it scaffolds, delegates, and gates, not makes policy.

## Invocation

User provides: model_id, hardware, dtype, tp.

Lead (you) scaffolds artifact directory, orchestrates the **campaign loop** — an iterative pipeline of 7 stages that continues until `f < min_e2e_improvement_pct` (mechanical threshold — see Campaign Stop Condition). Each iteration (round) discovers, debates, and implements optimizations against the current bottleneck landscape.

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
Stage 3: Candidate Proposal + Debate [round team: N ammo-champion agents]            → debate/summary.md
Stage 4+5: Parallel Tracks          [round team reused: per-track impl-champion + impl-validator pairs + DA audit]
  + Round N+1 Debate (overlapped)   [round team: N ammo-champion agents, if round 2+] → debate/campaign_round_{N+1}/summary.md
Stage 6: Integration Validation     [main session direct]                       → SHIP or round-fail
Stage 7: Campaign Evaluation        [main session direct]                       → next round, campaign_complete, or campaign_exhausted
Stage 7b: Report Generation          [general-purpose subagent, background]       → REPORT.md (on campaign_complete or campaign_exhausted)
```

Stages 1-6 form the **inner loop** of a round. Stage 7 executes the mechanical threshold check (`f` vs `min_e2e_improvement_pct`) → continue or stop. A single round-scoped team persists from Stage 3 through Stage 5; it is created at debate start and deleted after all implementation tracks complete. During Stages 4-5 (round 2+), the orchestrator may launch the next round's debate concurrently -- debate champions are spawned into the same round team alongside implementation agents (see Overlapped Debate below).

The campaign continues until the top bottleneck's share of decode latency (`f`) falls below `min_e2e_improvement_pct` — at that point, Amdahl's Law guarantees no single-component optimization can yield the minimum improvement (see Campaign Stop Condition below).

## Campaign Loop

Each round of the campaign follows this structure:

```
Round N:
  1. Profile (Stages 1-2) — re-profile against patched baseline (skip for round 1)
  2. Debate (Stage 3) — full adversarial debate
  3. Implement + Validate (Stages 4-5) — parallel worktree tracks
     + Overlapped: Round N+1 debate starts during implementation (round 2+, same team)
  4. Integrate (Stage 6) — ship or round-fail
  5. Campaign Evaluation (Stage 7) — E2E threshold check → next round or stop
```

### Campaign Stop Condition (MECHANICAL — NO DISCRETION)

After each round's integration:

1. Read the top bottleneck's share of total decode latency (`f`) from profiling data.
2. If `f < min_e2e_improvement_pct` (default: 1.0%): even complete elimination of that component cannot yield the minimum E2E improvement (Amdahl's Law) → **stop**.
3. If `f >= min_e2e_improvement_pct`: **continue unconditionally**. Start a new round.
4. No deflation applied — Amdahl's ceiling is a physical bound, not an estimate. See `references/validation-defaults.md` § Minimum E2E Improvement Threshold.

**The orchestrator has ZERO discretion here.** The following are NOT valid reasons to stop or ask the user:
- "All Triton approaches have been exhausted" → pivot to CUDA C++/CUTLASS/DeepGEMM
- "The remaining bottleneck is near its physical ceiling" → the ceiling is captured by `f`; if `f >= threshold`, the math says there's room
- "A new round is unlikely to find better candidates" → the orchestrator cannot predict debate outcomes
- "The campaign has been running for many rounds" → round count is not a stop criterion
- "Implementation complexity is increasing" → complexity is scored in debate, not a campaign-level gate

**Technology pivot guidance**: When a round exhausts one technology class (e.g., Triton), the next round's debate prompt MUST explicitly direct champions to explore alternative technology classes. This is automatic, not a user decision. The pivot order is:

1. Triton (default, lowest complexity)
2. CUDA C++ with manual SMEM/register management
3. CUTLASS/DeepGEMM integration (if hardware supports)
4. PTX-level optimization (last resort)

**After SHIP**: Re-profile first (bottleneck landscape shifted), then check the NEW top bottleneck. If `f >= threshold`, start next round immediately.
**After EXHAUSTED**: Check threshold against EXISTING profiling data (no re-profile needed). If `f >= threshold`, start next round immediately with debate directed to unexplored approaches.

### Campaign State Transitions

```
active → (f >= threshold after ship) → active (re-profile → new round)
active → (f >= threshold after exhaust) → active (new round, pivot technology)
active → (f < threshold after ship) → campaign_complete
active → (f < threshold after exhaust) → campaign_exhausted
active → (user explicitly requests pause) → paused
paused → (user explicitly requests resume) → active
```

Note: there is NO transition from `active` to `campaign_complete` or `campaign_exhausted` based on orchestrator judgment. Only the mechanical `f < threshold` condition triggers termination.

### In-Flight Tracks During Re-profiling

When a candidate ships and triggers re-profiling, other tracks from the same round may still be running:

1. Let all in-flight implementations complete (do NOT terminate).
2. Validate against the ORIGINAL round's baseline (not the new one).
3. If they pass: they also ship (additional cumulative gain).
4. Record all results in the current round's `campaign.rounds` entry.
5. Next round starts only after all current-round tracks complete.

### Overlapped Debate (Round 2+ Only)

During Stages 4-5 of round N (N >= 2), the orchestrator launches the next round's debate concurrently with implementation. Debate champions are spawned into the **same round team** that contains implementation agents. This saves 35-70 minutes of wall-clock time per extra round.

**When to launch**: Immediately after spawning all implementation agents for round N. The debate uses the EXISTING bottleneck_analysis.md (from the most recent profiling) -- it does NOT wait for re-profiling.

**How it works**:
1. Spawn 2-4 ammo-champion agents into the existing round team.
2. Moderate debate using the standard protocol (Phase 0 -> rounds -> selection).
3. Interleave debate moderation with implementation monitoring: broadcast debate phase starts, then check for impl track completions, then wait for debate phase completions.
4. When debate finishes: score winners, shut down debate champions via `shutdown_request`. Record winners in `debate.next_round_overlap.selected_winners`.
5. When all implementation tracks complete: proceed to Stage 6 integration.
6. After integration: winners from the overlapped debate become the implementation candidates for round N+1 (subject to lazy invalidation -- see Campaign Loop Transition above).

**Communication boundaries** (ENFORCED):
- Debate champions communicate with: each other (via file artifacts), orchestrator (via SendMessage).
- Implementation agents communicate with: their paired validator (via SendMessage), orchestrator (via SendMessage).
- Debate champions MUST NOT message implementation agents. Implementation agents MUST NOT message debate champions.
- The orchestrator enforces this by not providing cross-workstream agent names in spawn prompts.

**Lazy invalidation**: After re-profiling in the next round, the orchestrator checks each overlapped-debate winner's f-value against the new profiling data. If `f_old >= 0.05` AND `|f_new - f_old| / f_old > 0.3`, the candidate is discarded (the target kernel's share shifted too much). If `f_old < 0.05`, skip invalidation for that candidate (the kernel is too small to measure f-shift reliably). Otherwise, the candidate proceeds to implementation. This replaces the old eager re-scoring protocol.

**GPU allocation**: Debate champions may run brief GPU micro-benchmarks using the pool (`--num-gpus 1`).

**If round N is round 1**: Do NOT launch overlapped debate. Round 1 has no prior profiling data for the next round's debate to use.

**If all implementation tracks complete before debate finishes**: Wait for debate to complete before proceeding to Stage 6. Do not terminate the debate. Exception: if the debate has not completed within 90 minutes of launch, the orchestrator should: (1) collect any completed debate artifacts, (2) shut down all debate champions via `shutdown_request`, (3) mark `debate.next_round_overlap.phase` as `null` and `active` as `false`, (4) proceed to Stage 6 with only the implementation results.

**If debate finishes before all implementation tracks complete**: Record winners. Continue monitoring implementation tracks.

**If campaign terminates during overlapped debate**: If the campaign transitions to `campaign_complete` or `campaign_exhausted` while `debate.next_round_overlap.active` is `true`, the orchestrator must: (1) shut down all overlapped debate champions via `shutdown_request`, (2) discard overlapped debate results, (3) clear `debate.next_round_overlap` to initial state (`active: false, phase: null, selected_winners: [], profiling_basis: null, f_values_at_proposal: {}`).

## Orchestration Model

### Stages 1-2: Main Session + Subagents

- Lead (you) invokes ammo-researcher as a subagent via Task tool for profiling, source analysis, bottleneck mining.
- Lead delegates profiling and E2E baseline capture to ammo-researcher (who uses the sweep script with `--nsys-profile --labels baseline`). Use `--labels baseline` to avoid running a meaningless opt sweep (opt_env is still a placeholder at this stage). The lead does NOT duplicate work assigned to the ammo-researcher (i.e. run benchmarks, extract profiling results).
- No TeamCreate. No persistent agents. Subagent returns results directly.
- Lead runs gates (`verify_phase1_baseline.py`, Stage 2 review) between stages.
      
**Profiling strategy selection (lead decides BEFORE dispatching researcher)**:
For TP > 1 or models > 10B params, the lead should instruct the researcher to use two-step delimited capture. The researcher handles this automatically when using the sweep script with `--nsys-profile` (it sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `--trace-fork-before-exec=true`, and `--capture-range=cudaProfilerApi`). See `references/nsys-profiling-guide.md` §3.1B for background on why full-run capture hangs on multi-GPU models.

When `--nsys-profile` is used, the sweep script automatically restricts `cudagraph_capture_sizes` to match `workload.batch_sizes` from target.json. This reduces the CUDA graph capture surface from ~50 default sizes to only the profiled batch sizes, mitigating `--cuda-graph-trace=node` replay hangs. This is NOT a parity violation — the profiled sizes are exact matches in vLLM's default capture list, so the graphs are identical to production.
The lead should also instruct the researcher to run `scripts/nsys_probe.py` first to estimate profiling cost and determine safe `--nsys-output-len` values. See `references/nsys-profiling-guide.md` §3.10.

### Stage 2b: Baseline ncu Sanity Check

After bottleneck mining completes, the lead instructs the ammo-researcher to run a baseline ncu sanity check on the **top-3 bottleneck kernels** (by f_decode). This catches pathological baselines (dispatch bugs, near-zero SM utilization) before champions begin debate.

**Command per kernel**:
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes.sum.per_second,smsp__inst_executed.sum \
    --kernel-name <baseline_kernel> python baseline_invocation.py
```

**Red flag thresholds** (any one triggers investigation before debate begins):
- SM utilization < 10% for non-trivial kernels (indicates dispatch bug — e.g., B200 n_groups=8 case showed 0.6%)
- Achieved DRAM BW < 20% of theoretical peak for BW-bound kernels
- Instruction count < 50% of expected for target shape

If a red flag fires, the researcher must investigate before the lead proceeds to Stage 3. Findings are appended to `bottleneck_analysis.md` and shared with all champions.

**Cost**: ~60 seconds per kernel, ~3-5 minutes total. GPU is already dedicated to profiling at this stage.

### Stage 3: Candidate Proposal + Adversarial Debate

- **TeamCreate**: `ammo-round-{round_id}-{model_short}-{hardware}` — this is the **round team**, created once and reused through Stages 4-5.
- Spawn 2-4 ammo-champion agents into the round team. Each reads the grounded bottleneck_analysis.md independently.
- **Monitor spawn (REQUIRED)**: After spawning each champion, spawn an `ammo-transcript-monitor` (Sonnet) as a **team member** (`team_name=round_team_name`, `run_in_background=True`) — one per champion. Monitors provide continuous adversarial oversight via transcript reading. See `orchestration/debate-protocol.md` § Monitor Spawn Pattern for the exact Agent() call.
- **Phase 0 (Proposals)**: Each champion independently proposes 1-2 optimization candidates with micro-experiment-backed feasibility math. Champions derive candidates from the profiling data — NOT from pre-scored candidate lists.
- **Debate rounds**: Champions argue for their proposals, critique others, rebut. See `orchestration/debate-protocol.md`.
- Main session selects 2-3 winners using scoring rubric (`references/debate-scoring-rubric.md`).
- **After selection**: Shut down debate champions via `shutdown_request` (they are no longer needed). The **round team persists** — implementation agents will be spawned into it in Stages 4-5. Do NOT call TeamDelete here.
- **Debate is always mandatory.** Minimum 2 debate rounds required — no convergence shortcut (see `orchestration/debate-protocol.md` § No Convergence Shortcut).

### Stages 4-5: Parallel Worktree Tracks (Adversarial Validation)

Each track uses a **champion + independent validator** pair to prevent reward hacking (observed: cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation). All implementation agents join the **existing round team** created in Stage 3 — no new TeamCreate calls.

Execute these steps **in order**:

1. **Spawn implementation agents into the round team**: Per winning candidate:
   - Spawn an `ammo-impl-champion` (Opus, `isolation: worktree`) into the existing round team.
   - **Monitor spawn (REQUIRED)**: Immediately spawn an `ammo-transcript-monitor` (Sonnet) as a **team member** (`team_name=round_team_name`, `run_in_background=True`) for that impl-champion. See `orchestration/debate-protocol.md` § Monitor Spawn Pattern.
   - **Do NOT spawn `ammo-impl-validator` here.** The validator is spawned later by the orchestrator when the champion sends a `VALIDATION_REQUEST` via SendMessage (see `orchestration/parallel-tracks.md` § Validation Spawn Protocol).
   - The champion implements; when ready, it sends `VALIDATION_REQUEST` to the orchestrator. The orchestrator spawns the validator, who independently writes its own correctness tests, benchmark scripts, and runs E2E sweeps. The validator dual-reports raw results to both champion and orchestrator.

2. **Launch overlapped debate (round 2+ only)**: If `campaign.current_round >= 2`, spawn 2-4 ammo-champion agents into the same round team. **Monitor spawn (REQUIRED)**: After spawning each debate champion, spawn an `ammo-transcript-monitor` as a team member in background. Follow the standard debate protocol while also monitoring implementation tracks. See "Overlapped Debate" section above. For round 1, skip this step.

3. **Monitor and gate**: While implementation agents run, actively monitor progress. Interleave debate moderation (if active) with implementation gating. As each champion completes (DA Stop hook passed), run its compilation gate (T9) and update state.json. Do NOT stop or go idle until all implementation agents have returned results AND the overlapped debate (if launched) has completed.

4. **TeamDelete after all tracks complete**: Once all implementation tracks have finished (passed or failed) and results are collected, AND the overlapped debate (if launched) has completed, call TeamDelete on the round team. This is the only TeamDelete in the round lifecycle.

See `orchestration/parallel-tracks.md` for the full team structure, phase-transition protocol, and three-layer verification model.

### Stage 6: Integration Validation

- If multiple candidates pass (PASS or GATED_PASS) and target different components: cherry-pick both, re-run E2E.
- If same component: pick best E2E.
- If GATED_PASS tracks produce merge conflicts during cherry-pick: spawn dedicated resolver agent (`.claude/agents/ammo-resolver.md`) + DA reviewer.
- If none pass: round EXHAUSTED (not campaign-level — campaign evaluates in Stage 7).
- See `orchestration/integration-logic.md`.

### Stage 7: Campaign Evaluation (AUTONOMOUS)

This stage is fully autonomous — no user interaction. The orchestrator executes the mechanical threshold check and proceeds immediately.

- After integration: record round results, read `f` from profiling data, compare to `min_e2e_improvement_pct`.
- `f >= threshold` → continue unconditionally (re-profile if SHIP, pivot technology if EXHAUSTED — see Campaign Stop Condition above).
- `f < threshold` → set `campaign_complete` or `campaign_exhausted`, spawn report (T20).
- After an EXHAUSTED round where `f >= threshold`: the next round's debate prompt MUST direct champions to unexplored technology classes per the pivot order in Campaign Stop Condition.
- See Campaign Stop Condition and Campaign State Transitions above for the full mechanical protocol.
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
T5b: Baseline ncu sanity check (top-3 kernels by f_decode) [ammo-researcher subagent]   <- T5
T6:  TeamCreate round team + champion proposals + debate (Phase 0 -> rounds -> selection -> shutdown champions) [main + round team] <- T5b
T7:  GATE: Debate winner selection (proposals + summary.md exist) [main]                <- T6

  +- Per winning candidate (parallel, all in existing round team) ----------------------+
  | Spawn impl-champion-{id} + impl-validator-{id} into round team                     |
  | T8a_{id}: Research (validator) + plan reading (champion)   [round team]    <- T7   |
  | T8b_{id}: Implement kernel (champion only)                 [round team]    <- T8a  |
  | T8c_{id}: Independent validation Gates 5.1/5.2/5.3 (validator) [round team] <- T8b |
  | T8cx_{id}: [IF GATING_REQUIRED] Crossover probing (validator benchmarks,   |
  |            champion implements gating, validator re-validates)  [round team] <- T8c |
  | T8d_{id}: Kill criteria evaluation + validation_results.md (champion) [round team] <- T8c/T8cx |
  |           (frontmatter Stop hook = DA: Amdahl, baseline, parity, independent validation) |
  | T9_{id}: GATE: compilation check                           [main]          <- T8d  |
  | T10_{id}: State update                                     [main]          <- T9   |
  +---------------------------------------------------------------------------------+

  === Overlapped Debate (during Stages 4-5, round 2+ only) ===

  T_overlap_start: Spawn debate champions into round team        [main + round team]   <- T7 (after impl agents spawned)
  T_overlap_p0:    Phase 0 proposals + eligibility gate          [main + round team]   <- T_overlap_start
  T_overlap_debate: Debate rounds (interleaved with impl monitoring) [main + round team] <- T_overlap_p0
  T_overlap_select: Score and select winners                     [main]                <- T_overlap_debate
  T_overlap_shutdown: Shutdown debate champions                  [main]                <- T_overlap_select

  Note: T_overlap tasks interleave with per-track T8-T10 tasks. The orchestrator
  alternates between debate moderation and implementation gating.

T11: GATE: All tracks have results AND overlapped debate (if any) complete  [main]  <- all T10, T_overlap_shutdown
T11b: TeamDelete round team                               [main]               <- T11
T12: Integration validation (if multiple PASS or any GATED_PASS) [main]       <- T11b
T13: Round decision (SHIP / GATED_SHIP / round-EXHAUSTED) [main]              <- T12

=== Campaign Loop (Stage 7) ===

T14: Record round in campaign.rounds                      [main]               <- T13
T15: Campaign evaluation                                  [main]               <- T14
  IF SHIP:
    T16: Re-profile (baseline capture on patched code)    [ammo-researcher]    <- T15
    T17: Bottleneck mining on new baseline                [ammo-researcher]    <- T16
    T18: Mechanical threshold check (f vs min_e2e_improvement_pct) [main]      <- T17
      IF f < threshold: CAMPAIGN COMPLETE
      ELSE: new Round (T6 debate → ...)
  IF round-EXHAUSTED:
    T16b: Mechanical threshold check (existing profile, no re-profile) [main]  <- T15
      IF f < threshold: CAMPAIGN EXHAUSTED
      ELSE: new debate round from existing data (→ T6)
T19: GATE: campaign evaluation                            [main]               <- T15..T18
T20: Generate optimization report                         [general-purpose subagent, background] <- T19 (campaign_complete or campaign_exhausted)

```

## Non-Negotiables (BLOCKING)

These are NOT advisory. Violation blocks stage progression.

1. **Production parity**: CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `TORCH_COMPILE_DISABLE=1`, `--enforce-eager`, `VLLM_TORCH_COMPILE_LEVEL=0`. *(Reminded by `ammo-pretool-guard.sh` PreToolUse hook — warns but does not block)* Note: restricting `cudagraph_capture_sizes` to match profiled batch sizes during nsys capture is acceptable and does not violate production parity — these sizes are exact matches in the default capture list.
2. **vLLM baseline**: Compare against production kernel, NOT naive PyTorch.
3. **Numerical correctness**: `torch.allclose()` is mandatory in every correctness test.
4. **GPU sequencing**: E2E benchmarks sequential via GPU lock. Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements. *(Reminded by `ammo-pretool-guard.sh` — warns on raw `vllm bench latency`)*
5. **GPU isolation**: GPU commands MUST use the pool reservation pattern:
   `CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N) && CUDA_VISIBLE_DEVICES=$CVD <command>`.
   The PostToolUse hook auto-releases GPUs when the command completes. Lease expiry handles crashes.
   *(Enforced by ammo-pretool-guard.sh PreToolUse — one-shot block on first missing pattern, then trusts agent judgment)*
6. **Full-model E2E**: Do not skip because "weights aren't available" — download them.
6. **E2E delta math**: `E2E_improvement ~ f x kernel_speedup`, where `f` = component share of total latency. If `f` is small, large kernel wins yield small E2E gains — this is expected, not a bug. For BS-dependent optimizations, compute per-BS `f(BS) × kernel_speedup(BS)` — different batch sizes may have different `f` values. A partial regression at some batch sizes does not negate the optimization if it is gatable — see tiered verdict system in `references/validation-defaults.md`.
7. **Custom kernel mandate**: Stage 3 proposals MUST involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code. Config-only, flag-flipping, and parameter-tuning proposals are rejected outright in the Phase 0 eligibility gate.
8. **Autonomous campaign loop**: The orchestrator MUST NOT ask the user whether to continue, pause, or stop the campaign. The stop condition is purely mechanical: `f_top_bottleneck < min_e2e_improvement_pct` → stop, otherwise → continue. No qualitative judgment ("we've exhausted approaches", "diminishing returns feel likely") overrides this. Technology pivots (e.g., Triton → CUDA C++/CUTLASS) are expected between rounds and do NOT require user confirmation. The only user interaction during a campaign is: (a) the initial invocation, (b) blocker escalation if a gate fails with no recovery path, (c) the final report. If the orchestrator believes the threshold should change, it must propose the change and wait for approval — it cannot unilaterally decide the campaign is done.

## Hook Enforcement

Hooks in `.claude/settings.local.json` enforce the campaign protocol mechanically:

| Hook Event | Script | Purpose |
|------------|--------|---------|
| **Stop** | `ammo-stop-guard.sh` | Blocks session end if campaign is active (one-shot: blocks once, then allows) |
| **PreToolUse** (Bash) | `ammo-pretool-guard.sh` | Production-parity reminders (N1/N4) + GPU pool pattern guard (one-shot block if reservation pattern missing) |
| **PostToolUse** (Bash) | `ammo-gpu-release.sh` | GPU auto-release: detects reservation pattern in completed command, releases by session ID |
| **PreCompact** | `ammo-precompact.sh` | Saves campaign state checkpoint before compaction |
| **SessionStart** | `ammo-postcompact.sh` | Injects resume context after compaction |
| **WorktreeCreate** | `worktree-create-with-build.sh` | Sets up build environment in new worktrees |
| **WorktreeRemove** | `worktree-remove-cleanup.sh` | Cleans up worktree resources |

Inline DA verification (integrated into helper agents — replaces non-functional Stop hook DAs for team members):
- **ammo-transcript-monitor** → Periodic transcript-based DA review of champion artifacts. In debate (Stage 3): 7-point DA audit (custom kernel mandate, evidence tier verification with claim-evidence matching, CUDA graph methodology, Amdahl consistency, E2E grounding, steady-state target, BS-gated sanity); writes to `debate/monitor_audits/`. In implementation (Stages 4-5): scope adherence, methodology checks, reward-hacking detection; writes to `tracks/{op_id}/monitor_audits/`.
- **ammo-impl-validator** → After Gates 5.1/5.2/5.3: 5+ point DA section in validation report (Amdahl sanity, cross-track awareness, kernel-to-E2E coherence, scope adherence, Gate 5.2 cross-check; plus conditional GATED_PASS checks).
- **ammo-researcher** Stop → DA checks for ungrounded claims (subagent — Stop hooks work correctly).

## State Management

`state.json` in artifact directory tracks stage, status, debate, parallel tracks, and integration.

**Session ID**: The lead MUST record the session ID in state.json at campaign start: `"session_id": "<uuid>"`. This enables the eval pipeline to extract ground-truth timing and agent cost data from session logs automatically.

**Stage timestamps and agent costs**: Auto-extracted from session logs by `parse_session_logs.py` during eval. No manual recording needed — the `stage_timestamps` and `agent_costs` fields in state.json are populated by the eval pipeline, not by the lead.

```json
{
  "target": {"model_id": "...", "hardware": "...", "dtype": "...", "tp": 1},
  "stage": "1_baseline",
  "summary": "Initialized.",
  "gpu_resources": {"gpu_count": 1, "gpu_model": "...", "memory_total_gib": 0, "cuda_visible_devices": "0"},
  "debate": {
    "team_name": null,       /* round-scoped: ammo-round-{round_id}-{model_short}-{hardware}; persists Stage 3 through Stage 5 */
    "candidates": [],
    "rounds_completed": 0,
    "max_rounds": 4,
    "selected_winners": [],
    "selection_rationale": null,
    "next_round_overlap": {
      "active": false,       /* whether an overlapped debate is running during Stages 4-5 */
      "phase": null,         /* "phase_0" | "debating" | "selecting" | "selection_complete" | null */
      "selected_winners": [],
      "profiling_basis": null,
      "f_values_at_proposal": {}
    }
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
  "parallel_tracks": {},  /* per-track: { status, verdict, correctness, kernel_speedup, e2e_speedup, per_bs_verdict, gating, validation_results_path }
                              verdict: "PASS" | "GATING_REQUIRED" | "GATED_PASS" | "FAIL"
                              per_bs_verdict: { "1": "PASS", "8": "NOISE", "32": "REGRESSED" } (null if not yet evaluated)
                              gating: null unless verdict is GATED_PASS, then: { mechanism, env_var, dispatch_condition, crossover_threshold_bs, crossover_probing: { probed_points, predicted_bs, confirmed_bs, time_minutes, converged }, pre_gating_results, post_gating_results } */
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
    "min_e2e_improvement_pct": 1.0,               /* stop when top bottleneck f < this %. See references/validation-defaults.md */
    "e2e_deflation_factor": 2.0,       /* multiply predicted E2E improvement by 1/this for scoring. Default: 2.0 (median observed overestimate is 3x, 2x is conservative). */
    "noise_tolerance_pct": 0.5,                  /* per-BS verdict: speedup >= (1.0 - this) = NOISE. From target.json gating block. */
    "catastrophic_regression_pct": 5.0,          /* per-BS verdict: speedup < (1.0 - this) = CATASTROPHIC. From target.json gating block. */
    "cumulative_e2e_speedup": 1.0,               /* multiplicative across shipped rounds. For GATED_PASS tracks: uses min post-gating speedup across all BS. */
    "rounds": [],                                /* per-round history (see schema below) */
    "shipped_optimizations": []                  /* op_ids that shipped across all rounds */
  }
}
```

Each entry in `campaign.rounds`:
```json
{
  "round_id": 1,
  "profiling_baseline_path": "runs/baseline_round_1/",
  "top_bottleneck_share_pct": 15.2,
  "round_team_name": "ammo-round-1-...",
  "selected_candidates": ["op001", "op002"],
  "implementation_results": {
    "op001": {"status": "PASSED", "e2e_speedup": 1.12},
    "op002": {"status": "FAILED", "reason": "correctness"},
    "op003": {"status": "GATED_PASS", "e2e_speedup": 1.025, "gating": {"env_var": "VLLM_OP003", "crossover_threshold_bs": 16, "regressing_bs": [32]}}
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
- **Shutdown**: Debate champions terminated via `shutdown_request` after selection; round team persists until all implementation tracks complete, then TeamDelete.
- **Cross-workstream isolation**: During overlapped debate, debate champions and implementation agents share the same team but MUST NOT communicate directly. The orchestrator enforces this by not providing cross-workstream agent names.
- **Interleaved moderation**: During overlapped operation, the orchestrator alternates between debate phase broadcasts/waits and implementation completion checks. Pattern: broadcast debate phase start -> check for impl completions -> wait for debate phase completions -> repeat.

## Helper Scripts

Run, don't modify:

- `scripts/new_target.py` — Scaffold artifact directory
- `scripts/collect_env.py` — Capture environment
- `scripts/verify_phase1_baseline.py` — Stage 1->2 gate
- `scripts/verify_validation_gates.py` — Stage 5 gate (supports `--track` for per-track validation)
- `scripts/run_vllm_bench_latency_sweep.py` — Batch E2E benchmark runner (GPU-locked). Supports `--labels baseline` for baseline-only sweeps (Stage 1), `--labels opt` for opt-only (Stage 5), or `--labels baseline,opt` (default, A/B comparison). Also supports `workload_matrix` for multi-dimensional sweeps and `--nsys-profile` for per-bucket nsys traces without model reload. **WARNING**: If `opt_env` in target.json still contains placeholder keys (e.g., `<ENABLE_FLAG>`), the script will fail fast — update `opt_env` before running with `--labels opt`.
- `scripts/generate_validation_report.py` — Structured reporting
- `scripts/gpu_status.py` — Print current GPU reservation state (orchestrator/human diagnostic)
- `scripts/gpu_force_clear.py` — Force-clear stale GPU reservations after crashes (orchestrator-only)

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
| Debate-phase agent rules | `references/debate-rules.md` |
| Implementation-phase agent rules | `references/impl-track-rules.md` |
| GPU reservation pattern | `references/gpu-pool.md` |
| Teammate messaging patterns | `references/agent-responsiveness-guide.md` |

**Architectural boundary**: Agent files reference `references/*.md` for domain rules. Orchestration docs (`orchestration/*.md`) are orchestrator-only — agents do not read them directly.

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
| **critical** | HALT current stage and escalate to user. Broadcast halt if debate team active. |
| **major** | Investigate, adjust constraints, re-run subagent. |
| **minor** | Document and continue. |

**Note**: "HALT" means pause the current stage and escalate — NOT terminate the campaign. Campaign termination is governed exclusively by the mechanical `f < threshold` check (see Campaign Stop Condition). Gate failures trigger escalation, not campaign termination — the campaign remains `active`. This is the blocker escalation permitted by Non-Negotiable #8(b).

Save blocker details to `{artifact_dir}/blockers/{stage}_{date}.md`.

## Resume Protocol

After interruption or compaction:

1. Read this skill file (you are the LEAD — delegate, don't implement).
2. Read `state.json` from artifact directory.
3. Check `campaign.status` and `stage` to determine where you are.
3b. Check GPU reservation state: run `python .claude/skills/ammo/scripts/gpu_status.py`. If stale reservations exist from the crashed session, clear them: `python .claude/skills/ammo/scripts/gpu_force_clear.py --all --session-id <crashed_session_id>`. If the crashed session ID is unknown, use `--force-no-session`. Re-spawned agents will have their GPUs auto-reserved by hooks when they run commands.
4. If Stage 3 debate active: check debate artifacts in `debate/` or `debate/campaign_round_N/`.
4b. If Stages 4-5 active AND `debate.next_round_overlap.active` is `true`:
   - Check `debate.next_round_overlap.phase` to determine debate progress.
   - If `phase` is `"selection_complete"`: Winners already selected. No action needed for debate.
   - If `phase` is non-null but not complete: Check debate artifacts in `debate/campaign_round_{N+1}/`. Restart debate from the last completed phase (debate is restartable -- champions are stateless, artifacts on disk capture progress).
   - If `phase` is null but `active` is true: Debate was launched but no progress. Re-spawn debate champions and start from Phase 0.
5. If Stages 4-5 active: check `parallel_tracks` in `state.json` for worktree paths and status. Resume monitoring and gating.
6. Resume from last completed gate.
7. Read `campaign.current_round` to determine which round is active.
8. If `campaign` key is missing (legacy state.json): treat as round 1 — initialize the campaign object.
9. The Stop hook will block session end while campaign is active — complete the current stage. Do NOT set `campaign.status` to `"paused"` autonomously to satisfy the Stop hook; setting `paused` requires explicit user request (see Campaign State Transitions). If you cannot complete the stage, escalate the blocker through the Escalation Protocol.
10. If `campaign.status` is `campaign_complete` or `campaign_exhausted` but no `REPORT.md` exists, spawn the report generation subagent (T20).

## Quick Start Examples

**Example 1**: `User: "Use ammo for Qwen3-30B-A3B on L40S TP=1"` ->

1. Scaffold artifact directory (campaign initialized).
2. Round 1: invoke ammo-researcher subagent for baseline + bottleneck mining.
3. Run gates, spawn debate team for top candidates.
4. Select winners, create parallel worktree tracks.
5. Implement + validate in parallel.
6. Integration if multiple pass → SHIP or round-EXHAUSTED.
7. Campaign evaluation: record round, read `f`, compare to `min_e2e_improvement_pct` (mechanical — no user interaction).
8. If `f >= threshold`: re-profile (if SHIP) or pivot technology (if EXHAUSTED) → new round immediately. Repeat until `f < threshold`.
9. When `f < threshold`: declare `campaign_complete` or `campaign_exhausted`, spawn report.

**Example 2**: Resume -> Read `state.json`, check `campaign.current_round` and active stage, resume from last gate.

**Example 3**: Mid-campaign resume after compaction -> SessionStart hook injects campaign context (round, status, cumulative speedup). Read `state.json`, resume current round.
