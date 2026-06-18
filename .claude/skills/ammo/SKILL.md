---
name: ammo
description: Profile and optimize GPU kernels for vLLM inference on NVIDIA GPUs. Use when targeting specific (model, hardware, dtype, TP, etc.) deployments to improve latency. Triggers on requests to speed up any vLLM kernel.
---

# AMMO - Automated Model Micro-Optimizer

Profile and optimize **GPU kernels** for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

**Artifact Layout**: All campaign files follow the round-scoped hierarchy under `{artifact_dir}/rounds/{N}/`. See `references/artifact-layout.md` for the canonical layout, path resolution rules, and prohibited patterns.

## Lead Role

You are the **lead orchestrator**. You scaffold, delegate, and gate — you never implement.

**Responsibilities**:
- Spawn subagents and assign work — never implement stages directly
- Manage state.json — read before each action, update `campaign.current_stage` and round stage timestamps at EVERY stage transition (see State Management below)
- Own all gate tasks (T3, T5, T7, T9, T11, T13, T19) — run verification scripts yourself
- Use SendMessage to communicate with teammates — text output is NOT visible to them

**Prohibited**:
- Do not write kernel code (CUDA C++, Triton, CUTLASS, or CuTeDSL) yourself
- Do not skip team creation "for efficiency"
- Do not implement directly — always delegate to subagents
- Do not override state.json configuration (thresholds, etc.) without user approval

**Configuration Fidelity**: The orchestrator MUST respect all `state.json` configuration flags as written. If the orchestrator believes a configuration should change (e.g., adjusting thresholds for a round), it MUST propose the change to the user and wait for approval before acting. The orchestrator does NOT have discretion to override configuration settings — it scaffolds, delegates, and gates, not makes policy.

## Invocation

User provides: model_id, hardware, dtype, tp, and optional workload config: input_len, output_len, batch_sizes, max_model_len, data_parallel_size, enable_expert_parallel. **Extract ALL parameters the user specifies** — especially input/output sequence lengths (users often state these as "input length 2048", "output 256", "ISL 128 OSL 1024", etc.). If the user does not specify input/output lengths, defaults are 64/512 (decode-heavy workload). Pass whichever the user specifies; omit individual flags the user didn't mention (each defaults independently: input→64, output→512).

Lead scaffolds the artifact directory via `new_target.py` and enters the campaign loop.

```bash
python .claude/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP> \
  [--batch-sizes <BATCH_SIZES>] [--input-len <INPUT_LEN> --output-len <OUTPUT_LEN>] [--max-model-len <MAX_MODEL_LEN>] \
  [--data-parallel-size <DP_SIZE>] [--enable-expert-parallel]
```

Quick notes before dispatching:

- `--input-len` / `--output-len` (defaults `64` / `512`): Pass user-specified values here. Omit ONLY if the user did not mention sequence lengths — never hardcode 64/512 when the user asked for different values.
- `--batch-sizes` (default `[1, 8, 32]`) pins the decode buckets used across all profiling and validation for the campaign — pick once, don't retune.
- `--max-num-seqs` is a vLLM serving flag, not a `new_target.py` flag — set via `bench.extra_args` post-patch.
- `--isl-osl=a:b,c:d,...` requires `workload_matrix` patching (cross product with batch sizes) — do NOT pass `--input-len`/`--output-len`. Note: a single pair (e.g., user says "input 2048 output 256") uses `--input-len`/`--output-len` directly; `workload_matrix` is ONLY for multiple comma-separated pairs.
- `--data-parallel-size > 1` triggers unconditional `--distributed-executor-backend external_launcher` injection; the sweep script fails fast on conflicting post-hoc overrides.
- `--ep N` is a legacy sizing field only — enable actual expert parallelism with `--enable-expert-parallel`.

Full parameter table, ISL/OSL `workload_matrix` patching, `--max-num-seqs` post-patch, EP flag semantics, and DP cross-track contract: `references/workload-invocation.md`.

## Campaign Workflow

```
Stage 1:  Baseline Capture          [main + ammo-researcher subagent]     → rounds/{N}/constraints.md
Stage 2:  Bottleneck Mining         [main + ammo-researcher subagent]     → rounds/{N}/mining/bottleneck_analysis.md (grounded)
Stage 3:  Proposals + Debate        [round team: 2-4 ammo-champion]       → rounds/{N}/debate/summary.md
Stage 4-5: Parallel Tracks          [round team reused: per-track impl-champion + monitor]
Stage 6:  Integration Validation    [main]                                → SHIP or round-fail
Stage 7:  Campaign Evaluation       [main]                                → next round or terminal
Stage 7b: Report Generation         [ammo-report-writer subagent, bg]     → REPORT.md (on terminal)
```

A single round-scoped team persists from Stage 3 through Stage 5 (created at debate start, deleted after all tracks complete). Stage 7 is mechanical — no user prompts.

### Campaign Stop Condition

**Stop iff `f < min_e2e_improvement_pct`.** Otherwise, continue unconditionally. The Stop hook (`ammo-stop-guard.sh`) blocks session end while `campaign.status == "active"` — trust it. Full Amdahl rationale and the full "invalid reasons to stop" list in `references/validation-defaults.md` §§ Minimum E2E Improvement Threshold, Invalid Reasons to Stop.

**After SHIP**: Run mining on new baseline (bottleneck landscape shifted), then check the NEW top bottleneck. If `f >= threshold`, start the next round immediately at `2_bottleneck_mining`. (No separate re-profile — Stage 6 provides the post-SHIP measurement via combined sweep or single-track short-circuit.) Because Stage 1 is skipped here, a round N>1 re-mine has no fresh `stage_1` audit; the hard gate exempts the same-round `stage_1` requirement for `2_bottleneck_mining` when `current_round > 1` *and the previous round carries an `audit` key* — entry stays guarded by that previous round's `stage_67` (see `orchestration/audit-protocol.md` §Post-SHIP re-mine exemption). If the predecessor has no audit key, the exemption fails closed.
**After EXHAUSTED**: Check threshold against existing profiling data (no re-profile). If `f >= threshold`, start the next round at `3_debate` — reuse the existing bottleneck analysis because the baseline hasn't changed (no SHIP landed). Inject the exhaustion note + `exhausted_technologies` filter into the debate prompt so champions pivot to a different approach on the same bottleneck.

**When to re-mine after EXHAUSTED (rare)**: If the previous round's mining was invalidated (e.g., wrong component attribution discovered during track analysis), set `rounds[$IDX].mining_invalidated: true` with a reason string before starting the new round. This overrides the default `→ 3_debate` path and enters `2_bottleneck_mining` instead. The rationale: re-mining is warranted when the diagnosis was wrong, not just when the fix failed.

**Technology selection**: Champions pick Triton/CuTeDSL/CUTLASS/CUDA C++ per-proposal via the selection function in `references/technology-selection.md`. There is no fixed pivot ladder. When a round EXHAUSTED, the orchestrator appends a structured entry to `state.json.campaign.rounds[$IDX].exhausted_technologies` (schema: `.claude/schemas/state.schema.json`) — champions filter by their op and shape bucket.

### Campaign State Transitions

```
active → (f >= threshold, SHIP)      → active (mining on new baseline → 2_bottleneck_mining)
active → (f >= threshold, EXHAUSTED) → active (reuse mining → 3_debate, pivot technology)
active → (f < threshold, SHIP)       → campaign_complete
active → (f < threshold, EXHAUSTED)  → campaign_exhausted
active ↔ (user pause/resume)         → paused
```

No transition to terminal based on orchestrator judgment — only the mechanical `f < threshold` check terminates.

### In-Flight Tracks During Re-profiling

When a candidate ships and triggers re-profiling, other tracks from the same round may still be running. Let them complete (do NOT terminate), validate them against the ORIGINAL round's baseline, and ship any that pass. Record all results under the current round's `campaign.rounds` entry. The next round starts only after all current-round tracks reach terminal status.

## Orchestration Model

One paragraph per stage. Deep protocol detail lives in `orchestration/*.md`.

### Stages 1-2: Baseline + Bottleneck Mining

Lead invokes `ammo-researcher` as a subagent (Agent tool, no `name`/`team_name` — never a team member) using the structured dispatch interface. Two dispatches, no intermediate orchestrator gates:

1. **Stage 1** (`task_type: baseline`): Researcher runs exactly two invocations: (a) clean E2E sweep `--round {N} --slot baseline --labels baseline --capture-golden-refs` (no profiling flags — authoritative timing), then (b) bounded Nsight Systems sweep `--round {N} --slot profiling --labels baseline --nsys-profile --nsys-mode node --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800` for attribution. The sweep script shifts `input_len` and uses vLLM's CUDA profiler to capture short selected-step windows. Add `--nsys-trace cuda-sw` on Blackwell (B200/B300); Hopper/Ampere use the sweep default `cuda` backend. Note: `--nsys-profile` restricts `cudagraph_capture_sizes` to a bounded union of workload batch sizes and, for selected-step speculative decode, effective decode token counts (`batch_size * positions_per_step`) — this is NOT a parity violation. The sweep script hard-errors if `--slot baseline` is combined with any profiling flag.
2. **Stage 2** (`task_type: mining`): Researcher mines the Stage 1 nsys traces → produces `rounds/{N}/mining/bottleneck_analysis.md` (with §Technology Landscape). Run targeted NCU only when claiming occupancy, bandwidth-counter values, or physical ceilings.

After T4 returns, lead runs T5 gate: `python .claude/skills/ammo/scripts/verify_stage2_gate.py {artifact_dir} --round {N}`. Exit 0 = PASS; exit 1 = FAIL (prints reason). Checks: `rounds/{N}/mining/bottleneck_analysis.md` exists with §Technology Landscape AND `rounds/{N}/sweeps/baseline/e2e_latency_results.json` exists with results. The researcher's Stop hook (Sonnet DA) independently audits for ungrounded claims.

**Structured dispatch contract** (see `.claude/agents/ammo-researcher.md` § Dispatch Interface):

```python
# Stage 1:
Agent(subagent_type="ammo-researcher", prompt="task_type: baseline\nartifact_dir: {artifact_dir}")

# Stage 2:
Agent(subagent_type="ammo-researcher", prompt="task_type: mining\nartifact_dir: {artifact_dir}")
```

### Stage 3: Candidate Proposal + Adversarial Debate

TeamCreate `ammo-round-{round_id}-{model_short}-{hardware}` (the **round team** — reused through Stage 5). Spawn 2-4 `ammo-champion` agents (team members, `run_in_background=True`). **Phase -1 (Target Claim — runs first)**: champions read `bottleneck_analysis.md`, run the component waterfall in `.claude/agents/ammo-champion.md` § Target Claim Phase, and broadcast `Claiming {component}` to the team — a **component-only** claim; the mechanism/category is deferred. The lead reviews the claim distribution holistically (4-check rubric in `orchestration/debate-protocol.md` § Phase -1) and either approves or sends per-champion (component) redirects. **Per-component analysis**: after approval, each champion analyzes the existing profiling data for its assigned component (the campaign's already-captured Stage 1-2 traces — no new capture) to choose the mechanism; the analysis scope is the assigned component. **Phase 0 (Proposals)**: each champion derives 1-2 candidates from that analysis (NOT pre-scored lists) with micro-experiment-backed feasibility math, and self-selects `kernel_replacement` / `kernel_fusion` / `dispatch_optimization` grounded in the existing profiling data. The Phase 0 Diversity Check (Lead) runs the precise per-technology exhausted check (soft, document-the-loop) that the category-free claim could not. Minimum 1 debate round (full A/B/C); conditional 2nd round if any champion declares open items after Phase C. After selection (2-3 winners via `references/debate-scoring-rubric.md`), shut down debate champions via `SendMessage(to=<champion_id>, message={"type": "shutdown_request"})` and confirm each one leaves the team roster (`shutdown_approved`) before spawning implementation agents — a champion approves only when its work is complete. The round team persists — do NOT TeamDelete here. Full protocol: `orchestration/debate-protocol.md`.

### Stages 4-5: Parallel Worktree Tracks

Per winner, spawn `ammo-impl-champion` (Opus, `isolation: worktree`) + `ammo-transcript-monitor` into the existing round team. The champion implements, then validates kernel correctness + speedup (Gates 5.1a + 5.2) and runs the sweep script for E2E (Gates 5.1b + 5.3a + 5.3b). The champion reports the verdict via `TRACK_COMPLETE`.

**INVARIANT (HARD — enforced by `ammo-state-validate.sh`)**: The orchestrator MUST NOT transition to Stage 6 until ALL tracks have a terminal status. Terminal: `{PASS, GATED_PASS, FAIL}`. Non-terminal: `{IN_PROGRESS, GATING_REQUIRED, GPU_BLOCKED}`. Writing `current_stage = "6_integration"` while any track is non-terminal is blocked. This prevents late-arriving verdicts from being silently dropped.

After all tracks terminal, call TeamDelete on the round team (the only TeamDelete in the round lifecycle). Full team structure, phase-transition protocol, two-layer verification: `orchestration/parallel-tracks.md`.

**Projection accuracy check (v4.1+)**: After all tracks are terminal, run `scripts/check_projection_accuracy.py {artifact_dir} --round {N} --track-id {op_id}` per track to compare each candidate's `projected_e2e_improvement_pct` against realized E2E from the sweep at `rounds/{N}/sweeps/opt/{op_id}/`. Appends a `## Projection Accuracy` section to `rounds/{N}/tracks/{op_id}/validation_results.md`. This is a diagnostic backstop (does not block integration) — flags over-projection > 2× for calibration.

### Stage 6: Integration Validation

Single passer → short-circuit (copy Stage 5 results into integration slot, set `integration.status = "single_pass"`, skip combined sweep). Multiple candidates PASS/GATED_PASS targeting different components: cherry-pick both, re-run E2E. Same component: pick best E2E. GATED_PASS merge conflicts: spawn `ammo-resolver` + DA reviewer. If none pass: round EXHAUSTED (round-level, not campaign-level — campaign evaluates in Stage 7). Full decision matrix + short-circuit procedure: `orchestration/integration-logic.md`.

**Baseline promotion on SHIP**: (1) Pre-SHIP mechanical checks pass (merge-conflict residue, dual-verdict override, opt returncode — inline, no auditor); (2) `git merge --no-ff tracks/{op_id}` into session mainline; (3) append shipped `opt_env` keys to `target.json:bench.baseline_env`; (4) clear `opt_env`; (5) capture golden-refs (`--labels baseline --capture-golden-refs --num-iters 1`, ~15s); (6) spawn T_AUDIT_S67. When ≥2 tracks pass, the Stage 6 integration sweep runs with `--fresh-cache`; when only 1 track passes, the short-circuit copies Stage 5 results (no separate sweep). Either way the T16 re-profile step is eliminated. Subsequent rounds measure against the cumulative post-SHIP baseline; `cumulative_speedup_vs_round1` stays anchored to the original round-1 baseline via direct ratio (see §State Management).

### Stage 7: Campaign Evaluation (AUTONOMOUS)

No user interaction. After T_AUDIT_S67 passes: record round results, read `f`, compare to `min_e2e_improvement_pct`. `f >= threshold` → continue (mining on new baseline if SHIP, pivot technology if EXHAUSTED). `f < threshold` → set `campaign_complete` or `campaign_exhausted`, spawn T20 report. On EXHAUSTED with `f >= threshold`, append an entry to `state.json.round.exhausted_technologies` (schema fields: `technology_class`, `failure_mode`, `applies_to_component`, `applies_to_shape_bucket`, `evidence_refs`, `expires_after_reprofile`) — champions filter this array via the selection function.

**T16 eliminated**: The former re-profile step (separate E2E sweep with `--fresh-cache` after SHIP) is no longer needed — Stage 6 integration either runs `--fresh-cache` (≥2 passers) or copies Stage 5 results (single passer short-circuit). The `integration.e2e_latency_combined` value is the post-SHIP baseline for cumulative speedup computation.

**Round transition**: All per-round state lives in `campaign.rounds[N-1]`. Advance with:

```python
round_entry = campaign["rounds"][current_round - 1]
round_entry["status"] = "completed" | "SHIPPED" | "EXHAUSTED" | "FAILED"
round_entry["integration"]["completed_at"] = now_iso()
if len(campaign["rounds"]) <= current_round:
    campaign["rounds"].append(new_round_entry(round_id=current_round + 1))
campaign["current_round"] = current_round + 1
```

Do NOT deep-copy or reset live state. Full protocol: `orchestration/integration-logic.md` § Round Transition.

### Stage 7b: Report Generation

On terminal status, spawn `ammo-report-writer` in the background (`run_in_background=True`). The orchestrator does not wait for it to declare the campaign done. The subagent's Stop hook runs an adversarial fact-checker that cross-references every claim against source artifacts. Template, chart specs, and quality checklist: `.claude/skills/ammo/report/SKILL.md`.

## Task Graph

```
=== Round N Inner Loop (Stages 1-6) ===

T1:  Scaffold artifact directory                          [main]
T2:  Baseline + rounds/{N}/constraints.md                 [ammo-researcher, task_type: baseline]    <- T1
T4:  Mining                                               [ammo-researcher, task_type: mining] <- T2
T5:  GATE: artifacts check                               [main]                        <- T4
T5.5: TeamCreate round team + spawn champions + Claim Phase (champions broadcast COMPONENT-only claims via the component waterfall) [main + round team] <- T5
T5.6: Orchestrator review of component claims (4-check rubric; approve or redirect on component; loop ≤ 3) [main] <- T5.5
T5.7: Per-component analysis (each champion analyzes the existing profiling data for its approved component; mechanism follows from that data) [round team] <- T5.6
T6:  Champion proposals (Phase 0; candidate + grounded mechanism/category) + Phase 0 eligibility gates + Diversity Check (per-technology exhausted, soft) + debate (rounds -> selection -> shutdown champions) [main + round team] <- T5.7
T7:  GATE: Debate winner selection (proposals + summary.md exist) [main]                <- T6

  +- Per winning candidate (parallel, all in existing round team) ----------------------+
  | Spawn impl-champion-{id} into round team                                           |
  | T8a_{id}: Research + plan reading (champion)               [round team]    <- T7   |
  | T8b_{id}: Implement kernel (champion)                 [round team]    <- T8a  |
  | T8c_{id}: kernel validation (correctness & speedup) + E2E sweep (5.1b + 5.3a + 5.3b) [round team] <- T8b |
  | T8cx_{id}: [IF GATING_REQUIRED] Crossover probing (champion benchmarks,    |
  |            implements gating, re-validates) [round team] <- T8c  |
  | T8d_{id}: Kill criteria evaluation + validation_results.md (champion) [round team] <- T8c/T8cx |
  | T9_{id}: GATE: compilation check                           [main]          <- T8d  |
  | T10_{id}: State update                                     [main]          <- T9   |
  +---------------------------------------------------------------------------------+

T11: GATE: All tracks have terminal results               [main]               <- all T10
T11b: TeamDelete round team                               [main]               <- T11
T12: Integration validation (if multiple PASS or any GATED_PASS) [main]       <- T11b
T13: Round decision (SHIP / GATED_SHIP / round-EXHAUSTED) [main]              <- T12

=== Campaign Loop (Stage 7) ===

T14: Record round in campaign.rounds                      [main]               <- T13
T15: Campaign evaluation                                  [main]               <- T14
  IF SHIP:
    T15b: Mining on new baseline (task_type: mining) [ammo-researcher] <- T15
    T15c: Mechanical threshold check (f vs min_e2e_improvement_pct) [main]     <- T15b
      IF f < threshold: CAMPAIGN COMPLETE
      ELSE: new Round (T6 debate → ...)
  IF round-EXHAUSTED:
    T15d: Mechanical threshold check (existing profile, no re-profile) [main]  <- T15
      IF f < threshold: CAMPAIGN EXHAUSTED
      ELSE: new debate round from existing data (→ T6)
T19: GATE: campaign evaluation                            [main]               <- T15..T15c
T20: Generate optimization report                         [ammo-report-writer subagent, background] <- T19 (campaign_complete or campaign_exhausted)

```

## Audit Gates (T_AUDIT)

After every major stage gate, the lead spawns an **`ammo-auditor`** (Opus, adversarial framing) that runs a four-phase verification: input inventory → cold stage-completion reconstruction → institutional checklist → reconciliation with blocker categories. The auditor dispatches `ammo-delegate` sub-agents in parallel for primary evidence, critically evaluates their findings, and emits a severity-rated verdict. **BLOCKING** findings halt the campaign until resolved via a review loop.

Spawn as a sub-agent: `subagent_type="ammo-auditor"`, `run_in_background=True`, no `name`. Each gate gets a fresh instance. The auditor always spawns regardless of track outcomes or round status. Phase 2 uses precondition gating for invariant rows whose artifacts don't exist.

**Team-member vs subagent (HARD)**: Only `ammo-champion`, `ammo-impl-champion`, and `ammo-transcript-monitor` are team members — spawn them with a `name` (and the round `team_name`). Every other agent type (`ammo-researcher`, `ammo-auditor`, `ammo-investigator`, `ammo-delegate`, `ammo-report-writer`, `ammo-resolver`) is a one-shot subagent: spawn with `subagent_type` + `prompt` only, never a `name` or `team_name`. A `name` registers a persistent team member that lingers in the roster; `team_name` auto-attaches to the active team. The `ammo-team-spawn-guard.sh` PreToolUse hook blocks violations in both directions.

### Trigger Points

| Trigger | When | Invariants file section | State field on PASS |
|---------|------|-------------------------|---------------------|
| **T_AUDIT_S1**  | Stage 1 baseline capture complete (`rounds[$IDX].baseline.completed_at` set) | `After Stage 1` (15 items) | `rounds[$IDX].audit.stage_1.passed_at` |
| **T_AUDIT_S2**  | Stage 2 bottleneck mining complete (`rounds[$IDX].bottleneck_mining.completed_at` set). Schema v4.1+ only — skipped on legacy campaigns. | `After Stage 2` (10 items) | `rounds[$IDX].audit.stage_2.passed_at` |
| **T_AUDIT_S45** | All Stage 4-5 tracks reach terminal status (`PASS` / `GATED_PASS` / `FAIL`) | `After Stages 4-5` (9 items) | `rounds[$IDX].audit.stage_45.passed_at` |
| **T_AUDIT_S67** | After SHIP: fires AFTER git merge + env promotion + golden-refs capture. After EXHAUSTED: fires after integration status set. | `After Stage 6-7` (22 items) | `rounds[$IDX].audit.stage_67.passed_at` |

Stage 3 (debate) is excluded — the adversarial debate structure (Phase B cross-critique + Phase C rebuttal + open-items declaration) provides inherent quality control.

Every audit also runs the **Pre-Check** and **Holistic Cross-Reference** checklists (delivered to the auditor via hook after Phase 1 completes).

**Backward compatibility**: Legacy campaigns with `audit.stage_6` and `audit.stage_7` (pre-consolidation) are still accepted by hooks — they fall through to the old field names. New campaigns use `audit.stage_67` exclusively.

### Dispatch Prompt

```
task: audit_gate
stage: stage_45
round: 2
```

The auditor discovers all paths (artifact_dir, state_file, verdict_file, transcript dir) via convention. Only `stage` and `round` are passed by the orchestrator.

### Verdict Handling

The auditor writes its verdict to `{artifact_dir}/rounds/{M}/audits/stage_{N}.md` (where N is `1`, `45`, or `67`; M is the round number). On return:

```
IF verdict.overall == "PASS":
    - Write rounds[$IDX].audit.stage_{N} = {"passed_at": <now_iso()>, "verdict_file": "rounds/{M}/audits/stage_{N}.md"}
    - Continue to next stage

IF verdict.overall == "BLOCKED":
    FOR each BLOCKING finding:
        - Read the finding's blocker category
        - Delegate fix to appropriate agent (see orchestration/audit-protocol.md § Delegation Matrix)
        - After fix applied: re-spawn ammo-auditor with the SAME stage context
    - Loop until PASS OR 3 cycles exhaust → campaign halts with `auditor_escalation` field in state.json

IF verdict.overall == "NEEDS_INVESTIGATION":
    - Spawn investigator for each HIGH finding
    - Downgrade to LOW (continue) or upgrade to BLOCKING (enter fix loop)
```

### Round Bootstrap

**When appending a new round** (campaign progression after SHIP or EXHAUSTED), include `"audit": {}` in the new round entry. The empty dict presence activates the audit gate in the state-validate and next-step-reminder hooks. `new_target.py` already seeds `"audit": {}` in `rounds[0]` at campaign scaffold time; subsequent rounds appended by the orchestrator MUST replicate the field.

Legacy gate: hooks skip audit enforcement when the `audit` key is entirely absent from the round (pre-dating this feature). Presence — even empty `{}` — activates enforcement.

### Hook Enforcement

Two hooks enforce this gate mechanically (belt-and-suspenders):

- **`ammo-state-validate.sh`** (HARD gate) — blocks `state.json` writes that set `current_stage` to `2_bottleneck_mining`, `3_debate`, `6_integration`, or `7_campaign_eval*` when the relevant `audit.stage_X.passed_at` is missing from the current round. Also blocks new-round starts (`current_round > 1`, `current_stage ∈ {1_baseline, 2_bottleneck_mining, 3_debate}`) without `audit.stage_67.passed_at` (or legacy `stage_7.passed_at`) on the previous round.
- **`ammo-next-step-reminder.sh`** (SOFT reminder) — when an audit has not passed at the current transition, emits "AUDIT REQUIRED: Spawn ammo-auditor (4-phase: inventory → reconstruction → checklist → reconciliation) ..." in place of the normal next-step reminder.

### References

- Full orchestrator protocol (spawn lifecycle, delegation matrix, loop termination): `orchestration/audit-protocol.md`
- Stage-specific checklists (delivered to auditor via PostToolUse hook after Phase 1)
- Auditor's own agent definition (four-phase procedure, evidence mandate): `.claude/agents/ammo-auditor.md`

## Autonomous Decision-Making

This campaign runs **unattended** — a question to the user can sit for hours while the campaign stalls and GPUs idle. If you're stuck at a fork, the default is to make the call that is most aligned with the goal of the workflow, otherwise if you're genuinely unclear, spawn `ammo-investigator` agent like:

```python
Agent(
  subagent_type="ammo-investigator", 
  run_in_background=True, 
  prompt="""CALLER: orchestrator
          MODE: decision_support
          CAMPAIGN_GOAL: maximize validated E2E improvement over the production-parity baseline without regressing correctness, advancing autonomously until f < min_e2e_improvement_pct.
          DECISION: <the fork in one sentence>
          OPTIONS: <branches you're choosing between>
          EVIDENCE TO CHECK: <artifact paths bearing on the choice>
          artifact_dir: {artifact_dir}"""
)
```

## Non-Negotiables (BLOCKING)

These are NOT advisory. Violation blocks stage progression.

1. **Production parity**: CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `TORCH_COMPILE_DISABLE=1`, `--enforce-eager`, `VLLM_TORCH_COMPILE_LEVEL=0`. *(Reminded by `ammo-pretool-guard.sh` PreToolUse hook — warns but does not block)* Note: restricting `cudagraph_capture_sizes` to match profiled batch sizes during nsys capture is acceptable and does not violate production parity; for selected-step speculative decode, include the effective decode token-count sizes (`batch_size * positions_per_step`) as well.
2. **vLLM baseline**: Compare against production kernel, NOT naive PyTorch.
3. **Numerical correctness**: `torch.allclose()` is mandatory in every correctness test.
4. **GPU sequencing**: E2E benchmarks sequential via GPU lock. Use `scripts/run_vllm_bench_latency_sweep.py` for all E2E measurements. *(Reminded by `ammo-pretool-guard.sh` — warns on raw `vllm bench latency`)*
5. **GPU isolation**: GPU commands MUST use the pool reservation pattern with agent-scoped session_id:
   `CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus N --session-id {op_id} --no-auto-release) && CUDA_VISIBLE_DEVICES=$CVD <command>`.
   The PostToolUse hook auto-releases GPUs when the command completes. Lease expiry handles crashes.
   If the pool is exhausted, retry with 30s backoff — see `references/gpu-pool.md` § Contention Handling.
   **Never kill processes on GPUs you don't own** — see `references/gpu-pool.md` § Process Isolation Rules.
   *(Enforced by ammo-pretool-guard.sh PreToolUse — one-shot block on first missing pattern, then trusts agent judgment)*
6. **Full-model E2E**: Do not skip because "weights aren't available" — download them.
7. **E2E delta math**: `E2E_improvement ~ f_e2e × (1 - 1/s)`, where `f_e2e` (NOT `f_decode`) = component share of **total E2E wall time**, and `s` = kernel speedup ratio. `f_e2e = f_decode × decode_busy × decode_share_of_e2e` — see `references/e2e-delta-math.md` for the conversion and red-flag conditions. If `f_e2e` is small, large kernel wins yield small E2E gains — this is expected, not a bug. For BS-dependent optimizations, compute per-BS `f_e2e(BS) × (1 - 1/s(BS))` — different batch sizes may have different `f_e2e` values. A partial regression at some batch sizes does not negate the optimization if it is gatable — see tiered verdict system in `references/validation-defaults.md`.
8. **Authored-mechanism mandate** (north star: *real engineering work, no flag-flipping*): Stage 3 proposals MUST author a mechanism that changes the execution characteristics of the forward pass — not retune a knob and let the compiler, library, or runtime author the difference. A proposal is **eligible** iff all three hold:
   - **(a) Authored mechanism logic or host-side structure** — NOT a retuned scheduling/tuning value where the kernel/cubin body is byte-identical and a compiler, autotuner, library tactic-table, policy list, predicate flip, or env var emits the speedup. *A constant in a `.py` file is still config, regardless of how large the measured win* (`num_warps`, `num_stages`, `BLOCK_SIZE_*`, `@autotune` tuples, tactic tables, `custom_ops` list edits, boolean predicate flips stay rejected — the kernel body is unchanged). A hand-written prefetch loop that achieves what `num_stages` would is eligible (logic authored); flipping `num_stages=3` is not (logic identical).
   - **(b) Profiled bottleneck** — it targets a bottleneck identified in the round's profiling artifacts.
   - **(c) Measured parity-safe win** — it produces a measured, production-parity (CUDA graphs + torch.compile) E2E win ≥ `min_e2e_improvement_pct`.

   **`category` is a non-binding descriptor, NOT the gate.** It selects the projection formula and routes validation; it never decides eligibility. The catalog (`references/optimization-categories.md`) lists named classes — `custom_kernel`, `kernel_fusion`, `weight_layout_transform`, `compute_graph_pass`, `execution_pipeline_restructuring`, `communication_strategy`, `attention_kv_layout`, plus the legacy aliases `kernel_replacement` (≡ `custom_kernel`) and `dispatch_optimization` (≡ `execution_pipeline_restructuring`) — but a novel mechanism that passes (a)+(b)+(c) is eligible even if it fits no listed class; it simply names a new descriptor and maps onto a projection slice.

   The Phase 0 eligibility gate enforces:
   - Populated **Technology Selection** block (baseline tech, proposed tech, hardware, op character, library coverage, justification, anti-regression check, CUDA-graph capture self-check if CuTeDSL).
   - Populated `## Category` block per `references/optimization-categories.md` § Verifying a Proposal's Category Block. Missing block → reject at eligibility gate (same severity as missing Technology Selection).
   *Illustration (not the definition): pure env-var flips, torch.compile/CUDA-graph settings, and autotune-JSON edits are rejected because none authors a mechanism — the cubin they produce is byte-identical to the baseline's.* See `references/optimization-categories.md` for the category catalog, projection formulas, and disambiguating examples (including the eligible-vs-rejected tuning-constant boundary).
9. **Autonomous campaign loop**: The orchestrator MUST NOT ask the user whether to continue, pause, or stop the campaign. The stop condition is purely mechanical: `f_top_bottleneck < min_e2e_improvement_pct` → stop, otherwise → continue. No qualitative judgment ("we've exhausted approaches", "diminishing returns feel likely") overrides this. Technology shifts between rounds (e.g., the debate picking a different class after a round EXHAUSTED) are expected and do NOT require user confirmation — the orchestrator just injects the exhaustion note and lets the selection function run. The only user interaction during a campaign is: (a) the initial invocation, (b) blocker escalation if a gate fails with no recovery path, (c) the final report. If the orchestrator believes the threshold should change, it must propose the change and wait for approval — it cannot unilaterally decide the campaign is done.
10. **Track-Level Fallback Ladder**: Before writing `verdict=FAIL` at any lifecycle stage (post-validation track FAIL, pre-validation "Implementation Infeasible", round EXHAUSTED), walk the ladder in order. Stop at first match; FAIL only if all applicable rungs exhaust.

    ```
    PASS                   — any BS passes raw E2E gate.
    GATED_PASS             — any BS PASS at a subset of configs; env-var-free.
    GATING_REQUIRED        → GATED_PASS via env-var gate. See validation-defaults.md § Ship precedents.
    RETRY_WITH_CONTINGENCY — per orchestration/parallel-tracks.md § RETRY_WITH_CONTINGENCY protocol.
    FAIL                   — all above exhausted.
    ```

    For pre-validation FAIL paths (Implementation Infeasible, round EXHAUSTED), rungs that require per-BS verdicts (GATED_PASS, GATING_REQUIRED) are not applicable; `RETRY_WITH_CONTINGENCY` and structural fallback (pick a different tech class next round) remain. This ladder is lifecycle-neutral: it applies wherever FAIL is about to be authored, regardless of which stage produced the decision. Single source of truth — do not duplicate this definition elsewhere; other files reference this item by anchor.

## Hook Enforcement

Hooks in `.claude/settings.local.json` enforce the campaign protocol mechanically:

| Hook Event | Matcher | Script | Purpose |
|------------|---------|--------|---------|
| **Stop** | — | `ammo-stop-guard.sh` | Blocks session end while `campaign.status == "active"` (one-shot) |
| **PreToolUse** | — | `ammo-msg-check.sh` | Champion/monitor messaging etiquette reminders |
| **PreToolUse** | `Bash` | `ammo-pretool-guard.sh` | N1/N4 parity reminders + GPU pool pattern guard (one-shot block) |
| **PreToolUse** | `Agent` | `ammo-team-spawn-guard.sh` | Enforces team-member vs subagent: champion/impl-champion/monitor need a valid round `team_name`; every other type is blocked from spawning with a `name`/`team_name` |
| **PreToolUse** | `Edit\|Write` | `ammo-env-default-guard.sh` | Detects unsafe default-env mutations |
| **PostToolUse** | `Bash` | `ammo-gpu-release.sh` | GPU auto-release when reservation pattern completes |
| **PostToolUse** | `Agent` | `ammo-monitor-reminder.sh` | Reminds orchestrator to spawn monitor after impl-champion spawn (debate champions excluded) |
| **PostToolUse** | `Write\|Edit\|Bash` | `ammo-state-validate.sh` | Validates state.json writes (canonical values, terminal-track invariant) |
| **PostToolUse** | `Bash\|Write\|Edit` | `ammo-next-step-reminder.sh` | Injects stage-specific next-step reminders (14-state lookup, 30s throttle) |
| **SubagentStop** | — | `ammo-subagent-release.sh` | Releases subagent GPU reservations |
| **PreCompact** | — | `ammo-precompact.sh` | Saves campaign state checkpoint before compaction |
| **SessionStart** | — | `ammo-postcompact.sh` | Injects resume context after compaction |
| **WorktreeCreate** | — | `worktree-create-with-build.sh` | Sets up build environment in new worktrees |
| **WorktreeRemove** | — | `worktree-remove-cleanup.sh` | Cleans up worktree resources |

Inline DA verification (integrated into helper agents — replaces non-functional Stop hook DAs for team members):
- **ammo-transcript-monitor** → Periodic transcript-based review. Implementation (Stages 4-5) only: scope adherence + reward-hacking detection → `rounds/{N}/tracks/{op_id}/monitor_audits/{monitor_id}_observations.md`. Debate champions do not have monitors — the adversarial debate structure (Phase B critique + Phase C rebuttal) provides quality control.
- **ammo-impl-champion** (self-validation) → kernel correctness & speedup: the champion writes its own correctness test + CUDA-graph speedup bench. Gates 5.1b/5.3 come from the champion's sweep script.
- **ammo-researcher** Stop → DA checks for ungrounded claims (subagent — Stop hooks work correctly).

## State Management

`kernel_opt_artifacts/state.json` is the single source of truth for campaign progress. The frontend and backend parse it with exact key matching — use only canonical values (see Stage Values below). A PostToolUse hook (`ammo-state-validate.sh`) fires after every Write/Edit to state.json and warns if values are non-canonical. Full schema: `.claude/schemas/state.schema.json`.

**Session ID**: The lead MUST record the session ID in state.json at campaign start: `"session_id": "<uuid>"`. This enables the eval pipeline to extract ground-truth timing and agent cost data from session logs automatically.

**Stage timestamps**: Each stage's `started_at`/`completed_at` lives directly on the round's stage sub-object (e.g., `campaign.rounds[$IDX].debate.started_at`). The lead sets `started_at` when entering a stage and `completed_at` when leaving. The eval pipeline may refine these with precise values from session logs. The `campaign.agent_costs` field is auto-populated by the eval pipeline.

**Round-centric hierarchy**: All mutable pipeline state lives under `campaign.rounds[N]`. There is no separate "live state" vs "archive" — the current round and past rounds have identical shapes. `new_target.py` pre-populates `rounds[0]` at bootstrap, so `campaign.rounds[campaign.current_round - 1]` always dereferences a valid entry.

**Authoritative config whitelist**: The complete set of authoritative `state.json.campaign.config.*` fields is defined by `.claude/schemas/state.schema.json` (see `campaign.config.properties`). Any config-field name not in the schema is non-authoritative — do not cite it as policy, do not invent it. The schema is the single source of truth; prose references in this file are derived views for humans.

```json
{
  "target": {"model_id": "...", "hardware": "...", "dtype": "...", "tp": 1, "ep": 1, "component": "auto"},
  "session_id": null,    /* lead records session UUID at campaign start */
  "gpu_resources": {"gpu_count": 1, "gpu_model": "...", "memory_total_gib": 0, "cuda_visible_devices": "0"},
  "campaign": {
    "schema_version": "4.0",             /* v4.0: map-based e2e_latency, per_bs_verdict siblings, kernel_speedup split, additionalProperties locking */
    "status": "active",                  /* active | paused | campaign_complete | campaign_exhausted */
    "current_round": 1,                  /* 1-based; 0-based array index is current_round - 1 */
    "current_stage": "1_baseline",       /* see Stage Values below */
    "config": {
      "min_e2e_improvement_pct": 0.25,                 /* example; actual value scaffolded by new_target.py. See references/validation-defaults.md */
      "noise_tolerance_pct": 0.5,                     /* per-BS verdict: speedup >= (1.0 - this/100) = NOISE. From target.json gating block. */
      "catastrophic_regression_pct": 5.0              /* per-BS verdict: speedup < (1.0 - this/100) = CATASTROPHIC. From target.json gating block. */
    },
    "cumulative_speedup_vs_round1": 1.0, /* round_1_baseline_latency_s ÷ current_integrated_latency_s. Direct ratio, recomputed on every SHIP from the Stage 6 integration measurement. NEVER multiplicative across rounds. */
    "round_1_baseline_latency_s": null,  /* anchor latency captured at round 1 Stage 1 (before any SHIP); frozen after round 1 */
    "shipped_optimizations": [],         /* always [{op_id, round, classification}] — never bare string op_ids */
    "agent_costs": [],                   /* auto-populated by eval pipeline from session logs */
    "rounds": [ /* at least one entry — see round shape below. Pre-populated by new_target.py. */ ]
  }
}
```

Each `campaign.rounds[N]` entry carries per-stage timestamps (`baseline`, `bottleneck_mining`, `debate`, `parallel_tracks`, `integration`, `campaign_eval`), a round `status` ∈ `{IN_PROGRESS, completed, EXHAUSTED, SHIPPED, FAILED}`, a `team_name`, a `debate` sub-object (`candidates`, `rounds_completed`, `max_rounds`, `selected_winners`), a `parallel_tracks.tracks` dict keyed by op_id, an `integration` sub-object (`status` ∈ `{pending, in_progress, validated, single_pass, combined, gated_pass, completed, exhausted, failed, skipped}`, plus passing/failed/selected candidates and conflict fields), and `shipped`/`dropped`/`cumulative_speedup_after`/`combined_e2e_*`/`note`/`round_summary`. Full shape: `.claude/schemas/state.schema.json`.

Each track entry (`campaign.rounds[$IDX].parallel_tracks.tracks[op_id]`) carries:
- `status` ∈ `{IN_PROGRESS, PASS, GATING_REQUIRED, GATED_PASS, FAIL, GPU_BLOCKED}` (terminal: `{PASS, GATED_PASS, FAIL}`)
- `verdict` ∈ `{PASS, GATING_REQUIRED, GATED_PASS, FAIL, null}`
- `classification` ∈ `{lossless, lossy, null}` (from debate summary, set at T7)
- `per_bs_verdict`: `{BS: PASS|NOISE|REGRESSED}` or null
- `gating` (populated only when verdict is GATED_PASS): `{mechanism, env_var, dispatch_condition, crossover_threshold_bs, crossover_probing, pre_gating_results, post_gating_results, regressing_bs}`
- `kernel_speedup`/`kernel_speedup_warm`/`kernel_speedup_cold`/`e2e_speedup`, `gate_5_1a`/`gate_5_2`, `worktree_branch`/`commit_sha`, `fail_reason`, `validation_results_path`, `remediation_items_status`
- `gate_5_1a_metrics` (object|null): `{overall, max_abs_err, shapes_tested}` — orchestrator-enriched after gate 5.1a passes (see § FE Metric Enrichment below)
- `gate_5_2_metrics` (object|null): `{weighted_speedup_cold, weighted_speedup_warm, shapes_tested}` — orchestrator-enriched after gate 5.2 passes

Downstream consumers and `ammo-state-validate.sh` match these canonical strings exactly — custom values (`"track_complete"`, legacy `"PASSED"`/`"FAILED"`) fall through to default handling. UPDATE CADENCE: update `status` and `verdict` immediately when each gate completes, not only at end-of-stage. Checkpoint list: `orchestration/parallel-tracks.md`.

### FE Metric Enrichment (Orchestrator-Owned)

The frontend reads numeric metrics from `state.json` directly — there is no `.metrics.json` sidecar layer. The orchestrator MUST enrich `state.json` with the fields below at the listed trigger points so L2/L3 dashboards render correctly. All fields are additive (the schema is already updated to allow them); skip silently if a source file is missing or malformed (log a warning, do not block).

| Trigger | Source file | Target state.json path | Fields to write |
|---------|-------------|------------------------|-----------------|
| Gate 5.1a passes (per track) | `rounds/{N}/tracks/{op_id}/validator_tests/gate_5_1a_results.json` | `campaign.rounds[$IDX].parallel_tracks.tracks[op_id].gate_5_1a_metrics` | `{overall: <PASS\|FAIL>, max_abs_err: <number>, shapes_tested: <int>}` |
| Gate 5.2 passes (per track) | `rounds/{N}/tracks/{op_id}/validator_tests/gate_5_2_results.json` | `campaign.rounds[$IDX].parallel_tracks.tracks[op_id].gate_5_2_metrics` | `{weighted_speedup_cold: <number>, weighted_speedup_warm: <number>, shapes_tested: <int>}` |
| T2 mining complete (after Stage 2 audit PASS) | `rounds/{N}/mining/bottleneck_analysis.md` § structured table + profile summary | `campaign.rounds[$IDX].bottleneck_mining` | `{top_component: <string>, top_f_decode_pct: <number>, amdahl_ceiling: <number>, decode_frac: <number 0..1>, component_breakdown: [{name, pct}, ...]}` (in addition to the existing `top_bottleneck_share_pct`) |
| T7 debate complete (after winner selection) | Debate artifacts in `rounds/{N}/debate/` (proposals/, *.md rationale files, summary.md) | `campaign.rounds[$IDX].debate` | `{proposal_count: <int>, rationale_count: <int>, scoreboard: <object>, champions_count: <int>, winners: <[op_id]>, result: <string>}` |

**Enrichment timing**:
- **Gate metrics**: write immediately when a gate result file lands (same Edit/Write turn as setting `gate_5_1a` or `gate_5_2` on the track). The orchestrator polls the champion's gate result files between turns.
- **Mining metrics**: write right after `bottleneck_mining.completed_at` is set, in the same orchestrator state-update step. Parse the structured table from `bottleneck_analysis.md` (markdown table — top row = top bottleneck). `amdahl_ceiling = top_f_decode_pct` (decimal fraction × 100 to express as percent improvement under infinite kernel speedup; see `references/e2e-delta-math.md`). `decode_frac` is the decode-phase share of total E2E (0..1) from the profile summary. `component_breakdown` mirrors the markdown table rows as `[{name, pct}, ...]` (sorted desc by pct) — used by the L2 mining tooltip to render the component table. **All five fields are required by the schema once `completed_at` is set; the validation hook will reject the state.json write otherwise.**
- **Debate metrics**: write at T7 alongside `selected_winners`. `proposal_count` = count of files in `rounds/{N}/debate/proposals/`. `rationale_count` = count of `*_argument.md` + `*_critique_*.md` + `*_rebuttal.md` across debate rounds. `champions_count` = number of distinct `champion_id` values that produced proposals. `winners` mirrors `selected_winners` for FE convenience. `scoreboard` is free-form (typically `{op_id: {weighted_total, feasibility, evidence_tier, expected_e2e_pct}}`). `result` is a one-liner summary, e.g., `"2 winners: OP-001 (lossless), OP-002 (quant)"`.

**Forward compatibility**: If a source file is malformed or absent, write the surrounding object with the fields it can fill and leave the rest as `null`. Never write a stale value — null is preferred to wrong. The schema permits `null` for every enrichment field.

### Stage Values

`1_baseline` -> `2_bottleneck_mining` -> `3_debate` -> `4_5_parallel_tracks` -> `6_integration` -> `7_campaign_eval` -> `7b_report` (on campaign termination) | {next round after SHIP — reset `current_stage` to `2_bottleneck_mining`} | {next round after EXHAUSTED — reset `current_stage` to `3_debate` (reuse existing mining), unless `rounds[$IDX].mining_invalidated` is set → then `2_bottleneck_mining`}

Stage values are round-agnostic — the round is tracked in `campaign.current_round`. Terminal pseudo-stages (`campaign_complete`/`campaign_exhausted`) do NOT exist in `current_stage` — terminal state is carried by `campaign.status`. When a campaign ends, `campaign.status` becomes the terminal value and `current_stage` stays at whatever stage was last active (typically `7_campaign_eval` or `7b_report`).

**Update cadence:**
- Update `campaign.current_stage` at every stage transition
- Set `campaign.rounds[$IDX].{stage}.completed_at` when completing a stage (where `$IDX = campaign.current_round - 1` and `{stage}` is the matching sub-object: `baseline`, `bottleneck_mining`, `debate`, `parallel_tracks`, `integration`, `campaign_eval`)
- Set `campaign.rounds[$IDX].{stage}.started_at` when entering a stage
- Stage `7b_report` has no dedicated sub-object — it runs in the background after the campaign reaches terminal status; report progress is tracked by the presence of `REPORT.md` on disk
- Update `campaign.current_round` only when starting a new round (integer >= 1)
- Update `campaign.rounds[$IDX].parallel_tracks.tracks[op_id].status` and `.verdict` immediately when each gate completes — this is what downstream consumers read for live progress
- When a gate result file lands (`gate_5_1a_results.json`, `gate_5_2_results.json`), populate the matching `gate_5_1a_metrics` / `gate_5_2_metrics` track field in the same Edit/Write turn that flips `gate_5_1a` / `gate_5_2` to `PASS|FAIL` — see § FE Metric Enrichment for the exact field map
- When `bottleneck_mining.completed_at` is set, also write `bottleneck_mining.{top_component, top_f_decode_pct, amdahl_ceiling, decode_frac, component_breakdown}` from `bottleneck_analysis.md` (all five required by schema once `completed_at` is set; the L2 mining tooltip consumes `decode_frac` for the phase bar and `component_breakdown` for the multi-row breakdown table)
- When `debate.completed_at` is set (T7), also write `debate.{proposal_count, rationale_count, scoreboard, champions_count, winners, result}`
- Use only canonical status/verdict values (listed above) — the validation hook blocks non-conforming writes

On `campaign_complete` or `campaign_exhausted`, report generation (T20) is spawned in the background before the session ends.

## Communication Patterns

- **Blocker escalation**: Subagent returns error -> lead investigates.
- **Debate moderation**: Lead broadcasts phase starts, champions message back on completion.
- **Critical stop**: Lead broadcasts to halt debate team if needed.
- **Shutdown**: Terminate an agent via `SendMessage(to=<agent_id>, message={"type": "shutdown_request"})`; it approves only when its work is complete, so confirm each `shutdown_approved` before relying on the agent being gone. Debate champions shut down after selection; round team persists until all implementation tracks complete, then TeamDelete.

## Helper Scripts

Run, don't modify:

- `scripts/new_target.py` — Scaffold artifact directory
- `scripts/collect_env.py` — Capture environment
- `scripts/verify_stage2_gate.py` — T5 gate (bottleneck_analysis.md + baseline artifacts). v2: `--round N` reads from `rounds/{N}/mining/bottleneck_analysis.md` and `rounds/{N}/sweeps/baseline/e2e_latency_results.json`.
- `scripts/gpu_status.py` — Print current GPU reservation state (orchestrator/human diagnostic)
- `scripts/run_vllm_bench_latency_sweep.py` — Batch E2E benchmark runner (GPU-locked). Supports `--labels baseline` (default) for baseline-only sweeps (Stage 1), `--labels opt` for opt-only (Stage 5), or `--labels baseline,opt` for A/B comparison. Also supports `workload_matrix` for multi-dimensional sweeps, `--nsys-profile --nsys-mode node --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1` for selected-step nsys traces (`--nsys-trace cuda-sw` on Blackwell (B200/B300)), and `--nsys-mode graph` for optional diagnostic enrichment. Gate 5.1b correctness: `--capture-golden-refs` (Stage 1, saves golden references) and `--verify-correctness --baseline-from $STAGE1_DIR` (Stage 5, compares GSM8K accuracy against golden refs; `opt_accuracy >= baseline_accuracy - tolerance` at n=1319 by default, with 1.0pp tolerance (configurable via `--correctness-tolerance-pct`; set to 0.0 for strict)). **WARNING**: If `opt_env` in target.json still contains placeholder keys (e.g., `<ENABLE_FLAG>`), the script will fail fast — update `opt_env` before running with `--labels opt`.
- `scripts/generate_validation_report.py` — Structured reporting
- `scripts/gpu_force_clear.py` — Force-clear stale GPU reservations after crashes (orchestrator-only)

## References

| Topic | File |
|-------|------|
| Technology selection (authoring language: Triton/CuTeDSL/CUTLASS/CUDA C++) | `references/technology-selection.md` |
| Nsys profiling | `references/nsys-profiling-guide.md` |
| Validation gates | `references/validation-defaults.md` |
| CUDA graph safety | `references/cudagraph-safety.md` |
| torch.compile contract (6 invariants for kernel integration) | `references/torch-compile-contract.md` |
| E2E latency | `references/e2e-latency-guide.md` |
| E2E delta math | `references/e2e-delta-math.md` |
| GPU hardware specs | `references/gpu-configs.md` |
| Optimization techniques | `references/optimization-techniques.md` |
| Fusion feasibility | `references/fusion-feasibility-heuristics.md` |
| Code templates | `references/code-templates.md` |
| Debate scoring | `references/debate-scoring-rubric.md` |
| Debate-phase agent rules | `references/debate-rules.md` |
| Implementation-phase agent rules | `references/impl-track-rules.md` |
| GPU reservation pattern | `references/gpu-pool.md` |
| Champion common patterns (delegation, messaging, triage) | `references/champion-common-patterns.md` |
| Audit invariants (T_AUDIT checklists) | Delivered to auditor via hook — not directly referenced |

**Architectural boundary**: Agent files reference `references/*.md` for domain rules. Orchestration docs (`orchestration/*.md`) are orchestrator-only — agents do not read them directly.

## Orchestration Docs

| Topic | File |
|-------|------|
| Debate protocol | `orchestration/debate-protocol.md` |
| Parallel tracks | `orchestration/parallel-tracks.md` |
| Integration logic | `orchestration/integration-logic.md` |
| Audit protocol (T_AUDIT gates) | `orchestration/audit-protocol.md` |

## Escalation Protocol

When a subagent returns an error or a gate fails:

| Severity | Action |
|----------|--------|
| **critical** | HALT current stage. Attempt resolution via `ammo-investigator` first (it can often find a safe path the rules don't spell out). Escalate to the **user** only if the investigation yields no recovery path. Broadcast halt if debate team active. |
| **major** | Investigate (rule lookup, or `ammo-investigator` if genuinely unclear), adjust constraints, re-run subagent. |
| **minor** | Document and continue. |

**Note**: "HALT" means pause the current stage and resolve — NOT terminate the campaign.

Save blocker details to `{artifact_dir}/blockers/{stage}_{date}.md` (top-level — blockers are campaign-scoped, not round-scoped).

## Resume Protocol

After interruption or compaction:

1. Read this skill file (you are the LEAD — delegate, don't implement).
2. Read `state.json` from artifact directory.
3. Check `campaign.status` and `campaign.current_stage` to determine where you are.
3b. Check GPU reservation state: run `python .claude/skills/ammo/scripts/gpu_status.py`. If stale reservations exist from the crashed session, clear them: `python .claude/skills/ammo/scripts/gpu_force_clear.py --all --session-id <crashed_session_id>`. If the crashed session ID is unknown, use `--force-no-session`. Re-spawned agents will have their GPUs auto-reserved by hooks when they run commands.
4. If Stage 3 debate active: check debate artifacts in `rounds/{N}/debate/` (proposals, summary.md, micro_experiments).
5. If Stages 4-5 active: check `campaign.rounds[campaign.current_round - 1].parallel_tracks.tracks` for worktree paths and per-track status. Resume monitoring and gating. Before proceeding, reconcile each track entry against on-disk gate result files under `rounds/{N}/tracks/{op_id}/validator_tests/` and `rounds/{N}/sweeps/opt/{op_id}/` — if the track entry is stale (missing fields that the gate files already contain), merge them in per the checkpoint list in `orchestration/parallel-tracks.md`.
6. Resume from last completed gate. If `current_stage = "2_bottleneck_mining"` and `bottleneck_mining.completed_at` is unset, dispatch `task_type: mining` to re-run mining.
7. Read `campaign.current_round` to determine which round is active.
8. If the `campaign` key or `campaign.rounds[0]` is missing (malformed state.json): re-run `new_target.py` to re-emit the bootstrap shape. The current schema requires at least one round entry.
9. **One-time mining-enrichment backfill** (added 2026-05): For every round where `bottleneck_mining.completed_at != null` but any of `decode_frac` / `component_breakdown` is `null`, re-parse `rounds/{N}/mining/bottleneck_analysis.md` (markdown table + profile summary) and write the missing fields in the same state.json update that resumes the round. This is required because `decode_frac` and `component_breakdown` were added to the schema after sidecar removal (commit `5a16b78`); pre-existing mining records do not yet carry them, and the validation hook will block the next state.json write on those rounds until they are populated. Do this BEFORE any other state.json mutation on resume — once one round is missing the fields, every subsequent write fails. If `bottleneck_analysis.md` is missing/malformed: when `completed_at` is null (mining still in progress), it is fine to leave the new fields as null. When `completed_at` is already set, the schema rejects null — surface a blocker rather than silently corrupting the record.
10. The Stop hook will block session end while campaign is active — complete the current stage. Do NOT set `campaign.status` to `"paused"` autonomously to satisfy the Stop hook; setting `paused` requires explicit user request (see Campaign State Transitions). If you cannot complete the stage, escalate the blocker through the Escalation Protocol.
11. If `campaign.status` is `campaign_complete` or `campaign_exhausted` but no `REPORT.md` exists, spawn the `ammo-report-writer` subagent (T20).

## Quick Start Examples

**Example 1**: `User: "Use ammo for Qwen3-30B-A3B on L40S TP=1"` ->

1. Scaffold artifact directory (campaign initialized).
2. Round 1: invoke ammo-researcher subagent for baseline + bottleneck mining.
3. Run gates, spawn debate team for top candidates.
4. Select winners, create parallel worktree tracks.
5. Implement + validate in parallel.
6. Integration if multiple pass → SHIP or round-EXHAUSTED.
7. Campaign evaluation: record round, read `f`, compare to `min_e2e_improvement_pct` (mechanical — no user interaction).
8. If `f >= threshold`: run mining on new baseline (if SHIP) or inject exhaustion note into debate prompt (if EXHAUSTED) → new round immediately. Champions run the technology-selection function per proposal. Repeat until `f < threshold`.
9. When `f < threshold`: declare `campaign_complete` or `campaign_exhausted`, spawn report.

**Example 2**: Resume -> Read `state.json`, check `campaign.current_round` and active stage, resume from last gate.

**Example 3**: Mid-campaign resume after compaction -> SessionStart hook injects campaign context (round, status, cumulative speedup). Read `state.json`, resume current round.
