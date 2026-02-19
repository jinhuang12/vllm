# AMMO Task Guide (Canonical)

This is the source of truth for AMMO staged execution.

## Stages

1. Constraints + baseline capture (production parity)
2. Bottleneck mining and ranking
3. Candidate selection and optimization plan
4. Implementation
5. Validation (gated)
6. Maintenance decision and integration

## How to use this guide

1. Create an `{artifact_dir}` per target envelope.
2. If `artifact_bundle.json` exists, resume from recorded `stage` and `status`.
3. If bundle version is legacy (`version=2`), migrate to v3 fields before Stage 3 planning.
4. Before each stage, write a 3-7 step micro-plan.
5. Execute all stage checklist items before advancing.
6. Update `artifact_bundle.json` after each stage or failed gate.
7. If a gate fails, set `status=needs_investigation` and follow `investigation-prompts.md`.
8. For unattended runs, always complete artifact synthesis after evidence capture; do not stop after profiling/mining.
9. Default execution mode is parallel top-3 in Stage 4 unless explicit serial override is recorded.

## Command discipline

- Commands must come from AMMO scripts and adapter manifests.
- Do not invent CLI flags.
- If command parameters are unknown, mark them missing and resolve explicitly.
- Use markdown quality bars/templates as the authoritative stage-completeness checks.
- For autonomous runs, helper validators become mandatory stage gates:
  - `scripts/check_constraints.py`
  - `scripts/check_optimization_plan.py`
  - `scripts/check_parallel_evidence.py`
  - `scripts/check_validation_results.py`
  - `scripts/check_autonomy_completion.py`

## Artifact completion guardrails

- Treat scaffolded artifacts as placeholders until rewritten with concrete evidence and decisions.
- A stage cannot be marked complete if required artifacts still contain reference-only scaffold text.
- If evidence exists but artifact synthesis is incomplete, keep `status=in_progress` (or `blocked` with blocker notes), not `pending`.
- After each stage, update both artifact content and `artifact_bundle.json` state fields in the same run.
- Do not declare run completion while required artifacts are scaffold-only.

## End-of-run self-review (mandatory)

Before stopping, confirm all required outputs are evidence-populated:
- `constraints.md` includes concrete baseline/profile evidence and adapter appendix.
- `optimization_plan.md` includes ranked, evidence-linked opportunities and selected hypotheses.
- `optimization_plan.md` includes a concrete baseline-vs-candidate differentiation contract (commands/env/flags and activation proof).
- `implementation_notes.md` records at least one implemented candidate (or explicit `blocked` reason with investigation artifact).
- `validation_results.md` includes explicit gate outcomes (`pass|fail|blocked|not_run`) with rationale.
- `validation_results.md` includes `AMMO_VALIDATION_SUMMARY_V1` machine-readable JSON block.
- `maintenance_decision.md` includes explicit recommendation and rationale tied to Stage 5.
- `artifact_bundle.json` reflects actual stage/status progression and populated constraints fields.
- autonomy fields (`exhaustion_state`, `validation_convergence`, `promotion_history`) are coherent when enabled.
- mandatory validators passed for autonomous completion.

If any check fails, keep run open and finish synthesis, or mark blocked with explicit reasons.

Common unattended failure pattern to avoid:
- baseline/profile/mining finished but `constraints.md` / `optimization_plan.md` / `validation_results.md` / `maintenance_decision.md` stayed scaffold-only.

## Directory convention

Required artifacts:
- `{artifact_dir}/artifact_bundle.json`
- `{artifact_dir}/constraints.md`
- `{artifact_dir}/optimization_plan.md`
- `{artifact_dir}/implementation_notes.md`
- `{artifact_dir}/validation_results.md`
- `{artifact_dir}/maintenance_decision.md`
- `{artifact_dir}/integration.md`

Supporting evidence:
- `{artifact_dir}/profiles/`
- `{artifact_dir}/investigation/`

## artifact_bundle.json state discipline

Track at minimum:
- `stage`
- `status`
- `constraints`
- `incumbent`
- `candidates`
- `validation`
- `decision`
- `last_update`

For parallel top-3 runs (bundle v4), also track:
- `execution_mode`
- `autonomy_mode`
- `orchestration`
- `parallel_candidates`
- `acceptance_validation`
- `exhaustion_state`
- `validation_convergence`
- `promotion_history`

## Stage checklists

### Stage 1: Constraints + baseline capture

- Lock target envelope: project, hardware, workload buckets, parity knobs.
- Scaffold with `scripts/new_run.py`.
- Capture baseline profiling artifacts (nsys and optional ncu).
- Separate launch/API vs kernel-time signals.
- Write `{artifact_dir}/constraints.md` from `references/constraints-template.md`.
- Replace any scaffold placeholder text in `constraints.md` with concrete run evidence.
- Record baseline metrics and initialize incumbent.
- Update `artifact_bundle.json.constraints`:
  - `status=complete|blocked`
  - `snapshot_id`
  - `evidence_links`
  - `parity_signature`
  - `adapter_required_fields_complete`

Exit criteria:
- baseline evidence is reproducible and parity-valid.
- `constraints.md` exists and is complete for the selected adapter.
- `artifact_bundle.json.constraints.status=complete`.
- constraints content meets `references/constraints-template.md` quality bar.
- `constraints.md` is not scaffold-only and cites concrete evidence paths.
- for autonomous runs, `scripts/check_constraints.py --artifact-dir <artifact_dir>` passes.

### Stage 2: Bottleneck mining and ranking

- Extract hotspot evidence from profiling artifacts.
- Run adapter-defined bottleneck mining workflow when provided, and treat adapter-declared mining artifacts as primary ranking evidence.
- Build ranked opportunity backlog with impact/feasibility/risk.
- Record incumbent interaction risk (preserve/modify/disable).

Exit criteria:
- evidence-backed ranked backlog exists.
- ranked backlog cites adapter-declared mining outputs (not ad-hoc hotspot guesses).
- ranked backlog is written into stage artifacts (not left as template placeholders).

### Stage 3: Candidate selection and optimization plan

- Select the top 3 feasible candidates from ranked backlog (or fewer with explicit blocker evidence).
- Mark candidates with explicit rank order; default implementation policy is parallel top-3.
- Record fallback policy for reduced-feasibility scenarios.
- Define expected profiler signature and kill criteria per candidate.
- Build dependency map against incumbent paths.
- Define rollback and enablement envelope.
- Define Stage-4 candidate differentiation plans for each selected candidate:
  - exact baseline command/env/flags,
  - planned candidate command/env/flags shape,
  - planned delta knob(s) or code path switch to be introduced in Stage 4,
  - planned activation proof method (log/profiler signature).
- Cite Stage 1 constraints snapshot ID and evidence links in `optimization_plan.md`.
- Do not leave `optimization_plan.md` in scaffold-only form.
- Do not leave execution-critical placeholders (for example `<candidate parity nsys command>`).
- Define subagent evidence package requirements and checkpoint cadence per candidate.

Exit criteria:
- executable plan with explicit acceptance thresholds.
- plan sections are traceable to `constraints.md` (no uncited assumptions).
- plan quality bar in `references/optimization-plan-template.md` is satisfied.
- all selected candidates have concrete Stage 4 tasks with feature-differentiated candidate paths and worktree ownership.
- for autonomous runs, `scripts/check_optimization_plan.py --artifact-dir <artifact_dir>` passes.

### Stage 4: Implementation

- Implement only Stage 3 candidates.
- Default execution order: launch selected top-3 candidates in parallel.
- Create one git worktree per candidate branch before subagent launch.
- Spawn one subagent per candidate worktree with strict file ownership and evidence contract.
- Orchestrator continuously monitors checkpoints and auto-interrupts on drift:
  - scope violation,
  - parity mismatch,
  - weak/invalid evidence claims,
  - blocker without escalation artifact.
- If a candidate is blocked, write `{artifact_dir}/investigation/blocker.<candidate>.md` and continue remaining candidates.
- Preserve fallback path outside envelope.
- Record implementation notes, checkpoint interventions, and activation evidence.
- Require each subagent to run local smoke + full local gate sequence.
- Default behavior is to execute Stage 4 when optimization implementation is requested.
- Stage 4 is where feature-differentiated candidate paths are created for each selected hypothesis (code/config/dispatch switch).
- If Stage 4 cannot proceed, mark `status=blocked`, write `{artifact_dir}/investigation/blocker.md`, and stop promotion gating until resolved.
- `Stage 4 = not_run` is allowed only for explicit user-approved planning-only scope.

Exit criteria:
- each candidate builds/runs with preserved fallback.
- each candidate path is feature-differentiated from baseline and can be toggled/rolled back.
- subagent evidence manifests exist for each attempted candidate.
- implementation notes record ranked order, subagent interventions, and blocker/fallback decisions.

### Stage 4.1: Parallel orchestration protocol (default)

1. Select `OP-001..OP-003` feasible set from Stage 3.
2. Create worktrees under `{artifact_dir}/parallel/worktrees/<candidate_id>`.
3. Create candidate branches (`ammo/<run>/<candidate_id>`).
4. Spawn subagents and pass:
   - candidate scope, files, and constraints,
   - differentiation contract,
   - local gate and evidence requirements,
   - checkpoint cadence and interrupt policy.
5. Monitor in a loop:
   - poll agent status,
   - review checkpoint evidence,
   - interrupt and steer immediately on contract violations.
6. Collect final subagent evidence manifests and run orchestrator acceptance validation.
7. Promote one or more winners via sequential stack + revalidate.

### Stage 4.2: Bounded-exhaustion autonomy protocol

When `autonomy_mode=bounded_exhaustion`:

1. After each completed wave, re-run Stage 2 mining.
2. Build refreshed ranked backlog and select the next feasible top-3 set.
3. Continue waves while feasible candidates remain above threshold.
4. Track each wave in `artifact_bundle.json.exhaustion_state`:
   - `cycle`
   - `newly_mined_count`
   - `eligible_remaining_count`
   - `terminate_reason`
5. Terminate only with explicit reason:
   - `no_new_above_threshold`
   - `all_remaining_blocked`
   - `all_remaining_non_improving`
   - `manual_stop`
6. No fixed wall-clock/GPU-hour cap is required; termination is evidence-driven by bounded-exhaustion criteria.
7. Convergence guard:
   - if identical failure signature repeats twice for same `(candidate, gate, command_fingerprint)`,
     mark candidate `blocked` and continue with remaining feasible candidates.
8. Keep `promotion_history` updated with acceptance hash per stack step.

### Stage 5: Validation (strict order)

Per candidate, run:
1. Correctness gate
2. Local performance gate
3. End-to-end gate

- Use `references/validation-defaults.md` for pass/fail policy.
- Record evidence and gate outcome in `validation_results.md` for:
  - subagent local validation,
  - orchestrator acceptance validation.
- If baseline and candidate paths are indistinguishable, set kernel/e2e gates to `blocked` (invalid A/B), not `pass`.
- Subagent `pass` is insufficient for promotion; orchestrator acceptance `pass` is required.
- Run `scripts/check_parallel_evidence.py --artifact-dir <artifact_dir>`.
- Run `scripts/check_validation_results.py --artifact-dir <artifact_dir>`.

Exit criteria:
- all gate outcomes recorded with evidence links.
- validation artifact includes explicit per-candidate local + acceptance gate outcomes, even for `blocked`/`not_run` cases.
- both Stage 5 checker scripts pass.

### Stage 6: Maintenance decision and integration

- Apply `references/roi-tier-policy.md`.
- Produce `maintenance_decision.md` from measured outcomes.
- If multiple candidates pass, integrate via sequential stack in priority/ROI order and revalidate after each merge.
- Document integration envelope, rollback, and fallback behavior.
- For autonomous completion, run `scripts/check_autonomy_completion.py --artifact-dir <artifact_dir>`.

Exit criteria:
- explicit ship/ship_restricted/reject decision with rationale.
- maintenance artifact includes explicit rationale tied to Stage 5 outcomes and stacked integration evidence when applicable.
- autonomy completion checker passes for `autonomy_mode=bounded_exhaustion`.

## Blocker policy

Per failure mode:
- maximum 3 hypothesis -> implement -> measure cycles
- maximum 2 deep kernel-analysis cycles before re-scope
- for parallel mode, limits apply per candidate and per orchestration cycle

If blocked:
1. write `{artifact_dir}/investigation/blocker.md`
2. set `status=blocked|escalate_human`
3. if blocker is Stage 1/constraints, do not proceed to Stage 3 planning
4. revise Stage 3 plan before further implementation
