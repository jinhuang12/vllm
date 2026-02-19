---
name: ammo
description: "Primary orchestrator for staged optimization of systems that heavily use NVIDIA GPUs. Use for production-parity constraints and baseline capture, profiler-driven opportunity ranking, default parallel top-3 implementation via Codex multi-agent worktree subagents, strict orchestration, hybrid validation (subagent local gates plus orchestrator acceptance gates), bounded-exhaustion autonomous looping until explicit termination criteria, and end-to-end ROI decisioning."
---

# AMMO

AMMO is a staged workflow skill for optimizing any NVIDIA GPU-heavy system.

AMMO is project-agnostic. It does not assume a specific framework, model family, or kernel type.

## Trigger contract

Use AMMO when the task is:
- identify and prioritize GPU bottlenecks from real profiling evidence,
- implement and validate candidate optimizations,
- decide whether to ship based on correctness and meaningful end-to-end impact.

## Search anchors

production parity, baseline truth snapshot, nsys, ncu, bottleneck ranking, incumbent protection, correctness gate, local gate, e2e gate, maintenance ROI, rollback envelope

## Workflow (canonical)

- Follow `orchestration/task-guide.md`.
- For Stage 4+, follow `orchestration/multi-agent-playbook.md` as the default execution policy.
- Keep per-target state in `{artifact_dir}/artifact_bundle.json`.
- Treat `{artifact_dir}/constraints.md` as the mandatory Stage 1 source of truth for Stage 3 planning.
- Use adapter manifests/docs for project-specific commands, evidence signatures, and runtime-specific utility scripts.
- If validation gates fail repeatedly, use `orchestration/investigation-prompts.md`.
- Use AMMO scripts/manifests as command truth; do not invent CLI flags.


## Scaffold discipline (mandatory)

- `scripts/new_run.py` scaffolds placeholder artifacts; those files are not stage-complete outputs.
- If baseline/profile/mining evidence is captured, the same run must rewrite stage docs with concrete, evidence-linked content.
- Never finish a run with reference-only scaffold text in required artifacts.
- If evidence is insufficient to complete a stage, mark the stage `blocked` with explicit blocker notes and update `artifact_bundle.json` accordingly.
- If subagents run in parallel, unresolved subagent artifacts or missing evidence manifests are stage-incomplete outputs.

## Hard failure conditions

Treat the run as incomplete/failed (not complete) if any are true at stop time:
- required stage artifacts still contain scaffold-only instruction text,
- `artifact_bundle.json` stage/status/constraints fields were not updated to reflect collected evidence,
- Stage 2 mining outputs exist but are not synthesized into ranked opportunities,
- Stage 4 is `not_run` without explicit user-approved planning-only scope,
- baseline and candidate validation paths are indistinguishable but reported as promotion-grade A/B evidence,
- highest-priority feasible candidate was not attempted and no blocker-driven fallback chain was documented,
- top-3 parallel mode is selected but fewer than required worktrees/subagents were launched without explicit blockers,
- subagent local validation claims are accepted without orchestrator acceptance re-validation,
- multiple candidates are stacked without incremental re-validation evidence,
- `validation_results.md` is missing machine-readable summary block required by `scripts/check_validation_results.py`,
- bounded-exhaustion runs end with `exhaustion_state.terminate_reason=pending`,
- any mandatory autonomy checker fails while run is marked complete.

## Non-negotiables

1. Measure production parity baseline before proposing optimizations.
2. Require correctness evidence before any performance claim.
3. Protect incumbent wins and compare against best-so-far.
4. Bound enablement envelope and preserve fallback behavior.
5. Make ship/reject decisions with explicit ROI and complexity scoring.

## Primary references

- `references/optimization-plan-template.md`
- `references/constraints-template.md`
- `references/validation-defaults.md`
- `references/roi-tier-policy.md`
- `references/maintenance-decision-template.md`
- `references/adapter-contract.md`
- `orchestration/multi-agent-playbook.md`
- `references/core/INDEX.md`
- `references/core/nsys-profiling-guide.md`
- `references/core/profiling-launch-vs-kernel.md`
- `references/core/e2e-delta-math.md`
- `references/core/cudagraph-safety.md`
- `references/core/fusion-feasibility-heuristics.md`
- `references/core/gpu-configs.md`
- `references/core/tiling-config.md`

## Validation references

- `validation/E2E_LATENCY_GUIDE.md`
- `validation/AUTONOMY_EVAL_SCENARIOS.md`

## Adapter references

- `references/adapters/adapter-template.md`
- `references/adapters/adapter-template.manifest.json`
- project-specific adapter docs/manifests under `references/adapters/`

## Schemas

- `schemas/artifact_bundle.v4.json`
- `schemas/artifact_bundle.v3.json`
- `schemas/artifact_bundle.v2.json` (legacy)
- `schemas/adapter_manifest.v1.json`
- `schemas/opportunity_record.v1.json`
- `schemas/maintenance_decision.v1.json`
- `schemas/subagent_evidence_manifest.v1.json`
- `schemas/autonomy_run_state.v1.json`
- `schemas/validation_record.v1.json`

## Scripts

- `scripts/new_run.py`: scaffold `{artifact_dir}` and `artifact_bundle.json`.
- `scripts/collect_env.py`: capture reproducible environment evidence.
- `scripts/run_adapter_bench.py`: execute adapter command templates for baseline/profile/e2e phases.
- `scripts/render_decision_report.py`: render maintenance decision summary from recorded evidence.
- `scripts/translate_state_to_bundle.py`: optional legacy state conversion into v3 `artifact_bundle.json`.
- `scripts/check_constraints.py`: Stage 1 quality gate for autonomous runs.
- `scripts/check_optimization_plan.py`: Stage 3 plan quality gate for autonomous runs.
- `scripts/check_parallel_evidence.py`: Stage 5 manifest/bundle consistency gate for autonomous runs.
- `scripts/check_validation_results.py`: validate Stage 5 evidence summary against bundle and manifests.
- `scripts/check_autonomy_completion.py`: validate bounded-exhaustion completion state and required checker pass status.
- Runtime-specific helper scripts are documented by their project adapter docs.

## Required outputs per target

- `{artifact_dir}/artifact_bundle.json`
- `{artifact_dir}/constraints.md`
- `{artifact_dir}/optimization_plan.md`
- `{artifact_dir}/implementation_notes.md`
- `{artifact_dir}/validation_results.md`
- `{artifact_dir}/maintenance_decision.md`
- `{artifact_dir}/integration.md`
- `{artifact_dir}/parallel/` (orchestrator + subagent run-state and evidence manifests)
- `{artifact_dir}/parallel/validation_records/` (per-gate JSON records)

## Example prompts

- "Use AMMO to profile this GPU-heavy service and rank optimization opportunities."
- "Implement the top two candidates and validate correctness plus end-to-end impact."
- "Tell me if this optimization is worth long-term maintenance effort."
