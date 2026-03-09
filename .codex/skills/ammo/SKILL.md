---
name: ammo
description: End-to-end vLLM kernel optimization for a specific deployment envelope, from baseline capture through debate, isolated implementation, and validation. Use when Codex must optimize and validate a real vLLM inference path for a concrete model, hardware, dtype, and TP/EP target. Do not use for profiling-only work, postmortem review, or general opportunity mining.
---

# AMMO

Optimize vLLM inference kernels and ship only candidates that beat a production-parity Stage 1 baseline without correctness regressions.

Run scripts from this skill directory, for example `.codex/skills/ammo/scripts/...`.

## Required Inputs

User or repo context must provide:

- `model_id`
- `hardware`
- `dtype`
- `tp`
- optional `ep`
- optional `component` (`auto` allowed)
- target batch sizes or a clear default envelope

If one of the blocking inputs is missing and cannot be discovered locally, stop and ask for it before starting Stage 1.

## Do Not Use AMMO For

- profiling-only or bottleneck-mining-only tasks
- review-only or postmortem-only tasks
- vague “make this faster” requests with no concrete deployment envelope

For profiling/mining without implementation/validation, use a narrower skill instead of AMMO.

## Stage 0 Preflight

Before Stage 1, the lead must scaffold the artifact directory and run preflight.

```bash
python .codex/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP>

python .codex/skills/ammo/scripts/preflight_check.py \
  kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_tp{tp}
```

Preflight is blocking. Do not proceed until:

- repo `.venv` is active for Python commands
- `import vllm` works
- visible GPUs match the requested hardware target
- required target inputs are present in `target.json`

`nsys` availability may be recorded as a warning only when the run can still continue with an explicitly reduced profiling plan.

## Orchestration

Use Codex multi-agent roles configured in `.codex/config.toml`:

- `ammo-researcher`
- `ammo-champion`
- `ammo-implementer`

Use `spawn_agent`, `send_input`, `wait`, and `close_agent` for role coordination.

The lead owns every blocking gate and every stage transition.

## 7-Stage Workflow

| Stage | Owners | Required Output |
|---|---|---|
| Stage 0: Preflight | lead | `runs/preflight_report.json` |
| Stage 1: Baseline capture | lead + researcher | `constraints.md` |
| Stage 2: Bottleneck mining | lead + researcher | `bottleneck_analysis.md` |
| Stage 3: Candidate debate | lead + 3 champions | `debate/summary.md` plus proposal JSON/MD |
| Stages 4-5: Worktree tracks | lead + implementers | `tracks/{op_id}/evidence.json` and `tracks/{op_id}/validation_results.md` |
| Stage 5 gate | lead | `runs/validation_gate_report.json` |
| Stage 6: Integration decision | lead | `state.json.integration` and `integration.md` |

## Debate Rules

- Stage 3 always uses 3 champion agents.
- Do not fall back to a single-agent debate.
- If a champion misses a required artifact deadline:
  - re-prompt once
  - respawn the same role once if still incomplete
  - if the debate still cannot produce 3 complete lanes, mark the run `BLOCKED`

Every champion writes both:

- `{artifact_dir}/debate/proposals/{champion_id}_proposal.json`
- `{artifact_dir}/debate/proposals/{champion_id}_proposal.md`

Winner eligibility is strict:

- proxy-only microbench math is not enough to win
- proposals must include integrated-path proof on the real vLLM dispatch or layer path for the exact target shape
- proxy-derived E2E claims must be labeled as bounds, not decisive winner evidence

Do not advance a winner into implementation unless it has integrated-path proof.

## Track Validation Rules

Each winning candidate gets:

- an isolated worktree
- one `ammo-implementer`
- one authoritative evidence file: `{artifact_dir}/tracks/{op_id}/evidence.json`
- one generated human summary: `{artifact_dir}/tracks/{op_id}/validation_results.md`

The implementer owns implementation and validation end to end.

Official optimized E2E runs must include:

- Stage 1 baseline reuse
- production-parity settings
- admissibility status
- explicit fast-path hit evidence

“Prepared” or enablement logs are not enough. Official optimized runs require direct proof such as structured hit counters or equivalent runtime evidence with `hits >= 1`.

## Stage Gates

Minimum gate commands:

```bash
python .codex/skills/ammo/scripts/verify_phase1_baseline.py \
  {artifact_dir} \
  --json-output {artifact_dir}/runs/phase1_gate_report.json

python .codex/skills/ammo/scripts/verify_validation_gates.py \
  {artifact_dir} \
  --json-output {artifact_dir}/runs/validation_gate_report.json
```

Do not advance on `WARN`.

Validation gate semantics:

- `overall_status` describes evidence completeness and consistency
- `track_outcomes` describe candidate pass/regress/fail outcomes
- Stage 6 may proceed only when evidence gates pass, even if no candidate passed Stage 5

## Non-Negotiables

1. Production parity for all performance evidence: CUDA graphs plus production-equivalent compile settings
2. vLLM production kernels as the correctness and performance baseline
3. Numerical correctness via `torch.allclose()` or equivalent
4. Stage 1 baseline reuse for all Stage 5 and Stage 6 E2E comparisons
5. E2E GPU sequencing through the lock-based sweep workflow
6. Full-model E2E validation before `SHIP`
7. Custom-kernel-only debate candidates
8. Structured evidence is authoritative; Markdown is summary only

## Worktree Management

Create worktrees explicitly:

```bash
bash .codex/skills/ammo/scripts/create_worktree_with_build.sh ammo-track-{op_id} ammo/{op_id}
```

Clean them up explicitly:

```bash
bash .codex/skills/ammo/scripts/remove_worktree_cleanup.sh .codex/worktrees/ammo-track-{op_id}
```

## State Management

`state.json` tracks stage, debate health, track metadata, and the integration decision.

Recommended per-track record:

```json
{
  "parallel_tracks": {
    "op001": {
      "status": "PASSED",
      "branch": "ammo/op001",
      "worktree_path": ".codex/worktrees/ammo-track-op001",
      "validation_results_path": "tracks/op001/validation_results.md",
      "evidence_path": "tracks/op001/evidence.json",
      "correctness": true,
      "kernel_speedup": 1.12,
      "e2e_speedup": 1.04
    }
  }
}
```

## Helper Scripts

- `scripts/new_target.py` — scaffold artifact directory plus `state.json`
- `scripts/preflight_check.py` — Stage 0 preflight gate
- `scripts/collect_env.py` — capture environment snapshot
- `scripts/verify_phase1_baseline.py` — Stage 1 gate
- `scripts/run_vllm_bench_latency_sweep.py` — E2E runner with GPU lock and run metadata
- `scripts/generate_validation_report.py` — render `validation_results.md` from `evidence.json`
- `scripts/verify_validation_gates.py` — Stage 5 structured gate
- `scripts/create_worktree_with_build.sh` — create worktree and thin `.venv`
- `scripts/remove_worktree_cleanup.sh` — clean up worktree

## References

Read as needed:

- `references/nsys-profiling-guide.md`
- `references/validation-defaults.md`
- `references/cudagraph-safety.md`
- `references/e2e-latency-guide.md`
- `references/e2e-delta-math.md`
- `references/gpu-configs.md`
- `references/optimization-techniques.md`
- `references/fusion-feasibility-heuristics.md`
- `references/code-templates.md`
- `references/debate-scoring-rubric.md`
- `references/validation-troubleshooting.md`
- `references/da-audit-checklist.md`
- `references/claude-codex-equivalents.md`

## Resume Protocol

After interruption or compaction:

1. read this file
2. read `{artifact_dir}/state.json`
3. read the latest gate reports in `{artifact_dir}/runs/`
4. reconstruct active worktrees from `parallel_tracks`
5. resume from the earliest incomplete gate
6. do not rely on memory when artifact files disagree
