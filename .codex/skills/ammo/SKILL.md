---
name: ammo
description: Profile and optimize vLLM GPU kernels for specific deployments using a stage-gated workflow with production-parity benchmarking, adversarial candidate debate, isolated worktree tracks, and strict correctness plus E2E latency gates.
---

# AMMO — Automated Model Micro-Optimizer

Optimize vLLM inference kernels and ship only changes that beat a production-parity Stage 1 baseline without correctness regressions.

Run scripts from this skill directory, for example `.codex/skills/ammo/scripts/...`.

## Invocation

User provides `model_id`, `hardware`, `dtype`, `tp`, and optionally `ep` plus `component` (`auto` allowed).

Scaffold a target artifact directory:

```bash
python .codex/skills/ammo/scripts/new_target.py \
  --artifact-dir kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_tp{tp} \
  --model-id <MODEL_ID> --hardware <HW> --dtype <DTYPE> --tp <TP>
```

## Orchestration Modes

### Preferred Mode: Custom AMMO Roles

Use Codex multi-agent roles configured in `.codex/config.toml`:

- `ammo-researcher`
- `ammo-champion`
- `ammo-implementer`

### Built-In Fallback

| AMMO role | Built-in fallback |
|---|---|
| `ammo-researcher` | `explorer` |
| `ammo-champion` | `explorer` |
| `ammo-implementer` | `worker` |

### Single-Agent Fallback

Run the same stage and gate semantics sequentially in the main thread. Do not relax gates or non-negotiables.

## 6-Stage Workflow

| Stage | Owners | Primary Output |
|---|---|---|
| Stage 1: Baseline capture | lead + researcher | `constraints.md` |
| Stage 2: Bottleneck mining | lead + researcher | `bottleneck_analysis.md` |
| Stage 3: Candidate debate | lead + 2-4 champions | `debate/summary.md` |
| Stages 4-5: Worktree tracks | lead + implementers per winner | `tracks/{op_id}/validation_results.md` |
| Stage 6: Integration validation | lead | ship or exhausted decision |

## Codex Tool Map

- create agents with `spawn_agent`
- route phase instructions with `send_input`
- collect completions with `wait`
- close finished agents with `close_agent`

Codex does not provide Claude-style lifecycle hooks. AMMO replaces those with explicit scripts, state tracking, and lead-owned gate execution.

## Task Graph

```text
T1  Scaffold artifact directory                                 [lead]
T2  Baseline capture + constraints.md                           [researcher] <- T1
T3  GATE: verify_phase1_baseline.py                             [lead]       <- T2
T4  Bottleneck mining                                           [researcher] <- T3
T5  GATE: Stage 2 review                                        [lead]       <- T4
T6  Champion proposals + debate rounds                          [lead+champions] <- T5
T7  GATE: Debate winner selection                               [lead]       <- T6
T8  Create worktree per winner                                  [lead]       <- T7
T9  Implement + validate per winner in worktree                 [implementer] <- T8
T10 GATE: import/build check per track                          [lead]       <- T9
T11 GATE: verify_validation_gates.py                            [lead]       <- all T10
T12 Integration validation                                      [lead]       <- T11
T13 Final decision (SHIP or EXHAUSTED)                          [lead]       <- T12
```

## Stage-Gate Discipline

The lead runs a gate check before every stage transition. Subagents create artifacts; they do not decide stage progression.

Minimum gate commands:

```bash
python .codex/skills/ammo/scripts/verify_phase1_baseline.py \
  {artifact_dir} \
  --json-output {artifact_dir}/runs/phase1_gate_report.json

python .codex/skills/ammo/scripts/verify_validation_gates.py \
  {artifact_dir} \
  --json-output {artifact_dir}/runs/validation_gate_report.json
```

For gates without a dedicated script, the lead must write an explicit checklist report under `{artifact_dir}/runs/`.

## Non-Negotiables

1. Production parity for all performance evidence: CUDA graphs plus `torch.compile`
2. vLLM production kernels as the correctness and performance baseline
3. Numerical correctness via `torch.allclose()` or equivalent
4. Stage 1 baseline reuse for all Stage 5 and Stage 6 E2E comparisons
5. E2E GPU sequencing through the lock-based sweep workflow
6. Full-model E2E validation before `SHIP`
7. Custom-kernel-only debate candidates

## Worktree Management

Create worktrees explicitly:

```bash
bash .codex/skills/ammo/scripts/create_worktree_with_build.sh ammo-track-{op_id} ammo/{op_id}
```

Clean them up explicitly:

```bash
bash .codex/skills/ammo/scripts/remove_worktree_cleanup.sh .codex/worktrees/ammo-track-{op_id}
```

These scripts replace the Claude `WorktreeCreate` and `WorktreeRemove` hooks.

## State Management

`state.json` tracks stage, debate progress, track metadata, and the integration decision.

Recommended per-track record:

```json
{
  "parallel_tracks": {
    "op001": {
      "status": "PASSED",
      "branch": "ammo/op001",
      "worktree_path": ".codex/worktrees/ammo-track-op001",
      "correctness": true,
      "kernel_speedup": 1.35,
      "e2e_speedup": 1.08,
      "validation_results_path": "kernel_opt_artifacts/.../tracks/op001/validation_results.md",
      "files_changed": ["csrc/...", "tests/..."]
    }
  }
}
```

## Helper Scripts

- `scripts/new_target.py` — scaffold artifact directory plus `state.json`
- `scripts/collect_env.py` — capture environment snapshot
- `scripts/verify_phase1_baseline.py` — Stage 1 gate
- `scripts/verify_validation_gates.py` — Stage 5 gate
- `scripts/run_vllm_bench_latency_sweep.py` — E2E runner with GPU lock
- `scripts/generate_validation_report.py` — structured validation reporting
- `scripts/create_worktree_with_build.sh` — Codex replacement for Claude worktree creation hook
- `scripts/remove_worktree_cleanup.sh` — Codex replacement for Claude worktree cleanup hook

## References

| Topic | File |
|---|---|
| Nsight profiling | `references/nsys-profiling-guide.md` |
| Validation defaults | `references/validation-defaults.md` |
| CUDA graph safety | `references/cudagraph-safety.md` |
| E2E latency method | `references/e2e-latency-guide.md` |
| E2E delta math | `references/e2e-delta-math.md` |
| GPU hardware specs | `references/gpu-configs.md` |
| Optimization techniques | `references/optimization-techniques.md` |
| Fusion feasibility | `references/fusion-feasibility-heuristics.md` |
| Code templates | `references/code-templates.md` |
| Debate scoring | `references/debate-scoring-rubric.md` |
| Validation troubleshooting | `references/validation-troubleshooting.md` |
| DA audit checklist | `references/da-audit-checklist.md` |
| Claude to Codex mapping | `references/claude-codex-equivalents.md` |

## Orchestration Docs

| Topic | File |
|---|---|
| Debate protocol | `orchestration/debate-protocol.md` |
| Parallel tracks | `orchestration/parallel-tracks.md` |
| Integration logic | `orchestration/integration-logic.md` |

## Resume Protocol

After interruption or compaction:

1. read this file
2. read `{artifact_dir}/state.json`
3. reconstruct active worktrees from `parallel_tracks`
4. resume from the earliest incomplete gate
5. do not rely on memory when artifact files disagree
