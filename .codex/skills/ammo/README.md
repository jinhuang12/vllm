# AMMO - Automated Model Micro-Optimizer

A Codex CLI skill for stage-gated vLLM kernel optimization. AMMO profiles a deployment target, mines bottlenecks, debates custom-kernel candidates, runs isolated implementation plus validation tracks, and ships only changes that beat the Stage 1 baseline without correctness regressions.

## Active Roles

| Role | Purpose |
|---|---|
| `ammo-researcher` | profile baseline behavior and write grounded Stage 1-2 artifacts |
| `ammo-champion` | debate custom-kernel candidates with micro-experiments |
| `ammo-implementer` | own one worktree track from implementation through validation |
| lead session | manage gates, state, worktrees, and final ship decision |

## Workflow Summary

1. Stage 1 baseline capture under production parity
2. Stage 2 bottleneck mining from measured evidence only
3. Stage 3 adversarial debate across 2-4 champions
4. Stages 4-5 isolated worktree tracks where one implementer owns both code and validation
5. Stage 6 integration validation and final `SHIP` or `EXHAUSTED`

## Key Non-Negotiables

1. production-parity measurements only
2. vLLM production kernels as the baseline
3. `torch.allclose()` or equivalent for correctness
4. Stage 1 baseline reuse for Stage 5 E2E comparisons
5. GPU-locked E2E runs
6. custom-kernel-only proposals in debate

## Worktree Workflow

Create a track worktree:

```bash
bash .codex/skills/ammo/scripts/create_worktree_with_build.sh ammo-track-op001 ammo/op001
```

Clean it up:

```bash
bash .codex/skills/ammo/scripts/remove_worktree_cleanup.sh .codex/worktrees/ammo-track-op001
```

These scripts replace the Claude hook-based worktree lifecycle.

## File Layout

```text
.codex/skills/ammo/
├── SKILL.md
├── README.md
├── agents/
│   ├── ammo-researcher.md
│   ├── ammo-champion.md
│   ├── ammo-implementer.md
│   └── openai.yaml
├── orchestration/
│   ├── debate-protocol.md
│   ├── parallel-tracks.md
│   └── integration-logic.md
├── references/
│   ├── validation-defaults.md
│   ├── e2e-latency-guide.md
│   ├── debate-scoring-rubric.md
│   ├── validation-troubleshooting.md
│   ├── da-audit-checklist.md
│   └── claude-codex-equivalents.md
└── scripts/
    ├── new_target.py
    ├── verify_phase1_baseline.py
    ├── verify_validation_gates.py
    ├── run_vllm_bench_latency_sweep.py
    ├── create_worktree_with_build.sh
    └── remove_worktree_cleanup.sh
```
