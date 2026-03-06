---
paths:
  - kernel_opt_artifacts/**/*
  - .codex/skills/ammo/**/*
  - .codex/agents/ammo-*.toml
---

# AMMO Lead Role

When working on AMMO, act as the lead orchestrator.

## Non-Negotiable Responsibilities

1. Spawn and coordinate specialized agents with `spawn_agent`, `send_input`, `wait`, and `close_agent`.
2. Own every blocking gate in the main session.
3. Maintain `state.json` at each stage transition.
4. Persist a gate report in `{artifact_dir}/runs/` before advancing stages.
5. Require `PASS` for every stage transition. Treat `WARN` as blocking.
6. Create and clean up worktrees explicitly with the AMMO helper scripts. Do not assume Claude-style lifecycle hooks exist.

## Prohibited Actions

- Do not skip gates.
- Do not advance on warnings.
- Do not merge candidates without explicit conflict analysis.
- Do not claim `SHIP` without correctness and E2E evidence.
- Do not implement kernel code directly in the lead session unless the user explicitly overrides the AMMO workflow.

## Lead Workflow Discipline

- Keep researcher, champion, and implementer responsibilities separated.
- Each track is owned by one `ammo-implementer` who performs both implementation and validation.
- Record blocker, retry, and route decisions in artifact files.
- Keep artifact provenance reproducible: commands, environment, outputs, and branch/worktree metadata.

## State File Location

`kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_tp{tp}/state.json`

## Resume Checklist

1. Read `.codex/skills/ammo/SKILL.md`.
2. Read the current `state.json`.
3. Identify the earliest incomplete gate.
4. Reconstruct active worktrees from `parallel_tracks.*.worktree_path`.
5. Resume from the gate, not from memory.

## Key File References

- Skill: `.codex/skills/ammo/SKILL.md`
- Researcher prompt: `.codex/skills/ammo/agents/ammo-researcher.md`
- Champion prompt: `.codex/skills/ammo/agents/ammo-champion.md`
- Implementer prompt: `.codex/skills/ammo/agents/ammo-implementer.md`
- Worktree create script: `.codex/skills/ammo/scripts/create_worktree_with_build.sh`
- Worktree cleanup script: `.codex/skills/ammo/scripts/remove_worktree_cleanup.sh`
- Validation gate: `.codex/skills/ammo/scripts/verify_validation_gates.py`
- Claude-to-Codex mapping: `.codex/skills/ammo/references/claude-codex-equivalents.md`
