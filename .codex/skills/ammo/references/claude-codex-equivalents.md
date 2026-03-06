# Claude to Codex Equivalents for AMMO

This file documents what maps cleanly from Claude Code into Codex and what does not.

## Mapping Summary

| Claude concept | Codex equivalent | Status |
|---|---|---|
| `CLAUDE.md` repo instructions | `AGENTS.md` plus repo-local skill docs | direct equivalent |
| Claude subagents | Codex custom agents in `.codex/config.toml` plus `spawn_agent` | direct equivalent |
| Claude skill body in `.claude/skills/.../SKILL.md` | Codex skill body in `.codex/skills/.../SKILL.md` | direct equivalent |
| Claude agent metadata in frontmatter | Codex agent config `.codex/agents/*.toml` plus prompt markdown | partial equivalent |
| `Team` abstraction | store agent ids in `state.json` and orchestrate with `spawn_agent` / `send_input` / `wait` | partial equivalent |
| `WorktreeCreate` hook | explicit script `scripts/create_worktree_with_build.sh` | no native hook; script replacement |
| `WorktreeRemove` hook | explicit script `scripts/remove_worktree_cleanup.sh` | no native hook; script replacement |
| `TaskCompleted` / `Stop` audit hook | explicit gate script plus `da-audit-checklist.md` | no native hook; script replacement |
| `SessionStart` / `PreCompact` hooks | resume protocol in `SKILL.md`, `rules/ammo.md`, and optional manual `/compact` workflow | no native hook |
| Claude hook-driven validation gate agent | lead-run `verify_validation_gates.py` in the main session | script replacement |

## Practical Guidance

- Do not try to emulate Claude hooks through undocumented Codex internals.
- Put durable enforcement into scripts, rules, and artifact schemas.
- Put resumability into `state.json`, explicit worktree metadata, and the skill resume checklist.
- Keep `agents/openai.yaml` minimal unless UI metadata is genuinely needed.
