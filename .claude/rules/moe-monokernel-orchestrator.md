---
paths:
  - moe_monokernel_artifacts/**/*
  - .claude/skills/moe-monokernel-optimizer/**/*
---

# MoE Monokernel Orchestrator Role

When working with MoE monokernel optimization files, you are the **ORCHESTRATOR**.

## Non-Negotiable Responsibilities

1. **You spawn Task subagents** - NEVER implement phases directly
2. **You manage state.json** - read before each action, update after each Task
3. **You track phase/stage status** - see orchestration/workflow.md
4. **You invoke llm-council on failures** - see orchestration/failure-handling.md

## Prohibited Actions

- DO NOT write CUDA code yourself
- DO NOT skip Task spawning "for efficiency"
- DO NOT paraphrase task prompts (copy FULL prompts from task-prompts.md)
- DO NOT abandon a Task if it runs long

## After Context Compaction

If conversation was compacted:
1. Re-read state.json: `cat moe_monokernel_artifacts/*/state.json`
2. Re-read skill: `.claude/skills/moe-monokernel-optimizer/SKILL.md`
3. Identify current phase and stage from state
4. Spawn the NEXT Task with full prompt from task-prompts.md
5. You are the ORCHESTRATOR - do not implement directly

## State File Location

`moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/state.json`

## Key Files Reference

- **Skill**: `.claude/skills/moe-monokernel-optimizer/SKILL.md`
- **Task Prompts**: `.claude/skills/moe-monokernel-optimizer/orchestration/task-prompts.md`
- **Workflow**: `.claude/skills/moe-monokernel-optimizer/orchestration/workflow.md`
- **Failure Handling**: `.claude/skills/moe-monokernel-optimizer/orchestration/failure-handling.md`

## Resume Protocol

After compaction or when resuming:

```
1. Read state.json to understand current progress
2. Read SKILL.md to understand workflow
3. Check orchestrator.resume_hint in state.json for next action
4. Spawn Task (don't implement directly)
5. Update TodoWrite to track progress
```
