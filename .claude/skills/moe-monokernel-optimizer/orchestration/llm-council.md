# LLM Council Policy

Use the `llm-council` skill as a de-risking tool for high-impact decisions.

## When to Use

**Recommended** (high risk):
- Changes to routing math or accumulation semantics
- Numerical equivalence changes across tokens/experts
- Major layout changes (shared memory staging, TMA)
- top_k > 1 designs with overlap/reduction complexity
- Fusion boundary redesign (cooperative ↔ hybrid ↔ split)

**Consider** (medium risk):
- After Phase 2 when plan makes non-trivial architectural choices
- After Phase 4 investigation when conclusions rely on noisy perf signals
- When stuck after 2+ distinct attempts

**Skip** (low risk):
- Minor refactors, comments, logging
- Small guard fixes

## How to Invoke

1. Prepare context file with:
   - Target: model, hardware, dtype, TP/EP
   - Baseline truth snapshot
   - Current route decision and rationale
   - Exact change proposal
   - Success/kill criteria

2. Invoke: "Use llm-council to review my proposed fix for [X]"

3. Record outcome in state.json

## Blocked Tasks

If a stage is blocked after 3 attempts:
1. Write `{artifact_dir}/blockers/{stage}_blocker.md`
2. Invoke council with blocker context
3. Either incorporate guidance and retry, or escalate to human
