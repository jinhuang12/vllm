# LLM Council Policy

Use the separate `llm-council` skill as a **de-risking tool** for correctness-sensitive or high-impact decisions.

This file defines:
- when council review is required vs recommended
- what to prepare for review
- how to integrate feedback

## Risk-tier policy

**Required (high risk)**
- Changes to **math/semantics** (routing scoring, renorm, scaling factors, accumulation/reduction)
- Any change that affects **numerical equivalence** across tokens/experts
- Major layout changes (shared-memory staging, cp.async/TMA, reorderings)
- `top_k > 1` designs with overlap/reduction complexity
- A large fusion-boundary redesign (cooperative ↔ hybrid ↔ split)

**Recommended (medium risk)**
- After Phase 2 when the plan makes non-trivial architectural choices
- After a Phase 4 investigation when conclusions rely on noisy perf signals
- When stuck after 2 distinct attempts

**Optional (low risk)**
- Minor refactors, comments, logging, small guard fixes

## How to invoke

1. Create `.llm-council/context.md` (or `{artifact_dir}/investigation/council_context.md`) with:
   - target: model, hardware, dtype, TP/EP
   - baseline truth snapshot (bucket timings)
   - current route decision and why
   - the exact change proposal (diff or bullet list)
   - success criteria + kill criteria
   - links to relevant artifacts (`constraints.md`, `optimization_plan.md`, traces)
2. In chat, invoke council by name (example):
   - “Use llm-council to review my proposed fix for …”
3. Summarize accepted vs rejected feedback and why, and record it in `state.json`.

## Blocked tasks

If a stage is blocked after **3** attempts:
1. Write `{artifact_dir}/blockers/{stage}_blocker.md` (what is blocked + what evidence you have).
2. Invoke council with the blocker context.
3. Either:
   - incorporate the guidance and retry once, or
   - declare `escalate_human` with a handoff summary.
