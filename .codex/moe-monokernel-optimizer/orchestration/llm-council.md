# LLM Council Policy

Use `llm-council` as a **de-risking tool** for correctness‑sensitive or high‑impact changes. This file is the single source of truth for when and how to invoke it.

## Risk‑Tier Policy

**Required (high‑risk)**
- Kernel math / accumulation / reduction changes
- Memory layout or shared‑memory staging changes (cp.async/TMA)
- `top_k > 1` routing + accumulation behavior
- Major tiling or fusion‑boundary changes with cross‑batch impact

**Recommended (medium‑risk)**
- After Phase 2 when the plan makes non‑trivial architectural choices
- After Phase 4 if conclusions rely on noisy perf signals
- When stuck after 2+ distinct attempts

**Optional (low‑risk)**
- Mechanical refactors, docs, small glue changes with low blast radius

## Checkpoints

For **high‑risk** stages, prefer two checkpoints:
1. **Approach checkpoint**: before writing significant code
2. **Implementation checkpoint**: after code + validation, before marking complete

For **medium‑risk** stages, one checkpoint (usually final) is sufficient.

## How to Invoke

1. Invoke the `llm-council` skill by name in chat (e.g., “Use llm-council to review …”).
2. Follow its instructions to prepare `.llm-council/context.md`.
3. Run the critics (parallel by default; sequential optional).
4. Summarize accepted/rejected feedback and why.

## Blocked Tasks

If a stage is blocked after 3 attempts:
1. Read `{artifact_dir}/blockers/{stage}_blocker.md`.
2. Invoke `llm-council` for a fresh perspective.
3. Update blocker/state files with the council’s feedback.
4. Retry the stage with the new context.

## State Tracking (Recommended)

If invoked, record a short summary in `{artifact_dir}/state.json` for resumability. If not invoked, do **not** block completion.
