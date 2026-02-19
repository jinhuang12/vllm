# LLM Council Policy (AMMO)

Use the separate `llm-council` skill as a de-risking tool for correctness-sensitive or high-impact optimization decisions.

This policy defines:
- when council review is required vs recommended
- what context to prepare
- how to merge council feedback into AMMO artifacts

## Risk-tier policy

Required (high risk):
- math/semantic changes that can alter numerical equivalence
- fusion-boundary redesigns across kernel/graph/runtime ownership
- changes that modify or disable incumbent optimizations
- distributed execution semantic changes (partitioning/reduction ordering/communication overlap)
- adapter-contract or framework-boundary changes affecting parity workflows

Recommended (medium risk):
- after Stage 3 when plan includes non-trivial architecture choices
- after Stage 5 investigation when conclusions depend on noisy profiler signals
- when blocked after 2 distinct attempts on the same failure mode

Optional (low risk):
- refactors, logging, comments, and guard fixes with no semantic or perf-model impact

## How to invoke

1. Create council context file:
- preferred: `{artifact_dir}/investigation/council_context.md`

2. Include minimum context:
- target envelope (framework, model, hardware, dtype, TP/EP, buckets)
- constraints snapshot (`constraints.md` + snapshot ID + parity signature)
- baseline truth snapshot and incumbent metrics
- ranked opportunities and selected hypotheses (from Stage 3)
- exact proposed change (diff summary or patch scope)
- success criteria and kill criteria
- links to artifacts (`artifact_bundle.json`, `optimization_plan.md`, `validation_results.md`, traces)

3. Invoke council by name in chat:
- "Use llm-council to review this Stage 3 plan"
- "Use llm-council to review this blocker in Stage 5"

4. Integrate outcomes:
- record accepted vs rejected council feedback with rationale
- update `artifact_bundle.json` summary and relevant stage status
- update plan/implementation docs before continuing

## Blocked stages

If a stage remains blocked after 3 attempts:
1. Write `{artifact_dir}/investigation/blocker.md`.
2. Invoke council with blocker context.
3. Either:
- retry once with revised plan, or
- mark `escalate_human` with handoff summary and stop.
