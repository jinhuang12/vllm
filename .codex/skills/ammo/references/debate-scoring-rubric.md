# Debate Scoring Rubric

Used by the lead to evaluate champion arguments and select winners after Stage 3.

## Scoring Criteria

| Criterion | Weight | How Scored |
|---|---|---|
| Custom kernel gate | GATE | If the proposal does not require new or substantially modified CUDA, Triton, or CUTLASS kernel code, total score is forced to 0.0. |
| Proposal quality | 15% | Grounding in measured data, correct BW math, correct hardware interpretation |
| Feasibility evidence quality | 25% | Micro-experiments, roofline math, ISA checks, methodology rigor |
| E2E impact potential | 20% | Component share `f` times measured kernel speedup |
| Survived critiques | 25% | Penalties for material unaddressed critiques |
| Implementation complexity | 10% | Lower complexity, lower graph-safety risk, fewer touched files |
| Complementarity | 5% | Bonus for targeting a different component than other viable winners |

## Methodology Cap

If a micro-experiment uses methodology that would be invalid under `validation-defaults.md` for the claimed evidence, cap feasibility evidence at 5/10.

Examples:

- kernel timing without CUDA graph capture
- eager-mode measurements used to claim production-parity speedups
- unsupported cache-warmed speedups used without cold-cache disclosure

## Scoring Scale

| Score | Meaning |
|---|---|
| 9-10 | strong evidence, no material gaps |
| 7-8 | solid evidence, minor gaps |
| 5-6 | adequate but uncertain |
| 3-4 | weak evidence or major open issues |
| 0-2 | insufficient evidence or fatal flaw |

## Winner Selection Rules

1. Apply the custom kernel gate first.
2. Eliminate candidates below 5.0 weighted total.
3. Prefer 2-3 winners.
4. Prefer complementary winners over multiple candidates targeting the same component.
5. If multiple candidates target the same component, keep only the highest-scoring one unless the lead has concrete evidence they can coexist.

## Handling Conflicting Experimental Data

If two champions report measurements for the same shape that differ by more than 1.5x:

1. rerun a standardized CUDA-graphed micro-benchmark, or
2. compare exact methodology and prefer the production-parity result, or
3. cap feasibility at 5/10 if the disagreement remains unresolved

Do not advance unresolved disputed claims into Stage 4.

## Output

Write `{artifact_dir}/debate/summary.md` with:

- per-candidate score breakdown
- selected winners and rationale
- critiques that changed the outcome
- unresolved caveats that implementation must address
