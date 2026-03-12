# Debate Scoring Rubric

Used by the main session to evaluate champion arguments and select winners after the Stage 3 adversarial debate.

## Scoring Criteria

| Criterion | Weight | How Scored |
|-----------|--------|-----------|
| **Custom kernel gate** | GATE | Does the proposal involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code? If **NO** → score forced to **0.0**, candidate eliminated regardless of other criteria. Config-only, flag-flipping, and parameter-tuning proposals automatically fail this gate. |
| Proposal quality | 15% | Did the champion correctly derive their candidate from grounded profiling data? Were bandwidth calculations correct? Did they identify hardware-specific effects (L2 cache, architecture quirks)? Deduct heavily if the proposal relies on ungrounded assumptions. |
| Feasibility evidence quality | 25% | Roofline calcs, ISA checks, micro-experiment prototype results. ALL kernel speedup estimates MUST come from the champion's own micro-experiments. Deduct for hand-waving or unsupported claims. Methodology audit: if the micro-experiment uses methodology that would be INVALID under validation-defaults.md (e.g., no CUDA graph capture, eager mode), the feasibility score is capped at 5/10 regardless of other evidence quality. Cache audit: for BW-bound kernels (AI < breakeven), if warm/cold cache times are not both reported, cap feasibility at 5/10. For fusion proposals where test data < 25% of production pipeline working set AND warm/cold > 1.5x, deduct 2 points. |
| E2E impact potential | 20% | Component share (f) × micro-experiment-backed kernel speedup. Higher potential = higher score. Penalize if f is too small for meaningful E2E gain. |
| Survived critiques | 25% | Count unaddressed critiques from other champions. Each unaddressed material critique deducts points. Conceded + mitigated critiques are neutral. |
| Implementation complexity | 10% | Lower complexity preferred. Score based on: lines of CUDA/Triton code, number of files modified, CUDA graph safety risk, likelihood of regressions. |
| Complementarity | 5% | Bonus for targeting a different component than other candidates. Enables "ship all that pass" if combined with another winner. |

## Scoring Scale

Per criterion: 0-10 points, then weighted.

| Score | Meaning |
|-------|---------|
| 9-10 | Strong evidence, no material gaps |
| 7-8 | Solid evidence with minor gaps |
| 5-6 | Adequate evidence but notable uncertainties |
| 3-4 | Weak evidence, major gaps or unaddressed critiques |
| 0-2 | Insufficient evidence or fatally flawed |

## Winner Selection Rules

0. **Custom kernel gate**: Any candidate failing the custom kernel gate (no new or substantially modified kernel code) scores 0.0 total and is eliminated before threshold comparison.
1. **Minimum threshold**: Candidates scoring below 5.0 weighted total are eliminated regardless of rank.
2. **Number of winners**: Select 2-3 candidates. Prefer 3 if ≥3 GPUs available for parallel tracks.
3. **Complementarity preference**: If two candidates target different components and both score ≥5.0, prefer selecting both over two candidates targeting the same component.
4. **Same-component tiebreak**: If multiple candidates target the same component, select only the highest-scoring one for that component.

## Handling Conflicting Experimental Data

When two champions present contradictory micro-experiment results for the same kernel/shape with >1.5x discrepancy (e.g., one claims 1.34x speedup, another measures 0.83x), scoring is blocked for the disputed claim until resolved. Resolution options:

1. **Standardized tiebreaker**: Run a CUDA-graphed benchmark with agreed methodology
2. **Methodology disclosure**: Both champions disclose exact measurement code; the one using production-parity methodology (CUDA graphs + torch.compile) takes precedence
3. **Unresolved**: Cap the disputed claim's feasibility score at 5/10

The lead MUST NOT advance a candidate to Stage 4 with unresolved >1.5x measurement discrepancies.

## Output

Write selection rationale to `{artifact_dir}/debate/summary.md` with:
- Per-candidate scores (table with per-criterion breakdown)
- Selected winners and rationale
- Key arguments that influenced the decision
- Notable critiques that eliminated candidates
