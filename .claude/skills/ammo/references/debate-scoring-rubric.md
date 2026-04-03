# Debate Scoring Rubric

Used by the main session to evaluate champion arguments and select winners after the Stage 3 adversarial debate.

## Lossy Classification Rule

Before scoring, classify each proposal as **lossless** or **lossy** using the dtype boundary rule:

**Rule:** If the optimization introduces a precision reduction at ANY point in the dataflow — where the output dtype has fewer mantissa/exponent bits than the input dtype — the proposal is classified as **lossy**.

| Scenario | Classification | Rationale |
|----------|---------------|-----------|
| BF16 activations quantized to FP8 before fused GEMM | **Lossy** | New precision reduction introduced |
| Fusing two GEMMs on an already-FP8 model | **Lossless** | No new precision reduction — model was already quantized |
| FP32 accumulator → BF16 output (same as baseline) | **Lossless** | Accumulator precision matches baseline — no new truncation |
| Switching from FP32 to BF16 accumulator for larger tiles | **Lossy** | Accumulator precision reduced vs. baseline for performance |
| BF16 weights cast to FP8 for faster tensor core MMA | **Lossy** | Weight precision reduced for performance |
| INT4 dequant fused with GEMM (model already INT4) | **Lossless** | No new quantization — just fusing the existing dequant step |

Champions must self-declare "lossless" or "lossy" in their Phase 0 proposal, citing the dtype boundary rule. This is a required field (see debate-protocol.md § Proposal Eligibility Gate).

## Scoring Criteria

| Criterion | Weight | How Scored |
|-----------|--------|-----------|
| **Custom kernel gate** | GATE | Does the proposal involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code? If **NO** → score forced to **0.0**, candidate eliminated regardless of other criteria. Config-only, flag-flipping, and parameter-tuning proposals automatically fail this gate. |
| Proposal quality | 15% | Did the champion correctly derive their candidate from grounded profiling data? Were bandwidth calculations correct? Did they identify hardware-specific effects (L2 cache, architecture quirks)? Deduct heavily if the proposal relies on ungrounded assumptions. **Per-BS f-values required**: Champions must report f_decode for each target batch size, not a single aggregate. Different batch sizes can have different f values (e.g., f_decode at BS=1 may differ from BS=32 due to different kernel dispatch paths). Using f_total instead of f_decode without justification deducts 2 points. |
| Feasibility evidence quality | 25% | Roofline calcs, ISA checks, micro-experiment prototype results. ALL kernel speedup estimates MUST come from the champion's own micro-experiments. Deduct for hand-waving or unsupported claims. Methodology audit: if the micro-experiment uses methodology that would be INVALID under validation-defaults.md (e.g., no CUDA graph capture, eager mode), the feasibility score is capped at 5/10 regardless of other evidence quality. Cache audit: for BW-bound kernels (AI < breakeven), if warm/cold cache times are not both reported, cap feasibility at 5/10. For fusion proposals where test data < 25% of production pipeline working set AND warm/cold > 1.5x, deduct 2 points. **Theoretical-only cap**: If the proposal has NO empirical kernel benchmark (only roofline calculations, ISA inspection, or ncu --query-metrics), feasibility is capped at **3/10** regardless of theoretical quality. **Baseline provenance**: The micro-experiment baseline must use the production API path and memory layout (see debate-protocol.md "Baseline Provenance Rule"). If the baseline BW diverges >10% from Stage 2 ncu/nsys data for the same shape AND the champion has not provided ncu launch-grid cross-reference, cap feasibility at **3/10** (NCU Trigger 4). |
| E2E impact potential | 20% | Component share (f) × micro-experiment-backed kernel speedup (s), **with 2x deflation factor applied** for lossless proposals. The debate systematically overpredicts E2E by median 3x (empirical finding from root cause analysis). Score the DEFLATED projection: `effective_E2E = 1 + f × (1 - 1/s) / 2`. Penalize if f is too small for meaningful E2E gain. If deflated E2E lower bound < 1.5× min_e2e_improvement_pct, apply a 1-point scoring penalty (marginal territory). Per-BS impact estimates REQUIRED (not single aggregate). BS-dependent regressions acceptable if gatable. **For lossy proposals**: see "Lossy E2E Impact Scoring" below. |
| Survived critiques | 25% | Count unaddressed critiques from other champions. Each unaddressed material critique deducts points. Conceded + mitigated critiques are neutral. **Undisclosed precision reduction**: If a critique reveals that a "lossless" proposal actually introduces precision reduction, apply a **2-point deduction** — this is an unaddressed material critique about accuracy risk. |
| Implementation complexity | 10% | Lower complexity preferred. Score based on: lines of CUDA/Triton code, number of files modified, CUDA graph safety risk, likelihood of regressions. |
| Complementarity | 5% | Bonus for targeting a different component than other candidates. Enables "ship all that pass" if combined with another winner. |

### Lossy E2E Impact Scoring

For proposals classified as lossy, the E2E impact criterion requires **separate speedup accounting**:

**Champion must report two projections:**
- Kernel speedup from the **lossless component** (`s_L`): fusion, tiling, memory layout, scheduling, etc.
- Additional speedup from the **quantization component** (`s_Q`): dtype reduction enabling faster MMA, reduced BW, etc.
- Total kernel speedup: `s_T = s_L × s_Q`

**Decomposition must be empirically backed** by micro-experiment evidence (e.g., running the fused kernel with original precision vs. reduced precision). If the decomposition is unsupported or the components are architecturally inseparable, the entire speedup `s_T` is treated as quantization-inclusive and gets the 4x deflation.

**Scoring math** (let `f` = component share):

```
lossless_contribution = f × (1 - 1/s_L) / 2       # 2x deflation on lossless portion
quant_contribution    = f × (1/s_L - 1/s_T) / 4   # 4x deflation on quantization portion
effective_E2E         = 1 + lossless_contribution + quant_contribution
```

The 4x deflation on the quantization portion reflects the higher probability of Stage 5 accuracy failure. This is an initial estimate — recalibrate after N=5 lossy tracks have completed the pipeline.

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
- **Per-winner structured classification block** (machine-readable, consumed by orchestrator when spawning impl-champion and transcript monitor):

```markdown
## Track {OP_ID}
- classification: {lossless|lossy}
```

