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

## Scoring Is On Content, Not Volume

Scores reward evidence and reasoning, never prose length, section count, or visible diligence. A champion who clears every criterion in tight, compact form scores exactly as well as one who writes three times as much — and the compact one is easier to score correctly, so prefer it on ties. Specifically:

- **"Survived critiques"** counts whether each material critique was *addressed*, not how many words it took. A one-row concession or a one-sentence counter in a risk register fully addresses a critique. Walking every critique in its own multi-paragraph section earns nothing extra.
- **The feasibility caps below** are satisfied by the *presence of the evidence* — a methodology checklist line, both warm/cold numbers, a baseline-provenance pointer. Narrative wrapped around that evidence does not raise the score, and concise presentation does not lower it. Do not penalize a proposal for being short when the required evidence is present.

Champions are instructed to write per `references/writing-style.md` (length targets, no protocol-echo, no honesty-narration). Hold them to evidence, not word count.

## Scoring Criteria

| Criterion | Weight | How Scored |
|-----------|--------|-----------|
| **Mechanism/structure authorship gate** | GATE | Does the proposal **author mechanism logic or host-side structure** (per `SKILL.md` NN#8) — either new/substantially-modified kernel code in one of the four authoring classes (**Triton, CuTeDSL, CUTLASS, CUDA C++**), OR authored host-side structure (load-time weight restructuring, scheduling/dispatch/comm code, an Inductor/FX graph pass)? If **NO** → score forced to **0.0**, candidate eliminated regardless of other criteria. A **retuned constant** where the kernel/cubin body is byte-identical (`num_warps`, `num_stages`, `BLOCK_SIZE_*`, `@autotune` tuples, tactic tables, `custom_ops` list edits, predicate flips) and env-var/config-only flips automatically fail this gate — *a constant in `.py` is still config, regardless of the measured win.* |
| **Technology Selection block gate** | GATE | Does the proposal include a populated **Technology Selection** block (baseline tech, proposed tech, hardware, op character, library coverage, justification, anti-regression check, CUDA-graph self-check if CuTeDSL)? See `references/technology-selection.md` § Required proposal fields. If the block is **absent or has empty fields** → score forced to **0.0**, candidate eliminated. If the block is present but a field contains an *internally-contradictory fill* (e.g., `Anti-regression check: not applicable` when proposed is strictly higher-abstraction than baseline, or `Baseline technology: unknown` without supporting evidence), the block gate also forces 0.0 — the lead does not downgrade to a scoring cap when the block's own assertion is false. |
| Proposal quality | 15% | Did the champion correctly derive their candidate from grounded profiling data? Were bandwidth calculations correct? Did they identify hardware-specific effects (L2 cache, architecture quirks)? Deduct heavily if the proposal relies on ungrounded assumptions. **Per-BS f-values required**: Champions must report `f_e2e` (and `f_decode` as the diagnostic ranking column) for each target batch size, not a single aggregate. Different batch sizes can have different values (e.g., at BS=1 vs BS=32 dispatch paths and `decode_busy` shift). Using `f_total` instead of `f_decode` (ranking) or omitting the `f_e2e` conversion deducts 2 points. The 2-point deduction for using `f_decode` as the Amdahl multiplier under any workload-dilution red flag is scored on **E2E impact potential**, not here — keep them separate. |
| Feasibility evidence quality | 25% | Roofline calcs, ISA checks, micro-experiment prototype results. ALL kernel speedup estimates MUST come from the champion's own micro-experiments. Deduct for hand-waving or unsupported claims. Methodology audit: if the micro-experiment uses methodology that would be INVALID under validation-defaults.md (e.g., no CUDA graph capture, eager mode), the feasibility score is capped at 5/10 regardless of other evidence quality. Cache audit: for BW-bound kernels (AI < breakeven), if warm/cold cache times are not both reported, cap feasibility at 5/10. For fusion proposals where test data < 25% of production pipeline working set AND warm/cold > 1.5x, deduct 2 points. **Theoretical-only cap**: If the proposal has NO empirical kernel benchmark (only roofline calculations, ISA inspection, or ncu --query-metrics), feasibility is capped at **3/10** regardless of theoretical quality. **Baseline provenance**: The micro-experiment baseline must use the production API path and memory layout (see debate-protocol.md "Baseline Provenance Rule"). If the baseline BW diverges >10% from Stage 2 ncu/nsys data for the same shape AND the champion has not provided ncu launch-grid cross-reference, cap feasibility at **3/10** (NCU Trigger 4). **Anti-regression cap (technology selection)**: If the proposed authoring technology is strictly higher in the abstraction ranking than the baseline's (ranking: Triton > CuTeDSL ≈ CUTLASS > CUDA C++; library baselines cuBLAS/FlashAttn/FlashInfer/DeepGEMM/torch.compile-Triton are treated as rank 0 / below all custom classes — see `references/technology-selection.md` § Anti-regression rule) and the champion has NOT provided Tier 2+ micro-experiment evidence that their implementation beats the actual production kernel at the target shape under production-parity methodology, feasibility is capped at **3/10**. This catches proposals like "rewrite this Hopper CUTLASS GEMM in Triton" or "rewrite this cuBLAS FP8 GEMM in Triton" that sound cheaper but almost never win. Note: if the anti-regression fill in the Technology Selection block is *internally contradictory* (e.g., "not applicable" when the rule clearly applies), the block gate fires first and eliminates the candidate — the cap only applies to candidates that cleared both Phase 0 gates. Multiple caps on the same criterion combine as **`min(cap_1, cap_2, …)`**, not additive — a proposal hit by both theoretical-only and anti-regression is capped at 3/10, not 0/10. **Scope (regime):** the abstraction-ranking anti-regression cap applies only to the **kernel-authoring regime** (`kernel_replacement`, `kernel_fusion`, `custom_kernel`, `attention_kv_layout`, and the merged-kernel case of `weight_layout_transform`). For **inter-kernel-slice / host-side-structure** proposals (`dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass`, and structure-only `weight_layout_transform`) there is no library-baseline ranking — anti-regression applies to the **E2E metric only** (this relaxes the cap; it never tightens it). |
| E2E impact potential | 20% | The `f` in the Amdahl projection is **`f_e2e`, NOT `f_decode`** (see `references/e2e-delta-math.md` § f_e2e: The Correct Amdahl Multiplier). Score `f_e2e × (1 - 1/s)` (kernel categories) or the matching per-category projection formula (see below). Penalize if `f_e2e` is too small for meaningful E2E gain. If the projected E2E gain < 1.5× `min_e2e_improvement_pct`, apply a 1-point scoring penalty (advisory — scoring-only). Per-BS impact estimates REQUIRED (not single aggregate). BS-dependent regressions acceptable if gatable. **f_decode-without-conversion penalty**: if the champion's projection plugs `f_decode` into the Amdahl formula AND ANY workload-dilution red flag fires for the target BS (`prefill_share_of_e2e ≥ 0.10`, OR `decode_busy < 0.85`, OR `input_len ≥ 512`), apply a **2-point deduction** on this criterion. The conversion `f_e2e = f_decode × decode_busy × decode_share_of_e2e` is mandatory under any red flag. **Per-category projection formula requirement (v4.1+)**: champions whose Phase 0 proposal includes a `## Category` field MUST use the projection formula matching the **slice** that category attacks (any cataloged category, legacy alias, or novel descriptor — see `references/optimization-categories.md` § Per-Category Projection Formulas, also in `references/e2e-delta-math.md`; novel descriptors use the nearest-analogue regime formula). Using the wrong-slice formula scores **0/10** on E2E impact. **For lossy proposals**: see "Lossy E2E Impact Scoring" below. |
| Survived critiques | 25% | Count unaddressed critiques from other champions. Each unaddressed material critique deducts points. Conceded + mitigated critiques are neutral. **Undisclosed precision reduction**: If a critique reveals that a "lossless" proposal actually introduces precision reduction, apply a **2-point deduction** — this is an unaddressed material critique about accuracy risk. |
| Implementation complexity | 10% | Lower complexity preferred. Score based on: lines of CUDA/Triton code, number of files modified, CUDA graph safety risk, likelihood of regressions. |
| Complementarity | 5% | Bonus for targeting a different component than other candidates. Enables "ship all that pass" if combined with another winner. **Extended bonus (same-component, different authoring classes)**: two candidates targeting the **same** component qualify for the partial-complementarity bonus only when BOTH (a) their `Proposed technology` fields are distinct authoring classes per the taxonomy (not just renamed), AND (b) the candidates are **algorithmically independent** — different scheduling, tiling, or fusion strategy, not just different tooling wrapped around substantively the same algorithm. The lead rejects the bonus when the two proposals' Kernel Code Scope descriptions are identical except for the technology field. When honored, this bonus lets the debate de-risk technology choice by running two tools head-to-head in Stage 5. |

### Lossy E2E Impact Scoring

A lossy proposal's E2E impact uses the standard projection — `effective_E2E = 1 + f_e2e × (1 - 1/s_T)`, where `s_T` is the total kernel speedup and `f = f_e2e` for the component (see `references/e2e-delta-math.md`). Score the accuracy and numerics risk where it can be measured: the precision-reduction critique deduction in the Survived-critiques row above, and Gate 5.1a correctness at validation.

**Report the decomposition as evidence.** Split the kernel speedup into the **lossless component** (`s_L`: fusion, tiling, memory layout, scheduling) and the **quantization component** (`s_Q`: dtype reduction enabling faster MMA, reduced BW), with `s_T = s_L × s_Q`, and back the split with micro-experiment evidence (e.g., running the fused kernel at original vs. reduced precision). The split's job is to make the quant claim auditable: a champion who cannot show the quant component actually delivers `s_Q` has an unsupported claim, scored under Feasibility evidence quality. E2E impact is projected from `s_T` directly.

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

0. **Gate precedence (structural gates fire BEFORE scoring)**: The two Phase 0 gates — Mechanism/structure authorship gate (NN#8) and Technology Selection block — are *structural* checks that fire at the eligibility stage, before any criterion is scored. A candidate that fails either gate scores **0.0 total** and is eliminated before threshold comparison. The gates check (a) the proposal authors mechanism logic OR host-side structure (kernel code in one of the four authoring classes, OR load-time weight restructuring / scheduling / dispatch / comm / graph-pass host-side code) and is NOT a byte-identical-cubin retuned constant or config/env flip, (b) presence and non-emptiness of the Technology Selection block, and (c) that the block's fills are not internally contradictory (e.g., `Anti-regression check: not applicable` when the ranking comparison says the rule clearly applies, or `Baseline technology: unknown` unsupported by investigation evidence). The **feasibility caps** (anti-regression 3/10, theoretical-only 3/10, baseline-provenance 3/10, methodology 5/10, cache audit 5/10) are *scoring* rules that apply only to candidates that cleared both structural gates. When multiple caps apply to feasibility, combine them as `min(...)`, not additive. Rationale: a champion who populates the block honestly but lacks the beats-baseline evidence gets the 3/10 cap and can still compete on other criteria; a champion who lies about the anti-regression status on the block itself gets eliminated — we do not reward fabricated compliance.
1. **Minimum threshold**: Candidates scoring below `campaign.config.debate_elimination_score_floor` (schema default `5.0`) weighted total are eliminated regardless of rank.
2. **Number of winners**: Select 2-3 candidates. Prefer 3 if ≥3 GPUs available for parallel tracks.
3. **Complementarity preference**: If two candidates target different components and both score ≥5.0, prefer selecting both over two candidates targeting the same component.
4. **Same-component tiebreak**: If multiple candidates target the same component, select only the highest-scoring one for that component.

## Handling Conflicting Experimental Data

When two champions present contradictory micro-experiment results for the same kernel/shape with a materially large discrepancy (advisory — scoring-only, NOT a ship/retract gate), scoring is blocked for the disputed claim until resolved. Resolution options:

1. **Standardized tiebreaker**: Run a CUDA-graphed benchmark with agreed methodology
2. **Methodology disclosure**: Both champions disclose exact measurement code; the one using production-parity methodology (CUDA graphs + torch.compile) takes precedence
3. **Unresolved**: Cap the disputed claim's feasibility score at 5/10 (scoring-only advisory cap).

The lead MUST NOT advance a candidate to Stage 4 with unresolved material-magnitude measurement discrepancies.

## Output

**Authoritative cross-agent contract**: write structured winner data to `state.json.campaign.rounds[N-1].debate.selected_candidates` as an **array** (one entry per winner — debate typically selects 2-4 per round). Each entry has:
- `op_id`
- `track_assignment` (enum: `lossless`, `quant`, `structural`)
- `score_breakdown` (feasibility, evidence_tier, expected_e2e_pct, weighted_total)
- `stage_4_validation_obligations` (array of enum: `roofline_check`, `cuda_graph_check`, `noise_tolerance_check`, `crossover_probe`, `numerics_sweep`, `activation_range_check`)
- `cited_evidence` (array of `file:line` or artifact paths)

`op_id` values here must match the entries in `selected_winners` (the list of chosen op_id strings). Each Stage 4 impl-champion reads the entry matching its assigned `op_id`. Do not hand-author prose that substitutes for these typed fields.

**Rendered view (humans only)**: `{artifact_dir}/rounds/{CR}/debate/summary.md` is produced by `scripts/render_debate_summary.py` from `state.json`. Do NOT write `summary.md` directly — agents have no write path to it. If `summary.md` and `state.json` disagree, `state.json` wins; regenerate by re-running the renderer.

