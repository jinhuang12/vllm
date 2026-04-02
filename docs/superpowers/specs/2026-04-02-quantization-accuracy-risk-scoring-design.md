# Accuracy-Risk Treatment for Quantization-Inclusive Optimizations

## Problem

Champions blend precision reduction into kernel fusions/rewrites (e.g., "fused SiLU+GEMM with FP8 tensor cores" when the input was BF16). This inflates speedup projections while hiding accuracy degradation risk that only surfaces at Stage 5 — wasting a full implementation cycle if correctness gates fail.

The pipeline already has differentiated validation for quantization at Stage 5 (`topk_relaxed` mode, GSM8K accuracy gate, wider tolerances), but the debate scoring rubric treats all optimizations identically. Proposals that derive their speedup primarily from precision reduction compete on equal footing with genuinely lossless kernel rewrites.

## Goals

1. Make champions explicitly account for the quantization component of their speedup projections
2. Apply risk-adjusted scoring that reflects the higher Stage 5 failure probability of lossy proposals
3. Ensure lossy tracks face appropriately stricter correctness validation
4. Catch undisclosed precision reduction through existing adversarial mechanisms

## Non-Goals

- Banning quantization optimizations (they're acceptable if accuracy is preserved)
- Adding new review steps or agents
- Adding new gate types or benchmark scripts (existing gates are parameterized, not new ones)
- Giving lossless optimizations an explicit scoring bonus (the friction on lossy is sufficient)

## Design

### 1. Lossy Classification Rule

**Dtype boundary rule (mechanical):** If the optimization introduces a precision reduction at ANY point in the dataflow — where the output dtype has fewer mantissa/exponent bits than the input dtype — the proposal is classified as **lossy**.

Examples:
| Scenario | Classification | Rationale |
|----------|---------------|-----------|
| BF16 activations quantized to FP8 before fused GEMM | **Lossy** | New precision reduction introduced |
| Fusing two GEMMs on an already-FP8 model | **Lossless** | No new precision reduction — model was already quantized |
| FP32 accumulator → BF16 output (same as baseline) | **Lossless** | Accumulator precision matches baseline — no new truncation |
| Switching from FP32 to BF16 accumulator for larger tiles | **Lossy** | Accumulator precision reduced vs. baseline for performance |
| BF16 weights cast to FP8 for faster tensor core MMA | **Lossy** | Weight precision reduced for performance |
| INT4 dequant fused with GEMM (model already INT4) | **Lossless** | No new quantization — just fusing the existing dequant step |

**Champion self-declaration:** Champions must declare their proposal as "lossless" or "lossy" in the Phase 0 proposal, citing the dtype boundary rule. This is a required field, not optional.

**Adversarial audit:** Other champions critique undisclosed precision reduction during debate. The transcript monitor flags implementation-time precision reduction that contradicts the debate classification.

### 2. Debate Scoring — Separate Accounting (Stage 3)

For proposals classified as lossy, the **E2E impact potential** criterion (20% weight) requires separate speedup accounting:

**Champion must report two projections:**
- Kernel speedup from the **lossless component** (fusion, tiling, memory layout, scheduling, etc.)
- Additional speedup from the **quantization component** (dtype reduction enabling faster MMA, reduced memory bandwidth, etc.)

**Scoring math:**

Let `f` = component share, `s_L` = lossless kernel speedup, `s_T = s_L × s_Q` = total kernel speedup (lossless × quantization):

```
lossless_contribution = f × (1 - 1/s_L) / 2       # 2x deflation on lossless portion
quant_contribution    = f × (1/s_L - 1/s_T) / 4   # 4x deflation on quantization portion
effective_E2E         = 1 + lossless_contribution + quant_contribution
```

The 4x deflation on the quantization portion reflects the higher probability of Stage 5 accuracy failure. This is an initial estimate — recalibrate after N=5 lossy tracks have completed the pipeline.

**Decomposition must be empirically backed:** The champion's claimed split between `s_L` and `s_Q` must be supported by micro-experiment evidence (e.g., running the fused kernel with original precision vs. reduced precision). If the decomposition is unsupported or the two components are architecturally inseparable, the entire speedup `s_T` is treated as quantization-inclusive and gets the 4x deflation. This incentivizes both clean separation in kernel design and honest accounting in proposals.

**Undisclosed precision reduction:** If another champion's critique reveals undisclosed quantization in a "lossless" proposal, the orchestrator applies a **2-point deduction** under the "Survived critiques" criterion (25% weight) at scoring time — this is an unaddressed material critique.

### 3. Stage 5 Validation — Mandatory Stricter Gates

Tracks classified as lossy at debate time receive mandatory additional validation parameters:

| Parameter | Lossless tracks | Lossy tracks |
|-----------|----------------|--------------|
| `--correctness-mode` | `exact_greedy` (default) | `topk_relaxed` (forced) |
| `--correctness-num-questions` | 30 (script default) | **100** |
| GSM8K accuracy gate | Off | On (implied by `topk_relaxed`) |

These are enforced via the sweep script invocation in the champion's E2E validation step. The lossy parameters also apply to GATING_REQUIRED re-validation sweeps — if the track is lossy, every sweep invocation uses `topk_relaxed` and 100 questions.

The champion reads the `classification` field for their op_id from the structured per-candidate section in `debate/summary.md` (see Structured Handoff below).

No new gate types are added. The existing `topk_relaxed` + "zero questions lost" mechanism is sufficient — the gap was that mixed proposals weren't being classified correctly, so they ran with `exact_greedy` and the accuracy impact wasn't caught until late or not at all.

### 4. Structured Handoff (Debate → Implementation)

The orchestrator must write a machine-readable classification block per winning candidate in `debate/summary.md`:

```markdown
## Track OP007
- classification: lossy
- correctness_mode: topk_relaxed
- correctness_num_questions: 100
```

For lossless tracks:
```markdown
## Track OP008
- classification: lossless
- correctness_mode: exact_greedy
- correctness_num_questions: 30
```

The impl-champion reads the `classification` field for their op_id. **If absent, treat as lossy and flag to orchestrator** — this is the safe default.

The transcript monitor spawn prompt must also include the classification so it can apply the undisclosed-quantization CRITICAL check without parsing the debate summary:
```
classification: {lossless|lossy}
```

### 5. Existing Actor Guidance Updates

**`ammo-champion.md` (debate champion):**
- Add "Precision Classification" requirement to Phase 0 proposal template
- Add critique guidance: "When evaluating opponent proposals, check for undisclosed precision reduction. If the claimed speedup comes from a dtype change not classified as lossy, this is a material critique."

**`debate-scoring-rubric.md` (orchestrator scoring):**
- Add "Lossy Classification" section with the dtype boundary rule and examples table
- Modify "E2E impact potential" criterion with separate accounting formula and 4x deflation
- Add undisclosed-precision-reduction penalty under "Survived critiques"
- Add structured classification output requirement to the "Output" section

**`ammo-impl-champion.md` (implementation champion):**
- Modify correctness-mode selection guidance: read `classification` field from `debate/summary.md`
- Add: "If your track is classified as lossy (or classification is absent), you MUST use `--correctness-mode topk_relaxed --correctness-num-questions 100` for ALL sweep invocations including GATING_REQUIRED re-validation"

**`validation-defaults.md` (reference):**
- Add lossy track parameter overrides table to Gate 5.1b section
- Add note in GATING_REQUIRED section: lossy parameters carry through re-validation sweeps

**`ammo-transcript-monitor.md` (DA monitor):**
- Add "Undisclosed quantization" CRITICAL check: "If the debate proposal was classified as lossless but the implementation introduces lower-precision tensor core ops, FP8 casts, or quantization scales not present in the baseline, flag as CRITICAL."
- Monitor spawn prompt must include `classification: {lossless|lossy}` to enable this check

**`SKILL.md` (orchestrator):**
- Update sweep script description: correctness mode selection is based on lossy classification, not model dtype
- Add `classification` field to `parallel_tracks` schema in state.json, recorded at T7 (debate winner selection)

**`orchestration/integration-logic.md` (Stage 6):**
- Parameterize combined correctness check: if ANY integrated track is lossy → `topk_relaxed --correctness-num-questions 100`; if ALL lossless → `exact_greedy --correctness-num-questions 30`

**`ammo-impl-validator.md` (kernel validation sub-agent):**
- Champion's spawn prompt includes `classification` field so validator selects appropriate Gate 5.1a tolerances
- For lossy tracks: use FP8-appropriate tolerances from validation-defaults.md, not BF16 tolerances

**`orchestration/parallel-tracks.md` (track management):**
- Transcript monitor spawn prompt includes `classification: {lossless|lossy}`
- Impl-champion spawn prompt includes `classification` field

**`orchestration/debate-protocol.md` (debate orchestration):**
- Add Precision Classification check to Phase 0 Proposal Eligibility Gate: "Does the proposal include a `Precision Classification` field (lossy/lossless) citing the dtype boundary rule? If absent, message the champion to add it before Round 1 begins."

## Files Changed

| File | Change type | What changes |
|------|------------|--------------|
| `.claude/skills/ammo/references/debate-scoring-rubric.md` | Moderate | New classification section, modified E2E impact criterion, undisclosed-quantization penalty, structured output |
| `.claude/agents/ammo-champion.md` | Small | Precision classification in proposal template, critique guidance |
| `.claude/agents/ammo-impl-champion.md` | Small | Correctness-mode selection based on debate classification |
| `.claude/skills/ammo/references/validation-defaults.md` | Small | Lossy track parameter overrides in Gate 5.1b, GATING_REQUIRED note |
| `.claude/agents/ammo-transcript-monitor.md` | Small | New CRITICAL check for undisclosed quantization |
| `.claude/skills/ammo/SKILL.md` | Small | Sweep script description update, `classification` in state.json schema |
| `.claude/skills/ammo/orchestration/integration-logic.md` | Small | Parameterize Stage 6 combined correctness check by lossy classification |
| `.claude/agents/ammo-impl-validator.md` | Small | Classification in spawn prompt for tolerance selection |
| `.claude/skills/ammo/orchestration/parallel-tracks.md` | Small | Classification in impl-champion and monitor spawn prompts |
| `.claude/skills/ammo/orchestration/debate-protocol.md` | Small | Precision Classification check in Phase 0 eligibility gate |

10 files, all documentation/agent guidance. No code changes. No new scripts.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Champions game the classification by claiming "the model is already FP8" when they're the ones adding the quantization step | Dtype boundary rule is mechanical — traces the actual dataflow, not the model's existing precision |
| 4x deflation is too harsh and kills valid quantization proposals | The deflation only applies to the quantization PORTION of the speedup. If the lossless component is strong, the proposal still scores well. Champions who cleanly separate their accounting aren't penalized much. |
| 100 GSM8K questions takes too long | ~3.3x the default time (100 vs 30) for correctness phase. Acceptable given these tracks have higher accuracy risk. |
| Champions always declare "lossless" to avoid friction | Other champions have adversarial incentive to catch this (it's a material critique that deducts points from the competitor). Transcript monitor catches it at impl time. |
