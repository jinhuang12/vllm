# Gate 5.1b v2: Unified Accuracy Gate

**Date**: 2026-04-03
**Status**: Draft
**Scope**: AMMO Stage 5 correctness validation (Gate 5.1b)
**Supersedes**: `2026-04-01-gate-5-1b-redesign-design.md` (archived as v1)

## Background

Gate 5.1b v1 introduced E2E greedy decode correctness via GSM8K, replacing the bypassed component-level tensor capture. It used three token-level comparison modes (`exact_greedy`, `topk_relaxed`, `first_divergence_topk`) selected based on lossless/lossy classification, plus a "zero questions lost" accuracy superset check.

### Why v1 Failed

Investigation of Round 1 tracks revealed:

1. **Token-level comparison is structurally broken for autoregressive models.** Once one token diverges (from accumulation order, cuBLAS tile selection, or quantization noise), autoregressive cascade makes all downstream positions incomparable. op008 (BF16 GEMM merge) showed 10.9% topk containment failure; op007 (FP8 MLP) showed 48.3%. Both are cascade noise, not real correctness problems.

2. **`max_tokens=256` caused false accuracy gate failures.** 5 of 7 question-level changes across both tracks were truncation artifacts from the sweep script's 256-token limit (inherited from upstream vLLM GSM8K eval). The upstream tolerance of 8% absorbs this noise; our zero-tolerance per-question gate amplified it.

3. **The lossless/lossy distinction is unnecessary at the E2E gate level.** cuBLAS uses BF16 reduced-precision accumulation by default (`allow_bf16_reduced_precision_reduction=True`). Any GEMM shape change — fusion, merge, tiling — triggers different accumulation patterns via cuBLAS heuristic lookup tables, producing cascade divergence indistinguishable from quantization noise at the token level. Classification still matters for Gate 5.1a tolerances and debate scoring, but not for E2E correctness.

4. **vLLM's own CI validates FP8/INT8/W4A16 with accuracy-only gates** (3-8% tolerance). Per-kernel optimizations (AMMO's scope) are strictly less lossy than full-model quantization.

## Design

### Gate 5.1b: `opt_accuracy >= baseline_accuracy`

A single gate for all tracks regardless of classification:

```
opt_gsm8k_accuracy >= baseline_gsm8k_accuracy
```

- **n = 200** questions (configurable via `--correctness-num-questions`)
- **Percentage comparison** — allows question-level churn (different questions correct) as long as aggregate accuracy is maintained
- **No token-level gating** — token-level data is computed and logged as diagnostics, never determines the verdict
- **No mode selection** — `--correctness-mode` flag removed entirely

### Changes from v1

| Aspect | v1 | v2 |
|--------|----|----|
| Gate logic | 3 token-level modes + accuracy superset | `opt_accuracy >= baseline_accuracy` |
| Classification dependency | Lossless→first_divergence_topk/30q, Lossy→topk_relaxed/100q | None (classification affects 5.1a only) |
| Default sample size | 30 (lossless) / 100 (lossy) | 200 (all tracks) |
| max_tokens | 256 | 1024 |
| Accuracy check | "Zero questions lost" superset | Percentage comparison (allows churn) |
| Token-level | Hard gate | Informational diagnostic |
| `--correctness-mode` | Required (3 choices) | Removed |
| `--max-divergent-positions` | exact_greedy parameter | Removed |
| `--max-topk-failures-pct` | topk_relaxed parameter | Removed |

### Sampling Config

```python
SamplingParams(
    temperature=0.0,
    max_tokens=1024,
    stop=["Question", "Assistant:", "<|separator|>"],
    seed=42,
    logprobs=5,  # retained for diagnostic logging
)
```

### Correctness Verdict File

```json
{
    "gate": "5.1b",
    "verdict": "PASS",
    "num_questions": 200,
    "baseline_accuracy": 0.76,
    "optimized_accuracy": 0.76,
    "accuracy_delta": 0.0,
    "baseline_correct_count": 152,
    "optimized_correct_count": 152,
    "questions_lost": [13, 45],
    "questions_gained": [7, 85],
    "duration_s": 120.5,
    "diagnostics": {
        "divergent_questions": 45,
        "first_divergence_positions_p50": 28,
        "first_divergence_positions_p95": 112,
        "churn_rate": 0.026,
        "note": "Token-level data is informational only and does not affect the verdict."
    },
    "_diagnostic_notes": "p50/p95 computed across questions where first_divergence_pos >= 0; set to -1 if no questions diverge. churn_rate = (questions_lost + questions_gained) / num_questions."
}
```

### Metadata Mismatch Detection

Golden refs metadata (Stage 1 `--capture-golden-refs`) records `num_questions` and `max_tokens`. At Stage 5 `--verify-correctness`:

- If `golden_refs.num_questions != current num_questions` → exit code 4 (infrastructure error) with clear message
- If `golden_refs.max_tokens != current max_tokens` → exit code 4 (infrastructure error) with clear message

### Baseline Accuracy Floor

If `baseline_correct_count == 0` and `num_questions > 0`, fail with exit code 4 (infrastructure error — model cannot solve any GSM8K questions; environment suspect). This is not a correctness FAIL (exit code 3) but an infrastructure alarm.

### GSM8K Data

Bundle 200 test questions + 5 train examples locally at `data/gsm8k_subset.json`. The 200 questions are the **first 200 questions of the GSM8K test split, in original order**. The current 30-question subset is questions 0-29 of this same ordering — the expansion is an append, not a reshuffle. Download fallback for `--correctness-num-questions > 200`.

### Migration (In-Flight Tracks)

Tracks with golden refs captured under v1 (n=30, max_tokens=256) **must re-capture golden refs** at n=200, max_tokens=1024 before running v2 verification. The metadata mismatch check is a hard fail — no graceful degradation. Re-capture uses the same Stage 1 baseline branch and model; only the sweep parameters change.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All phases passed |
| 2 | Bucket constraint violation |
| 3 | Correctness FAIL (`opt_accuracy < baseline_accuracy`) |
| 4 | Infrastructure error (missing golden refs, n mismatch, max_tokens mismatch, baseline_accuracy=0, download failure) |

### CLI Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--capture-golden-refs` | bool | False | Stage 1: capture GSM8K outputs as golden references |
| `--verify-correctness` | bool | False | Stage 5: compare against golden refs |
| `--correctness-num-questions` | int | 200 | Number of GSM8K questions |

Removed: `--correctness-mode`, `--max-divergent-positions`, `--max-topk-failures-pct`.

## Scope of Classification After This Change

| Purpose | Uses classification? | Where |
|---------|---------------------|-------|
| Gate 5.1a tolerances | Yes (BF16 vs FP8 tolerances) | `ammo-impl-validator.md` |
| Debate scoring deflation | Yes (2x lossless vs 4x lossy) | `debate-scoring-rubric.md` |
| Transcript monitor precision check | Yes (undisclosed precision reduction) | `ammo-transcript-monitor.md` |
| Gate 5.1b correctness | **No** | This change |

## Files Modified

| File | Change |
|------|--------|
| `scripts/run_vllm_bench_latency_sweep.py` | Rewrite `_compare_correctness()`, remove mode dispatch, max_tokens→1024, default n→200. Remove public flags `--correctness-mode`, `--max-divergent-positions`, `--max-topk-failures-pct` (L1498-1506). Remove hidden subprocess flags `--_correctness-mode`, `--_correctness-num-questions`, `--_max-divergent-positions`, `--_max-topk-failures-pct` (L1515-1518). Remove subprocess forwarding (L2064-2068). Remove child getattr calls (L1970-1973). Add metadata mismatch check (num_questions + max_tokens). Add baseline_accuracy=0 floor check. |
| `scripts/test_correctness_comparator.py` | Full replacement — see Test Cases section below |
| `references/validation-defaults.md` | Rewrite Gate 5.1b section |
| `agents/ammo-impl-champion.md` | Remove mode selection, update sweep command template |
| `orchestration/parallel-tracks.md` | Remove mode-derivation comment lines; **preserve** `Classification: {classification}` line, update its inline comment to reference Gate 5.1a and debate scoring (NOT correctness mode) |
| `orchestration/integration-logic.md` | Remove mode selection in Stage 6 |
| `SKILL.md` | Update sweep invocation docs |
| `references/debate-scoring-rubric.md` | Remove classification→mode derivation sentence |
| `agents/ammo-transcript-monitor.md` | Update "lossy validation gates" reference |
| `data/gsm8k_subset.json` | Expand from 30→200 test questions (append questions 30-199 from GSM8K test split, preserving original order) |
| `VERSION` | Changelog entry |

## Test Cases (for `test_correctness_comparator.py`)

The new `_compare_correctness()` has a simpler signature (no mode, no max_divergent_positions, no max_topk_failures_pct). Minimum required tests:

| # | Test | Expected |
|---|------|----------|
| 1 | `opt_accuracy == baseline_accuracy` (both get same questions right) | PASS |
| 2 | `opt_accuracy > baseline_accuracy` (opt gains questions) | PASS |
| 3 | `opt_accuracy < baseline_accuracy` by 1 question | FAIL |
| 4 | `opt_accuracy < baseline_accuracy` by N questions, verify `questions_lost` populated | FAIL |
| 5 | Question churn: lost 2, gained 2 (same accuracy) | PASS |
| 6 | `baseline_accuracy = 0` (all wrong) | Infrastructure error / special handling |
| 7 | `n = 0` (empty question list) | FAIL |
| 8 | All outputs empty (0 tokens) | FAIL |
| 9 | `questions_lost` and `questions_gained` are mutually exclusive and correct | PASS (verify lists) |
| 10 | Diagnostics populated (divergent_questions, first_divergence_positions) but don't affect verdict | PASS (verify diagnostics present) |
| 11 | `accuracy_delta` computed correctly | PASS (verify math) |
| 12 | Large n (200 questions) with 1-question accuracy drop | FAIL |
