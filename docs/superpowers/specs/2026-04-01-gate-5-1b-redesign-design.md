# Gate 5.1b Redesign: E2E Greedy Decode Correctness via GSM8K

**Date**: 2026-04-01
**Status**: Draft
**Scope**: AMMO Stage 5 correctness validation (Gate 5.1b)

## Problem

Gate 5.1b (component-level baseline tensor capture/compare) is bypassed 81% of the time (13 of 16 tracks). Most vLLM modules depend on runtime infrastructure (`ForwardContext`, `attn_metadata`, KV cache) that cannot be synthesized outside the full engine. Champions write `NOT_APPLICABLE.md` justifications, and the validator rubber-stamps them. The gate catches nothing.

### Failure Modes Not Caught

1. **Silent numerical drift**: FP8 quantization introduces accumulation errors that pass kernel-level `torch.allclose` but cause wrong model outputs
2. **Integration bugs**: Missing bias, wrong reshape, transposed dimensions — the component works in isolation but breaks when wired into the model

## Design

Replace component-level tensor capture/compare with E2E greedy decode comparison using GSM8K prompts. The correctness check piggybacks on the sweep script's already-loaded `LLM` object — zero extra model loads.

### Architecture: Two-Phase Sweep

The sweep script (`run_vllm_bench_latency_sweep.py`) gains a correctness phase that runs **once per model load**, before the per-bucket latency loop:

```
Model Load (already exists)
  │
  ├── Phase 1: Correctness (NEW — runs once)
  │     └── GSM8K greedy decode → save golden refs (Stage 1) or compare (Stage 5)
  │
  └── Phase 2: Latency (unchanged — per-bucket loop)
        └── Fixed-length dummy prompts, per-BS timing, ignore_eos
```

### Phase 1: Correctness via GSM8K

**Prompt source**: GSM8K 5-shot prompts via adapted `_build_gsm8k_prompts()` from `tests/evals/gsm8k/gsm8k_eval.py`.

**Why GSM8K**:
- Diverse natural-language math problems — exercises reasoning, arithmetic, varied vocabulary
- 5-shot prefix is ~400-500 tokens — naturally exercises M=400 prefill shapes (covers large-M GEMM paths)
- Well-known benchmark with ground-truth labels — enables accuracy scoring as a bonus signal
- Stop sequences (`["Question", "Assistant:", "<|separator|>"]`) produce variable-length outputs — exercises different decode lengths naturally

**Sampling config**:
```python
SamplingParams(
    temperature=0.0,      # Greedy decode — deterministic
    max_tokens=256,        # Sufficient for GSM8K answers
    stop=["Question", "Assistant:", "<|separator|>"],
    seed=42,              # Explicit seed for reproducibility
    logprobs=5,           # Top-5 logprobs at every position
)
```

**Question count**: 30 questions (configurable via `--correctness-num-questions`). At ~256 max tokens each, this is ~30s of generation after warmup. Total phase time: ~1-2 min including prompt building. For `topk_relaxed` mode, 30 questions yields ~2,600 token positions — sufficient to detect gross errors (5% threshold with ~2,600 samples gives 95% CI width of ~1.7%). If boundary-case flakiness is observed, increase to 50 questions.

**Output capture**: For each question, save:
```json
{
    "prompt_index": 0,
    "token_ids": [12, 345, ...],
    "text": "Let me solve...",
    "logprobs": [
        {"top_logprobs": {"12": -0.01, "345": -3.2, ...}},
        ...
    ],
    "num_tokens": 87,
    "gpu_name": "NVIDIA L40S"
}
```

**JSON serialization note**: `logprobs.top_logprobs` keys are stringified token IDs (JSON requires string keys). The comparator must convert to `int` before matching against `token_ids`. vLLM's `Logprob` objects must be unwrapped: `{str(token_id): lp.logprob for token_id, lp in output.logprobs[pos].items()}`.

**Hardware metadata**: `golden_refs.json` includes `gpu_name` from `torch.cuda.get_device_name()`. At comparison time, mismatch triggers a warning (not a hard fail — same-arch GPUs with different naming are OK).

Saved to `{out_dir}/json/golden_refs.json` (Stage 1) or `{out_dir}/json/opt_outputs.json` (Stage 5).

### Stage 1: Capture Golden References

```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir $ARTIFACT_DIR \
    --labels baseline \
    --capture-golden-refs          # NEW flag
```

The `--capture-golden-refs` flag tells Phase 1 to:
1. Build 30 GSM8K prompts
2. Run greedy decode with logprobs=5
3. Run **self-consistency check**: decode the same prompts a second time and verify outputs match (validates determinism assumption for this environment). If self-consistency fails, warn and automatically switch to `topk_relaxed` mode for Stage 5 comparisons (recorded in golden refs metadata).
4. Save full outputs (token_ids + logprobs + text + gpu_name) to `{out_dir}/json/golden_refs.json`
5. Also save GSM8K accuracy as a metadata field (bonus signal, not gating)
6. Proceed to Phase 2 (latency) as normal

**Self-consistency cost**: ~30s additional (same 30 questions, second pass). Total Phase 1 in Stage 1: ~2-3 min.

### Stage 5: Compare Optimized Outputs

```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir $ARTIFACT_DIR \
    --labels opt \
    --baseline-from $STAGE1_DIR \    # Existing flag — now also copies golden_refs.json
    --verify-correctness             # NEW flag — runs comparison after Phase 1
```

The `--verify-correctness` flag tells Phase 1 to:
1. Build identical 30 GSM8K prompts (same seed, same questions)
2. Run greedy decode with logprobs=5
3. Load golden refs from `{out_dir}/json/golden_refs.json` (copied by `--baseline-from`)
4. Run custom comparator (see below)
5. Write structured verdict to `{out_dir}/json/correctness_verdict.json`
6. **If FAIL**: exit with nonzero code — do NOT proceed to Phase 2 latency
7. **If PASS**: proceed to Phase 2 (latency) as normal

### Custom Comparator

Two modes, selected per-track via `--correctness-mode`:

#### Mode 1: `exact_greedy` (default for non-quantization tracks)

For each question, for ALL token positions:
- Require `baseline_token_id == optimized_token_id`
- Allow configurable divergence tolerance: up to `--max-divergent-positions N` positions (default: 0) may differ before FAIL
- Report: total positions checked, divergent positions, first divergence position

```python
for q in range(num_questions):
    baseline_ids = golden_refs[q]["token_ids"]
    opt_ids = opt_outputs[q]["token_ids"]
    min_len = min(len(baseline_ids), len(opt_ids))
    divergent = 0
    for pos in range(min_len):
        if baseline_ids[pos] != opt_ids[pos]:
            divergent += 1
    # Length mismatch counts as divergence too
    divergent += abs(len(baseline_ids) - len(opt_ids))
    if divergent > max_divergent_positions:
        FAIL
```

#### Mode 2: `topk_relaxed` (for FP8/quantization tracks)

For each question, for ALL token positions:
- Check bidirectional top-K containment: the optimized token must appear in the baseline's top-K logprobs, AND the baseline token must appear in the optimized's top-K logprobs
- K=5 (matches `logprobs=5` in sampling params)
- **Length-mismatch penalty**: positions beyond `min_len` count as containment failures (prevents truncated outputs from silently passing)
- Allow configurable failure budget: up to `--max-topk-failures-pct P` percent of total positions may fail containment (default: 5%)
- Report: total positions checked, containment failures, failure rate, worst positions, empty-output warnings

```python
total_positions = 0
containment_failures = 0
empty_output_count = 0
for q in range(num_questions):
    baseline = golden_refs[q]
    opt = opt_outputs[q]
    b_len = len(baseline["token_ids"])
    o_len = len(opt["token_ids"])
    if b_len == 0 and o_len == 0:
        empty_output_count += 1
        continue
    min_len = min(b_len, o_len)
    max_len = max(b_len, o_len)
    for pos in range(min_len):
        total_positions += 1
        # Keys are strings after JSON round-trip — convert to int
        b_topk = {int(k) for k in baseline["logprobs"][pos]["top_logprobs"]}
        o_topk = {int(k) for k in opt["logprobs"][pos]["top_logprobs"]}
        b_token = baseline["token_ids"][pos]
        o_token = opt["token_ids"][pos]
        if o_token not in b_topk or b_token not in o_topk:
            containment_failures += 1
    # Length mismatch: every extra position is a containment failure
    length_penalty = max_len - min_len
    total_positions += length_penalty
    containment_failures += length_penalty

if total_positions == 0:
    FAIL  # All questions produced empty output — something is fundamentally broken
if empty_output_count > num_questions * 0.1:
    WARN  # >10% empty outputs — prompts may be hitting stop sequences immediately
failure_rate = containment_failures / total_positions
if failure_rate > max_topk_failures_pct / 100:
    FAIL
```

**Critical**: No early break. Both modes check ALL positions for every question. This fixes the `check_logprobs_close` bug (line 254 of `tests/models/utils.py`) where a `break` on first divergence meant FP8 tracks only checked ~5-10 positions instead of hundreds.

### Correctness Verdict File

Written to `{out_dir}/json/correctness_verdict.json`:

```json
{
    "gate": "5.1b",
    "verdict": "PASS",
    "mode": "topk_relaxed",
    "num_questions": 30,
    "total_positions": 2847,
    "divergent_positions": 12,
    "failure_rate_pct": 0.42,
    "threshold_pct": 5.0,
    "gsm8k_accuracy_baseline": 0.73,
    "gsm8k_accuracy_optimized": 0.73,
    "accuracy_delta": 0.0,
    "duration_s": 45.2,
    "per_question_summary": [
        {"idx": 0, "baseline_tokens": 87, "opt_tokens": 87, "divergent": 0},
        ...
    ]
}
```

### `--baseline-from` Extension

The existing `--baseline-from` import loop (line 1634) copies `baseline_*.json` files. Extend it to also copy `golden_refs.json`:

```python
# After existing baseline JSON copy loop:
golden_src = baseline_from_json / "golden_refs.json"
golden_dst = json_dir / "golden_refs.json"
if golden_src.exists():
    shutil.copy2(str(golden_src), str(golden_dst))
    print(f"Imported golden references from {golden_src}")
elif verify_correctness:
    raise SystemExit(
        f"--verify-correctness requires golden_refs.json but "
        f"--baseline-from has none at {golden_src}"
    )
```

### GSM8K Data: Bundled Subset + Download Fallback

A minimal GSM8K subset (30 test questions + 5 few-shot train examples, ~15KB) is bundled at `.claude/skills/ammo/data/gsm8k_subset.json`. The correctness phase loads from this file first, falling back to the full GitHub download only if `--correctness-num-questions` exceeds the bundled count.

This eliminates the network dependency for the default 30-question configuration. Air-gapped and CI environments work without any download.

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | All phases passed |
| 2 | Bucket constraint violation (existing — `max_model_len` too small) |
| 3 | Correctness comparison FAIL (real divergence — optimization broke outputs) |
| 4 | Infrastructure error in correctness phase (GSM8K download failure, missing golden refs, serialization error) |

Exit code 3 vs 4 allows the orchestrator to distinguish "optimization broke correctness" (fail track) from "the test infrastructure had a problem" (retry).

## What Changes

### New Flags on `run_vllm_bench_latency_sweep.py`

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--capture-golden-refs` | bool | False | Stage 1: capture GSM8K outputs as golden references (mutually exclusive with `--verify-correctness`) |
| `--verify-correctness` | bool | False | Stage 5: compare against golden refs, fail-fast on mismatch (mutually exclusive with `--capture-golden-refs`) |
| `--correctness-mode` | str | `exact_greedy` | Comparator mode: `exact_greedy` or `topk_relaxed` |
| `--correctness-num-questions` | int | 30 | Number of GSM8K questions for correctness |
| `--max-divergent-positions` | int | 0 | (exact_greedy) max allowed token mismatches across all questions |
| `--max-topk-failures-pct` | float | 5.0 | (topk_relaxed) max % of positions failing top-K containment |

### Files Modified

| File | Change |
|------|--------|
| `.claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` | Add Phase 1 correctness logic, new CLI flags, comparator |
| `.claude/skills/ammo/references/validation-defaults.md` | Rewrite Gate 5.1b section to describe E2E correctness gate |
| `.claude/skills/ammo/orchestration/parallel-tracks.md` | Update pass criteria: 5.1b is now deterministic script output |

### Files Deleted

| File | Reason |
|------|--------|
| `.claude/skills/ammo/references/tensor-capture-template.py` | Replaced by golden ref capture in sweep script |
| `.claude/skills/ammo/references/tensor-compare-template.py` | Replaced by custom comparator in sweep script |

### Validator Scope Change

**Before**: Validator runs Gates 5.1a, 5.1b, 5.2, 5.3 independently.

**After**: Validator runs Gate 5.1a only (synthetic kernel correctness tests). Gates 5.1b, 5.2, 5.3 are all deterministic outputs of the sweep script:

| Gate | Owner | Mechanism |
|------|-------|-----------|
| 5.1a | Validator | Independent synthetic kernel tests (unchanged) |
| 5.1b | Sweep script | `--verify-correctness` verdict in `correctness_verdict.json` |
| 5.2 | Champion | Isolated kernel benchmark under CUDA graphs (unchanged mechanism — NOT the sweep script) |
| 5.3a | Sweep script | nsys kernel proof (existing `--nsys-profile`) |
| 5.3b | Sweep script | E2E latency comparison (existing) |

**Gate 5.2 clarification**: Gate 5.2 measures *isolated kernel-level* GPU time under CUDA graphs (per `validation-defaults.md`). This is fundamentally different from E2E latency (5.3b). The sweep script runs E2E, not isolated kernels. Gate 5.2 remains champion-owned — the champion writes a dedicated kernel benchmark script (or the validator did previously). With validator scoped to 5.1a only, the champion takes full ownership of 5.2.

The validator remains for 5.1a because synthetic kernel tests require module-specific knowledge (shapes, dtypes, edge cases) that cannot be standardized into a script. The validator's scope shrinks from ~4 gates to 1.

### NOT_APPLICABLE.md — Retired

The `NOT_APPLICABLE.md` escape clause is eliminated. E2E correctness works for all module types — there is no module that cannot be exercised by running the full model. Existing `NOT_APPLICABLE.md` files in completed tracks are left as historical artifacts.

## Interaction With Existing Gates

- **Gate 5.1a** (synthetic kernel tests): Unchanged. Validator still writes independent tests.
- **Gate 5.1b** (this redesign): Script-driven, deterministic, universal.
- **Gate 5.2** (kernel benchmark): Moves from validator to champion. The sweep script already computes per-bucket speedup in Phase 2 — champion reviews these numbers directly.
- **Gate 5.3a** (nsys proof): Unchanged. Sweep script runs nsys profiled iteration.
- **Gate 5.3b** (E2E latency): Unchanged. Sweep script runs per-BS latency comparison.

## Timing Budget

| Phase | Duration | Notes |
|-------|----------|-------|
| GSM8K prompt build | ~5s | Download cached, 30 prompts |
| Greedy decode (30 questions) | ~30-60s | Post-warmup, depends on model size |
| Comparison (Stage 5 only) | ~1s | In-memory token/logprob comparison |
| **Total Phase 1** | **~1-2 min** | Well within 5 min budget |

Phase 2 (latency) timing is unchanged.

## Fail-Fast Execution Order

The full gate sequence runs cheapest-first, bailing at the first failure:

```
1. Gate 5.1a  (~seconds)  — Validator synthetic kernel tests (pytest)
   └── FAIL? → Stop. Don't load model.
2. Gate 5.1b  (~1-2 min)  — Sweep script Phase 1: GSM8K correctness
   └── FAIL? → Exit code 1. Don't run latency.
3. Gate 5.3a  (~85s)      — Sweep script: nsys kernel proof
   └── FAIL? → Stop. Kernel not dispatching.
4. Gate 5.2   (~minutes)  — Champion: isolated kernel benchmark
   └── FAIL? → No measurable speedup. Stop.
5. Gate 5.3b  (~10-15 min) — Sweep script Phase 2: E2E latency sweep
   └── Per-BS verdicts → track verdict
```

**Key**: 5.1a runs *before* the sweep script is invoked. If synthetic kernel tests fail, there's no point loading the model. The sweep script itself runs 5.1b → 5.3a → 5.3b in sequence (5.3a via a separate `--nsys-profile` invocation between Phase 1 and Phase 2). Gate 5.2 is champion-owned and can run in parallel with the sweep.

### Diagnostic Output on Failure

When the comparator returns FAIL, the verdict file includes diagnostic fields to help the champion narrow down the root cause:

```json
{
    "verdict": "FAIL",
    "diagnostics": {
        "first_divergence": {
            "question_idx": 3,
            "position": 7,
            "baseline_token": 1234,
            "opt_token": 5678,
            "baseline_top5": {"1234": -0.01, "5678": -2.1, ...},
            "opt_top5": {"5678": -0.01, "1234": -4.3, ...},
            "prompt_snippet": "...the total cost is $"
        },
        "divergence_summary": "12 of 30 questions diverged, first at position 7",
        "length_mismatches": 3,
        "empty_outputs": 0
    }
}
```

## Stage 6 Integration Correctness Check (Cumulative Quantization Guard)

### Problem: Per-Track Gates Don't Catch Cumulative Drift

If a campaign ships 5 FP8 quantization tracks, each passing `topk_relaxed` individually at ~4% failure rate, the stacked model may have 15-20% failure rate against the BF16 baseline. FP8 errors compound through two mechanisms:
- **Additive through residual stream**: Each quantized layer adds ~O(eps_fp8) error, growing as ~O(N * eps_fp8) with N layers
- **Multiplicative through attention**: Softmax amplifies small logit differences into large attention weight shifts

The current Stage 6 integration logic (`integration-logic.md:60-66`) runs per-component pytest tests + E2E latency sweep but has **no combined accuracy/correctness check** against the original BF16 baseline.

### Fix: Run `--verify-correctness` After Cherry-Pick (Multi-Candidate Only)

When Stage 6 cherry-picks **multiple** tracks onto the integration branch, run the sweep with `--verify-correctness` comparing the fully-optimized model against the current round's Stage 1 golden refs:

```bash
python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir $ARTIFACT_DIR \
    --labels opt \
    --baseline-from $STAGE1_DIR \
    --verify-correctness \
    --correctness-mode topk_relaxed \
    --correctness-num-questions 50     # More questions for integration
```

**Single candidate**: Skip Stage 6 correctness — Stage 5 already ran `--verify-correctness` against the same golden refs. Ship directly per existing `integration-logic.md`.

**Multiple candidates**: Mandatory. Each track passed individually in Stage 5, but the combination has never been validated. This is where cumulative quantization drift surfaces.

### Thresholds

**Token-level (topk_relaxed)**: 5% failure rate threshold at Stage 6 (same as per-track). W8A16 quantization errors are structurally independent across layers — combined rate should not be dramatically higher than individual rates. If the combined model can't pass at 5%, the bisection protocol drops the least valuable track until it passes.

**GSM8K accuracy (HARD GATE for quantization tracks): "Zero Questions Lost"**

The optimized model cannot get ANY question wrong that the baseline got correct. This is a superset check, not a percentage threshold:

```python
baseline_correct = {i for i, (pred, label) in enumerate(zip(baseline_preds, labels)) if pred == label}
opt_correct = {i for i, (pred, label) in enumerate(zip(opt_preds, labels)) if pred == label}
lost_questions = baseline_correct - opt_correct  # Questions baseline got right but opt got wrong
if len(lost_questions) > 0:
    FAIL  # Quantization flipped a correct answer to incorrect
```

| Scenario | Token-level gate | Accuracy gate |
|----------|-----------------|---------------|
| BF16-only optimizations (fusions, memory layout) | `exact_greedy` (0 divergent positions) | Not applied (exact token match implies identical accuracy) |
| Single FP8 quantization track (Stage 5) | `topk_relaxed` 5% | Zero questions lost (HARD) |
| Combined N FP8 tracks (Stage 6) | `topk_relaxed` 5% | Zero questions lost (HARD) |

**Why zero-tolerance**: Quantization should be lossless from the user's perspective. A percentage-point threshold (e.g., 2pp) is unmeasurable at n=30 (granularity = 3.3pp per question). The superset check is measurable at any n, has no statistical ambiguity, and matches the intent: "did quantization flip any correct answer?"

**Note**: The optimized model MAY get questions right that the baseline got wrong (quantization noise can occasionally help). Only lost questions count as failures.

### Verdict File Extension

```json
{
    "gsm8k_accuracy_gate": {
        "enabled": true,
        "baseline_accuracy": 0.73,
        "baseline_correct_indices": [0, 1, 3, 5, 7, ...],
        "optimized_accuracy": 0.73,
        "optimized_correct_indices": [0, 1, 3, 5, 7, ...],
        "lost_questions": [],
        "gained_questions": [12],
        "verdict": "PASS"
    }
}
```

The accuracy gate is enabled when `--correctness-mode topk_relaxed` (quantization tracks). It is informational-only for `exact_greedy` mode (BF16 tracks should have identical outputs). The `lost_questions` array lists indices where baseline was correct but optimized was wrong — any non-empty list is a FAIL.

### Stage 6 Failure Handling

If the combined integration correctness check fails:
1. **Bisect**: Try subsets of quantization tracks (e.g., drop the track with the worst individual failure rate)
2. **Re-run**: Verify the smaller combination passes
3. **Ship the largest passing subset**
4. **Record**: Which tracks were dropped and why, in `integration.correctness_bisect` in state.json

### Should We Discourage Quantization Candidates in Stage 3?

**No.** Quantization often provides the best speedups (FP8 GEMMs are 2x faster on tensor cores). Discouraging quantization in debate leaves the biggest performance gains on the table. Instead:
- **Stage 5 (per-track)**: `topk_relaxed` catches individual regressions
- **Stage 6 (combined)**: Integration correctness check catches cumulative drift
- **GSM8K accuracy**: Hard gate at both Stage 5 and Stage 6 prevents accuracy death by a thousand cuts
- **Bisection**: If combined fails, the orchestrator finds the maximal passing subset

This approach ships the most quantization tracks possible while guaranteeing zero accuracy regression (no correct answers lost).

## Edge Cases

1. **Model doesn't support GSM8K well** (e.g., code-only models): The primary correctness gate is token-level agreement, not accuracy. Even if both baseline and optimized produce wrong answers, they must produce the *same* wrong answers (or within top-K for FP8). The "zero questions lost" accuracy gate is relative — if baseline gets 2/30 right and optimized gets 2/30 right (same 2 questions), that's PASS. If optimized gets 1/30 right and lost one of baseline's correct answers, that's FAIL regardless of low absolute accuracy.

2. **torch.compile / CUDA graph non-determinism**: FP reordering under torch.compile or non-deterministic atomics in CUDA graph replay can cause token divergence even without optimization changes. Mitigated by the **self-consistency check** in Stage 1: if the same prompts produce different outputs across two runs, the golden refs metadata records `deterministic: false` and the recommended comparison mode is automatically set to `topk_relaxed`. The champion can override with `--correctness-mode exact_greedy` if they're confident in their environment, but the default follows the empirical evidence.

4. **Golden refs captured on different GPU**: Results should be reproducible across same-arch GPUs (L40S→L40S) with `seed=42` and `temperature=0.0`. Cross-architecture (L40S→H100) is not supported — golden refs are per-hardware.

## Success Criteria

- Gate 5.1b has 0% N/A bypass rate (down from 81%)
- Catches silent numerical drift that component-level tests miss
- Catches integration bugs (wrong token outputs = immediate FAIL)
- Adds < 2 minutes to sweep time
- No extra model load
- Validator scope reduced to Gate 5.1a only
- Cumulative quantization drift caught by Stage 6 integration correctness check (multi-candidate only)
- GSM8K "zero questions lost" hard gate for quantization tracks (per-track + combined)
