# Gate 5.1b Strictness Analysis: Is exact_greedy Too Strict for AMMO?

**Date**: 2026-04-02
**Status**: Decision pending
**Context**: Op008 (merged GDN proj GEMM) failed exact_greedy despite being provably lossless

---

## Background

Gate 5.1b is the E2E correctness gate in the AMMO validation pipeline (Stage 5). It runs GSM8K few-shot greedy decode on the optimized model and compares outputs against golden references captured from the unmodified baseline in Stage 1.

The gate has two modes, selected based on lossy/lossless classification from the debate stage:

| Mode | Used for | Mechanism |
|------|----------|-----------|
| `exact_greedy` | Lossless tracks (no precision reduction) | Every generated token must match the baseline. Zero tolerance. |
| `topk_relaxed` | Lossy tracks (precision reduction introduced) | Bidirectional top-5 containment per position. Up to 5% failure rate allowed. |

Both modes include a GSM8K accuracy superset check: the optimized model cannot get ANY question wrong that the baseline got correct ("zero questions lost").

## The Op008 Case

**Optimization**: Merge two separate F.linear operations in Qwen3.5-4B's Gated Delta Network layers — `in_proj_qkvz` [12288, 2560] and `in_proj_ba` [64, 2560] — into a single combined [12352, 2560] GEMM. Applied to 24 of 32 layers.

**Gate results**:

| Gate | Verdict | Detail |
|------|---------|--------|
| 5.1a (kernel correctness) | PASS | 0.0 max error — bit-exact at F.linear level |
| 5.2 (kernel speedup) | PASS | 1.155x warm, 1.150x cold |
| 5.1b (E2E correctness) | **FAIL** | 9/30 questions diverged, 837/4938 token positions mismatched |
| 5.3a/5.3b (E2E latency) | NOT RUN | Blocked by 5.1b failure |

**GSM8K accuracy**: 66.67% baseline, 66.67% optimized. Zero questions lost. One question *gained* under nsys profiling.

**Root cause**: The merged [12352, 2560] GEMM dispatches a different CUTLASS tile configuration than the separate [12288, 2560] + [64, 2560] GEMMs. Different tile sizes produce different FP32 accumulation order within each output element's dot product. These BF16 ULP (unit in the last place) differences compound across 24 GDN layers x 512 autoregressive decode steps, causing token-level divergence.

This is not a bug. It is an inherent property of floating-point arithmetic: `(a + b) + c != a + (b + c)` due to rounding. Any optimization that changes the shape of a GEMM — which is nearly every kernel fusion or merge — will trigger a different CUTLASS config and potentially different accumulation order.

## Empirical Analysis: Would topk_relaxed Have Helped?

We simulated the topk_relaxed comparator on op008's existing logprobs data (golden_refs.json + opt_outputs.json from the exact_greedy run):

```
Total positions:           4938
Token divergences:         799  (16.18%)
Top-5 containment failures: 813  (16.46%)

Verdict at 5% threshold:  FAIL
Verdict at 2% threshold:  FAIL
Verdict at 1% threshold:  FAIL
```

**topk_relaxed also fails** — and the containment failure rate (16.46%) is nearly identical to the raw token divergence rate (16.18%). This seems to contradict the expectation that accumulation-order divergence would produce near-tie tokens that stay within each other's top-5.

### The Autoregressive Cascade Explanation

Deeper analysis reveals why:

**At every first-divergent position**, both tokens ARE in each other's top-5 (9/9 questions pass containment at the initial divergence point). The initial flip is always a legitimate near-tie — e.g., in Q13 at position 8, the baseline token has logprob -0.634 and the opt token has logprob -0.759 (a 0.125 nats difference).

**But generation is autoregressive.** Once one token differs, the model receives a different input at the next step. After a few positions, the two models are generating from completely different contexts — effectively two different conversations. Comparing top-5 sets between these diverged contexts is meaningless.

Evidence from Q13 (worst case, 215/256 positions diverged):

```
Regions of agreement/divergence:
  pos   0-  7 (  8 tokens): same
  pos   8-  8 (  1 tokens): diff    <-- initial flip
  pos   9- 28 ( 20 tokens): same    <-- brief reconvergence
  pos  29- 37 (  9 tokens): diff
  pos  38- 38 (  1 tokens): same
  pos  39-134 ( 96 tokens): diff    <-- full cascade
  pos 135-145 ( 11 tokens): same
  pos 146-255 (110 tokens): diff    <-- no recovery

Post-divergence reconvergence: 13.4% of positions
```

The pattern is clear: initial near-tie flip at position 8, brief reconvergence where both paths happen to agree, then full cascade where the two generation paths are permanently separated.

### What Each Metric Actually Measures

| Metric | Op008 result | What it detects |
|--------|-------------|-----------------|
| `exact_greedy` | FAIL (17% divergence) | Any token-level change, including legitimate near-ties |
| `topk_relaxed` per-position | FAIL (16.5% containment) | Autoregressive cascade, not quality degradation |
| First-divergence containment | PASS (9/9 questions) | Whether the initial flip was a legitimate near-tie |
| GSM8K accuracy superset | PASS (66.67% -> 66.67%, 0 lost) | Whether final answer quality is preserved |

## The Core Problem

**exact_greedy is too strict**: Any optimization that changes GEMM dispatch (fusions, merges, padding changes, different Triton kernels) will produce different accumulation order. This is ~100% of AMMO's optimization targets.

**topk_relaxed per-position is the wrong metric**: Once autoregressive cascade begins, per-position top-5 comparison is noise. It measures how far two diverged generation paths have separated, not whether the optimization degraded quality.

**GSM8K accuracy is the right signal**: It measures what we actually care about — does the model still solve problems correctly? Op008 preserves accuracy perfectly.

## Additional Context

- Even torch.compile recompilation across versions can change accumulation order, producing the same cascade pattern.
- The self-consistency check in Stage 1 (running golden ref capture twice) sometimes detects non-determinism from torch.compile itself, recommending topk_relaxed. But topk_relaxed doesn't help either, as shown above.
- The lossless/lossy classification was designed to catch precision reduction (BF16->FP8), not accumulation reorder. Both lossy and reorder tracks fail exact_greedy, but for fundamentally different reasons.

## Decision Options

### Option A: Accuracy-only gate for all tracks

Replace both `exact_greedy` and `topk_relaxed` with a GSM8K accuracy-only check for all tracks:
- **Lossless**: Zero questions lost, 30 questions
- **Lossy**: Zero questions lost, 100 questions (higher sample because precision reduction has higher accuracy risk)

Drop token-level comparison entirely. It is a cascade detector, not a quality detector.

**Pros**: Simplest. Directly measures what we care about. Op008 passes.
**Cons**: Loses early warning from first-divergence containment. A subtle bug that shifts probabilities slightly might not show up in 30 questions but would fail token matching.

### Option B: First-divergence containment + accuracy

Check top-5 containment only at the first divergent position per question (not all positions). Then check GSM8K accuracy superset.

- If first-divergence containment fails: something fundamentally changed the probability distribution (real bug, not just accumulation reorder). FAIL.
- If accuracy check fails: optimization degraded quality. FAIL.
- Otherwise: PASS.

**Pros**: Catches cases where the optimization shifts probabilities beyond a near-tie (real bugs), while allowing legitimate cascade from accumulation reorder. More signal than accuracy-only.
**Cons**: More complex. First-divergence containment is a novel metric — not battle-tested.

### Option C: Keep topk_relaxed but only count pre-cascade positions

Modify the comparator to stop counting containment failures after the first divergence in each question. Only the initial divergent position (and a small window around it) is checked for containment.

**Pros**: Reuses existing topk_relaxed infrastructure.
**Cons**: Arbitrary window size. Still fundamentally a per-position metric applied to an autoregressive problem.

### Option D: Graduated — accuracy-only for lossless, topk_relaxed for lossy

- **Lossless tracks**: Accuracy-only (zero questions lost, 30q). Rationale: if dtype is preserved, any divergence is accumulation reorder. Accuracy is the only meaningful signal.
- **Lossy tracks**: Keep topk_relaxed + accuracy (100q). Rationale: precision reduction CAN cause systematic distribution shifts (not just cascade from a single flip), so per-position containment may still carry signal.

**Pros**: Targeted relaxation where the evidence supports it. Keeps stricter checks where precision reduction introduces real risk.
**Cons**: Two different comparator modes still exist. The argument that topk_relaxed doesn't help for lossy tracks either (cascade is cascade regardless of cause) may apply.
