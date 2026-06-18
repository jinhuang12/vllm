# E2E Delta Math (Stop‑Condition)

It's easy to get "good" microbench speedups and still see tiny end‑to‑end gains. The Amdahl multiplier you plug into the formula has to be the kernel's contribution to the **end‑to‑end wall clock**, not its share within a single phase. This document defines the correct multiplier (`f_e2e`), shows how to derive it from sweep + profiling outputs, and gives the per-category projection formulas champions must use.

## f_e2e: The Correct Amdahl Multiplier

Let:
- `T_e2e` = total end-to-end wall time of one request (prefill + decode).
- `T_component` = wall-clock time spent in the target component **across the whole request**.
- `f_e2e = T_component / T_e2e` — the component's share of E2E wall time.
- `s` = component speedup factor (`T_component_new / T_component`; `s < 1` is faster, but here we use `s ≥ 1` so `1 - 1/s` is a fraction).

Then the standard Amdahl form for E2E improvement is:

```
E2E_improvement_pct = f_e2e × (1 - 1/s)
```

### Why `f_decode` is wrong for prediction

`f_decode` (component share within decode-step GPU time) is the **ranking** metric — it tells us which kernel dominates the decode phase. It is *not* the Amdahl multiplier for E2E gains, because:

1. Decode time is only a fraction of E2E (`decode_share_of_e2e < 1` whenever prefill > 0).
2. Decode wall-time is not all kernel-busy time (`decode_busy < 1` whenever inter-kernel overhead exists).

Using `f_decode` directly over-projects E2E gains by `1 / (decode_busy × decode_share_of_e2e)`, which on production stacks (decode_busy≈0.57, decode_share_of_e2e≈0.82 in the motivating Qwen3.6 campaign) is a **2.1×** over-projection — and reaches **5–50×** on prefill-heavy or low-busy workloads.

`f_decode` stays in the ranking table as a diagnostic column; `f_e2e` is the column champions plug into Amdahl.

## Conversion Formula

```
f_e2e = f_decode × decode_busy × decode_share_of_e2e
```

| Variable | Definition | Data source |
|----------|-----------|-------------|
| `f_decode` | Component's share of decode-step GPU time | nsys trace; ranking metric only |
| `decode_busy` | `decode_kernel_time / decode_wall_time`. Fraction of decode wall-time that is actual GPU kernel work (vs scheduling, launch gaps). | Tier-0: `sum(kernel_dur in decode region) / decode_wall_time`. Tier-1: `Σ all_kernel_dur / (avg_latency × num_profiled_iters)` (sweep-level aggregate; **never** mix Chrome-trace timestamps with `RequestOutput.metrics` `time.monotonic` clocks). |
| `decode_share_of_e2e` | `decode_avg_s / (prefill_avg_s + decode_avg_s)`. Fraction of E2E wall time that is decode (vs prefill). | `e2e_latency_results.json` — the sweep emits `prefill_avg_s` and `decode_avg_s` from `RequestOutput.metrics`. Tier-C fallback: `OL / (IL + OL)` (strict under-estimate). |

The researcher computes `decode_busy` and `decode_share_of_e2e` once per BS in Stage 2 and writes both into the `## Workload Dilution` table in `bottleneck_analysis.md`. Champions read those values; they do not re-derive them.

## Inductor Baseline Parity (Gate 5.2 → E2E Translation)

Gate 5.2 measures kernel speedup against the production function called in isolation (inside a CUDA graph, but NOT inside `torch.compile`). This validates your kernel is functionally correct and faster than the raw call. But it can drastically overstate the E2E opportunity.

The reason: vLLM's Inductor pass pipeline includes pattern-matching fusion passes (`RMSNormQuantFusionPass`, `ActQuantFusionPass`, etc.) that rewrite multi-kernel chains into single fused ops at compile time. If your target chain is already fused by Inductor in the compiled baseline, Gate 5.2's "2.7x speedup" is measured against something that doesn't actually run in production — the real competition is the Inductor-fused kernel.

### The Check

Before projecting E2E from Gate 5.2 kernel speedup, verify the ACTUAL per-kernel time of the baseline chain inside the compiled CUDA graph:

1. Open the Stage 1 nsys trace (`rounds/{N}/profiling/nsys/baseline_bs{BS}.nsys-rep`)
2. Grep for your target kernel names in the decode window
3. If the unfused chain (e.g., separate `rms_norm` + `dynamic_scaled_fp8_quant`) does NOT appear — but a single fused kernel does (e.g., `rms_norm_dynamic_per_token_quant`) — then Inductor has already fused your target

### Corrected Projection

When Inductor has fused the baseline:

```
effective_speedup = inductor_fused_kernel_time / your_kernel_time   (from nsys)
E2E_improvement = f_e2e × (1 - 1/effective_speedup)
```

NOT:

```
E2E_improvement = f_e2e × (1 - 1/gate_5_2_speedup)   ← uses unfused baseline, over-projects
```

If `effective_speedup < 1.1x`, the optimization cannot meaningfully improve E2E regardless of how large Gate 5.2 reports.

### Which Passes to Check

```bash
grep -rn "FusionPass\|PatternMatcherPass" vllm/compilation/passes/fusion/
```

The main ones as of this writing: `RMSNormQuantFusionPass` (rms+quant), `ActQuantFusionPass` (activation+quant), `RopeKVCacheFusionPass`, `AllReduceRMSFusionPass`. If your target overlaps with any of these, your Gate 5.2 baseline is likely already fused in production.

## Worked Example — Qwen3.6-27B-FP8, H100, BS=8

From the motivating campaign (see Appendix A of the design spec):

| Quantity | Value | Source |
|----------|-------|--------|
| `f_decode` (DeepGEMM gate_up) | 0.066 | nsys decode-region kernel time |
| `decode_busy` | 0.57 | `9.1 s / 15.9 s` from `## Workload Dilution` |
| `decode_share_of_e2e` | 0.82 | `15.9 s / 19.4 s` from `RequestOutput.metrics` |

```
f_e2e = 0.066 × 0.57 × 0.82 = 0.031
```

A 1.18× kernel speedup (the BW-utilization-limited ceiling for this DeepGEMM shape) projects:

```
E2E_improvement = 0.031 × (1 - 1/1.18) = 0.031 × 0.153 = 0.005  (0.47%)
```

…which may be below `min_e2e_improvement_pct` depending on the configured threshold. A naïve `f_decode`-based projection gives `0.066 × 0.153 = 1.0%`, over-projecting by **2.1×** and giving the champion false confidence the proposal is shippable. After conversion, the same evidence correctly classifies the proposal as marginal. Always compare against the campaign's configured `min_e2e_improvement_pct` (see `references/validation-defaults.md` for default).

## Red Flags — When Conversion Is Mandatory

The `f_decode → f_e2e` conversion is **mandatory** whenever ANY of these workload-dilution red flags fires for the target BS:

| Red flag | Condition | Why |
|----------|-----------|-----|
| **Prefill non-trivial** | `prefill_share_of_e2e ≥ 0.10` | Decode is ≤ 90% of E2E; even a modest decode share inflates `f_decode` vs `f_e2e`. |
| **Pipeline not saturated** | `decode_busy < 0.85` | Substantial inter-kernel overhead means kernel time ≠ decode wall time. |
| **Long context** | `input_len ≥ 512` | Prefill cost grows with `IL`; the safe heuristic is to always convert when `IL` is non-trivial, even if `RequestOutput.metrics` is unavailable. |

If none of the red flags fire (all-decode or near-saturated workload), `f_decode ≈ f_e2e` within ~5% and the conversion is a no-op. The scoring rubric still requires champions to show the conversion was performed (or that all three red-flag conditions were checked and none fired) — see `debate-scoring-rubric.md`.

## Prefill-Active Components — Scope Note

The `f_e2e = f_decode × decode_busy × decode_share_of_e2e` formula handles **decode-only** kernels — kernels whose contribution to E2E is entirely through the decode phase. Some kernels run during BOTH prefill and decode (e.g., attention, embedding, RMSNorm). For these:

- The published `f_e2e` is a **lower bound** on the kernel's true E2E contribution.
- Actual gains from speeding up the kernel may *exceed* the projection because the kernel also accelerates prefill.
- The researcher flags such components with `prefill-active? = Yes` in the Top Components table; champions targeting them should treat their projection as conservative and note the upside in their proposal.

The bound is conservative (under-projects, never over-projects), so a `f_e2e × (1 - 1/s) ≥ min_e2e_improvement_pct` proposal is safe to ship.

## Four-Slice Composition Model

Total E2E wall-time decomposes into four exclusive slices:

```
total_e2e = prefill_time + decode_kernel_time + decode_inter_kernel_time + other
```

Equivalently, in the dimensionless shares the researcher publishes:

```
1.0 = prefill_share_of_e2e + (decode_busy × decode_share_of_e2e) + inter_kernel_share + other_share
        prefill              decode_kernel                          decode_inter_kernel  other (small)
```

Where:

- `prefill_share_of_e2e` = `prefill_avg_s / total_e2e_s` — prefill's slice of E2E.
- `decode_busy × decode_share_of_e2e` — the slice that is actual decode-step kernel execution (sum of all `f_e2e` values for decode-only kernels equals this slice, modulo measurement noise).
- `inter_kernel_share = (1 - decode_busy) × decode_share_of_e2e` — decode wall time spent NOT in kernels (scheduling, launch gaps, host-side overhead).
- `other` — small residual (warmup, KV management, cleanup); typically < 0.02.

Each optimization category targets a different slice. Pick the projection formula that matches your category — using the wrong slice's `f` is the most common projection error and is worth a 2-point deduction in the scoring rubric.

## Per-Category Projection Formulas

The Amdahl form `f × (1 - 1/s)` only applies cleanly to single-component kernel work. Other categories target overhead slices (inter-kernel time, prefill, dispatch) and have category-specific projection formulas. **Champions MUST use the formula that matches the SLICE their declared `Category` attacks** — project by slice, not by enum value. Using the wrong-slice formula scores 0/10 on the E2E impact criterion.

| Category | Projection formula | Variables defined |
|----------|--------------------|-------------------|
| `kernel_replacement` | `f_e2e(component) × (1 - 1/s)` | `s` = kernel speedup ratio (new vs baseline at target shape, under CUDA graphs + torch.compile). |
| `kernel_fusion` | `f_e2e(chain) × (1 - 1/s_fused)` | `f_e2e(chain)` = sum of `f_e2e` over all kernels in the fused chain. `s_fused` = `chain_time / fused_time`. |
| `custom_kernel`, `attention_kv_layout` | `f_e2e(component) × (1 - 1/s)` | Decode-kernel slice — same form as `kernel_replacement`. |
| `weight_layout_transform` | `f_e2e(chain) × (1 - 1/s_fused)` | Decode-kernel slice — same form as `kernel_fusion`: `f_e2e(chain)` = sum over the merged GEMMs; `s_fused` = `chain_time / merged_time`. |
| `dispatch_optimization`, `execution_pipeline_restructuring`, `communication_strategy`, `compute_graph_pass` | `inter_kernel_share × host_fraction × elimination_fraction` | Inter-kernel slice — `host_fraction` = host-attributable share of `inter_kernel_share`; `elimination_fraction` = measured host-side latency reduction (dispatches removed, comm latency cut, or graph-op-count drop). |

**Novel descriptor:** use the formula of whichever of the two regimes (decode-kernel vs inter-kernel) the proposal's declared `Slice targeted` matches.

See `references/optimization-categories.md` for the by-slice gate routing, evidence requirements, and disambiguating examples.

## Practical Usage in Phase 0 / Phase 4

1. Read `f_decode`, `decode_busy`, `decode_share_of_e2e`, `inter_kernel_share`, `prefill_share_of_e2e` from the `## Workload Dilution` table in `bottleneck_analysis.md`.
2. If the researcher already published `f_e2e` for your component (which they do for decode-only candidates in the Top Components table), use it directly — no conversion needed.
3. Otherwise compute `f_e2e` per the formula above.
4. Pick the projection formula from the per-category table matching your declared `Category`.
5. Solve for the kernel speedup `s` (or analogous variable) needed to clear the `min_e2e_improvement_pct` threshold.
6. If the required `s` is implausibly large given the BW/compute ceiling, switch to:
   - a different target where `f_e2e` is larger, OR
   - document the limitation and stop the proposal.

## Crossover Prediction from Kernel Data

When an optimization improves some batch sizes but regresses others, the crossover batch size can be predicted from kernel-level measurements without running expensive E2E benchmarks at every intermediate BS.

### Per-BS Delta Math

The standard formula generalizes to per-BS:

```
predicted_e2e_improvement(BS) = f_e2e(BS) × (1 - T_kernel_opt(BS) / T_kernel_base(BS))
```

Where `f_e2e(BS)` is the per-BS row in the `## Workload Dilution` table. **Use `f_e2e(BS)`, not `f_decode(BS)`** — see Red Flags above.

### Why f Varies with Batch Size

The component share is NOT constant across batch sizes:
- At BS=1, `decode_busy` is often higher (smaller launch gaps relative to kernel work) and a kernel may be 8% of E2E (`f_e2e = 0.08`).
- At BS=32, the same kernel may be 3% (`f_e2e = 0.03`) because (a) the kernel scales superlinearly, (b) `decode_busy` may drop as more launches fit into a step, (c) decode_share may shift if prefill scales differently.

Extract per-BS `f_e2e` from the per-bucket `## Workload Dilution` rows. For intermediate BS values not directly profiled, linearly interpolate between adjacent rows. This is approximate but sufficient for crossover prediction.

### Finding the Crossover

```
crossover_bs = max(BS) where predicted_e2e_improvement(BS) >= noise_tolerance_pct / 100
```

### When Kernel Prediction Is Unreliable

If the warm-cache vs cold-cache kernel speedup ratio exceeds 1.5× at any probed BS, the kernel prediction may not reflect production behavior (where L2 cache pressure from the full model pipeline dominates). In this case:
- For narrow BS ranges (< 15 values): fall back to E2E binary search.
- For wide BS ranges (>= 15 values): gate to exact PASS batch sizes only (skip probing).

See `references/crossover-probing.md` for the full probing protocol.
