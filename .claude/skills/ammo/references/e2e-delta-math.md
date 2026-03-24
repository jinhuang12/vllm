# E2E Delta Math (Stop‑Condition)

It’s easy to get “good” microbench speedups and still see tiny end‑to‑end gains.

## Relationship

Let:
- `T_total = T_component + T_other`
- `f = T_component / T_total` (component share of end‑to‑end)
- `s = T_component_new / T_component` (component speedup factor; `s < 1` is faster)

Then end‑to‑end speedup is:
```
T_total_new = s*T_component + T_other
improvement = 1 - (T_total_new / T_total) = 1 - (s*f + (1 - f))
           = f * (1 - s)
```

So the best‑case end‑to‑end improvement is upper‑bounded by the fraction of time spent in the target component.

## Practical usage in Phase 1 / Phase 4

1. Measure `f` using Nsight Systems (or by timing a config that disables component optimizations for comparison).
2. The minimum required E2E improvement is `min_e2e_improvement_pct` from `state.json` (default: 1%). See `references/validation-defaults.md` for how this threshold is applied at each decision point.
3. Solve for required component improvement:
```
min_e2e_improvement_pct <= f * (1 - s)  =>  (1 - s) >= min_e2e_improvement_pct / f
```

If `min_e2e_improvement_pct / f` is implausibly large, switch to:
- a simpler optimization approach to harvest the most reliable µs wins, or
- document the limitation and stop.

## Which `f` to use: f_decode vs f_total

For decode-heavy workloads (output_len >> input_len), the E2E latency is dominated by decode steps. Use **f_decode** (component share within the FULL CUDA graph decode phase), not f_total (full trace including warmup and prefill).

**Common trap**: A kernel may show f_total = 0.23 in the full nsys trace but f_decode = 0.0 because it only runs during prefill or autotuning warmup. Optimizing such a kernel yields ~0% E2E improvement in production.

**Rule**: Always verify your target component's f_decode before committing to an optimization. If f_decode ≈ 0 and the workload is decode-heavy, the optimization ceiling is near zero regardless of kernel speedup.

**Transient overhead**: Triton `@triton.autotune` probing, torch.compile graph capture, and JIT compilation appear in traces but are one-time startup costs. They inflate f_total but have f_decode = 0. These are NOT optimization targets for serving workloads.

## Crossover Prediction from Kernel Data

When an optimization improves some batch sizes but regresses others, the crossover batch size can be predicted from kernel-level measurements without running expensive E2E benchmarks at every intermediate BS.

### Per-BS Delta Math

The standard formula `E2E_improvement = f × (1 - s)` generalizes to per-BS:

```
predicted_e2e_improvement(BS) = f_decode(BS) × (1 - T_kernel_opt(BS) / T_kernel_base(BS))
```

Where:
- `f_decode(BS)` = component share of total decode latency at batch size BS
- `T_kernel_opt(BS)` = optimized kernel time at batch size BS
- `T_kernel_base(BS)` = baseline kernel time at batch size BS

### Why f Varies with Batch Size

The component share `f` is NOT constant across batch sizes:
- At BS=1 (decode), a kernel may be 8% of total decode latency (`f = 0.08`)
- At BS=32, the same kernel may be only 3% (`f = 0.03`) because other components scale differently

Extract per-BS `f_decode` from the Stage 1 nsys per-bucket traces. Each batch-size bucket has its own profiling trace with kernel-level timing breakdowns.

For intermediate BS values (not directly profiled), linearly interpolate between measured `f` values at adjacent profiled batch sizes. This is approximate but sufficient for crossover prediction.

### Finding the Crossover

The crossover batch size is where the predicted E2E improvement drops below the noise tolerance:

```
crossover_bs = max(BS) where predicted_e2e_improvement(BS) >= noise_tolerance_pct / 100
```

### When Kernel Prediction Is Unreliable

If the warm-cache vs cold-cache kernel speedup ratio exceeds 1.5x at any probed BS, the kernel prediction may not reflect production behavior (where L2 cache pressure from the full model pipeline dominates). In this case:
- For narrow BS ranges (< 15 values): fall back to E2E binary search
- For wide BS ranges (>= 15 values): gate to exact PASS batch sizes only (skip probing)

See `references/crossover-probing.md` for the full probing protocol.
