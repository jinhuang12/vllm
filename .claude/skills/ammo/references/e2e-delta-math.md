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
2. Decide the target end‑to‑end improvement `I_target` (e.g., 5% at BS≤8).
3. Solve for required component improvement:
```
I_target <= f * (1 - s)  =>  (1 - s) >= I_target / f
```

If `I_target / f` is implausibly large, switch to:
- a simpler optimization approach to harvest the most reliable µs wins, or
- document the limitation and stop.

## Which `f` to use: f_decode vs f_total

For decode-heavy workloads (output_len >> input_len), the E2E latency is dominated by decode steps. Use **f_decode** (component share within the FULL CUDA graph decode phase), not f_total (full trace including warmup and prefill).

**Common trap**: A kernel may show f_total = 0.23 in the full nsys trace but f_decode = 0.0 because it only runs during prefill or autotuning warmup. Optimizing such a kernel yields ~0% E2E improvement in production.

**Rule**: Always verify your target component's f_decode before committing to an optimization. If f_decode ≈ 0 and the workload is decode-heavy, the optimization ceiling is near zero regardless of kernel speedup.

**Transient overhead**: Triton `@triton.autotune` probing, torch.compile graph capture, and JIT compilation appear in traces but are one-time startup costs. They inflate f_total but have f_decode = 0. These are NOT optimization targets for serving workloads.
