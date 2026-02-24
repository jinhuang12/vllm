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
