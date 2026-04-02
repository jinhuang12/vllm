# Crossover Probing Protocol

Determines the batch-size threshold where an optimization transitions from beneficial to harmful. Used when the track-level verdict is `GATING_REQUIRED` (some batch sizes PASS, others REGRESSED).

## When Triggered

Crossover probing activates when:
- At least one tested BS has verdict `PASS` (speedup >= 1.0)
- At least one tested BS has verdict `REGRESSED` (speedup below noise tolerance but above catastrophic)
- The champion has evaluated gating feasibility and spawned a sub-agent to probe

If ALL tested BS are PASS/NOISE, no probing needed. If ALL are REGRESSED/CATASTROPHIC, the track FAILs — no probing can help.

## Non-Monotonic Guard

If the PASS and REGRESSED batch sizes are interleaved (e.g., BS=1 PASS, BS=8 REGRESSED, BS=32 PASS), there is no single crossover point. In this case:
- **Skip probing entirely**
- Gate the optimization to the exact set of PASS batch sizes only (no interpolation)
- The dispatch condition is an explicit set membership check, not a threshold comparison

## Kernel-Informed Protocol (Primary)

Instead of expensive E2E binary search, use cheap kernel-level data to predict the crossover:

### Phase 1: Kernel Sweep (~1-2 minutes)
Run kernel-level benchmarks at intermediate BS values between the known beneficial and regressed ranges. Kernel benchmarks are cheap (~seconds each, no model load needed).

Example: If BS=8 is PASS and BS=32 is REGRESSED, test kernels at BS=12, 16, 20, 24.

### Phase 2: Delta Math Prediction
For each intermediate BS, compute:
```
predicted_e2e_delta(BS) = f_decode(BS) × (1 - T_kernel_opt(BS) / T_kernel_base(BS))
```

Where `f_decode(BS)` is the component's share of total decode latency at that batch size (from Stage 1 nsys per-bucket traces). See `references/e2e-delta-math.md` §Crossover Prediction.

The predicted crossover BS is where `predicted_e2e_delta` crosses the noise tolerance threshold (drops below `noise_tolerance_pct / 100`).

### Phase 3: E2E Confirmation (1-2 runs)
Run 1-2 E2E benchmarks at the predicted crossover BS to confirm:
- If E2E confirms (speedup at predicted crossover is PASS or NOISE): `crossover_threshold = confirmed BS`
- If E2E disconfirms: adjust by ±1-2 BS and re-confirm (max 2 adjustments)

## Wide-Range Fallback

If the kernel sweep is inconclusive (warm/cold speedup ratio > 1.5x at intermediate BS) AND the range spans > 15 batch-size values:
- **Skip probing entirely**
- Gate to exact PASS batch sizes only (same as non-monotonic guard)
- This avoids burning 30 minutes on an unresolvable search

For narrow ranges (< 15 BS values) where kernel sweep is inconclusive, fall back to E2E binary search:
```
While hi - lo > 1:
    mid = (lo + hi) // 2
    Run E2E at BS=mid
    If speedup >= (1.0 - noise_tolerance): lo = mid
    Else: hi = mid
crossover_threshold = lo
```

## Time Budget

- Kernel sweep: ~1-2 minutes
- E2E confirmation: ~5-10 minutes (1-2 runs)
- Total: ~10 minutes typical (vs 25+ minutes for pure E2E binary search)
- **Hard timeout: 30 minutes**. If not converged, use `lo` from last iteration (conservative).

## One Attempt Rule

After crossover probing completes and the champion implements gating:
- The champion spawns a sub-agent for kernel re-validation (5.1a + 5.2) AND re-runs the sweep (5.1b + 5.3a + 5.3b)
- If any gate shows REGRESSED or CATASTROPHIC at any BS: **track FAILs**
- Do NOT attempt nested gating (no recursive probing)

## State Recording

Record crossover probing results in `state.json:parallel_tracks.{op_id}.gating.crossover_probing`:

```json
{
  "method": "kernel_informed",
  "probed_points": [
    {"bs": 12, "kernel_speedup": 1.15, "predicted_e2e_delta": 0.012},
    {"bs": 16, "kernel_speedup": 1.08, "predicted_e2e_delta": 0.006},
    {"bs": 20, "kernel_speedup": 0.95, "predicted_e2e_delta": -0.004}
  ],
  "predicted_bs": 16,
  "confirmed_bs": 16,
  "converged": true,
  "time_minutes": 8.5
}
```

## Conservative Bias

When in doubt, use the last known-beneficial BS as the threshold. A slightly narrower beneficial range is better than shipping a regression. The env var gating always provides an escape hatch for operators.
