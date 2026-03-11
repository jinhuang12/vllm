# Campaign Loop

The campaign loop transforms AMMO from a single-pass pipeline into an iterative optimization loop that continues until diminishing returns. Each round discovers, debates, and implements optimizations against the current bottleneck landscape.

## Loop Structure

```
Round N:
  1. Profile (Stages 1-2) — re-profile against patched baseline (skip for round 1)
  2. Debate (Stage 3) — full adversarial debate; may overlap with round N-1 implementation
  3. Implement + Validate (Stages 4-5) — parallel worktree tracks
  4. Integrate (Stage 6) — ship or round-fail
  5. Campaign Evaluation (Stage 7) — diminishing returns check → next round or stop
```

## Async Pipeline: Debate Overlaps Implementation

While round N implementers work on debate winners:

1. Orchestrator may start round N+1 debate from existing bottleneck data.
2. New debate follows the full adversarial protocol — no lighter screening.
3. Debate winners are placed in `campaign.pending_queue`, NOT sent to implementation yet.

### When a round N candidate ships (triggers re-profile)

- Let the in-progress debate finish (do not abort).
- Re-validate queued debate winners against the new profiling data (see Re-validation below).
- Discard winners whose target bottleneck dropped below the diminishing returns threshold.
- Remaining winners enter implementation when slots open in the next round.

### When round N completes without any ship

- Queued debate winners can proceed to implementation immediately (no re-validation needed since the profiling data hasn't changed).

## Re-validation After Re-profiling

When the bottleneck landscape shifts (because a candidate shipped), queued candidates may be stale. For each candidate in `campaign.pending_queue`:

1. Check if the target kernel still appears in the updated `bottleneck_analysis.md`.
2. Recalculate expected E2E impact using the new f-values.
3. If `new_f × kernel_speedup < 1%` E2E improvement: discard (not worth implementing).
4. If still viable: candidate proceeds to implementation in the next available slot.

This is a feasibility recheck, NOT a full re-debate. The debate evidence (micro-experiments, kernel analysis) is still valid — only the f-value has changed.

## Diminishing Returns

After each round's integration, the orchestrator evaluates whether to continue:

1. Read the top bottleneck's share of total decode latency from the profiling data.
2. Compare against `campaign.diminishing_returns_threshold_pct` (default: 3%).
3. If below threshold: no single kernel optimization can yield meaningful E2E gains → **stop**.

The threshold is set at campaign creation time via `new_target.py --diminishing-returns-threshold`.

### After SHIP

Re-profile first (the bottleneck landscape has shifted), then check the NEW top bottleneck against the threshold.

### After EXHAUSTED (no candidates shipped this round)

Check the threshold against the EXISTING profiling data (no re-profile needed since nothing changed). If above threshold, start a new debate round from existing data.

## Campaign State Transitions

```
active → (round completes, threshold not met) → active (next round)
active → (threshold met after ship) → campaign_complete
active → (threshold met after exhaust) → campaign_exhausted
active → (user requests pause) → paused
paused → (user requests resume) → active
```

## In-Flight Tracks During Re-profiling

When a candidate ships and triggers re-profiling, other tracks from the same round may still be running:

1. Let all in-flight implementations complete (do NOT terminate).
2. Validate their results against the ORIGINAL round's baseline (not the new one).
3. If they pass: they also ship (additional cumulative gain).
4. Record all results in the current round's entry in `campaign.rounds`.
5. The next round starts only after all current-round tracks have completed.

## Hook Enforcement

The campaign loop is enforced by hooks in `.claude/settings.local.json`:

| Hook | Event | What it prevents |
|------|-------|-----------------|
| Campaign eval gate (`ammo-gate-guard.sh`) | TaskCompleted | Skipping diminishing returns check |
| Stop guard (`ammo-campaign-stop-guard.sh`) | Stop | Ending session mid-campaign |
| Agent gate validator | TaskCompleted | Skipping re-profiling after ship |
| Session resume hooks | PreCompact/SessionStart | Losing campaign state during compaction |
| Debate scope guard (`ammo-gate-guard.sh`) | TaskCompleted | Overwriting previous round's debate artifacts |

## Campaign Object Schema

See `state.json` schema in SKILL.md. Key fields:

```json
{
  "campaign": {
    "status": "active | paused | campaign_complete | campaign_exhausted",
    "current_round": 1,
    "diminishing_returns_threshold_pct": 3,
    "cumulative_e2e_speedup": 1.0,
    "rounds": [{ "round_id": 1, "...": "..." }],
    "shipped_optimizations": [],
    "pending_queue": []
  }
}
```

Each round entry:
```json
{
  "round_id": 1,
  "profiling_baseline_path": "runs/baseline_round_1/",
  "top_bottleneck_share_pct": 15.2,
  "debate_team_name": "ammo-debate-...",
  "selected_candidates": ["op001", "op002"],
  "implementation_results": {
    "op001": {"status": "PASSED", "e2e_speedup": 1.12},
    "op002": {"status": "FAILED", "reason": "correctness"}
  },
  "shipped": ["op001"],
  "cumulative_speedup_after": 1.12
}
```
