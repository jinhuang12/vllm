# Stage 6: Integration Validation

After all parallel tracks complete in Stage 5, the main session determines how to combine and ship the results. This stage handles conflict detection, combined validation, and final decision-making.

## Decision Matrix

| Scenario | Action |
|----------|--------|
| Single candidate passes | Ship directly -- E2E already validated in Stage 5 |
| Multiple pass, different components | Cherry-pick both onto integration branch, re-run correctness + E2E |
| Multiple pass, same component | Pick the candidate with the best E2E speedup, ship that one |
| Single GATED_PASS candidate | Ship with gating dispatch intact — env var enabled, dispatch active |
| PASS + GATED_PASS, different components | Cherry-pick both onto integration branch, re-run E2E at all BS |
| PASS + GATED_PASS, same component | Pick the PASS candidate (cleaner integration) |
| Two GATED_PASS, different components | Cherry-pick both; if merge conflict, spawn resolver agent |
| Two GATED_PASS, same component | Pick candidate with best weighted E2E across all BS |
| None pass | Mark optimization target as `EXHAUSTED` in state.json |

## Conflict Detection

### Step 1: Identify Changed Files Per Track

For each passing track, compute the file diff against main:

```bash
# Worktree branch is named like: '{op_id}-{desc}' (i.e. op002-triton-gemm-silu-fusion)
git diff --name-only main...{op_id}-{desc}

### Step 2: Classify Overlap

Compare the file sets between all pairs of passing tracks:

- **Disjoint file sets** (no overlap): candidates modify different components and are combinable.
- **Overlapping file sets**: candidates modify the same component. Pick the one with the best E2E speedup.

Example with two passing tracks:

```bash
# Get changed files for each track
FILES_OP001=$(git diff --name-only main...op001-{desc})
FILES_OP002=$(git diff --name-only main...op002-{desc})

# Check for overlap
comm -12 <(echo "$FILES_OP001" | sort) <(echo "$FILES_OP002" | sort)
# Empty output = disjoint = combinable
# Non-empty output = overlapping = pick best
```

## Combined Validation Workflow

When multiple passing candidates modify **different components**, create an integration branch and validate the combination.

```bash
# Create integration branch from main
git checkout -b ammo/integration main

# Cherry-pick each passing track
git cherry-pick {op_id_1}
git cherry-pick {op_id_2}

# Run correctness tests for both components
pytest tests/path/to/component_1_tests.py
pytest tests/path/to/component_2_tests.py

# Run combined E2E benchmark
python scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir}
```

### Combined Result Evaluation

| Condition | Decision |
|-----------|----------|
| Combined E2E >= max(individual E2E results) | Ship the combined integration branch |
| Combined E2E < max(individual E2E results) | Ship the single track with the best individual E2E |
| Combined correctness fails | Fall back to shipping tracks individually (best E2E first) |

If a cherry-pick produces a merge conflict, treat the candidates as overlapping (same-component) and pick the one with the best E2E speedup.

### GATED_PASS Track Evaluation

When combining a GATED_PASS track with other tracks:
- Re-run E2E at ALL batch sizes including the gated track's non-beneficial range
- Verify no interaction effects between the gated dispatch and other optimizations
- If cherry-pick produces merge conflict on a GATED_PASS track: spawn resolver agent (see below)

## State Tracking

The integration section of `state.json` records all decisions and results:

```json
{
  "integration": {
    "status": "pending | validated | single_pass | combined | exhausted",
    "passing_candidates": [
      {
        "op_id": "op001",
        "verdict": "PASS",
        "e2e_speedup": 1.12,
        "files_changed": ["vllm/attention/backends/flash_attn.py"]
      },
      {
        "op_id": "op002",
        "verdict": "PASS",
        "e2e_speedup": 1.08,
        "files_changed": ["csrc/quantization/gptq_marlin.cu"]
      },
      {
        "op_id": "op003",
        "verdict": "GATED_PASS",
        "e2e_speedup": 1.025,
        "files_changed": ["vllm/model_executor/layers/some_layer.py"],
        "gating": {
          "env_var": "VLLM_OP003",
          "crossover_threshold_bs": 16,
          "regressing_bs": [32]
        }
      }
    ],
    "conflict_analysis": {
      "method": "file_set_overlap",
      "overlapping_pairs": [],
      "combinable_pairs": [["op001", "op002"]]
    },
    "combined_patch_branch": "ammo/integration",
    "combined_e2e_result": {
      "latency_ms": 42.3,
      "speedup_vs_baseline": 1.18,
      "speedup_vs_best_individual": 1.05
    },
    "final_decision": {
      "action": "ship_combined",
      "branch": "ammo/integration",
      "included_candidates": ["op001", "op002"],
      "total_e2e_speedup": 1.18
    }
  }
}
```

### Status Values

| Status | Meaning |
|--------|---------|
| `pending` | Integration has not started yet |
| `validated` | Single candidate validated successfully |
| `single_pass` | One candidate selected (sole passer or best among overlapping) |
| `combined` | Multiple candidates merged and validated successfully |
| `gated_pass` | One or more GATED_PASS candidates integrated with dispatch gating |
| `exhausted` | No candidates passed validation; no optimization to ship |

## Campaign Loop Transition (Stage 7)

After Stage 6 makes a SHIP or EXHAUSTED decision for the current round, the orchestrator evaluates whether to continue the campaign. See the Campaign Loop section in `SKILL.md` for the full protocol.

### If SHIP (one or more candidates passed)

1. Record shipped candidates in `campaign.shipped_optimizations`.
2. Update `campaign.cumulative_e2e_speedup` (multiplicative: `old x round_speedup`).

For GATED_PASS tracks, use the **minimum post-gating speedup across all batch sizes** as the `e2e_speedup` value (conservative — avoids needing production BS distribution data).

3. Record the round in `campaign.rounds` with all results.
4. Trigger re-profiling: invoke `ammo-researcher` subagent for Stage 1 baseline capture on the patched codebase.
5. After re-profile: run bottleneck mining (Stage 2) on the new baseline.

**Lazy invalidation with GATED_PASS**: When re-profiling after a GATED_PASS track ships, profile at ALL campaign batch sizes (not just one). The gated optimization's f-shift is BS-dependent — f changes only at gated-on batch sizes, not at gated-off batch sizes. Use the **maximum f-shift across all BS** for the lazy invalidation test to be conservative.

5b. **Lazy invalidation of overlapped debate winners** (if `debate.next_round_overlap.selected_winners` is non-empty):
   - For each winner in `debate.next_round_overlap.selected_winners`:
     - Retrieve `f_old` from `debate.next_round_overlap.f_values_at_proposal[op_id]`.
     - Compute `f_new` from the new bottleneck_analysis.md.
     - If `f_old < 0.05`: skip invalidation for this candidate (kernel too small for reliable f-shift measurement).
     - If `f_old >= 0.05` AND `|f_new - f_old| / f_old > 0.3`: discard the candidate. Record as `{"event": "candidate_invalidated", "op_id": "...", "reason": "f_shift", "f_old": ..., "f_new": ...}`.
     - Otherwise: retain the candidate.
   - If any candidates survive invalidation: move them to `debate.selected_winners`. Clear `debate.next_round_overlap` to initial state. Skip Step 8 (debate) -- proceed directly to Stages 4-5.
   - If all candidates are invalidated: clear `debate.next_round_overlap` to initial state. Proceed to Step 8 (fresh debate).
6. Read the new top bottleneck's share of total decode latency.
7. If `top_bottleneck_share < campaign.min_e2e_improvement_pct`: set `campaign.status = "campaign_complete"`. Done.
8. Else: increment `campaign.current_round`, enter Stage 3 for the next round.

### If EXHAUSTED (no candidates passed this round)

1. Record the failed round in `campaign.rounds`.
2. Check diminishing returns against the CURRENT profiling data (no re-profile since nothing changed).
   - If `top_bottleneck_share < threshold`: set `campaign.status = "campaign_exhausted"`. Done.
   - Else: start a new debate round from existing bottleneck data (skip re-profiling, skip Stage 2).

### Hook Enforcement

The Stop hook (`ammo-stop-guard.sh`) blocks the session from ending while the campaign is active. The orchestrator must either complete the current stage or set `campaign.status` to `"paused"` before the session can end.

## Overlapped Debate and Implementation Interaction

During overlapped operation, these interactions may occur:

### A Track Ships While Debate Is Running

When an implementation track passes all gates while the overlapped debate is still in progress:
- Record the track result in `parallel_tracks` as usual.
- Do NOT terminate the debate. Let it complete naturally.
- The debate's winners will be validated against post-ship profiling data in the next round (lazy invalidation).

### Debate Finishes While Tracks Are Running

When the overlapped debate completes before all implementation tracks:
- Score winners and shut down debate champions.
- Record winners in `debate.next_round_overlap.selected_winners`.
- Continue monitoring remaining implementation tracks.
- Do NOT start the next round until all current-round tracks complete.

### All Tracks Fail But Debate Produced Winners

If all implementation tracks for round N fail (round EXHAUSTED), but the overlapped debate produced viable winners:
- Record round N as EXHAUSTED.
- The overlapped debate winners are still valid -- they were based on the same profiling data.
- Skip lazy invalidation (no re-profiling occurred since nothing shipped).
- Move overlapped debate winners directly to `debate.selected_winners` for round N+1 implementation.
- Clear `debate.next_round_overlap` to initial state after moving winners.

**However**: If round EXHAUSTED AND the diminishing returns threshold is met (campaign_exhausted), discard the overlapped debate winners and shut down debate champions. See "Campaign Terminates During Overlapped Debate" below.

### Campaign Terminates During Overlapped Debate

If the campaign transitions to `campaign_complete` or `campaign_exhausted` while `debate.next_round_overlap.active` is `true`:
1. Shut down all overlapped debate champions (and delegates) via `shutdown_request`.
2. Discard overlapped debate results -- they will never be used.
3. Clear `debate.next_round_overlap` to initial state.
4. Proceed with TeamDelete and campaign termination as normal.

This can occur in two scenarios:
- **SHIP + diminishing returns met**: A track shipped, re-profiling shows top bottleneck below threshold. The overlapped debate was running in parallel but its results are now irrelevant.
- **All tracks EXHAUSTED + diminishing returns met**: No tracks shipped, existing profiling shows top bottleneck below threshold. The debate winners are technically valid but the campaign is ending.

## Resolver Agent for Merge Conflicts

When cherry-picking a GATED_PASS track (or combining multiple GATED_PASS tracks) produces merge conflicts, the orchestrator spawns a dedicated resolver.

### When Invoked

- Cherry-pick of a GATED_PASS track onto integration branch produces git merge conflicts
- Two GATED_PASS tracks targeting different components but touching overlapping files (e.g., both register env vars in `vllm/envs.py`)

### Workflow

1. **Orchestrator** spawns a resolver agent (`.claude/agents/ammo-resolver.md`, Opus) with:
   - The conflicting files and conflict markers
   - Both tracks' gating metadata (env vars, dispatch conditions, crossover thresholds)
   - The optimization intent for each track

2. **Resolver** proposes a merged version preserving both gating dispatches

3. **Orchestrator** spawns a DA reviewer (Sonnet) to verify:
   - Correct dispatch ordering (more specific conditions first)
   - No interaction effects between gating conditions
   - Env var namespace conflicts (each optimization must have a unique env var)
   - torch.compile safety of the merged dispatch logic

4. If DA approves: merged version committed to integration branch
5. If DA rejects: resolver revises based on DA feedback (max 2 iterations), then escalates to orchestrator

### Priority Dispatch Chain (Overlapping Call Sites)

For the rare case where two gated optimizations dispatch at the same call site, use a priority chain instead of nested conditionals:

```python
AMMO_DISPATCH_CHAIN = [
    # (condition, kernel_fn, name) — evaluated in order, first match wins
    (lambda M: 2 <= M <= 16, fused_qkv_fn, "op012_fused_qkv"),
    (lambda M: 2 <= M <= 32, selective_fn, "op007_selective"),
]

def ammo_dispatch(layer, x, weight, bias=None):
    M = x.numel() // x.shape[-1]
    for condition, kernel_fn, name in AMMO_DISPATCH_CHAIN:
        if condition(M):
            return kernel_fn(layer, x, weight, bias)
    return default_fn(layer, x, weight, bias)
```

### State Recording

Record resolver invocation in `integration`:
```json
{
  "resolver_invoked": true,
  "resolver_outcome": "approved" | "rejected" | "escalated",
  "conflicting_tracks": ["op001", "op003"]
}
```
