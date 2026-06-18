# Stage 6: Integration Validation

After all parallel tracks complete in Stage 5, the main session determines how to combine and ship the results. This stage handles conflict detection, combined validation, and final decision-making.

**Precondition (HARD):** All `parallel_tracks.tracks[*].status` must be in `{PASS, GATED_PASS, FAIL}`. The state-validate hook (`.claude/hooks/ammo-state-validate.sh`) blocks any write that sets `current_stage = 6_integration` while non-terminal tracks exist (`IN_PROGRESS`, `GATING_REQUIRED`, `GPU_BLOCKED`). This prevents late-arriving verdicts from being silently dropped — see Track A17 (observed in session 6327c5d6).

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

## Single-Track Short-Circuit

When exactly one track has `status ∈ {PASS, GATED_PASS}`, integration adds no information — Stage 5 already validated the optimization end-to-end. Skip the combined sweep and promote directly.

### Precondition

```bash
PASSING=$(jq -r '[.campaign.rounds[(.campaign.current_round - 1)].parallel_tracks.tracks | to_entries[] | select(.value.status == "PASS" or .value.status == "GATED_PASS") | .key] | .[]' state.json)
NUM_PASSING=$(printf '%s' "$PASSING" | grep -c . || true)
# NUM_PASSING == 1 → short-circuit; 0 → exhausted; >= 2 → full ceremony below
```

### Procedure (single passer)

1. **Copy Stage 5 sweep results into the integration slot** so Pre-SHIP mechanical checks find them at the canonical path:

   ```bash
   OP_ID=$(printf '%s' "$PASSING" | head -1)
   CR=$(jq -r '.campaign.current_round' state.json)
   TRACK_VERDICT=$(jq -r --arg op "$OP_ID" '.campaign.rounds[(.campaign.current_round - 1)].parallel_tracks.tracks[$op].status' state.json)
   mkdir -p "{artifact_dir}/rounds/${CR}/sweeps/integration/json"
   cp "{artifact_dir}/rounds/${CR}/sweeps/opt/${OP_ID}/json/"opt_bs*.json "{artifact_dir}/rounds/${CR}/sweeps/integration/json/"
   cp "{artifact_dir}/rounds/${CR}/sweeps/opt/${OP_ID}/e2e_latency_results.json" "{artifact_dir}/rounds/${CR}/sweeps/integration/e2e_latency_results.json"
   ```

2. **Determine integration status** — preserve GATED_PASS semantics:

   ```bash
   if [ "$TRACK_VERDICT" = "GATED_PASS" ]; then
       INTEG_STATUS="gated_pass"
   else
       INTEG_STATUS="single_pass"
   fi
   ```

3. **Write state.json integration fields**:

   ```bash
   IDX=$(( $(jq -r '.campaign.current_round' state.json) - 1 ))
   NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)
   jq --argjson idx "$IDX" --arg op "$OP_ID" --arg verdict "$TRACK_VERDICT" \
      --arg status "$INTEG_STATUS" --arg now "$NOW" '
     .campaign.rounds[$idx].integration.status = $status |
     .campaign.rounds[$idx].integration.started_at = $now |
     .campaign.rounds[$idx].integration.passing_candidates = [{"op_id": $op, "verdict": $verdict}] |
     .campaign.rounds[$idx].integration.selected_candidates = [$op] |
     .campaign.rounds[$idx].integration.combined_patch_branch = null
   ' state.json > state.json.tmp && mv state.json.tmp state.json
   ```

4. **Write `e2e_latency_combined` and `per_bs_verdict`** from the copied results:

   ```bash
   E2E_MAP=$(jq -c '
     [.results[] | {
       key: (.batch_size | tostring),
       value: {
         avg: (.opt.aggregate.mean_latency // .opt.avg_s),
         p50: (.opt.aggregate.p50 // .opt.avg_s)
       } | with_entries(select(.value != null))
     }] | from_entries
   ' "{artifact_dir}/rounds/${CR}/sweeps/integration/e2e_latency_results.json")

   # Compute per_bs_verdict by comparing integration vs baseline
   PER_BS=$(jq -c --argjson idx "$IDX" --argjson lat "$E2E_MAP" '
     ($lat | to_entries) | map({
       key: .key,
       value: (if .value.avg <= (.campaign.rounds[$idx].baseline.e2e_latency[.key].avg // 999)
               then "PASS" else "REGRESSED" end)
     }) | from_entries
   ' state.json)

   jq --argjson idx "$IDX" --argjson lat "$E2E_MAP" --argjson pbv "$PER_BS" '
     .campaign.rounds[$idx].integration.e2e_latency_combined = $lat |
     .campaign.rounds[$idx].integration.per_bs_verdict = $pbv
   ' state.json > state.json.tmp && mv state.json.tmp state.json
   ```

5. **Run Pre-SHIP mechanical checks** (unchanged — they read from `sweeps/integration/` which now exists).

6. **Proceed to SHIP path** (cherry-pick the single track's branch into session mainline, env promotion, golden capture, T_AUDIT_S67). After the merge, write `commit_sha` and `completed_at`:

   ```bash
   jq --argjson idx "$IDX" --arg sha "$(git rev-parse HEAD)" --arg now "$(date -u +%Y-%m-%dT%H:%M:%SZ)" '
     .campaign.rounds[$idx].integration.commit_sha = $sha |
     .campaign.rounds[$idx].integration.completed_at = $now
   ' state.json > state.json.tmp && mv state.json.tmp state.json
   ```

The short-circuit saves 5–15 minutes of redundant E2E sweeping and eliminates the python-environment mismatch risk (Stage 5 sweeps always use the per-op worktree's `.venv/bin/python`).

---

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

# Run combined E2E benchmark (--fresh-cache for clean compile, gate-quality measurement)
# v2: writes to {artifact_dir}/rounds/{CR}/sweeps/integration/
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir {artifact_dir} --round {CR} --slot integration --fresh-cache

# Run combined correctness check (mandatory for multi-candidate integration)
# Same slot — overwrites e2e_latency_results.json with verified outputs
.venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
    --artifact-dir {artifact_dir} --round {CR} --slot integration --labels opt \
    --baseline-from {stage1_dir} --verify-correctness

# If correctness fails: bisect — drop track with worst individual failure rate, re-run
```

### Combined Result Evaluation

| Condition | Decision |
|-----------|----------|
| Combined E2E >= max(individual E2E results) | Ship the combined integration branch |
| Combined E2E < max(individual E2E results) | Ship the single track with the best individual E2E |
| Combined correctness fails | Fall back to shipping tracks individually (best E2E first) |
| Combined correctness fails (accuracy gate) | Bisect: drop worst track, re-validate smaller combination |

If a cherry-pick produces a merge conflict, treat the candidates as overlapping (same-component) and pick the one with the best E2E speedup.

### GATED_PASS Track Evaluation

When combining a GATED_PASS track with other tracks:
- Re-run E2E at ALL batch sizes including the gated track's non-beneficial range
- Verify no interaction effects between the gated dispatch and other optimizations
- If cherry-pick produces merge conflict on a GATED_PASS track: spawn resolver agent (see below)

## Per-Track Environment Variable Isolation

**Convention: all AMMO-introduced `VLLM_*` flags default to off (`0`/`False`).**

Each track's gating env var (e.g., `VLLM_MOE_TWO_STREAM`, `VLLM_MOE_TRITON_ROUTER` — mechanism-derived, never `op_id`-derived; see `skills/ammo/references/impl-track-rules.md` § Env Flag Naming (PR-Ready)) is registered in `vllm/envs.py` with default value `0` (disabled). The E2E sweep harness (`run_vllm_bench_latency_sweep.py`) sanitizes the inherited environment by stripping all `VLLM_*` vars NOT named in `baseline_env` or `opt_env` before building run specs, then re-adds the current run's vars from `target.json`. This prevents cross-track contamination where a prior round's gating flag silently activates stale optimizations during subsequent sweeps.

**Preserve-keys contract**: `_sanitize_vllm_op_env(env, preserve_keys=...)` drops any `^VLLM_` key NOT in `preserve_keys`. Callers pass `set(baseline_env.keys())` for baseline runs and `set(baseline_env.keys()) | set(opt_env.keys())` for opt runs, so promoted flags (Track A6) and the current round's experimental flag both survive while stale shell-inherited flags are removed.

**Why this matters**: In a multi-round campaign, worktree branches accumulate `envs.py` edits from all prior tracks, and shipped flags get promoted into `baseline_env` (Track A6). Without sanitization, a Round 1 `VLLM_OP001=True` shell export persists into Round 2 sweeps, contaminating both baseline and opt measurements. This was observed in production: op003's Triton dispatch was silently blocked by op001's CUTLASS flag, and op004's first sweep showed a 0.83x false-negative regression.

**Requirements**:
1. `ammo-impl-champion.md` mandates a default-off (`=0`, never `=1`) flag in `vllm/envs.py`, named for the mechanism per `skills/ammo/references/impl-track-rules.md` § Env Flag Naming (PR-Ready) — never the `op_id`
2. `run_vllm_bench_latency_sweep.py` calls `_sanitize_vllm_op_env(env, preserve_keys=...)` with the appropriate preserve set per run
3. Integration E2E sweeps must explicitly set all shipped gating flags via `baseline_env` (promoted on SHIP, per Track A6) — never rely on envs.py defaults

## State Tracking

The integration section lives under `campaign.rounds[$IDX].integration` in `state.json` (where `$IDX = campaign.current_round - 1`). All decisions and results for the round are recorded there:

```json
{
  "campaign": {
    "current_round": 1,
    "rounds": [
      {
        "round_id": 1,
        "baseline": {
          "e2e_latency": {"128": {"avg": 7.66, "p50": 7.55, "p90": 8.0}},
          "per_bs_verdict": null
        },
        "integration": {
          "started_at": "2026-04-23T12:00:00Z",
          "completed_at": null,
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
            }
          ],
          "selected_candidates": [],
          "conflict_analysis": {
            "method": "file_set_overlap",
            "overlapping_pairs": [],
            "combinable_pairs": [["op001", "op002"]]
          },
          "combined_patch_branch": "ammo/integration",
          "e2e_latency_combined": {
            "128": {"avg": 6.5, "p50": 6.4, "p90": 7.0}
          },
          "per_bs_verdict": {"128": "PASS"},
          "commit_sha": "abc123def456",
          "final_decision": {
            "action": "ship_combined",
            "branch": "ammo/integration",
            "included_candidates": ["op001", "op002"],
            "total_e2e_speedup": 1.18
          },
          "resolver_invoked": null,
          "resolver_outcome": null,
          "conflicting_tracks": null
        }
      }
    ]
  }
}
```

### Field Ownership

| Field | Written by | Stage |
|---|---|---|
| `baseline.e2e_latency` | `ammo-researcher` | Stage 1 (baseline sweep) |
| `baseline.per_bs_verdict` | `ammo-researcher` | Stage 1 (null initially; populated by track eval) |
| `integration.e2e_latency_combined` | `ammo-orchestrator` | Stage 6 (integration sweep) |
| `integration.per_bs_verdict` | `ammo-orchestrator` | Stage 6 (per-BS verdict after integration) |
| `integration.commit_sha` | `ammo-orchestrator` | Stage 6 (post-merge mainline HEAD) |

### Writing `e2e_latency_combined` at Stage 6

After the combined integration E2E sweep, write the results as a map keyed by batch size — the same shape as `baseline.e2e_latency`:

```bash
IDX=$(( $(jq -r '.campaign.current_round' state.json) - 1 ))
CR=$(jq -r '.campaign.current_round' state.json)
# Build e2e_latency_combined map from integration sweep results (v2 path)
E2E_MAP=$(jq -c '
  [.results[] | {
    key: (.batch_size | tostring),
    value: {
      avg: (.opt.aggregate.mean_latency // .opt.avg_s),
      p50: (.opt.aggregate.p50 // .opt.avg_s)
    } | with_entries(select(.value != null))
  }] | from_entries
' "{artifact_dir}/rounds/${CR}/sweeps/integration/e2e_latency_results.json")
jq --argjson idx "$IDX" --argjson lat "$E2E_MAP" --arg sha "$(git rev-parse HEAD)" \
  '.campaign.rounds[$idx].integration.e2e_latency_combined = $lat |
   .campaign.rounds[$idx].integration.commit_sha = $sha' \
  state.json > state.json.tmp && mv state.json.tmp state.json
```

The `per_bs_verdict` is computed by comparing each BS key's `avg` in `e2e_latency_combined` against `baseline.e2e_latency` using the gating thresholds from `target.json`.

### Status Values

| Status | Meaning |
|--------|---------|
| `pending` | Integration has not started yet |
| `validated` | Single candidate validated successfully |
| `single_pass` | One candidate selected (sole passer or best among overlapping) |
| `combined` | Multiple candidates merged and validated successfully |
| `gated_pass` | One or more GATED_PASS candidates integrated with dispatch gating |
| `exhausted` | No candidates passed validation; no optimization to ship |

## Pre-SHIP Mechanical Checks (Inline — No Auditor)

Before executing the SHIP merge, the orchestrator runs three fast mechanical checks inline. These are NOT auditor items — they are cheap, deterministic, and must block the merge if they fail:

```bash
# 1. Merge-conflict residue (grep integrated code for conflict markers)
git diff main...ammo/integration -- '*.py' '*.cu' | grep -qE '^[+].*(<<<<<<|======|>>>>>>)' && echo "BLOCKED: merge-conflict residue" && exit 1

# 2. Dual-verdict override check (no regression/catastrophic verdict in integration results)
INTEG_RESULTS="{artifact_dir}/rounds/${CR}/sweeps/integration/e2e_latency_results.json"
jq -e '
  [.results[] | select(.per_bs_verdict // {} | to_entries[] | .value == "REGRESSED" or .value == "CATASTROPHIC")] | length == 0
' "$INTEG_RESULTS" || echo "BLOCKED: regression detected in integration sweep"

# 3. Opt leg returncode == 0
jq -e '.results[] | select(.label == "opt") | .returncode == 0' "$INTEG_RESULTS" || echo "BLOCKED: opt leg crashed"
```

Only if ALL three pass does the orchestrator proceed to the SHIP merge below.

## SHIP Path: Baseline Promotion

When the round's terminal status is `SHIPPED` (single PASS, combined PASS, GATED_PASS, or any variant that lands code on session mainline), the orchestrator executes these steps in order:

1. **Cherry-pick / merge op-branch(es) into session mainline** per the Decision Matrix + Combined Validation Workflow above.

2. **Promote env flags to baseline** — the shipped track's experimental flag becomes the new baseline for the next round:

   ```bash
   # Merge shipped opt_env into baseline_env, then clear opt_env
   jq '.bench.baseline_env += .bench.opt_env | .bench.opt_env = {}' target.json | sponge target.json
   ```

   Keys that used to be in `opt_env` now live in `baseline_env`; future champions populate `opt_env` from scratch for their new proposals.

3. **Capture golden-refs for next round** (~15s lightweight pass after env promotion). Use the dedicated `golden_capture` slot of the CURRENT round so the existing baseline sweep is preserved:

   ```bash
   .venv/bin/python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
       --artifact-dir {artifact_dir} --round {CR} --slot golden_capture \
       --labels baseline --capture-golden-refs --num-iters 1
   ```

   This runs against the newly-promoted baseline (opt_env is now empty, baseline_env contains all shipped flags). The golden-refs are used by the next round's Gate 5.1b correctness comparison.

4. **Spawn T_AUDIT_S67** — the consolidated Stage 6+7 adversarial audit that verifies integration invariants AND post-SHIP state (env promotion, shipped_optimizations, new baseline vs R1). On PASS, write `audit.stage_67.passed_at` and proceed to `7_campaign_eval`.

Cumulative E2E speedup is computed at read time by the backend normalizer from `rounds[0].baseline.e2e_latency` vs the latest `integration.e2e_latency_combined`. Agents do not write this value.

**Note**: The former T16 re-profile step is eliminated. The Stage 6 integration sweep (with `--fresh-cache`) already produces a clean measurement under the promoted env. The `integration.e2e_latency_combined` value serves as the next round's effective baseline for `cumulative_speedup_vs_round1` computation.

## Round Transition

Per-round state always lives in place under `campaign.rounds[N-1]` (1-based `round_id`, 0-based array index). When Stage 6 completes for round N:

1. Set `campaign.rounds[N-1].integration.completed_at` and the terminal `integration.status`.
2. Populate round-level summary fields on the same entry: `shipped`, `dropped`, `combined_e2e_speedup_x`, `combined_e2e_delta_pp`, `cumulative_speedup_after`, `round_summary`, and terminal round `status` (`"completed"` | `"SHIPPED"` | `"EXHAUSTED"` | `"FAILED"`).
3. Append a fresh `rounds[N]` entry matching the bootstrap shape from the schema (all stage sub-objects with null timestamps, `status: "IN_PROGRESS"`, `round_id: N + 1`).
4. Increment `campaign.current_round = N + 1`.

Round N's track outcomes, integration verdicts, and failed candidates remain in `campaign.rounds[N-1]` — consumers read them directly from the same location the orchestrator wrote them.

## Campaign Loop Transition (Stage 7)

See SKILL.md § Campaign Loop (T15b-T15c) for the mining workflow and campaign stop condition. The former T16 re-profile step is eliminated — Stage 6 integration sweep with `--fresh-cache` already provides the post-SHIP gate-quality measurement.

### Integration-Specific Addenda

**GATED_PASS speedup accounting**: For GATED_PASS tracks, use the **minimum post-gating speedup across all batch sizes** as the `e2e_speedup` value (conservative — avoids needing production BS distribution data).

**Lazy invalidation with GATED_PASS**: When re-profiling after a GATED_PASS track ships, profile at ALL campaign batch sizes (not just one). The gated optimization's f-shift is BS-dependent — f changes only at gated-on batch sizes, not at gated-off batch sizes. Use the **maximum f-shift across all BS** for the lazy invalidation test to be conservative.

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

Record resolver invocation on the current round's integration entry (`campaign.rounds[$IDX].integration`):
```json
{
  "resolver_invoked": true,
  "resolver_outcome": "approved" | "rejected" | "escalated",
  "conflicting_tracks": ["op001", "op003"]
}
```
