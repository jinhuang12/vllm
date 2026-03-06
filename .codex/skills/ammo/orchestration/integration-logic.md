# Stage 6: Integration Validation

After all tracks finish Stage 5, the lead decides how to combine and ship results.

## Decision Matrix

| Scenario | Action |
|---|---|
| One candidate passes | ship that branch directly |
| Multiple candidates pass with disjoint file sets | cherry-pick all onto an integration branch and re-run correctness plus E2E |
| Multiple candidates pass with overlapping file sets | ship the best individual candidate |
| No candidates pass | mark the target `EXHAUSTED` |

## Conflict Detection

### Step 1: Gather changed files per passing track

```bash
git diff --name-only main...ammo/{op_id}
```

### Step 2: Classify overlap

- disjoint file sets: combinable
- overlapping file sets: choose the best single candidate

If cherry-pick conflicts occur, treat the candidates as overlapping and ship the best single candidate.

## Combined Validation Workflow

When candidates are combinable:

```bash
git checkout -b ammo/integration main
git cherry-pick ammo/{op_id_1}
git cherry-pick ammo/{op_id_2}
pytest tests/path/to/component_1_tests.py
pytest tests/path/to/component_2_tests.py
python .codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py --artifact-dir {artifact_dir}
```

Use the same Stage 1 baseline source of truth for combined E2E reporting.

## Combined Result Evaluation

| Condition | Decision |
|---|---|
| Combined E2E >= best individual E2E | ship combined branch |
| Combined E2E < best individual E2E | ship best single candidate |
| Combined correctness fails | fall back to best single candidate |

## State Tracking

Record integration decisions in `state.json.integration`:

```json
{
  "integration": {
    "status": "pending | validated | single_pass | combined | exhausted",
    "passing_candidates": [
      {"op_id": "op001", "e2e_speedup": 1.12, "files_changed": ["..."]},
      {"op_id": "op002", "e2e_speedup": 1.08, "files_changed": ["..."]}
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
      "action": "ship_combined | ship_single | exhausted",
      "branch": "ammo/integration",
      "included_candidates": ["op001", "op002"],
      "total_e2e_speedup": 1.18,
      "reason": "Evidence-based explanation"
    }
  }
}
```
