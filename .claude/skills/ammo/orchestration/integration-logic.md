# Stage 6: Integration Validation

After all parallel tracks complete in Stage 5, the main session determines how to combine and ship the results. This stage handles conflict detection, combined validation, and final decision-making.

## Decision Matrix

| Scenario | Action |
|----------|--------|
| Single candidate passes | Ship directly -- E2E already validated in Stage 5 |
| Multiple pass, different components | Cherry-pick both onto integration branch, re-run correctness + E2E |
| Multiple pass, same component | Pick the candidate with the best E2E speedup, ship that one |
| None pass | Mark optimization target as `EXHAUSTED` in state.json |

## Conflict Detection

### Step 1: Identify Changed Files Per Track

For each passing track, compute the file diff against main:

```bash
git diff --name-only main...ammo/{op_id}
```

### Step 2: Classify Overlap

Compare the file sets between all pairs of passing tracks:

- **Disjoint file sets** (no overlap): candidates modify different components and are combinable.
- **Overlapping file sets**: candidates modify the same component. Pick the one with the best E2E speedup.

Example with two passing tracks:

```bash
# Get changed files for each track
FILES_OP001=$(git diff --name-only main...ammo/op001)
FILES_OP002=$(git diff --name-only main...ammo/op002)

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
git cherry-pick ammo/{op_id_1}
git cherry-pick ammo/{op_id_2}

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

## State Tracking

The integration section of `state.json` records all decisions and results:

```json
{
  "integration": {
    "status": "pending | validated | single_pass | combined | exhausted",
    "passing_candidates": [
      {
        "op_id": "op001",
        "e2e_speedup": 1.12,
        "files_changed": ["vllm/attention/backends/flash_attn.py"]
      },
      {
        "op_id": "op002",
        "e2e_speedup": 1.08,
        "files_changed": ["csrc/quantization/gptq_marlin.cu"]
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
| `exhausted` | No candidates passed validation; no optimization to ship |
