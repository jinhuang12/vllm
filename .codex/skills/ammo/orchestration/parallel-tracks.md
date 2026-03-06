# Stages 4-5: Parallel Worktree Track Management

Each winner from Stage 3 gets an isolated git worktree, branch, and single-agent implementation plus validation track.

## Worktree Creation

Codex has no Claude-style worktree lifecycle hooks. The lead creates worktrees explicitly:

```bash
bash .codex/skills/ammo/scripts/create_worktree_with_build.sh ammo-track-{op_id} ammo/{op_id}
```

The helper script creates `.codex/worktrees/ammo-track-{op_id}`, copies runtime build artifacts, and builds a thin worktree `.venv` that points at the main repo environment.

## GPU Assignment

| Track | `CUDA_VISIBLE_DEVICES` | Usage |
|---|---|---|
| Track 0 | `0` | kernel benchmarks and micro-validation |
| Track 1 | `1` | kernel benchmarks and micro-validation |
| Track 2 | `2` | kernel benchmarks and micro-validation |
| E2E benchmarks | all required GPUs | serialized via lock script |

All E2E runs must use `.codex/skills/ammo/scripts/run_vllm_bench_latency_sweep.py` or an equivalent lock-based wrapper.

## Worktree Build Rules

| Change Type | Required Action | Time |
|---|---|---|
| Pure Python, Triton, configs | edit, test, commit; no rebuild | Immediate |
| `csrc/` changes | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s |

Do not run `pip install -e .`, `python setup.py build_ext --inplace`, or any other full rebuild path unless the plan explicitly requires it.

## Per-Track Execution Pipeline

### Step 1: Lead creates the worktree

- Record `worktree_path` and `branch` in `state.json.parallel_tracks.{op_id}`.
- Pass the worktree path, artifact path, and assigned GPU to the implementer.

### Step 2: Implementer owns implementation plus validation

Spawn `ammo-implementer` in the assigned worktree. Required outputs:

- code changes
- correctness tests
- kernel benchmark results
- E2E benchmark results based on Stage 1 baseline reuse
- `{artifact_dir}/tracks/{op_id}/validation_results.md`
- committed branch on `ammo/{op_id}`

### Step 3: Compilation gate (lead)

```bash
cd .codex/worktrees/ammo-track-{op_id}
source .venv/bin/activate
python -c "import vllm; print('compilation OK')"
```

If this fails, mark the track failed and do not accept validation results.

### Step 4: Track state update (lead)

Update `state.json.parallel_tracks.{op_id}` with a terminal record. Recommended fields:

```json
{
  "status": "PASSED | FAILED | REGRESSED",
  "branch": "ammo/op001",
  "worktree_path": ".codex/worktrees/ammo-track-op001",
  "correctness": true,
  "kernel_speedup": 1.35,
  "e2e_speedup": 1.08,
  "validation_results_path": "{artifact_dir}/tracks/op001/validation_results.md",
  "files_changed": ["..."]
}
```

## Pass Criteria

A track passes only if all of these are true:

- correctness passes
- kernel speedup is positive on target buckets
- E2E meets the kill criteria or a narrower validated enablement envelope
- validation cites the Stage 1 baseline and passes the contamination and Amdahl sanity checks

Before Stage 6, the lead must run the global validation gate:

```bash
python .codex/skills/ammo/scripts/verify_validation_gates.py \
  {artifact_dir} \
  --json-output {artifact_dir}/runs/validation_gate_report.json
```

`WARN` is blocking. Do not advance without `PASS`.

## Cleanup

After Stage 6, remove worktrees explicitly:

```bash
bash .codex/skills/ammo/scripts/remove_worktree_cleanup.sh .codex/worktrees/ammo-track-{op_id}
```

Retain the integration branch until the final ship decision is complete.
