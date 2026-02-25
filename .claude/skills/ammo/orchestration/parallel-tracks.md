# Stages 4-5: Parallel Worktree Track Management

Each winning candidate from Stage 3 gets its own git worktree, branch, and subagent pipeline. Tracks run in parallel across GPUs; steps within a track run sequentially.

## Worktree Creation

The **main session** creates one worktree per winning candidate:

```bash
git worktree add .claude/worktrees/ammo-track-{op_id} -b ammo/{op_id}
```

Example for three winners:

```bash
git worktree add .claude/worktrees/ammo-track-op001 -b ammo/op001
git worktree add .claude/worktrees/ammo-track-op002 -b ammo/op002
git worktree add .claude/worktrees/ammo-track-op003 -b ammo/op003
```

## GPU Assignment

| Track | `CUDA_VISIBLE_DEVICES` | Usage |
|-------|----------------------|-------|
| Track 0 (op_id 1) | `0` | Kernel benchmarks, micro-validation |
| Track 1 (op_id 2) | `1` | Kernel benchmarks, micro-validation |
| Track 2 (op_id 3) | `2` | Kernel benchmarks, micro-validation |
| E2E benchmarks | All GPUs | Sequential via `flock` on `/tmp/ammo_gpu_locks/` |

E2E benchmarks require exclusive GPU access. The existing flock mechanism serializes them:

```bash
flock /tmp/ammo_gpu_locks/gpu_all.lock -c "run_e2e_benchmark.sh"
```

## Per-Track Execution Pipeline

Each track follows these four steps **sequentially**. All tracks run **in parallel** with each other.

### Step 1: Implementation (Subagent)

Main spawns an implementer subagent that writes the optimization code, then commits to the track branch.

```
Task(
    subagent_type="ammo-implementer",
    prompt="""
    You are implementing optimization {op_id} for the AMMO pipeline.

    Worktree path: .claude/worktrees/ammo-track-{op_id}
    Artifact dir: {artifact_dir}
    Optimization plan: {artifact_dir}/debate/summary.md (section for {op_id})
    Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md

    Instructions:
    1. Read the optimization plan and bottleneck analysis for {op_id}.
    2. Implement the optimization in the worktree.
    3. Ensure all changes compile cleanly.
    4. Commit all changes to the ammo/{op_id} branch with a descriptive message.
    5. Return a summary of files changed and the approach taken.
    """,
    cwd=".claude/worktrees/ammo-track-{op_id}"
)
```

The implementer subagent returns upon completion. Main resumes.

### Step 2: Compilation Gate (Main Session)

Main verifies the implementation compiles:

```bash
cd .claude/worktrees/ammo-track-{op_id}
python -c "import vllm; print('compilation OK')"
```

If compilation fails, the track is marked `FAILED` in `state.json` and no further steps run.

### Step 3: Validation (Subagent)

Main spawns a validator subagent that runs correctness tests, kernel benchmarks, and E2E benchmarks.

```
Task(
    subagent_type="ammo-validator",
    prompt="""
    You are validating optimization {op_id} for the AMMO pipeline.

    Worktree path: .claude/worktrees/ammo-track-{op_id}
    Artifact dir: {artifact_dir}
    GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}
    Optimization plan: {artifact_dir}/debate/summary.md (section for {op_id})

    Instructions:
    1. Run correctness tests for the modified component.
    2. Run kernel-level benchmarks (use only GPU {gpu_id}).
    3. Acquire the E2E GPU lock, then run E2E benchmarks.
    4. Compare results against the baseline in {artifact_dir}/baseline_profile.json.
    5. Write results to {artifact_dir}/tracks/{op_id}/validation_results.md.
    6. Return pass/fail status and key metrics.
    """,
    cwd=".claude/worktrees/ammo-track-{op_id}",
    env={"CUDA_VISIBLE_DEVICES": "{gpu_id}"}
)
```

The validator subagent returns upon completion. Main resumes.

### Step 4: State Update (Main Session)

Main reads the validation results and updates `state.json`:

```bash
# Read results
cat {artifact_dir}/tracks/{op_id}/validation_results.md
```

Update `state.json` field `parallel_tracks.{op_id}.result` with:

```json
{
  "status": "PASSED | FAILED | REGRESSED",
  "correctness": true,
  "kernel_speedup": 1.35,
  "e2e_speedup": 1.12,
  "validation_results_path": "{artifact_dir}/tracks/{op_id}/validation_results.md"
}
```

## Result Collection

After all tracks complete, main reads each track's outputs:

1. `{artifact_dir}/tracks/{op_id}/validation_results.md` -- detailed results
2. `state.json` field `parallel_tracks.{op_id}.result` -- structured summary

Main aggregates results to determine which candidates pass to Stage 6 integration.

### Pass Criteria

A track **passes** if all of the following hold:

- Correctness tests pass (no regressions)
- Kernel benchmark shows measurable speedup (>1% over baseline)
- E2E benchmark shows non-negative impact (>=1.0x)

## Worktree Cleanup

After Stage 6 integration is complete (or a track is abandoned), clean up:

```bash
git worktree remove .claude/worktrees/ammo-track-{op_id} --force
git branch -d ammo/{op_id}
```

Run cleanup for all tracks, including failed ones. The integration branch (if created in Stage 6) is retained until the final patch is shipped.
