# Stages 4-5: Parallel Worktree Track Management

Each winning candidate from Stage 3 gets its own git worktree, branch, and subagent pipeline. Tracks run in parallel across GPUs; steps within a track run sequentially.

## Worktree Creation

Worktrees are created automatically by the Agent tool when spawning `ammo-implementer` subagents (which have `isolation: worktree` in their definition). The `WorktreeCreate` hook (`worktree-create-with-build.sh`) pre-configures Python isolation, copies `.so` files, and creates a per-worktree `.venv`.

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

## Worktree Build Rules (CRITICAL — Read Before Running Any Commands)

The worktree-create hook pre-configures Python isolation. Agents MUST follow these rules:

| Change Type | Required Action | Time |
|-------------|----------------|------|
| **Pure Python** (model code, Triton kernels, configs) | Edit, test, commit. **NO rebuild.** | Immediate |
| **C++ kernel** (csrc/ changes) | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s (ccache) |

**Why no rebuild for Python**: The hook copies all `.so` files from the main repo and configures `.pth` files so worktree Python code takes priority over the main repo. Triton kernels JIT-compile at runtime.

**AVOID** — these are redundant and waste 10-15 minutes:
- `pip install -e .` — triggers full C++ rebuild unnecessarily
- `pip install -e . --no-build-isolation` — still rebuilds C++
- `python setup.py build_ext --inplace` — unnecessary if no C++ changes

## Per-Track Execution Pipeline

Each track follows these four steps **sequentially**. All tracks run **in parallel** with each other.

### Step 1: Implementation + Validation (ammo-implementer Subagent)

Main spawns an implementer subagent that writes the optimization code, runs validation (correctness tests, kernel benchmarks, E2E benchmarks), and writes `validation_results.md`. The implementer works in an isolated worktree (auto-created via `isolation: worktree`).

A frontmatter Stop hook (DA) on the implementer verifies `validation_results.md` is complete, runs an Amdahl's Law sanity check, verifies baseline citation and production parity, and checks for cross-track contamination risk before allowing it to stop. If any check fails, the hook blocks the implementer and tells it what to fix.

```
Agent(
    subagent_type="ammo-implementer",
    prompt="""
    You are implementing and validating optimization {op_id} for the AMMO pipeline.

    Artifact dir: {artifact_dir}
    Optimization plan: {artifact_dir}/debate/summary.md (section for {op_id})
    Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
    GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}

    ## Stage 1 Baseline (DO NOT RE-RUN)
    Baseline E2E latency files (captured from clean main in Stage 1):
    - Per-batch-size JSON: {artifact_dir}/runs/baseline_bs{N}.json
    - Summary table: {artifact_dir}/constraints.md ("Baseline E2E latency" section)
    - Kernel breakdown: {artifact_dir}/constraints.md ("Baseline Truth Snapshot" section)
    Use these for ALL E2E comparisons. Do NOT run a baseline from the worktree.

    ## Kill Criteria
    {kill_criteria_from_optimization_plan}

    Instructions:
    Phase 1 — Implementation:
    1. Read the optimization plan and bottleneck analysis for {op_id}.
    2. Implement the optimization.
    3. If C++ changes (csrc/): run cmake --preset release && cmake --build --preset release --target install
    4. Commit implementation.

    Phase 2 — Validation:
    5. Run correctness tests (Gate 5.1): torch.allclose() against vLLM production kernel.
    6. Run kernel benchmarks (Gate 5.2): both baseline and optimized captured in CUDA graphs.
    7. Run E2E benchmark (Gate 5.3): ONLY the optimized run. Compare against Stage 1 baseline.
       Use the sweep script:
       python .claude/skills/ammo/scripts/run_vllm_bench_latency_sweep.py \
         --artifact-dir {artifact_dir} --labels opt
       FORBIDDEN: Do NOT use raw `vllm bench latency` commands. The sweep script is mandatory for all E2E measurements.
    8. Evaluate all kill criteria with definitive PASS/FAIL verdicts.
    9. Write results to {artifact_dir}/tracks/{op_id}/validation_results.md.
    10. Commit validation results.
    11. Return: overall PASS/FAIL, key metrics, worktree path.
    """
)
```

The implementer subagent returns upon completion. The frontmatter Stop hook (DA) has already verified validation completeness, Amdahl's Law sanity, baseline citation, production parity, and cross-track contamination awareness before allowing the implementer to stop. Main resumes and records the worktree path from the agent result.

### Step 2: Compilation Gate (Main Session)

Main verifies the implementation compiles from the implementer's worktree:

```bash
cd {worktree_path}
source .venv/bin/activate
python -c "import vllm; print('compilation OK')"
```

If compilation fails, the track is marked `FAILED` in `state.json` and no further steps run.

### Step 3: State Update (Main Session)

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

**Note**: The DA audit is embedded in the implementer's frontmatter Stop hook — if the implementer returned successfully, the DA already passed (Amdahl's check, baseline citation, production parity, cross-track awareness). No separate DA artifact is produced.

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
- Implementer's frontmatter DA Stop hook passed (Amdahl's check, baseline citation, production parity)

## Worktree Cleanup

After Stage 6 integration is complete (or a track is abandoned), clean up:

```bash
git worktree remove {worktree_path} --force
```

Run cleanup for all tracks, including failed ones. The integration branch (if created in Stage 6) is retained until the final patch is shipped.

## In-Flight Tracks During Campaign Re-profiling

When a candidate ships and triggers re-profiling (see Campaign Loop section in `SKILL.md`), other tracks from the same round may still be running. These tracks are NOT terminated:

1. Let all in-flight implementations complete against the ORIGINAL round's baseline.
2. Validate their results using Stage 1 baseline from the current round (not the re-profiled baseline).
3. If they pass: they also ship as additional cumulative gain — update `campaign.cumulative_e2e_speedup` multiplicatively.
4. Record all track results in the current round's entry in `campaign.rounds`.
5. The next campaign round starts only after all current-round tracks have completed.

This ensures no work is wasted — an implementer that started before the re-profile can still contribute a valid optimization, even if the bottleneck landscape has shifted.
