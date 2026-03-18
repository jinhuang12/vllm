# Stages 4-5: Parallel Worktree Track Management

Each winning candidate from Stage 3 gets its own git worktree, branch, and implementation agent pair. All agents -- across all tracks -- belong to the **same round team** created at the start of Stage 3. The orchestrator can only lead one team at a time, so a single round-scoped team is used for the entire round lifecycle (debate through implementation). Tracks run in parallel across GPUs. Within a track, the champion and validator collaborate fluidly -- both may be active simultaneously, with GPU access coordinated via SendMessage. During round 2+, debate champions for the next round may also be present in the team (see Overlapped Debate below). Implementation and debate agents share the team but operate as independent workstreams -- they do not communicate directly.

## Team Structure (Single Round Team)

All implementation agents join the existing round team. The team was created at Stage 3 start (`ammo-round-{round_id}-{model_short}-{hardware}`) and persists through Stages 4-5.

```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
[Implementation Workstream]
+-- impl-champion-{op_id_1} (Opus)    -- implementation, kill-criteria evaluation
+-- impl-validator-{op_id_1} (Sonnet)  -- independent correctness tests, benchmarks, E2E sweep
+-- impl-champion-{op_id_2} (Opus)    -- implementation, kill-criteria evaluation
+-- impl-validator-{op_id_2} (Sonnet)  -- independent correctness tests, benchmarks, E2E sweep
[Overlapped Debate Workstream -- round 2+ only]
+-- champion-r{N+1}-1 (Opus)          -- next-round debate champion [shut down after selection]
+-- champion-r{N+1}-2 (Opus)          -- next-round debate champion [shut down after selection]
+-- delegate-r{N+1}-1a (Sonnet)       -- debate delegate [shut down after selection]
```

Each track uses an **adversarial validation model**: the champion implements, an independent validator validates. This separation prevents reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation).

**Why adversarial validation**: The AMMO pipeline has observed four types of reward hacking when the same agent both implements and validates. Separating implementation from validation into independent agents with differing goals catches issues in real-time. The existing DA Stop hook remains as a third verification layer.

**Why a single team**: The orchestrator can only lead ONE team at a time. Creating per-track teams (`ammo-impl-{op_id}`) is architecturally impossible — the orchestrator would lose contact with all but the last-created team. A single round-scoped team keeps all agents under one roof.

## Worktree Creation

Worktrees are created automatically by the Agent tool when spawning `ammo-impl-champion` subagents (which have `isolation: worktree` in their definition). The `WorktreeCreate` hook (`worktree-create-with-build.sh`) pre-configures Python isolation, copies `.so` files, and creates a per-worktree `.venv`.

The validator agent shares the champion's worktree (same track team). Both agents may be active simultaneously — GPU access is coordinated via SendMessage, with the champion having priority.

## GPU Assignment

| Track | `CUDA_VISIBLE_DEVICES` | Usage |
|-------|----------------------|-------|
| Track 0 (op_id 1) | `0` | Kernel benchmarks, micro-validation |
| Track 1 (op_id 2) | `1` | Kernel benchmarks, micro-validation |
| Track 2 (op_id 3) | `2` | Kernel benchmarks, micro-validation |
| E2E benchmarks | All GPUs | Sequential via `flock` on `/tmp/ammo_gpu_locks/` |

Within a track, GPU access is coordinated via SendMessage — the champion has priority, and the validator yields when the champion needs the GPU for compilation or smoke tests. During validation, the champion is idle and the validator has exclusive GPU access.

## Worktree Build Rules (CRITICAL)

| Change Type | Required Action | Time |
|-------------|----------------|------|
| **Pure Python** (model code, Triton kernels, configs) | Edit, test, commit. **NO rebuild.** | Immediate |
| **C++ kernel** (csrc/ changes) | `cmake --preset release && cmake --build --preset release --target install` | ~5-55s (ccache) |

**Only the champion compiles.** The validator never runs cmake. The validator only executes against committed, compiled code.

## Per-Track Execution Pipeline

### Orchestrator Spawns Implementation Agents

The round team already exists from Stage 3 (`ammo-round-{round_id}-{model_short}-{hardware}`). No new TeamCreate is needed. All implementation agents join the existing team.

```python
# existing_team_name = state.json -> debate.team_name
# e.g., "ammo-round-1-llama70b-h100"

# Per winning candidate — spawn into the EXISTING round team (no TeamCreate):
Agent(
    name=f"impl-champion-{op_id}",
    subagent_type="ammo-impl-champion",
    team_name=existing_team_name,    # Reuse round team, NOT a per-track team
    prompt="""
    You are implementing optimization {op_id} for the AMMO pipeline.
    Your validator is impl-validator-{op_id}.

    Artifact dir: {artifact_dir}
    Optimization plan: {artifact_dir}/debate/summary.md (section for {op_id})
    Bottleneck analysis: {artifact_dir}/bottleneck_analysis.md
    GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}

    ## Stage 1 Baseline (DO NOT RE-RUN)
    Baseline E2E latency files (captured from clean main in Stage 1):
    - Per-batch-size JSON: {artifact_dir}/runs/baseline_bs{N}.json
    - Summary table: {artifact_dir}/constraints.md ("Baseline E2E latency" section)
    - Kernel breakdown: {artifact_dir}/constraints.md ("Baseline Truth Snapshot" section)

    ## Kill Criteria
    {kill_criteria_from_optimization_plan}

    Workflow:
    1. Delegate research to your validator while you read debate artifacts
    2. Implement the kernel optimization (use validator for codebase lookups as needed)
    3. Commit implementation, then delegate validation (all 3 gates) to your validator
    4. Evaluate kill criteria from validator's raw data, write validation_results.md
    """
)

# Validator in the same round team (shared worktree via champion)
Agent(
    name=f"impl-validator-{op_id}",
    subagent_type="ammo-impl-validator",
    model="sonnet",
    team_name=existing_team_name,    # Same round team as champion
    prompt="""
    You are the independent validator for optimization {op_id}.
    Your champion is impl-champion-{op_id}.

    Artifact dir: {artifact_dir}
    GPU assignment: CUDA_VISIBLE_DEVICES={gpu_id}

    Wait for initial tasks from your champion via SendMessage.
    You support the champion throughout: research, profiling, codebase lookups, proactive advisories.
    When the champion requests validation, write your OWN independent tests and benchmarks (Gates 5.1, 5.2, 5.3).

    Key references:
    - Benchmark template: .claude/skills/ammo/references/kernel-benchmark-template.py
    - Validation defaults: .claude/skills/ammo/references/validation-defaults.md
    - CUDA graph safety: .claude/skills/ammo/references/cudagraph-safety.md
    """
)
```

### Collaboration Timeline

The champion and validator work as a team with fluid collaboration. Both may be active simultaneously, except during the validation phase when the champion must remain idle:

```
Early:     Validator researches + profiles (ncu, dispatch, shapes)
           Champion reads debate artifacts, receives research report
           Both active — validator assists champion as needed

Mid:       Champion implements kernel (sole source modifier)
           Validator assists: codebase lookups, proactive advisories, prep work
           Champion has GPU priority for compilation/smoke tests

Late:      Champion commits implementation, requests validation
           Validator runs independent Gates 5.1/5.2/5.3 (writes OWN tests/benchmarks)
           Champion IDLE during validation (validator needs stable code + GPU)

Final:     Champion evaluates validator's raw data, writes validation_results.md
           DA Stop hook fires on champion
```

### Key Rules (Not Phases)

1. **Only the champion modifies source files** (csrc/, vllm/, etc.). The validator reads files and writes to `{artifact_dir}/tracks/{op_id}/validator_prep/`.
2. **Champion has GPU priority.** Validator coordinates via SendMessage before GPU-intensive work (ncu profiling). Champion signals when it needs the GPU.
3. **Independent validation is non-negotiable.** When validating, the validator writes its OWN correctness tests and benchmarks from the optimization plan — not from the champion's scripts. This is the reward hacking prevention mechanism.
4. **During validation, champion is idle.** The validator needs stable committed code and exclusive GPU access for accurate measurements.

### Three Layers of Verification

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN correctness tests, OWN benchmark scripts, runs E2E sweep
  Reports raw structured results — no interpretation

Layer 2: Champion Review (Opus)
  Evaluates kill criteria against validator's raw data
  Cross-checks Gate 5.2 numbers against own smoke-test
  Writes final validation_results.md with evidence chain

Layer 3: DA Stop Hook (Sonnet, frontmatter on champion)
  Fires when champion attempts to stop
  Audits validation_results.md: completeness, Amdahl's consistency,
  production parity, independent validation existence, benchmark cross-check
```

### Handling Validation Failures

When the validator reports a gate failure:

1. Validator reports failure details to champion
2. Champion diagnoses root cause
3. Champion fixes implementation, recompiles if needed, commits
4. Champion re-delegates validation with new commit SHA
5. Validator re-runs ALL gates from scratch with fresh independent tests

The validator writes new tests each re-delegation cycle — the champion cannot "fix" by influencing the test methodology.

## Result Collection

After all tracks complete, main reads each track's outputs:

1. `{artifact_dir}/tracks/{op_id}/validation_results.md` — champion's final report
2. `{artifact_dir}/tracks/{op_id}/validator_tests/` — validator's independent scripts and results
3. `state.json` field `parallel_tracks.{op_id}.result` — structured summary

Main aggregates results to determine which candidates pass to Stage 6 integration.

### Pass Criteria

A track **passes** if all of the following hold:

- Gate 5.1: Validator's independent correctness tests pass (no regressions)
- Gate 5.2: Validator's independent kernel benchmark shows measurable speedup (>1% over baseline)
- Gate 5.3: E2E benchmark shows non-negative impact (>=1.0x)
- Champion's DA Stop hook passed (Amdahl's check, baseline citation, production parity, independent validation exists)
- No unresolved benchmark divergence between champion and validator (if both ran Gate 5.2)

## Team and Worktree Cleanup

After all implementation tracks have completed and results are collected:

1. **TeamDelete** the round team (`ammo-round-{round_id}-{model_short}-{hardware}`). This is the only TeamDelete in the round lifecycle -- it shuts down any remaining implementation agents. TeamDelete is called only after all implementation tracks complete AND the overlapped debate (if launched) has completed.

2. **Remove worktrees** for all tracks (after Stage 6 integration is complete or a track is abandoned):

```bash
git worktree remove {worktree_path} --force
```

Run cleanup for all tracks, including failed ones.

## In-Flight Tracks During Campaign Re-profiling

When a candidate ships and triggers re-profiling, other tracks from the same round may still be running:

1. Let all in-flight implementations complete against the ORIGINAL round's baseline
2. Validate using Stage 1 baseline from the current round (not re-profiled baseline)
3. If they pass: they also ship as additional cumulative gain
4. Record all track results in the current round's `campaign.rounds` entry
5. Next campaign round starts only after all current-round tracks complete

## Overlapped Debate Within the Round Team (Round 2+ Only)

During round 2+, the orchestrator spawns debate champions for the next campaign round into the same round team alongside implementation agents. This overlaps debate with implementation, saving 35-70 minutes per extra round.

### Communication Isolation

Debate champions and implementation agents MUST NOT communicate:
- The orchestrator does NOT provide implementation agent names in debate champion spawn prompts (and vice versa).
- Debate champions write to `debate/campaign_round_{N+1}/` (scoped paths).
- Implementation agents write to `tracks/{op_id}/`.
- The orchestrator is the only agent that communicates with both workstreams.

### Orchestrator Interleaving Pattern

The orchestrator manages both workstreams through the same team inbox:

1. Broadcast debate Phase 0 start to debate champions.
2. Check for any impl track completions (DA Stop hook notifications).
3. If an impl track completed: run T9 gate, update state.json.
4. Wait for debate Phase 0 completions from all champions.
5. Run eligibility gate on proposals.
6. Broadcast debate Round 1 Phase A.
7. Check for impl track completions again.
8. Continue alternating until both workstreams complete.

The orchestrator MUST NOT advance to Stage 6 integration until:
- ALL implementation tracks have completed (PASSED or FAILED)
- The overlapped debate has completed (winners selected, champions shut down) OR has exceeded the 90-minute timeout (in which case: shut down debate champions, discard partial results, proceed without overlapped debate winners)

### GPU Allocation During Overlap

| Agent Type | GPU Access |
|-----------|-----------|
| Implementation champions | Assigned GPU per track (as today) |
| Implementation validators | Shared GPU with their champion (coordinated via SendMessage) |
| Debate champions | CPU-based analysis only (roofline, ISA, ncu --query-metrics) |
| Debate delegates | CPU-based analysis only |

On systems with 3+ GPUs, the last GPU is soft-reserved for debate micro-experiments that need GPU access. It returns to the implementation pool when debate is idle.

### Debate Results Handling

When the overlapped debate completes:
1. Orchestrator scores candidates using the debate-scoring-rubric.md.
2. Orchestrator writes `debate/campaign_round_{N+1}/summary.md`.
3. Orchestrator records winners in `state.json` at `debate.next_round_overlap.selected_winners`.
4. Orchestrator records each winner's f-value at proposal time in `debate.next_round_overlap.f_values_at_proposal`.
5. Orchestrator sends `shutdown_request` to all debate champions and delegates.
6. Winners are used in round N+1 after lazy invalidation (see SKILL.md, Campaign Loop Transition).
