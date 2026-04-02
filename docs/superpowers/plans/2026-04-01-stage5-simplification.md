# Stage 5 Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repurpose `ammo-impl-validator` from orchestrator-spawned team member to champion-spawned sub-agent handling Gates 5.1a + 5.2, simplifying Stage 5 validation flow.

**Architecture:** The validator becomes a sub-agent the champion spawns via `Agent()`. Sub-agent handles kernel-level validation (5.1a correctness + 5.2 speedup), champion handles E2E-level validation (5.1b + 5.3a + 5.3b via single sweep). Orchestrator is aware but hands-off.

**Tech Stack:** Markdown configuration files (agent definitions, skill docs, references, tests). No code changes.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `.claude/agents/ammo-impl-validator.md` | REWRITE | Sub-agent definition (Gates 5.1a + 5.2) |
| `.claude/agents/ammo-impl-champion.md` | MAJOR EDIT | Spawns sub-agent, owns E2E gates |
| `.claude/skills/ammo/SKILL.md` | MODERATE EDIT | Simplified Stages 4-5, task graph |
| `.claude/skills/ammo/orchestration/parallel-tracks.md` | MAJOR EDIT | Team structure, remove Validation Spawn Protocol |
| `.claude/skills/ammo/references/impl-track-rules.md` | MODERATE EDIT | Verification model, GATING_REQUIRED |
| `.claude/skills/ammo/references/validation-defaults.md` | MODERATE EDIT | Gate ownership, ordering fixes |
| `.claude/skills/ammo/references/crossover-probing.md` | SMALL EDIT | Ownership clarification |
| `.claude/skills/ammo/references/kernel-benchmark-template.py` | SMALL EDIT | CHAMPION_FILL -> FILL |
| `.claude/agents/ammo-delegate.md` | SMALL EDIT | Template ownership ref |
| `.claude/agents/ammo-transcript-monitor.md` | MODERATE EDIT | Safety net check, gate completeness |
| `.claude/skills/ammo/tests/agents/test-impl-champion.md` | MODERATE EDIT | Scenarios for sub-agent model |
| `.claude/skills/ammo/tests/agents/test-orchestrator.md` | SMALL EDIT | Remove validator spawn refs |
| `.claude/skills/ammo/tests/agents/test-implementer.md` | SMALL EDIT | Update validator refs |

---

### Task 1: Rewrite ammo-impl-validator.md as sub-agent

**Files:**
- Rewrite: `.claude/agents/ammo-impl-validator.md`

The validator becomes a sub-agent spawned by the champion. It handles Gate 5.1a (kernel correctness) and Gate 5.2 (kernel speedup benchmark). It returns results via the Agent tool return, not SendMessage.

- [ ] **Step 1: Rewrite the entire file**

Replace the full contents of `.claude/agents/ammo-impl-validator.md` with:

```markdown
---
name: ammo-impl-validator
description: Kernel-level validation sub-agent for AMMO optimization tracks. Writes independent correctness tests (Gate 5.1a) and runs kernel speedup benchmarks (Gate 5.2). Spawned by impl-champion at validation time.
model: sonnet
---

# AMMO Kernel Validation Sub-Agent

You independently validate a champion's GPU kernel optimization by writing your OWN correctness tests and running kernel speedup benchmarks. You are spawned by the impl-champion as a sub-agent — your results return directly to the champion via the Agent tool.

Your scope: **Gate 5.1a** (kernel correctness) and **Gate 5.2** (kernel speedup benchmark under production parity). Gates 5.1b (E2E correctness) and 5.3 (E2E latency) are handled by the champion via the sweep script after you return.

# Environment (BLOCKING)
- **Python environment is pre-built.** Run `source .venv/bin/activate` before any Python command.
- **NEVER install packages.** Do not run `pip install`, `uv pip install`, or any installation command.
- **NEVER create a new venv.** The `.venv` already exists and is ready to use.
- If any import fails, report the error in your return — do not attempt to fix it.

## GPU Pool

GPU commands require pool reservation — see `references/gpu-pool.md`. Kernel benchmarks: `--num-gpus 1`.

## Independence Rule (NON-NEGOTIABLE)

**Write your OWN correctness tests and benchmarks.** Do NOT read or execute the champion's test files or benchmark scripts. Derive test methodology from the **optimization plan and debate summary (debate/summary.md)**, not from the implementation. This adversarial separation prevents reward hacking.

## Gate 5.1a: Independent Kernel Correctness Tests

Derive test methodology from:
1. The optimization plan (`{artifact_dir}/debate/summary.md`)
2. `{artifact_dir}/target.json` — `workload.batch_sizes`
3. `references/validation-defaults.md` — tolerance starting points

Your correctness tests must:
- Import vLLM's **production kernel** as baseline (not naive PyTorch)
- Use `torch.allclose()` with appropriate tolerances per dtype
- Test ALL batch sizes from target.json (no cherry-picking)
- Include adversarial cases: edge batch sizes (1, max), precision boundary values
- Check for NaNs/INFs in output
- Test under CUDA graph capture/replay (not just eager mode)

Write tests to `{artifact_dir}/tracks/{op_id}/validator_tests/test_correctness.py`.

## Gate 5.2: Kernel Speedup Benchmark

Run an independent kernel benchmark under production parity (CUDA graphs, production stream):

1. Adapt the benchmark template at `references/kernel-benchmark-template.py`
2. Benchmark both baseline (vLLM production kernel) and optimized kernel
3. Capture both warm-cache and cold-cache timings under CUDA graph replay
4. Test ALL batch sizes from target.json
5. Write results to `{artifact_dir}/tracks/{op_id}/validator_tests/gate_5_2_results.json`

See `references/validation-defaults.md` § Gate 5.2 for methodology requirements.

## DA Verification Checks

After completing Gates 5.1a and 5.2, run these DA checks:

1. **Cross-track awareness**: Read `state.json` `parallel_tracks`. If other tracks exist with C++ changes (`csrc/`) and THIS track is Python-only, FLAG: ".so contamination risk."
2. **Scope adherence**: Read `{artifact_dir}/debate/summary.md` for the planned scope of this op_id. Compare against files modified in the worktree (`git diff --name-only main`). If planned components were omitted without documented rationale, FLAG.

## Return Format

Return a structured report (this is your Agent tool return value):

```
## Kernel Validation Report: {op_id}

### Gate 5.1a: Kernel Correctness
- Batch sizes tested: [list all]
- Tolerances used: atol={}, rtol={}
- Per-size results: [pass/fail per batch size with max absolute error]
- NaN/INF check: [pass/fail]
- CUDA graph mode: [pass/fail]
- Overall: [PASS/FAIL]

### Gate 5.2: Kernel Speedup
- Per-BS results: [baseline_us, optimized_us, speedup per BS]
- Warm-cache / cold-cache ratio: [per BS]
- Overall: [table]

### DA Verification
1. Cross-track: [PASS/FAIL + detail]
2. Scope adherence: [PASS/FAIL + detail]

### Files Written
- validator_tests/test_correctness.py
- validator_tests/gate_5_2_results.json
```

## Hard Rules

1. **Independent tests are non-negotiable.** Write your OWN. Do NOT use the champion's scripts.
2. **Report raw data.** Pass/fail per test with max absolute error. The champion interprets significance.
3. **No source modification.** You do NOT edit kernel code, vLLM source, or csrc/ files.
4. **Write validation outputs to artifact dir.** Test files and results go to `{artifact_dir}/tracks/{op_id}/validator_tests/`.

## References

Read as needed from `.claude/skills/ammo/references/`:
- `impl-track-rules.md` — worktree build rules, verdict thresholds, track constraints
- `gpu-pool.md` — GPU reservation pattern
- `validation-defaults.md` — tolerances, gate definitions, production parity
- `cudagraph-safety.md` — CUDA graph capture checklist
- `gpu-configs.md` — hardware specs
- `kernel-benchmark-template.py` — Gate 5.2 benchmark template
```

- [ ] **Step 2: Verify the rewrite**

Read `.claude/agents/ammo-impl-validator.md` and confirm:
- Frontmatter says "Spawned by impl-champion" (not orchestrator)
- No mention of SendMessage, dual-reporting, or team membership
- Gate 5.2 section exists
- Return format uses structured text (not SendMessage)

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/ammo-impl-validator.md
git commit -m "refactor(ammo): rewrite impl-validator as champion-spawned sub-agent

Gate 5.1a + 5.2 kernel-level validation. Returns results via Agent
tool instead of SendMessage. No team membership or dual-reporting."
```

---

### Task 2: Update ammo-impl-champion.md — spawn sub-agent instead of VALIDATION_REQUEST

**Files:**
- Modify: `.claude/agents/ammo-impl-champion.md`

Replace the VALIDATION_REQUEST protocol with direct sub-agent spawning. Add Gate 5.2 delegation. Update the validation flow.

- [ ] **Step 1: Update the agent identity (line 10)**

Replace:
```
You implement GPU kernel optimizations for a specific track in the AMMO pipeline. When your implementation is committed and ready, you report to the orchestrator, who spawns an independent validator. The validator writes its OWN tests and benchmarks — this adversarial separation is non-negotiable.
```
With:
```
You implement GPU kernel optimizations for a specific track in the AMMO pipeline. When your implementation is committed and ready, you spawn a kernel validation sub-agent for Gates 5.1a + 5.2. The sub-agent writes its OWN tests and benchmarks — this adversarial separation is non-negotiable.
```

- [ ] **Step 2: Remove worktree validator instruction (line 39)**

Replace:
```
After entering the worktree, tell your validator which worktree to enter (they need to work on the same branch to see your changes).
```
With:
```
After entering the worktree, all your work happens here. When you spawn the kernel validation sub-agent, it inherits your worktree context from the spawn prompt.
```

- [ ] **Step 3: Replace "Requesting Validation" section (lines 106-127)**

Replace the entire section from `## Requesting Validation` through the end of the orchestrator spawn workflow with:

```markdown
## Kernel Validation (Gates 5.1a + 5.2)

After implementation is committed and your smoke test passes, spawn the kernel validation sub-agent:

```python
result = Agent(
    subagent_type="ammo-impl-validator",
    prompt=f"""Validate optimization {op_id}.
    Artifact dir: {artifact_dir}
    Debate plan: {artifact_dir}/debate/summary.md (section for {op_id})
    Target config: {artifact_dir}/target.json
    Batch sizes: {batch_sizes}

    Run Gate 5.1a (kernel correctness) and Gate 5.2 (kernel speedup).
    Write tests to {artifact_dir}/tracks/{op_id}/validator_tests/.
    Return structured results: 5.1a pass/fail per BS, 5.2 speedup per BS.
    GPU pool: CVD=$(python .claude/skills/ammo/scripts/gpu_reservation.py reserve --num-gpus 1) && CUDA_VISIBLE_DEVICES=$CVD <cmd>"""
)
```

The sub-agent returns results directly. Evaluate them:
- **If Gate 5.1a FAIL**: Fix the kernel, run the Self-Validation Gate checklist, re-spawn the sub-agent. Do NOT proceed to the E2E sweep — fix kernel correctness first.
- **If Gate 5.1a PASS**: Proceed to the E2E sweep (Gates 5.1b + 5.3a + 5.3b).

For re-validation after a fix, spawn a fresh sub-agent (each invocation is independent).
```

Find and remove these exact lines (the old VALIDATION_REQUEST message format):
```
SendMessage("team-lead", """
VALIDATION_REQUEST:
- op_id: {op_id}
- commit_sha: {sha}
- worktree_path: {worktree_path}
- artifact_dir: {artifact_dir}
- expect_kernel: "{optimized_kernel_function_name}"
- batch_sizes: {batch_sizes from target.json}
Ready for independent validation. I will remain available for re-validation cycles.
""")
```

And remove these lines:
```
The orchestrator will spawn an independent validator and tell you the validator's name. You then:
- Receive validation results from the validator via SendMessage
- Write `validation_results.md` based on the raw validation data
- If validation fails, fix issues, recommit, and message the validator directly for re-validation
- For GATING_REQUIRED tracks, implement gating and request re-validation
```

- [ ] **Step 4: Update "Making the Final Decision" section (line 130+)**

Replace:
```
When you receive validation results from the orchestrator-spawned validator (Gate 5.1a):
```
With:
```
When your kernel validation sub-agent returns results (Gates 5.1a + 5.2):
```

- [ ] **Step 5: Update Per-BS Verdict section — GATING_REQUIRED flow (line 149)**

Replace:
```
  5. Request re-validation of the gated version (message validator with commit SHA)
```
With:
```
  5. Spawn kernel validation sub-agent for re-validation of gated kernel (5.1a + 5.2)
  6. Re-run the sweep on gated code (`--labels opt --verify-correctness --nsys-profile --baseline-from $STAGE1_DIR`)
```

- [ ] **Step 6: Update Self-Validation Gate section (line 188)**

Replace:
```
5. **Message the validator**: Include your root cause reasoning in the re-validation request so the validator has context for what changed and why.
```
With:
```
5. **Re-spawn sub-agent**: Spawn a fresh kernel validation sub-agent. Include your root cause reasoning in the spawn prompt so the sub-agent has context for what changed.
```

- [ ] **Step 7: Update "Handling Validation Failures" section (line 197)**

Replace:
```
5. Commit and message the validator directly for re-validation with the new commit SHA
```
With:
```
5. Commit and spawn a fresh kernel validation sub-agent for re-validation
```

- [ ] **Step 8: Update GATED_PASS Output section**

Verify the Output section includes GATED_PASS verdict in the checklist. (This was already done in the prior harmonization — confirm it says "Overall PASS/FAIL/GATED_PASS verdict".)

- [ ] **Step 9: Report to orchestrator — add explicit step**

After the "Kernel Validation" section added in Step 3, add below the E2E Validation section:

```markdown
## Reporting to Orchestrator

After writing `validation_results.md`, report the final verdict to the orchestrator:

```
SendMessage("team-lead", """
TRACK_COMPLETE:
- op_id: {op_id}
- verdict: {PASS|FAIL|GATED_PASS}
- validation_results: {artifact_dir}/tracks/{op_id}/validation_results.md
- commit_sha: {sha}
""")
```
```

- [ ] **Step 10: Commit**

```bash
git add .claude/agents/ammo-impl-champion.md
git commit -m "refactor(ammo): champion spawns validator sub-agent directly

Replace VALIDATION_REQUEST protocol with Agent() spawn for Gates 5.1a
+ 5.2. Champion reports TRACK_COMPLETE to orchestrator after writing
validation_results.md. Explicit re-run sweep step for GATING_REQUIRED."
```

---

### Task 3: Update SKILL.md — simplify orchestrator's Stages 4-5

**Files:**
- Modify: `.claude/skills/ammo/SKILL.md`

Remove validator spawn from orchestrator, simplify task graph, update inline DA section.

- [ ] **Step 1: Update Stage 4+5 description (line 49)**

Replace:
```
Stage 4+5: Parallel Tracks          [round team reused: per-track impl-champion + impl-validator pairs + DA audit]
```
With:
```
Stage 4+5: Parallel Tracks          [round team reused: per-track impl-champion + DA audit]
```

- [ ] **Step 2: Simplify Stages 4-5 orchestration (lines 219-227)**

Replace:
```
Each track uses a **champion + independent validator** pair to prevent reward hacking (observed: cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation). All implementation agents join the **existing round team** created in Stage 3 — no new TeamCreate calls.

Execute these steps **in order**:

1. **Spawn implementation agents into the round team**: Per winning candidate:
   - Spawn an `ammo-impl-champion` (Opus, `isolation: worktree`) into the existing round team.
   - **Monitor spawn (REQUIRED)**: Immediately spawn an `ammo-transcript-monitor` (Sonnet) as a **team member** (`team_name=round_team_name`, `run_in_background=True`) for that impl-champion. See `orchestration/debate-protocol.md` § Monitor Spawn Pattern.
   - **Do NOT spawn `ammo-impl-validator` here.** The validator is spawned later by the orchestrator when the champion sends a `VALIDATION_REQUEST` via SendMessage (see `orchestration/parallel-tracks.md` § Validation Spawn Protocol).
   - The champion implements; when ready, it sends `VALIDATION_REQUEST` to the orchestrator. The orchestrator spawns the validator, who independently writes its own synthetic correctness tests (Gate 5.1a only). Gates 5.1b and 5.3 are handled by the champion via the sweep script with `--verify-correctness`. Gate 5.2 (isolated kernel benchmark) is champion-owned independently. The validator dual-reports raw results to both champion and orchestrator.
```
With:
```
Each track uses a champion with an independently-spawned kernel validation sub-agent to prevent reward hacking (observed: cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation). All implementation agents join the **existing round team** created in Stage 3 — no new TeamCreate calls.

Execute these steps **in order**:

1. **Spawn implementation agents into the round team**: Per winning candidate:
   - Spawn an `ammo-impl-champion` (Opus, `isolation: worktree`) into the existing round team.
   - **Monitor spawn (REQUIRED)**: Immediately spawn an `ammo-transcript-monitor` (Sonnet) as a **team member** (`team_name=round_team_name`, `run_in_background=True`) for that impl-champion. See `orchestration/debate-protocol.md` § Monitor Spawn Pattern.
   - The champion implements; when ready, it spawns `ammo-impl-validator` as a sub-agent for kernel-level validation (Gates 5.1a + 5.2). The champion then runs the sweep script for E2E validation (Gates 5.1b + 5.3a + 5.3b). The champion reports the final verdict to the orchestrator via `TRACK_COMPLETE` message.
```

- [ ] **Step 3: Update task graph (lines 278-287)**

Replace:
```
  | T8a_{id}: Research + plan reading (champion)               [round team]    <- T7   |
  | T8b_{id}: Implement kernel (champion only)                 [round team]    <- T8a  |
  |   T8b_val_{id}: Champion sends VALIDATION_REQUEST → orchestrator spawns impl-validator-{id} |
  | T8c_{id}: Gate 5.1a validation (validator) + E2E sweep with --verify-correctness (champion) [round team] <- T8b |
  | T8cx_{id}: [IF GATING_REQUIRED] Crossover probing (validator benchmarks,   |
  |            champion implements gating, validator re-validates)  [round team] <- T8c |
  | T8d_{id}: Kill criteria evaluation + validation_results.md (champion) [round team] <- T8c/T8cx |
```
With:
```
  | T8a_{id}: Research + plan reading (champion)               [round team]    <- T7   |
  | T8b_{id}: Implement kernel (champion only)                 [round team]    <- T8a  |
  | T8c_{id}: Kernel validation sub-agent (5.1a + 5.2) + E2E sweep (5.1b + 5.3a + 5.3b) [round team] <- T8b |
  | T8cx_{id}: [IF GATING_REQUIRED] Crossover probing (sub-agent benchmarks,  |
  |            champion implements gating, sub-agent re-validates) [round team] <- T8c  |
  | T8d_{id}: Kill criteria evaluation + validation_results.md (champion) [round team] <- T8c/T8cx |
```

- [ ] **Step 4: Update inline DA section (line 357)**

Replace:
```
- **ammo-impl-validator** → After Gate 5.1a: DA section in validation report (scope adherence, correctness methodology, cross-track awareness; plus conditional GATED_PASS checks). Gates 5.1b/5.3 results come from the champion's sweep script (`correctness_verdict.json` for 5.1b, sweep output for 5.3). Gate 5.2 (isolated kernel benchmark) is champion-owned independently.
```
With:
```
- **ammo-impl-validator** (champion-spawned sub-agent) → Gates 5.1a + 5.2: DA section in return results (scope adherence, cross-track awareness). Gates 5.1b/5.3 results come from the champion's sweep script.
```

- [ ] **Step 5: Commit**

```bash
git add .claude/skills/ammo/SKILL.md
git commit -m "refactor(ammo): simplify SKILL.md Stages 4-5 for sub-agent model

Remove validator spawn protocol from orchestrator. Champion manages
kernel validation internally. Task graph simplified."
```

---

### Task 4: Update parallel-tracks.md — team structure and spawn protocol

**Files:**
- Modify: `.claude/skills/ammo/orchestration/parallel-tracks.md`

Remove Validation Spawn Protocol, update team structure, rewrite verification model, fix spawn prompt (Issue A).

- [ ] **Step 1: Update header paragraph (line 1-3)**

Replace:
```
Each winning candidate from Stage 3 gets its own git worktree, branch, and implementation agent pair. All agents -- across all tracks -- belong to the **same round team** created at the start of Stage 3. The orchestrator can only lead one team at a time, so a single round-scoped team is used for the entire round lifecycle (debate through implementation). Tracks run in parallel across GPUs. Within a track, the champion and validator collaborate fluidly -- both may be active simultaneously, with GPU access coordinated via SendMessage. During round 2+, debate champions for the next round may also be present in the team (see Overlapped Debate below). Implementation and debate agents share the team but operate as independent workstreams -- they do not communicate directly.
```
With:
```
Each winning candidate from Stage 3 gets its own git worktree, branch, and implementation champion. All agents -- across all tracks -- belong to the **same round team** created at the start of Stage 3. The orchestrator can only lead one team at a time, so a single round-scoped team is used for the entire round lifecycle (debate through implementation). Tracks run in parallel across GPUs. Within a track, the champion manages validation internally by spawning a kernel validation sub-agent (not a team member). During round 2+, debate champions for the next round may also be present in the team (see Overlapped Debate below). Implementation and debate agents share the team but operate as independent workstreams -- they do not communicate directly.
```

- [ ] **Step 2: Update team structure diagram (lines 9-22)**

Replace:
```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
[Implementation Workstream]
+-- impl-champion-{op_id_1} (Opus)           -- implementation, E2E threshold evaluation
+-- monitor-impl-champion-{op_id_1} (Sonnet) -- transcript monitor (background, team member)
+-- impl-champion-{op_id_2} (Opus)           -- implementation, E2E threshold evaluation
+-- monitor-impl-champion-{op_id_2} (Sonnet) -- transcript monitor (background, team member)
+-- impl-validator-{op_id} (Sonnet)          -- spawned by orchestrator AT VALIDATION TIME ONLY
[Overlapped Debate Workstream -- round 2+ only]
```
With:
```
Round Team: ammo-round-{round_id}-{model_short}-{hardware}
[Implementation Workstream]
+-- impl-champion-{op_id_1} (Opus)           -- implementation + validation orchestration
+-- monitor-impl-champion-{op_id_1} (Sonnet) -- transcript monitor (background, team member)
+-- impl-champion-{op_id_2} (Opus)           -- implementation + validation orchestration
+-- monitor-impl-champion-{op_id_2} (Sonnet) -- transcript monitor (background, team member)
    (kernel validation sub-agents spawned by champions as needed -- NOT team members)
[Overlapped Debate Workstream -- round 2+ only]
```

- [ ] **Step 3: Update adversarial validation paragraph (line 24)**

Replace:
```
Each track uses an **adversarial validation model**: the champion implements, an orchestrator-spawned independent validator validates. This separation prevents reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation). Transcript monitors provide continuous DA oversight of champion work.
```
With:
```
Each track uses an **adversarial validation model**: the champion implements, then spawns a kernel validation sub-agent that independently writes its own correctness tests and benchmarks. This separation prevents reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation). Transcript monitors provide continuous DA oversight of champion work.
```

- [ ] **Step 4: Remove "validator shares worktree" line (line 34)**

Replace:
```
The validator agent shares the champion's worktree (same track team). Both agents may be active simultaneously — GPU access is coordinated via SendMessage, with the champion having priority.
```
With:
```
The champion spawns the kernel validation sub-agent with the worktree path in the spawn prompt. The sub-agent runs sequentially (Gates 5.1a + 5.2 first, then champion runs the E2E sweep).
```

- [ ] **Step 5: Update spawn prompt workflow (lines 87-93 — fix Issue A)**

In the spawn prompt template, replace step 5:
```
    5. Run E2E sweep: `run_vllm_bench_latency_sweep.py --verify-correctness --baseline-from {stage1_dir}`
```
With:
```
    5. Spawn kernel validation sub-agent for Gates 5.1a + 5.2 (see your agent definition § Kernel Validation)
    6. Run E2E sweep per your agent definition § E2E Validation (ONE command handles 5.1b + 5.3a + 5.3b)
```

- [ ] **Step 6: Remove validator from spawn code NOTE (lines 117-118)**

Replace:
```
# NOTE: impl-validator is NOT spawned here. It is spawned by the orchestrator
# when the impl-champion sends a VALIDATION_REQUEST. See "Validation Spawn Protocol" below.
```
With:
```
# NOTE: The champion spawns ammo-impl-validator as a sub-agent internally.
# The orchestrator does NOT spawn or manage the validator.
```

- [ ] **Step 7: Delete Validation Spawn Protocol section (lines 121-153)**

Delete the entire section from `### Validation Spawn Protocol` through the `SendMessage` notification line. Replace with:

```markdown
### Champion-Managed Validation

The champion manages kernel validation internally:
1. Champion spawns `ammo-impl-validator` as a sub-agent via `Agent()` (not a team member)
2. Sub-agent runs Gates 5.1a (kernel correctness) + 5.2 (kernel speedup), returns results
3. If 5.1a FAIL: champion fixes, re-spawns sub-agent (no wasted E2E sweep)
4. If 5.1a PASS: champion runs sweep (5.1b + 5.3a + 5.3b)
5. Champion combines all gate results into `validation_results.md`
6. Champion reports `TRACK_COMPLETE` to orchestrator via SendMessage

The orchestrator reads `validation_results.md` for gate decisions but does not participate in the validation loop.
```

- [ ] **Step 8: Rewrite "Two Layers of Verification" (lines 159-171)**

Replace:
```
### Two Layers of Verification

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN synthetic correctness tests (Gate 5.1a)
  Reports raw structured results — no interpretation

Layer 2: Champion (Opus)
  Runs sweep with --verify-correctness (Gates 5.1b/5.3)
  Evaluates E2E results against min_e2e_improvement_pct threshold
  Cross-checks Gate 5.1a against correctness_verdict.json
  Writes final validation_results.md with evidence chain
```
```
With:
```markdown
### Champion-Owned Validation with Kernel Sub-Agent

```
Kernel-Level (Sub-Agent, Sonnet):
  Gate 5.1a: Writes OWN kernel correctness tests, returns structured results
  Gate 5.2: Runs kernel speedup benchmark under CUDA graphs

E2E-Level (Champion, Opus):
  Gate 5.1b: Sweep --verify-correctness (GSM8K greedy decode)
  Gate 5.3a: Sweep --nsys-profile (kernel execution proof)
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Cross-checks Gate 5.1a against correctness_verdict.json
  Writes final validation_results.md with evidence chain
```

The sub-agent provides adversarial kernel-level verification. The champion provides E2E-level verification. The transcript monitor provides continuous DA oversight.
```

- [ ] **Step 9: Update result collection (lines 179-190)**

Replace:
```
1. `{artifact_dir}/tracks/{op_id}/validation_results.md` — champion's final report
2. `{artifact_dir}/tracks/{op_id}/validator_tests/` — validator's independent scripts and results
3. `state.json` field `parallel_tracks.{op_id}.result` — structured summary
```
With:
```
1. `{artifact_dir}/tracks/{op_id}/validation_results.md` — champion's final report (includes sub-agent results)
2. `{artifact_dir}/tracks/{op_id}/validator_tests/` — sub-agent's independent scripts and results
3. `state.json` field `parallel_tracks.{op_id}.result` — structured summary
```

- [ ] **Step 10: Update Pass Criteria section (line 199)**

Replace:
```
- Gate 5.1: Correctness — both sub-gates must pass:
  - 5.1a: Validator's independent synthetic correctness tests pass
  - 5.1b: Sweep script `--verify-correctness` verdict is PASS in `correctness_verdict.json` (deterministic — no N/A escape)
```
With:
```
- Gate 5.1: Correctness — both sub-gates must pass:
  - 5.1a: Kernel validation sub-agent's independent correctness tests pass
  - 5.1b: Sweep script `--verify-correctness` verdict is PASS in `correctness_verdict.json` (deterministic — no N/A escape)
```

- [ ] **Step 11: Commit**

```bash
git add .claude/skills/ammo/orchestration/parallel-tracks.md
git commit -m "refactor(ammo): update parallel-tracks.md for sub-agent model

Remove Validation Spawn Protocol. Update team structure (no validator
slot). Rewrite verification model. Fix spawn prompt missing flags."
```

---

### Task 5: Update reference files — impl-track-rules.md, validation-defaults.md, crossover-probing.md

**Files:**
- Modify: `.claude/skills/ammo/references/impl-track-rules.md`
- Modify: `.claude/skills/ammo/references/validation-defaults.md`
- Modify: `.claude/skills/ammo/references/crossover-probing.md`

- [ ] **Step 1: Update impl-track-rules.md build rules (line 12)**

Replace:
```
Only the champion compiles. The validator never runs cmake. The validator only executes against committed, compiled code.
```
With:
```
Only the champion compiles. The kernel validation sub-agent never runs cmake — it only executes against committed, compiled code.
```

- [ ] **Step 2: Update impl-track-rules.md source modification rules (lines 15-18)**

Replace:
```
## Source Modification Rules

- Only the champion modifies source files (`csrc/`, `vllm/`, etc.).
- The validator reads files and writes to `{artifact_dir}/tracks/{op_id}/validator_prep/` only.
- Champion has GPU priority. The validator coordinates via SendMessage before GPU-intensive work (ncu profiling, E2E sweeps). The champion signals when it needs the GPU.
```
With:
```
## Source Modification Rules

- Only the champion modifies source files (`csrc/`, `vllm/`, etc.).
- The kernel validation sub-agent reads files and writes to `{artifact_dir}/tracks/{op_id}/validator_tests/` only.
- The sub-agent runs sequentially before the champion's E2E sweep — no GPU coordination needed.
```

- [ ] **Step 3: Update impl-track-rules.md Independent Validation (lines 20-25)**

Replace:
```
## Independent Validation Principle

When validating, the validator writes its OWN correctness tests and benchmarks from the **optimization plan and debate summary** — not from the champion's scripts or implementation. This is the structural guarantee against reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation).

The validator can know everything about the codebase from its support work and still write unbiased validation tests, as long as tests are derived from what the optimization SHOULD do (the plan) rather than what it DOES do (the implementation).
```
With:
```
## Independent Validation Principle

The kernel validation sub-agent writes its OWN correctness tests and benchmarks from the **optimization plan and debate summary** — not from the champion's scripts or implementation. This is the structural guarantee against reward hacking (cherry-picked batch sizes, weakened assertions, inflated benchmarks, optimistic interpretation).

The sub-agent derives tests from what the optimization SHOULD do (the plan) rather than what it DOES do (the implementation).
```

- [ ] **Step 4: Rewrite impl-track-rules.md Two Layers (lines 26-39)**

Replace:
```
## Two Layers of Verification

Each track undergoes two layers of verification:

```
Layer 1: Independent Validator (Sonnet)
  Writes OWN synthetic correctness tests (Gate 5.1a only)
  Reports raw correctness results — no interpretation

Layer 2: Champion Review
  Evaluates E2E results against min_e2e_improvement_pct threshold
  Cross-checks Gate 5.2 numbers against own smoke-test
  Writes final validation_results.md with evidence chain
```
```
With:
```markdown
## Champion-Owned Validation

The champion owns all Stage 5 validation, with a sub-agent for kernel-level gates:

```
Kernel-Level (Sub-Agent):
  Gate 5.1a: Independent kernel correctness tests
  Gate 5.2: Kernel speedup benchmark under production parity

E2E-Level (Champion):
  Gate 5.1b: Sweep --verify-correctness
  Gate 5.3a: Sweep --nsys-profile (kernel proof)
  Gate 5.3b: Sweep E2E latency (per-BS verdicts)
  Writes final validation_results.md with evidence chain
```
```

- [ ] **Step 5: Update impl-track-rules.md GATING_REQUIRED (lines 41-69)**

Replace:
```
1. Validator reports per-BS verdict table to champion
2. Champion evaluates gating feasibility (is the dispatch site compatible with a gating mechanism?)
3. If feasible: champion requests validator to run crossover probing benchmarks
4. Validator runs kernel sweep + E2E confirmation per `crossover-probing.md`, reports probe results
```
With:
```
1. Sweep reports per-BS verdict table showing mixed results
2. Champion evaluates gating feasibility (is the dispatch site compatible with a gating mechanism?)
3. If feasible: champion spawns sub-agent for crossover probing benchmarks
4. Sub-agent runs kernel sweep + E2E confirmation per `crossover-probing.md`, returns probe results
```

And replace:
```
8. Champion requests validator to re-validate gated version
9. Validator re-validates at all BS — all must be PASS or NOISE
10. If re-validation passes: verdict = `GATED_PASS`. If fails: verdict = `FAIL`.
```
With:
```
8. Champion spawns sub-agent for re-validation of gated kernel (5.1a + 5.2)
9. Champion re-runs sweep on gated code (5.1b + 5.3a + 5.3b) — all BS must be PASS or NOISE
10. If both kernel re-validation and sweep pass: verdict = `GATED_PASS`. If either fails: verdict = `FAIL`.
```

- [ ] **Step 6: Update validation-defaults.md Gate 5.3b ordering (Issue C, line 251)**

Replace:
```
**Only runs after Gate 5.3a passes.**
```
With:
```
**Gate 5.3b latency results are only VALID if Gate 5.3a passes** — if the optimized kernel is not found in the nsys trace, E2E numbers are inadmissible.
```

- [ ] **Step 7: Update validation-defaults.md GATING_REQUIRED (Issue B, lines 277-285)**

Replace:
```
### GATING_REQUIRED Workflow

When the track verdict is `GATING_REQUIRED`:
1. Validator reports per-BS verdict table to champion
2. Champion evaluates gating feasibility
3. Champion requests validator to run crossover probing (see `references/crossover-probing.md`)
4. Champion implements gating mechanism per `references/code-templates.md` dispatch decision tree
5. Validator re-validates gated version — all BS must be PASS or NOISE
6. If re-validation passes: track status = `GATED_PASS`
7. If re-validation fails or gating infeasible: track status = `FAIL` (one gating attempt per track)
```
With:
```
### GATING_REQUIRED Workflow

When the track verdict is `GATING_REQUIRED`:
1. Sweep reports per-BS verdict table showing mixed results (some PASS + some REGRESSED)
2. Champion evaluates gating feasibility
3. Champion spawns sub-agent for crossover probing (see `references/crossover-probing.md`)
4. Champion implements gating mechanism per `references/code-templates.md` dispatch decision tree
5. Champion spawns sub-agent for re-validation of gated kernel (5.1a + 5.2)
6. Champion re-runs sweep on gated code (`--labels opt --verify-correctness --nsys-profile --baseline-from $STAGE1_DIR`) — all BS must be PASS or NOISE
7. If both kernel re-validation and sweep pass: track status = `GATED_PASS`
8. If either fails or gating infeasible: track status = `FAIL` (one gating attempt per track)
```

- [ ] **Step 8: Update crossover-probing.md ownership (line 10)**

Replace:
```
- The champion has evaluated gating feasibility and requested the validator to probe
```
With:
```
- The champion has evaluated gating feasibility and spawned a sub-agent to probe
```

- [ ] **Step 9: Update crossover-probing.md One Attempt Rule (lines 71-74)**

Replace:
```
After crossover probing completes and the champion implements gating:
- The validator re-validates the gated version at all BS
- If re-validation shows REGRESSED or CATASTROPHIC at any BS: **track FAILs**
```
With:
```
After crossover probing completes and the champion implements gating:
- The champion spawns a sub-agent for kernel re-validation (5.1a + 5.2) AND re-runs the sweep (5.1b + 5.3a + 5.3b)
- If any gate shows REGRESSED or CATASTROPHIC at any BS: **track FAILs**
```

- [ ] **Step 10: Commit**

```bash
git add .claude/skills/ammo/references/impl-track-rules.md \
       .claude/skills/ammo/references/validation-defaults.md \
       .claude/skills/ammo/references/crossover-probing.md
git commit -m "refactor(ammo): update references for sub-agent validation model

impl-track-rules.md: champion-owned validation, GATING_REQUIRED with
explicit re-run sweep step.
validation-defaults.md: fix 5.3a/5.3b ordering, GATING_REQUIRED flow.
crossover-probing.md: sub-agent ownership for probing."
```

---

### Task 6: Update minor files — kernel-benchmark-template.py, ammo-delegate.md, ammo-transcript-monitor.md

**Files:**
- Modify: `.claude/skills/ammo/references/kernel-benchmark-template.py`
- Modify: `.claude/agents/ammo-delegate.md`
- Modify: `.claude/agents/ammo-transcript-monitor.md`

- [ ] **Step 1: Update kernel-benchmark-template.py header (lines 5-6)**

Replace:
```
Independent kernel benchmark for adversarial validation. The champion (or their
delegate) adapts this template for the specific target kernel — filling in
```
With:
```
Independent kernel benchmark for adversarial validation. The kernel validation
sub-agent (or champion's delegate) adapts this template — filling in
```

- [ ] **Step 2: Update kernel-benchmark-template.py fill comment (line 22)**

Replace:
```
The champion (or their delegate) fills in the sections marked CHAMPION_FILL below.
```
With:
```
Fill in the sections marked FILL below.
```

- [ ] **Step 3: Replace all CHAMPION_FILL markers in kernel-benchmark-template.py**

Use replace_all to change all `CHAMPION_FILL` to `FILL` (6 occurrences at lines 35, 55, 69, 80, and in comments).

- [ ] **Step 4: Update ammo-delegate.md reference table (line 85)**

Replace:
```
| `kernel-benchmark-template.py` | Benchmark script template (champion-owned, Gate 5.2 — not validator-directed) |
```
With:
```
| `kernel-benchmark-template.py` | Gate 5.2 benchmark template (used by champion's kernel validation sub-agent) |
```

- [ ] **Step 5: Update ammo-transcript-monitor.md stop condition (line 105)**

Replace:
```
- The champion's transcript shows a completion message (e.g., "VALIDATION_REQUEST", "Track complete", "Implementation infeasible")
```
With:
```
- The champion's transcript shows a completion message (e.g., "TRACK_COMPLETE", "Track complete", "Implementation infeasible")
```

- [ ] **Step 6: Update ammo-transcript-monitor.md gate completeness (lines 207-208)**

Replace:
```
- Validation integrity: not sharing test scripts with validator
- Gate completeness: validator runs Gate 5.1a; champion runs Gates 5.1b/5.3 via sweep script (`correctness_verdict.json` for 5.1b, sweep output for 5.3); Gate 5.2 via isolated kernel benchmark. All must complete before declaring success
```
With:
```
- Validation integrity: not sharing test scripts with kernel validation sub-agent
- Gate completeness: champion spawns sub-agent for Gates 5.1a + 5.2; champion runs Gates 5.1b/5.3a/5.3b via sweep script. All must complete before declaring success
- **Missing sub-agent spawn**: champion runs sweep without prior `Agent(subagent_type="ammo-impl-validator")` call → CRITICAL (kernel correctness never independently verified)
```

- [ ] **Step 7: Update ammo-transcript-monitor.md sharing scripts violation (line 238)**

Replace:
```
- **Sharing test scripts with validator**: Champion sends test code to the validator. [implementation only]
```
With:
```
- **Sharing test scripts with sub-agent**: Champion's spawn prompt includes test code or points to champion's own test scripts. [implementation only]
```

- [ ] **Step 8: Update ammo-transcript-monitor.md blind fix-and-send (lines 254-255)**

Replace:
```
- **Blind fix-and-send**: Champion makes a code change (Edit tool) then immediately calls SendMessage to the validator with a re-validation request, with NO verification step in between — no Bash running pytest/python, no smoke test, no extended reasoning about correctness. The champion's agent definition requires a self-validation gate before re-requesting.
  - Message: `DA-MONITOR: [WARNING] You sent a re-validation request without running your own smoke test. Your Self-Validation Gate requires: (1) root cause reasoning, (2) smoke test, (3) fix-attempt counter check — before messaging the validator.`
```
With:
```
- **Blind fix-and-respawn**: Champion makes a code change (Edit tool) then immediately spawns a new validation sub-agent, with NO verification step in between — no Bash running pytest/python, no smoke test, no extended reasoning about correctness. The champion's agent definition requires a self-validation gate before re-spawning.
  - Message: `DA-MONITOR: [WARNING] You re-spawned the validation sub-agent without running your own smoke test. Your Self-Validation Gate requires: (1) root cause reasoning, (2) smoke test, (3) fix-attempt counter check — before re-spawning.`
```

- [ ] **Step 9: Update ammo-transcript-monitor.md independence limitation (line 274)**

Replace:
```
**Limitation**: This protocol is behavioral instructions, not a structural guarantee. The structurally independent verification layers (validator at Layer 1, champion review at Layer 2) remain the primary independence mechanisms.
```
With:
```
**Limitation**: This protocol is behavioral instructions, not a structural guarantee. The structurally independent kernel validation sub-agent (writes own tests from debate plan, not implementation) remains the primary independence mechanism.
```

- [ ] **Step 10: Commit**

```bash
git add .claude/skills/ammo/references/kernel-benchmark-template.py \
       .claude/agents/ammo-delegate.md \
       .claude/agents/ammo-transcript-monitor.md
git commit -m "refactor(ammo): update minor files for sub-agent model

kernel-benchmark-template.py: CHAMPION_FILL -> FILL.
ammo-delegate.md: update Gate 5.2 template ownership.
ammo-transcript-monitor.md: add missing sub-agent spawn check,
update gate completeness, fix blind-fix pattern."
```

---

### Task 7: Update test files

**Files:**
- Modify: `.claude/skills/ammo/tests/agents/test-impl-champion.md`
- Modify: `.claude/skills/ammo/tests/agents/test-orchestrator.md`
- Modify: `.claude/skills/ammo/tests/agents/test-implementer.md`

- [ ] **Step 1: Update test-impl-champion.md header (lines 1-4)**

Replace:
```
Role-boundary and constraint tests for the `ammo-impl-champion` subagent. Verifies the agent understands the Tiered Message Assessment Protocol, Self-Validation Gate, fix-attempt auto-escalation, delegation patterns, and interaction with validator/monitor messages.
```
With:
```
Role-boundary and constraint tests for the `ammo-impl-champion` subagent. Verifies the agent understands the Tiered Message Assessment Protocol, Self-Validation Gate, fix-attempt auto-escalation, delegation patterns, kernel validation sub-agent spawning, and interaction with sub-agent results/monitor messages.
```

- [ ] **Step 2: Update IC1 — "Validator reports" → "Sub-agent returns" (line 22)**

Replace:
```
**Context**: Validator sends: "Gate 5.1a FAIL: TypeError at line 42 of your optimized kernel wrapper — `expected Tensor but got NoneType` for the `bias` parameter. Full traceback: [traceback showing the line]." This is your first fix attempt.
```
With:
```
**Context**: Your kernel validation sub-agent returns: "Gate 5.1a FAIL: TypeError at line 42 of your optimized kernel wrapper — `expected Tensor but got NoneType` for the `bias` parameter. Full traceback: [traceback showing the line]." This is your first fix attempt.
```

- [ ] **Step 3: Update IC2 — sub-agent context (line 42)**

Replace:
```
**Context**: DA verification shows: "DA Verification FAIL — Amdahl sanity: actual E2E 3.2% but expected max 1.6% (f=0.08, s=1.25). Possible measurement error." This requires cross-checking constraints.md, Gate 5.2 raw data, and the Amdahl math. First encounter with this issue.
```
With:
```
**Context**: Your kernel validation sub-agent's DA verification shows: "DA Verification FAIL — Amdahl sanity: actual E2E 3.2% but expected max 1.6% (f=0.08, s=1.25). Possible measurement error." This requires cross-checking constraints.md, Gate 5.2 raw data from the sub-agent, and the Amdahl math. First encounter with this issue.
```

- [ ] **Step 4: Update IC4 — "Validator reported" → "Sub-agent returned" (line 82)**

Replace:
```
**Context**: Validator reported Gate 5.1a failure (tolerance exceeded for BS=32). You already tried one fix (adjusted tensor shapes) but the validator re-ran and reported the same gate failing again with a different error. This is your 2nd attempt at the same issue.
```
With:
```
**Context**: Your kernel validation sub-agent returned Gate 5.1a failure (tolerance exceeded for BS=32). You already tried one fix (adjusted tensor shapes) and re-spawned the sub-agent, which reported the same gate failing with a different error. This is your 2nd attempt at the same issue.
```

- [ ] **Step 5: Update IC5 — "validator reported" → "sub-agent returned" (line 100)**

Replace:
```
**Context**: Same Gate 5.1 issue from IC4. Delegate's recommended fix also didn't resolve it — the validator reported failure a third time. This is fix attempt #3.
```
With:
```
**Context**: Same Gate 5.1a issue from IC4. Delegate's recommended fix also didn't resolve it — the re-spawned sub-agent returned failure a third time. This is fix attempt #3.
```

- [ ] **Step 6: Update IC8 — "Validator sends" → "Sub-agent returns" (line 160)**

Replace:
```
**Context**: Validator sends: "Gate 5.1a FAIL: torch.allclose failed with atol=1e-5 for BF16 output." However, the debate plan specifies BF16 dtype, and `validation-defaults.md` says BF16 tolerance should be atol=1e-2, rtol=1e-2. The validator used an incorrect (too tight) tolerance.
```
With:
```
**Context**: Your kernel validation sub-agent returns: "Gate 5.1a FAIL: torch.allclose failed with atol=1e-5 for BF16 output." However, the debate plan specifies BF16 dtype, and `validation-defaults.md` says BF16 tolerance should be atol=1e-2, rtol=1e-2. The sub-agent used an incorrect (too tight) tolerance.
```

Update the expected behavior message (line 169):
Replace:
```
5. Message the validator: "Your Gate 5.1a used atol=1e-5, but validation-defaults.md specifies atol=1e-2 for BF16. Please re-run with correct tolerances."
```
With:
```
5. Re-spawn the sub-agent with a corrected spawn prompt specifying the right tolerances: "Use atol=1e-2 per validation-defaults.md for BF16."
```

- [ ] **Step 7: Update test-orchestrator.md — remove "impl-validator pairs" (lines 51, 97, 140)**

Replace all instances of `impl-champion + impl-validator pairs` with `impl-champion agents` (use replace_all).

Replace all instances of `impl-champion + impl-validator pair` with `impl-champion agent` (use replace_all).

- [ ] **Step 8: Update test-implementer.md Scenario I9 (line 242)**

Replace:
```
**Context**: The orchestrator-spawned validator reports Gate 5.1a results: "All correctness tests PASS. torch.allclose(atol=1e-3, rtol=1e-3) on [1, 32, 128]. Bit-exact match on all batch sizes. All tests passed on first run with zero numerical error."
```
With:
```
**Context**: Your kernel validation sub-agent returns Gate 5.1a results: "All correctness tests PASS. torch.allclose(atol=1e-3, rtol=1e-3) on [1, 32, 128]. Bit-exact match on all batch sizes. All tests passed on first run with zero numerical error."
```

Replace the expected behavior (lines 246-248):
```
**Expected behavior**: The champion should recognize trivially-passing validator tests as a signal that Gate 5.1a tests may be too weak. Message the validator asking them to strengthen their test suite — add adversarial cases (edge batch sizes, precision boundaries, CUDA graph capture/replay, varied sequence lengths). Do NOT accept trivially-passing Gate 5.1a as sufficient.
```
With:
```
**Expected behavior**: The champion should recognize trivially-passing sub-agent tests as a signal that Gate 5.1a tests may be too weak. Re-spawn the sub-agent with an explicit instruction to strengthen the test suite — add adversarial cases (edge batch sizes, precision boundaries, CUDA graph capture/replay, varied sequence lengths). Do NOT accept trivially-passing Gate 5.1a as sufficient.
```

- [ ] **Step 9: Update test-implementer.md Scenario I11 (lines 326-327)**

Replace:
```
6. Champion requests validator to re-validate gated version
```
With:
```
6. Champion spawns sub-agent for re-validation of gated kernel (5.1a + 5.2)
7. Champion re-runs sweep on gated code (5.1b + 5.3a + 5.3b)
```

- [ ] **Step 10: Commit**

```bash
git add .claude/skills/ammo/tests/agents/test-impl-champion.md \
       .claude/skills/ammo/tests/agents/test-orchestrator.md \
       .claude/skills/ammo/tests/agents/test-implementer.md
git commit -m "refactor(ammo): update test files for sub-agent validation model

test-impl-champion.md: validator messages -> sub-agent returns.
test-orchestrator.md: remove impl-validator pairs references.
test-implementer.md: update I9, I11 for sub-agent model."
```

---

### Task 8: Verification grep sweep

**Files:** None (read-only verification)

- [ ] **Step 1: Grep for eliminated patterns**

Run each grep and verify zero matches:

```bash
# 1. No VALIDATION_REQUEST
grep -r "VALIDATION_REQUEST" .claude/agents/ .claude/skills/ammo/

# 2. No dual-report/dual report
grep -ri "dual.report" .claude/agents/ .claude/skills/ammo/

# 3. No "orchestrator.*spawn.*validator"
grep -ri "orchestrator.*spawn.*validator" .claude/agents/ .claude/skills/ammo/

# 4. No orchestrator-validator messaging
grep -ri "team.lead.*validator\|validator.*team.lead" .claude/agents/ .claude/skills/ammo/

# 5. No "Spawned by orchestrator" in validator
grep -i "spawned by orchestrator" .claude/agents/ammo-impl-validator.md

# 6. No impl-validator in team structure
grep "impl-validator" .claude/skills/ammo/orchestration/parallel-tracks.md
```

Expected: ALL commands return 0 matches.

- [ ] **Step 2: Verify positive patterns exist**

```bash
# 1. Validator says "Spawned by impl-champion"
grep -i "spawned by impl-champion" .claude/agents/ammo-impl-validator.md

# 2. Champion has Agent() spawn for validator
grep "ammo-impl-validator" .claude/agents/ammo-impl-champion.md

# 3. Transcript monitor has "missing sub-agent spawn" check
grep -i "missing.*sub-agent.*spawn\|Missing.*validator.*sub-agent" .claude/agents/ammo-transcript-monitor.md

# 4. GATING_REQUIRED has explicit "re-run sweep" step
grep -i "re-run.*sweep\|re-runs sweep" .claude/skills/ammo/references/impl-track-rules.md
grep -i "re-run.*sweep\|re-runs sweep" .claude/skills/ammo/references/validation-defaults.md

# 5. Gate 5.2 in validator scope
grep "Gate 5.2" .claude/agents/ammo-impl-validator.md
```

Expected: ALL commands return 1+ matches.

- [ ] **Step 3: Commit verification results (if all pass)**

No commit needed — this is verification only. If any check fails, go back and fix the relevant file.
