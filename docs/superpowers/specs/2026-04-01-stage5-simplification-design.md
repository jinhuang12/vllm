# AMMO Stage 5 Simplification: Validator as Champion Sub-Agent

## Problem

The current Stage 5 validation architecture uses the `ammo-impl-validator` as an **orchestrator-spawned independent team member**. This creates unnecessary complexity:

1. **Orchestrator coordination overhead**: Champion sends `VALIDATION_REQUEST` to orchestrator, orchestrator spawns validator into the round team, orchestrator relays results.
2. **Dual-reporting**: Validator must SendMessage to both champion AND orchestrator.
3. **GPU coordination via messaging**: Champion and validator coordinate GPU access through SendMessage since both are team members competing for pool slots.
4. **Team member slot**: Validator occupies a team member slot in the round team, complicating the team structure diagram and overlapped debate communication isolation.
5. **Gate 5.2 orphaned**: Kernel speedup benchmarking (Gate 5.2) is awkwardly described as "champion-owned independently" with no clear agent responsible for writing and running the benchmark.

The sweep script now handles Gates 5.1b + 5.3a + 5.3b in a single invocation. The validator's only unique contribution is Gate 5.1a (independent kernel correctness tests). Adding Gate 5.2 to the validator's scope creates a clean "kernel-level validation" sub-agent.

## Design

### New Gate Ownership Model

| Gate | Owner | Method |
|------|-------|--------|
| 5.1a (kernel correctness) | Sub-agent | Independent tests from debate plan |
| 5.2 (kernel speedup) | Sub-agent | Kernel benchmark under CUDA graphs, production parity |
| 5.1b (E2E correctness) | Champion | Sweep `--verify-correctness` |
| 5.3a (kernel proof) | Champion | Sweep `--nsys-profile` |
| 5.3b (E2E latency) | Champion | Sweep E2E latency measurement |

**Split principle**: Sub-agent = kernel-level validation. Champion = E2E-level validation.

### Execution Flow Within a Track

```
1. Champion implements kernel, commits
2. Champion spawns sub-agent via Agent(subagent_type="ammo-impl-validator")
   -> Sub-agent runs Gate 5.1a (kernel correctness tests under CUDA graphs)
   -> Sub-agent runs Gate 5.2 (kernel speedup benchmark under production parity)
   -> Sub-agent returns structured results (Agent tool return, NOT SendMessage)
3. Champion evaluates sub-agent results:
   - If 5.1a FAIL -> fix kernel, re-spawn sub-agent (no wasted E2E sweep)
   - If 5.1a PASS -> proceed to sweep
4. Champion runs ONE sweep: --labels opt --verify-correctness --nsys-profile --baseline-from $STAGE1_DIR
   -> Gate 5.1b (Phase 1 correctness): stops on failure (exit code 3)
   -> Gate 5.3b (Phase 2 latency): per-BS verdicts
   -> Gate 5.3a (kernel proof): nsys trace verified post-sweep
5. Champion combines all gate results into validation_results.md
6. Champion reports final verdict to orchestrator (SendMessage)
```

**Key improvement**: Kernel correctness (5.1a) gates the E2E sweep. If 5.1a fails, the champion fixes before burning 15-30 min on a sweep. Currently this sequencing is not enforced.

### Sub-Agent Spawn Template

The champion spawns the validator as a foreground sub-agent:

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

The sub-agent returns results directly via the Agent tool return value. No SendMessage, no dual-reporting, no team membership.

### GATING_REQUIRED Flow

When per-BS verdicts show mixed results (some PASS + some REGRESSED):

```
1. Initial sweep shows mixed per-BS verdicts -> track enters GATING_REQUIRED
2. Champion spawns sub-agent for crossover probing:
   -> Sub-agent runs kernel sweep across BS range
   -> Sub-agent returns crossover threshold
3. Champion implements gating mechanism (env var VLLM_{OP_NAME}=0, dispatch condition)
4. Champion spawns sub-agent for re-validation of gated kernel (5.1a + 5.2)
5. Champion re-runs sweep on gated code (5.1b + 5.3a + 5.3b) -- explicit step
6. If all BS PASS/NOISE -> GATED_PASS. If not -> FAIL.
```

One gating attempt per track. No nested gating.

### Orchestrator Role Change

**Before**: Orchestrator spawns validator, receives dual-report, coordinates champion-validator interaction.

**After**: Orchestrator is aware but hands-off:
- Orchestrator spawns impl-champion only (no validator spawn)
- Champion manages the sub-agent internally
- Champion reports final verdict to orchestrator via SendMessage
- Orchestrator reads `validation_results.md` for gate decisions
- Orchestrator's task graph simplified: `T8b_val_{id}` entry removed, `T8c_{id}` simplified

### Transcript Monitor Safety Net

The transcript monitor gains a new CRITICAL check for implementation stage:

> **Missing validator sub-agent spawn**: Champion commits implementation and proceeds to the E2E sweep (or signals track completion) without first spawning the validator sub-agent for Gates 5.1a + 5.2. Evidence: sweep command (`run_vllm_bench_latency_sweep.py`) appears in transcript with no prior `Agent(subagent_type="ammo-impl-validator")` call. Severity: **CRITICAL** -- kernel correctness was never independently verified.

### Independence Principle (Preserved)

The sub-agent still writes its OWN tests derived from the debate plan (`debate/summary.md`), not from the champion's implementation. The adversarial separation is maintained by:

1. Sub-agent derives test methodology from the optimization plan, not the implementation
2. Sub-agent does not read or execute the champion's test/benchmark scripts
3. Sub-agent writes to `{artifact_dir}/tracks/{op_id}/validator_tests/` (isolated output dir)

The structural guarantee weakens slightly (champion spawns sub-agent vs orchestrator spawning independently), but the sub-agent's instructions explicitly mandate independence. The transcript monitor provides a backstop.

### DA Verification Checks (Preserved)

The sub-agent retains the DA verification checks currently in the validator:
1. **Cross-track awareness**: Check `state.json` for other tracks with C++ changes that could contaminate
2. **Scope adherence**: Compare files modified vs debate plan scope

These are included in the sub-agent's return results for the champion to incorporate into `validation_results.md`.

## What Gets Eliminated

| Removed | Rationale |
|---------|-----------|
| `VALIDATION_REQUEST` protocol | Champion spawns sub-agent directly |
| Validation Spawn Protocol section (parallel-tracks.md) | Orchestrator no longer spawns validator |
| Dual-reporting (validator -> champion + orchestrator) | Sub-agent returns to champion only |
| SendMessage GPU coordination between champion/validator | Sequential: sub-agent runs first, then champion runs sweep |
| Validator as team member in round team | Sub-agent, not team member |
| Orchestrator task graph entries for validator spawn/coordination | Simplified task graph |

## What Gets Preserved

| Preserved | How |
|-----------|-----|
| Independence principle | Sub-agent writes OWN tests from debate plan |
| Adversarial separation | Sub-agent doesn't read champion's scripts |
| DA verification checks | Moved into sub-agent scope |
| Transcript monitor oversight | Still monitors champion; gains new "missing spawn" check |
| Production parity | All benchmarks under CUDA graphs + torch.compile |
| All 5 gates | Same gates, cleaner ownership split |

## File Changes

### Agent Files

| File | Change | Severity |
|------|--------|----------|
| `ammo-impl-validator.md` | Rewrite: team member -> sub-agent. Add Gate 5.2 scope. Remove SendMessage/dual-reporting. Remove orchestrator spawn references. Keep independence principle and DA checks. | MAJOR |
| `ammo-impl-champion.md` | Replace VALIDATION_REQUEST with Agent() spawn template. Add Gate 5.2 delegation. Update flow: spawn sub-agent -> evaluate -> run sweep. Remove "orchestrator spawns validator" references (lines 10, 112-127, 131). | MAJOR |
| `ammo-delegate.md` | Line 79: clarify kernel-benchmark-template ownership now that sub-agent handles Gate 5.2. Lines 94-99: update post-merge testing references. | MINOR |
| `ammo-transcript-monitor.md` | Line 105: change stop condition from VALIDATION_REQUEST to track completion. Lines 207-208: update gate completeness (sub-agent runs 5.1a+5.2, champion runs 5.1b+5.3). Line 238: update "sharing scripts with validator" rule. Add new CRITICAL check: champion skips validator sub-agent spawn. | MODERATE |

### Orchestration Files

| File | Change | Severity |
|------|--------|----------|
| `SKILL.md` | Stages 4-5 (lines 219-234): remove validator spawn, simplify to "champion manages validation internally". Task graph: remove T8b_val, simplify T8c. Line 49: update stage description. Line 281: simplify. Line 357: update inline DA section. | MAJOR |
| `parallel-tracks.md` | Remove Validation Spawn Protocol (lines 121-153). Update team structure diagram (remove validator slot, lines 16-22). Rewrite Two Layers -> single champion-owned flow with sub-agent. Update spawn prompt (fix Issue A: add missing --labels opt, --nsys-profile, --correctness-mode). | MAJOR |

### Reference Files

| File | Change | Severity |
|------|--------|----------|
| `impl-track-rules.md` | Rewrite Two Layers (lines 26-39) -> champion owns all validation with sub-agent for kernel-level. Update GATING_REQUIRED (lines 41-69): add explicit "re-run sweep" step (Issue B), update crossover probing ownership. Update source modification rules. | MAJOR |
| `validation-defaults.md` | Fix 5.3a/5.3b ordering language (Issue C: "results only valid if 5.3a passes"). Update Gate 5.1a description to note sub-agent ownership. Fix GATING_REQUIRED section (Issue B). | MODERATE |
| `crossover-probing.md` | Clarify ownership: champion spawns sub-agent for kernel crossover sweep, champion evaluates E2E confirmation. | MODERATE |
| `kernel-benchmark-template.py` | Change CHAMPION_FILL markers to generic FILL (sub-agent now uses this for Gate 5.2). Update header comment. | MINOR |

### Test Files

| File | Change | Severity |
|------|--------|----------|
| `tests/agents/test-impl-champion.md` | Update 10 scenarios: validator message handling -> sub-agent spawn/result handling. Remove two-layer model references. Add scenario for champion forgetting to spawn sub-agent. | MAJOR |
| `tests/agents/test-orchestrator.md` | Remove validator spawn details from scenarios. Simplify Stages 4-5 test expectations. | MINOR |
| `tests/agents/test-implementer.md` | Update validator references to match sub-agent model (already partially rewritten in prior harmonization). | MINOR |

### Files NOT Changed

| File | Reason |
|------|--------|
| `ammo-researcher.md` | Zero validator references (confirmed by audit) |
| `ammo-resolver.md` | Zero validator references (confirmed by audit) |
| `ammo-champion.md` | Debate agent, no validation involvement |
| `gpu-pool.md` | Reservation pattern unchanged |
| `cudagraph-safety.md` | Capture checklist unchanged |
| `code-templates.md` | Gating dispatch patterns unchanged |
| `debate-protocol.md` | No validator references in debate flow |
| `integration-logic.md` | Uses champion results, no validator interaction |
| `e2e-delta-math.md` | Math unchanged |
| `e2e-latency-guide.md` | Methodology unchanged |

## Verification

After implementation, verify:
1. Grep for "VALIDATION_REQUEST" across all `.claude/` files -> zero hits
2. Grep for "dual.report" across all `.claude/` files -> zero hits
3. Grep for "orchestrator.*spawn.*validator" across all `.claude/` files -> zero hits
4. Grep for "team.lead.*validator\|validator.*team.lead" -> zero hits (no orchestrator-validator messaging)
5. `ammo-impl-validator.md` frontmatter no longer says "Spawned by orchestrator"
6. `parallel-tracks.md` team structure diagram has no `impl-validator` line
7. Transcript monitor has "missing validator spawn" check
8. GATING_REQUIRED workflow has explicit "re-run sweep" step in all files
