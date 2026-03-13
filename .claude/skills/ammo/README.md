# AMMO - Automated Model Micro-Optimizer

A Claude Code skill for GPU kernel optimization in vLLM. Given a deployment target (model, hardware, dtype, TP), AMMO profiles the inference pipeline, identifies bottlenecks, debates optimization strategies adversarially, implements and validates candidates in parallel worktrees, and ships the ones that pass.

## Workflow Diagram

```
                              AMMO 6-Stage Pipeline
 ============================================================================

 Stage 1: Baseline Capture                    Stage 2: Bottleneck Mining
 ┌─────────────────────────┐                  ┌──────────────────────────────┐
 │  Lead scaffolds target   │                  │  ammo-researcher subagent    │
 │  ammo-researcher profiles│─── GATE ────────>│  Grounded data only:         │
 │  under production parity │  verify_phase1   │  - Top-K kernels by GPU time │
 │  (CUDA graphs + compile) │  _baseline.py    │  - Component share f         │
 │                          │                  │  - Bandwidth utilization     │
 │  Output: constraints.md  │                  │  - Physical ceilings         │
 └─────────────────────────┘                  │  NO estimates, NO projections│
                                               │                              │
                                               │  Output: bottleneck_analysis │
                                               └──────────┬───────────────────┘
                                                          │
                                                    GATE: no ungrounded
                                                    estimates allowed
                                                          │
                                                          v
 Stage 3: Adversarial Debate
 ┌────────────────────────────────────────────────────────────────────────────┐
 │  TeamCreate: ammo-debate-{component}-{model}-{hw}                         │
 │                                                                            │
 │  ┌──────────┐  ┌──────────┐  ┌──────────┐                                │
 │  │Champion 1│  │Champion 2│  │Champion 3│  (2-4 ammo-champion agents)     │
 │  └────┬─────┘  └────┬─────┘  └────┬─────┘                                │
 │       │              │              │                                      │
 │  Phase 0: Independent proposals + micro-experiments                       │
 │       │              │              │                                      │
 │  Rounds 1-2: Evidence ──> Critique ──> Rebuttal                           │
 │       │              │              │                                      │
 │       └──────────────┼──────────────┘                                     │
 │                      v                                                     │
 │  Lead scores via rubric ──> Select 2-3 winners ──> TeamDelete             │
 │                                                                            │
 │  Output: debate/summary.md                                                │
 └────────────────────────────────────────────────────┬───────────────────────┘
                                                      │
                                                      v
 Stages 4-5: Parallel Worktree Tracks
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 │  Track A (worktree)              Track B (worktree)                       │
 │  ┌─────────────────────┐         ┌─────────────────────┐                  │
 │  │ ammo-implementer    │         │ ammo-implementer    │                  │
 │  │ - Write kernel code │         │ - Write kernel code │                  │
 │  │ - Correctness tests │         │ - Correctness tests │                  │
 │  │ - Kernel benchmarks │         │ - Kernel benchmarks │   GPU isolated   │
 │  │ - E2E benchmarks    │         │ - E2E benchmarks    │   E2E via flock  │
 │  └─────────┬───────────┘         └─────────┬───────────┘                  │
 │     Stop hook (DA):                Stop hook (DA):                        │
 │     Amdahl's check,               Amdahl's check,                        │
 │     baseline, parity              baseline, parity                        │
 │       GATE: compiles?                 GATE: compiles?                     │
 │            │                               │                              │
 │            v                               v                              │
 │  tracks/OP-001/                  tracks/OP-002/                           │
 │  validation_results.md           validation_results.md                    │
 └────────────────────────────────────────────────────┬───────────────────────┘
                                                      │
                                                      v
 Stage 6: Integration Validation
 ┌────────────────────────────────────────────────────────────────────────────┐
 │                                                                            │
 │  Disjoint files?  ──yes──>  Cherry-pick both, re-run E2E ──>  SHIP       │
 │       │                                                                    │
 │       no (overlapping)                                                     │
 │       │                                                                    │
 │       └──> Pick best E2E single candidate ──>  SHIP                       │
 │                                                                            │
 │  None pass?  ──>  EXHAUSTED                                               │
 └────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
.claude/skills/ammo/
├── SKILL.md                              # Main orchestration (6-stage pipeline, task graph, non-negotiables)
├── README.md                             # This file
├── orchestration/
│   ├── debate-protocol.md                # Stage 3: team setup, phases, convergence criteria
│   ├── parallel-tracks.md                # Stages 4-5: worktree creation, GPU assignment, pass criteria
│   └── integration-logic.md              # Stage 6: conflict detection, cherry-pick, decision matrix
├── references/
│   ├── debate-scoring-rubric.md          # 6-criterion weighted scoring (min 5.0 to advance)
│   ├── e2e-delta-math.md                 # f x kernel_speedup = E2E improvement
│   ├── cudagraph-safety.md               # Stream usage, no allocations during capture
│   ├── e2e-latency-guide.md              # vllm bench latency methodology
│   ├── validation-defaults.md            # Correctness tolerances, gate thresholds
│   ├── validator-troubleshooting.md      # Investigation playbook (max 3 hypothesis cycles)
│   ├── da-audit-checklist.md             # Devil's advocate audit investigation template
│   ├── nsys-profiling-guide.md           # nsys/ncu capture and CSV export
│   ├── fusion-feasibility-heuristics.md  # ROI math for fusion candidates
│   ├── gpu-configs.md                    # Hardware specs (SMEM, registers, TMA availability)
│   ├── optimization-techniques.md        # Technique catalog T1-T14
│   └── code-templates.md                 # GPU kernel patterns (token-major, expert-major)
└── scripts/
    ├── new_target.py                     # Scaffold artifact directory + state.json
    ├── collect_env.py                    # Capture environment snapshot
    ├── verify_phase1_baseline.py         # Stage 1→2 gate
    ├── verify_validation_gates.py        # Stage 5 gate (per-track)
    ├── run_vllm_bench_latency_sweep.py   # E2E benchmarks with GPU lock (flock)
    └── generate_validation_report.py     # Structured reporting

.claude/agents/
├── ammo-researcher.md      # Profiling + bottleneck mining (grounded data only, NO estimates)
├── ammo-champion.md        # Debate: proposes candidates, runs micro-experiments, argues with data
└── ammo-implementer.md     # Implements kernel + runs full validation (correctness, kernel bench, E2E) in isolated worktree
```

## Specialized Agents

| Agent | Role | Key Constraint |
|-------|------|----------------|
| **ammo-researcher** | Profiles baseline, mines bottlenecks | Cannot make feasibility estimates or E2E projections |
| **ammo-champion** | Proposes optimizations, argues in debate | Must back claims with micro-experiments |
| **ammo-implementer** | Implements kernel + runs full validation (correctness, kernel bench, E2E) | Works in isolated worktree; frontmatter Stop hook (DA) enforces validation + Amdahl's sanity |

The **lead** (main Claude session) orchestrates all stages, manages `state.json`, owns all gates, spawns DA audit agents after each implementer completes, and never writes kernel code directly.

## Non-Negotiables

1. **Production parity** - CUDA graphs + torch.compile in ALL measurements
2. **vLLM baseline** - Compare against production kernel, not naive PyTorch
3. **Numerical correctness** - `torch.allclose()` mandatory in every test
4. **GPU sequencing** - E2E benchmarks sequential via flock
5. **Full-model E2E** - Download weights, never skip
6. **E2E delta math** - `improvement = f x kernel_speedup` (small `f` = small E2E, not a bug)

> **Note**: The example below predates the **Custom Kernel Mandate** (Non-Negotiables #7). Under current rules, OP-001 (Triton config autotuning — config-only, zero code changes) would be rejected at the Phase 0 eligibility gate. Only OP-002 (new fused CUDA kernel) would advance to debate rounds. The example is retained for workflow illustration.

## Example Session: Qwen3.5-35B-A3B-FP8 on L40S (TP=2)

Below is a walkthrough of a completed AMMO session from `kernel_opt_artifacts/auto_Qwen3.5-35B-A3B-FP8_L40S_fp8_tp2/`.

### Invocation

```
User: "Use ammo for Qwen/Qwen3.5-35B-A3B-FP8 on L40S TP=2"
```

### Stage 1-2: Baseline + Bottleneck Mining

The `ammo-researcher` profiled with nsys under production parity (CUDA graphs + torch.compile level 3) on 2x L40S GPUs. Key findings from `constraints.md`:

- **Top kernels by GPU time**: `w8a8_block_fp8_triton_block_scaled_mm` (f=0.231), `fused_moe_triton` (f=0.250), attention (f=0.141)
- **Hardware**: 4x L40S (sm_89), 44.4 GiB each, 142 SMs
- **Missing Triton configs**: 7 shape/dtype combos had no L40S-specific tuned configs

### Stage 3: Adversarial Debate

Three champions debated. After 1 round (shortened — C2 and C3 converged on the same candidate independently):

| Candidate | Champion | Target | Score | E2E Estimate |
|-----------|----------|--------|-------|-------------|
| ~~**OP-001**: Triton config autotuning~~ | C1 | ~~Dense FP8 matmul (f=0.231) + MoE GEMM (f=0.250)~~ | ~~8.15~~ | ~~4-8%~~ | *(Would be rejected at Phase 0 — config-only, no kernel code)* |
| **OP-002**: SiLU + block-FP8 quant fusion | C2+C3 | Activation + quant chain (f=0.051) | **7.20** | 1.5-3% |

Key debate moments:
- C2/C3 correctly critiqued C1's proxy kernel micro-experiment (missing block-scale dequant), revising E2E from 9.6-12.8% down to 4-8%
- C1 critiqued C2/C3's small component share (f=0.049), but the fusion was still viable as a complementary win
- Both selected because they target **completely different components** (zero file overlap)

### Stages 4-5: Parallel Tracks

Two worktrees ran in parallel:

**Track OP-001** (config autotuning) — ~~would be rejected under Custom Kernel Mandate~~:
- Generated 7 Triton JSON configs for L40S-specific shapes
- Zero code changes — config files only *(now disqualifying)*
- Kernel speedup: 1.03-3.25x across shapes
- E2E: +1.9-5.2% (BS=4-64), -0.7% BS=1 (within noise)
- Verdict: **CONDITIONAL PASS** *(would not reach this stage under current rules)*

**Track OP-002** (SiLU + FP8 quant fusion):
- New fused CUDA kernel across 5 source files
- 42/42 correctness tests passing
- Kernel speedup: 2.2x
- CUDA graph safe
- Verdict: **PASS**

### Stage 6: Integration

Conflict analysis: **No conflicts**. OP-001 adds JSON config files, OP-002 modifies CUDA/Python files.

Cherry-picked both onto `ammo/combined` branch. Combined E2E results:

| Batch Size | Baseline (s) | Combined (s) | Improvement |
|-----------|-------------|-------------|-------------|
| BS=1 | 3.762 | 3.415 | **+9.2%** |
| BS=4 | 5.230 | 5.071 | +3.0% |
| BS=8 | 6.866 | 6.600 | +3.9% |
| BS=16 | 9.616 | 9.375 | +2.5% |
| BS=32 | 14.198 | 13.742 | +3.2% |
| BS=64 | 20.826 | 19.951 | +4.2% |

**Average: +4.3% E2E improvement. Zero regressions. 84/84 correctness tests pass.**

Final decision: **SHIP** (branch: `ammo/combined`, 3 commits)

### Artifact Directory Layout

```
kernel_opt_artifacts/auto_Qwen3.5-35B-A3B-FP8_L40S_fp8_tp2/
├── state.json                    # Stage: complete, Status: SHIP
├── target.json                   # Deployment target + benchmark config
├── constraints.md                # Baseline profile (environment, top kernels, model arch)
├── bottleneck_analysis.md        # Grounded bottleneck data
├── optimization_plan.md          # Planned optimizations
├── implementation_notes.md       # Implementation details
├── validation_results.md         # Overall validation
├── integration.md                # Integration strategy
├── integration_results.md        # Final combined results
├── nsys/                         # Nsight Systems traces + CSV exports
├── investigation/                # Any investigation artifacts
├── blockers/                     # Blocker documentation (if any)
├── debate/
│   ├── proposals/                # Per-champion initial proposals
│   ├── round_1/                  # Critiques and rebuttals
│   ├── micro_experiments/        # Champion micro-experiment scripts + results
│   └── summary.md                # Scoring, winner selection, rationale
└── tracks/
    ├── OP-001/                   # Config autotuning track
    │   ├── *.json                # 7 tuned Triton configs
    │   ├── tune_dense_fp8.py     # Tuning script
    │   ├── tune_moe_fp8.py       # MoE tuning script
    │   ├── test_correctness_op001.py
    │   └── validation_results.md
    └── OP-002/                   # SiLU+quant fusion track
        └── validation_results.md
```

---

## Changes from Deprecated Version (ammo-old)

The skill was rewritten from the ground up. Here is what changed:

### Scope: MoE-only -> General GPU Kernels

| | Old | New |
|---|---|---|
| **Name** | Automated **MoE** Model Optimizer | Automated Model **Micro**-Optimizer |
| **Scope** | MoE kernel fusion only (fused_moe, monokernels) | Any GPU kernel bottleneck (matmul, attention, activation, quantization, ...) |
| **Target** | Fixed fusion boundary decision tree (cooperative monokernel / hybrid / split) | Bottleneck-driven — profiles first, decides technique from data |

### Architecture: Monolithic -> Specialized Agents

| | Old | New |
|---|---|---|
| **Agents** | Generic `general-purpose` Task subagents | 3 specialized types: `ammo-researcher`, `ammo-champion`, `ammo-implementer` + orchestrator-spawned DA audit |
| **Roles** | Subagent does entire phase | Each agent has strict role boundaries (researcher can't estimate, implementer implements + validates, DA audits methodology) |
| **Context** | All context passed via task prompts | Agents have dedicated `.claude/agents/` definitions with built-in guardrails |

### Decision Making: LLM Council -> Adversarial Debate

| | Old | New |
|---|---|---|
| **Candidate selection** | Orchestrator picks optimization route via decision tree | 2-4 champion agents propose independently, debate with micro-experiments, scored by rubric |
| **De-risking** | Optional LLM Council for high-risk decisions | Mandatory adversarial debate with evidence requirements |
| **Estimates** | Champions and researcher could both estimate | Strict separation: researcher provides grounded data only, champions derive estimates from their own micro-experiments |

### Execution: Sequential -> Parallel Worktrees

| | Old | New |
|---|---|---|
| **Tracks** | Single optimization attempt, sequential phases | 2-3 candidates implemented in parallel git worktrees |
| **Isolation** | Shared codebase | Each track gets `git worktree add .claude/worktrees/ammo-track-{id}` |
| **GPU management** | Manual | Kernel benchmarks on separate GPUs, E2E sequential via `flock` |
| **Integration** | Manual merge | Automated conflict detection + cherry-pick + re-validation |

### Workflow: 5 Phases -> 6 Stages

| Old (5 phases) | New (6 stages) |
|---|---|
| Phase 1: Constraints | Stage 1: Baseline Capture |
| Phase 2: Planning | Stage 2: Bottleneck Mining (grounded data only) |
| *(no equivalent)* | **Stage 3: Adversarial Debate** (new) |
| Phase 3: Implementation | Stage 4: Parallel Implementation (in worktrees) |
| Phase 4: Validation | Stage 5: Parallel Validation (per-track) |
| Phase 5: Integration | Stage 6: Integration Validation |

### Removed

- **MoE-specific references**: `route-selection-decision-tree.md`, `architecture-pattern.md`, `algorithmic-branching.md`, `tiling-config.md`, `expert-grouping.md`, `hybrid-large-grid-fusion.md`, `router-design.md`, `moe-parameters-template.md`, `profiling-launch-vs-kernel.md`, `scope-and-support.md`, `state-schema.md`
- **Hardcoded examples**: `LLAMA4_MONOKERNEL_PATCH.md`, `MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md`, `QWEN3_BASELINE.md`, `MODELS_COMPARISON.md`, `W1_EPILOGUE_FUSION.md`
- **LLM Council**: Replaced by adversarial debate
- **Task prompts file**: `orchestration/task-prompts.md` replaced by agent definitions in `.claude/agents/`
- **Verification commands**: `orchestration/verification-commands.md` replaced by gate scripts

### Added

- **Debate infrastructure**: `orchestration/debate-protocol.md`, `references/debate-scoring-rubric.md`, debate artifact structure
- **Parallel execution**: `orchestration/parallel-tracks.md`, `orchestration/integration-logic.md`
- **New references**: `e2e-latency-guide.md`, `validator-troubleshooting.md`
- **New scripts**: `verify_validation_gates.py` (per-track), `generate_validation_report.py`
- **Agent definitions**: 3 specialized agents in `.claude/agents/ammo-*.md` (implementer now includes validation; DA audit spawned by orchestrator)
- **Lead rules**: Merged into `SKILL.md` Lead Role section (prohibits lead from writing kernel code)

---

## Known Bug: SubagentStop Hook Matcher Bypass (Claude Code)

**Severity**: Critical — causes infinite hook cascade, process freeze, 100%+ CPU
**Affected versions**: 2.1.63, 2.1.68, 2.1.69 (all tested; likely all versions)
**Status**: Unfixed upstream — file at https://github.com/anthropics/claude-code/issues
**Discovered**: 2026-03-05 during AMMO session `bd4b9a66`

### Summary

The `SubagentStop` hook `matcher` field in `settings.local.json` is **never evaluated**. All `SubagentStop` hooks fire on every subagent stop, regardless of the matcher value. When the hook itself spawns a subagent (e.g., `type: "agent"`) and that subagent fails, its stop re-triggers the same hook, creating an infinite cascade.

### Root Cause

In the Claude Code hook matching function (extracted from binary via `strings`):

```javascript
// The query for SubagentStop is set from agent_type
case "SubagentStop": I = L.agent_type; break;

// Filtering logic:
let K = (I
  ? f.filter((J) => !J.matcher || matchFn(I, J.matcher))  // query truthy → filter by matcher
  : f                                                        // query falsy → return ALL hooks
).flatMap(...)
```

`L.agent_type` is **always an empty string** `""` for `SubagentStop` events (even though `SubagentStart` correctly populates it). Since `""` is falsy in JavaScript, the ternary skips the `.filter()` branch and returns all hooks unfiltered. The matcher `"ammo-implementer"` is never checked.

### Cascade Mechanism

1. `ammo-implementer` subagent stops → `SubagentStop` fires
2. Hook spawns agent to verify validation → agent calls API with invalid Bedrock model ID → 400 error
3. Hook agent stops → triggers `SubagentStop` again (matcher bypassed)
4. New hook agent spawned → same error → stops → triggers hook → **infinite loop**
5. Each iteration accumulates V8 heap objects → RSS grows to 5+ GB → GC thrashing → 100%+ CPU → terminal freezes

In the observed incident: **14,319 hook agents** were spawned over 2 hours before the process was killed.

### Workaround

Since the matcher doesn't work, add a guard **inside the hook prompt** to break the cycle:

```json
{
  "SubagentStop": [{
    "matcher": "ammo-implementer",
    "hooks": [{
      "type": "agent",
      "prompt": "FIRST: Check the agent_id in $ARGUMENTS. If it starts with 'hook-agent-', return {\"ok\": true} immediately — do NOT proceed. Only verify validation for real implementer agents.\n\n... (rest of prompt)",
      "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
      "timeout": 600
    }]
  }]
}
```

Also ensure the `model` field uses a valid Bedrock model ARN (not `claude-sonnet-4-6`) to prevent the 400 error that amplifies the cascade.

### Detection

Signs of an active cascade:
- Claude Code terminal frozen, high CPU (`ps aux | grep claude` shows 100%+ CPU)
- Debug log growing rapidly: `tail -f ~/.claude/debug/<session-id>.txt` shows repeating `SubagentStop` + `hook-agent-*` entries
- RSS oscillating wildly (GC thrashing): `watch -n1 'grep VmRSS /proc/<pid>/status'`

Recovery: `kill -9 <pid>` (SIGTERM may not work due to blocked event loop)

### Adversarial Verification

An independent adversarial review (120 tool calls across the Claude Code binary and debug logs) confirmed all findings:

**Why `agent_type` is empty — asymmetric design:**
- `SubagentStart` has a **dedicated function** (`nmA`) that explicitly passes the agent name as `agent_type`:
  ```javascript
  async function* nmA(H, $, A, L) {
      let D = {..., hook_event_name: "SubagentStart", agent_id: H, agent_type: $};
      yield* rx({hookInput: D, matchQuery: $, ...});
  }
  ```
- `SubagentStop` uses a **generic stop-hook function** (`$QA`) that reads `agentType` from the execution context — but the context's `agentType` is `undefined` by the time stop hooks fire, so `M ?? ""` coalesces to `""` (falsy), bypassing the filter.

**Empirical evidence (exhaustive):**
- **28,638 / 28,638** `SubagentStop` events in the incident debug log had empty `agent_type` — zero exceptions
- Every single one matched `"Matched 1 unique hooks for query 'no match query'"` despite the `"matcher": "ammo-implementer"` filter
- `SubagentStart` in the same session correctly showed `query: ammo-researcher`

**Built-in vs custom agents:**
- Built-in agent types (e.g., `claude-code-guide`) DO populate `agent_type` on stop — observed in a separate healthy session where `SubagentStop with query: claude-code-guide` appeared
- Custom agents defined in `.claude/agents/` (e.g., `ammo-implementer`) do NOT — `agentType` is lost during context propagation from the agent definition to the stop-hook execution path

**Matcher function (`QPM`) is never reached:**
- The ternary `(I ? f.filter(...) : f)` short-circuits when `I` is falsy
- Even if `QPM` were called with `QPM("", "ammo-implementer")`, it would correctly return `false` — the bug is upstream of the matcher, not in it
