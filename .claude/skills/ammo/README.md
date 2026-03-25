# AMMO - Automated Model Micro-Optimizer

A Claude Code skill for GPU kernel optimization in vLLM. Given a deployment target (model, hardware, dtype, TP), AMMO profiles the inference pipeline, identifies bottlenecks, debates optimization strategies adversarially, implements and validates candidates in parallel worktrees, and ships the ones that pass. The campaign loops until `f < min_e2e_improvement_pct` (mechanical threshold — no orchestrator discretion).

## Campaign Workflow

```
                            AMMO Campaign Pipeline
 ================================================================================

 The campaign is an iterative loop of 7 stages. Each iteration (round) discovers,
 debates, and implements optimizations. The loop repeats until the top bottleneck
 falls below the mechanical stop threshold (default 1.0%).

 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                          ROUND N (Stages 1-7)                                  │
 │                                                                                │
 │  Stage 1: Baseline Capture               Stage 2: Bottleneck Mining            │
 │  ┌──────────────────────────┐             ┌───────────────────────────────┐     │
 │  │  Lead scaffolds target    │             │  ammo-researcher subagent     │     │
 │  │  ammo-researcher profiles │── GATE ───>│  Grounded data only:          │     │
 │  │  under production parity  │ verify_    │  - Top-K kernels by GPU time  │     │
 │  │  (CUDA graphs + compile)  │ phase1_    │  - Component share f          │     │
 │  │                           │ baseline   │  - Bandwidth utilization      │     │
 │  │  Output: constraints.md   │ .py        │  - Physical ceilings          │     │
 │  └──────────────────────────┘             │  NO estimates, NO projections │     │
 │                                            │                               │     │
 │                                            │  Output: bottleneck_analysis  │     │
 │                                            └──────────────┬────────────────┘     │
 │                                                           │                      │
 │                                                     GATE: no ungrounded          │
 │                                                     estimates allowed             │
 │                                                           │                      │
 │                                                           v                      │
 │  Stage 3: Adversarial Debate                                                     │
 │  ┌──────────────────────────────────────────────────────────────────────────┐    │
 │  │  TeamCreate: ammo-round-{round_id}-{model_short}-{hardware}              │    │
 │  │                                                                          │    │
 │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │    │
 │  │  │Champion 1│  │Champion 2│  │Champion 3│  (2-4 ammo-champion agents)   │    │
 │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘                              │    │
 │  │       │              │              │                                    │    │
 │  │  Phase 0: Independent proposals + micro-experiments                     │    │
 │  │       │  GATE: Custom Kernel Mandate (config-only → reject)             │    │
 │  │  Rounds 1-2: Evidence ──> Critique ──> Rebuttal                         │    │
 │  │       │              │              │                                    │    │
 │  │       └──────────────┼──────────────┘                                   │    │
 │  │                      v                                                   │    │
 │  │  Lead scores via rubric ──> Select 2-3 winners ──> shutdown champions   │    │
 │  │  Output: debate/summary.md  (round team persists for Stages 4-5)       │    │
 │  └──────────────────────────────────────────────────────────┬───────────────┘    │
 │                                                              │                   │
 │                                                              v                   │
 │  Stages 4-5: Parallel Worktree Tracks (Adversarial Validation)                  │
 │  ┌──────────────────────────────────────────────────────────────────────────┐    │
 │  │                                                                          │    │
 │  │  STEP 1: Spawn impl-champion + impl-validator pairs into round team     │    │
 │  │  ┌─────────────────────────┐       ┌─────────────────────────┐          │    │
 │  │  │ Track A (worktree)      │       │ Track B (worktree)      │          │    │
 │  │  │ ammo-impl-champion      │       │ ammo-impl-champion      │          │    │
 │  │  │ + ammo-impl-validator   │       │ + ammo-impl-validator   │          │    │
 │  │  │ - Write kernel code     │       │ - Write kernel code     │          │    │
 │  │  │ - Independent validation│       │ - Independent validation│  GPU     │    │
 │  │  │ - E2E via sweep script  │       │ - E2E via sweep script  │ isolated │    │
 │  │  └─────────┬───────────────┘       └─────────┬───────────────┘          │    │
 │  │            │                                  │                          │    │
 │  │  STEP 2: Launch overlapped debate (round 2+ only, same round team)     │    │
 │  │  STEP 3: Monitor and gate (do NOT stop until ALL complete)              │    │
 │  │  - Gate each impl-champion (T9: compilation check, T10: state update)   │    │
 │  │  - Interleave debate moderation with impl gating                        │    │
 │  │  - Wait for ALL impl tracks + overlapped debate before Stage 6          │    │
 │  │  STEP 4: TeamDelete round team after all complete                       │    │
 │  │                                                                          │    │
 │  └──────────────────────────────────────────────────────────┬───────────────┘    │
 │                                                              │                   │
 │                                                              v                   │
 │  Stage 6: Integration Validation                                                 │
 │  ┌──────────────────────────────────────────────────────────────────────────┐    │
 │  │  Disjoint files?  ──yes──>  Cherry-pick both, re-run E2E ──>  SHIP     │    │
 │  │       │                                                                  │    │
 │  │       no (overlapping)                                                   │    │
 │  │       └──> Pick best E2E single candidate ──>  SHIP                     │    │
 │  │                                                                          │    │
 │  │  None pass?  ──>  round EXHAUSTED (not campaign-level)                  │    │
 │  └──────────────────────────────────────────────────────────┬───────────────┘    │
 │                                                              │                   │
 └──────────────────────────────────────────────────────────────┼───────────────────┘
                                                                │
                                                                v
 Stage 7: Campaign Evaluation
 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                                                                                 │
 │  IF SHIP:                              IF round EXHAUSTED:                      │
 │    1. Record shipped candidates          1. Record failed round                 │
 │    2. Update cumulative speedup          2. Mechanical threshold check           │
 │    3. Re-profile (new baseline)             on EXISTING profile (no re-profile) │
 │    4. Bottleneck mining on new baseline     │                                   │
 │    5. Mechanical threshold check             │                                   │
 │       │                                     │                                   │
 │       v                                     v                                   │
 │  top bottleneck f < 1.0%?              top bottleneck f < 1.0%?                 │
 │    YES → campaign_complete               YES → campaign_exhausted               │
 │    NO  → next round (Stage 3)            NO  → new debate from existing data    │
 │                                                                                 │
 │  On campaign_complete or campaign_exhausted:                                    │
 │    → Spawn report subagent (background) → REPORT.md                            │
 └─────────────────────────────────────────────────────────────────────────────────┘
```

## File Structure

```
.claude/skills/ammo/
├── SKILL.md                              # Main orchestration (campaign loop, task graph, non-negotiables)
├── README.md                             # This file (workflow diagram, test suite)
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
│   ├── nsys-profiling-guide.md           # nsys/ncu capture and CSV export
│   ├── fusion-feasibility-heuristics.md  # ROI math for fusion candidates
│   ├── gpu-configs.md                    # Hardware specs (SMEM, registers, TMA availability)
│   ├── optimization-techniques.md        # Technique catalog T1-T14
│   └── code-templates.md                 # GPU kernel patterns (token-major, expert-major)
├── eval/                                 # Skill evaluation pipeline
│   └── ...
├── report/
│   └── SKILL.md                          # Report generation skill (for T20 subagent)
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
├── ammo-delegate.md        # Research assistant for champions during debate (Sonnet model)
├── ammo-impl-champion.md   # Implements kernel in isolated worktree, aggregates validation results
└── ammo-impl-validator.md  # Independent validation: correctness tests, benchmarks, E2E sweeps
```

## Specialized Agents

| Agent | Role | Key Constraint |
|-------|------|----------------|
| **ammo-researcher** | Profiles baseline, mines bottlenecks | Cannot make feasibility estimates or E2E projections |
| **ammo-champion** | Proposes optimizations, argues in debate | Must back claims with micro-experiments |
| **ammo-delegate** | Research assistant for champions (optional) | No GPU benchmarks, no source modifications, 15-min timeout |
| **ammo-impl-champion** | Implements kernel in isolated worktree, aggregates validation results | Works in isolated worktree; frontmatter Stop hook (DA) enforces validation + Amdahl's sanity |
| **ammo-impl-validator** | Independent validation: correctness tests, benchmarks, E2E sweeps | Paired with impl-champion; writes own tests independently to prevent reward hacking |

The **lead** (main Claude session) orchestrates all stages, manages `state.json`, owns all gates, and never writes kernel code directly.

## Non-Negotiables

1. **Production parity** - CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, `VLLM_TORCH_COMPILE_LEVEL=0`
2. **vLLM baseline** - Compare against production kernel, not naive PyTorch
3. **Numerical correctness** - `torch.allclose()` mandatory in every test
4. **GPU sequencing** - E2E benchmarks sequential via flock. Must use `run_vllm_bench_latency_sweep.py` — raw `vllm bench latency` is FORBIDDEN
5. **Full-model E2E** - Download weights, never skip
6. **E2E delta math** - `improvement = f x kernel_speedup` (small `f` = small E2E, not a bug)
7. **Custom kernel mandate** - Stage 3 proposals MUST involve writing new or substantially modifying existing CUDA/Triton/CUTLASS kernel code. Config-only, flag-flipping, and parameter-tuning proposals are rejected at Phase 0.
8. **Autonomous campaign loop** — The orchestrator MUST NOT ask the user whether to continue. Stop condition is mechanical: `f < min_e2e_improvement_pct` → stop, else continue. No qualitative judgment overrides this.

## Hook Enforcement

| Hook Event | Script | Purpose |
|------------|--------|---------|
| **Stop** | `ammo-stop-guard.sh` | Blocks session end if campaign is active (file-based circuit breaker: blocks once, then allows) |
| **PreToolUse** (Bash) | `ammo-pretool-guard.sh` | Warns on `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, raw `vllm bench latency` (does not block) |
| **PreCompact** | `ammo-precompact.sh` | Saves campaign state checkpoint before compaction |
| **SessionStart** | `ammo-postcompact.sh` | Injects resume context after compaction |
| **WorktreeCreate** | `worktree-create-with-build.sh` | Sets up build environment in new worktrees |
| **WorktreeRemove** | `worktree-remove-cleanup.sh` | Cleans up worktree resources |

---

## Conformance Test Suite

48 scenarios across 4 test files verify that the orchestrator and all subagents correctly understand and follow the AMMO workflow.

| Test File | Agent | Scenarios | Count |
|-----------|-------|-----------|-------|
| [`tests/agents/test-orchestrator.md`](tests/agents/test-orchestrator.md) | Lead orchestrator | 20 (overlapped debate, resume, campaign eval, integration, violations) | 20 |
| [`tests/agents/test-researcher.md`](tests/agents/test-researcher.md) | ammo-researcher | 8 (grounded data, profiling strategy, production parity, steady-state) | 8 |
| [`tests/agents/test-champion.md`](tests/agents/test-champion.md) | ammo-champion | 10 (kernel mandate, micro-experiments, CUDA graphs, cache, delegation, debate) | 10 |
| [`tests/agents/test-implementer.md`](tests/agents/test-implementer.md) | ammo-impl-champion + ammo-impl-validator | 10 (baseline reuse, sweep script, parity, scope, Amdahl, build, contamination) | 10 |

### Run All Tests

```
Run the AMMO conformance test suite. Execute each test file in
.claude/skills/ammo/tests/agents/ (test-orchestrator.md, test-researcher.md,
test-champion.md, test-implementer.md). For each file, spawn a Sonnet subagent
that reads the referenced agent definitions, role-plays as the target agent,
and answers each scenario. Grade responses against "Expected Behavior".
Report pass/fail per scenario and overall.
```

### Run a Single Agent's Tests

```
Run the AMMO orchestrator conformance tests from
.claude/skills/ammo/tests/agents/test-orchestrator.md
```

Each test file is self-contained with: scenario descriptions, expected behavior, reference outputs from baseline runs (in `<details>` blocks), and grading criteria.

---

> **Note on the example below**: It predates the **Custom Kernel Mandate** (Non-Negotiables #7). Under current rules, OP-001 (Triton config autotuning — config-only, zero code changes) would be rejected at the Phase 0 eligibility gate. Only OP-002 (new fused CUDA kernel) would advance to debate rounds. The example is retained for workflow illustration.

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
