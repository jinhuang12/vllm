# AMMO - Automated Model Micro-Optimizer

An AI-driven fully autonomous optimization pipeline built specifically for vLLM. AMMO takes a specific vLLM deployment configuration (i.e model, hardware type, vLLM configuration) and systematically finds and implements kernel-level performance optimizations to reduce inference latency.

## Campaign Workflow

```
                            AMMO Campaign Pipeline
 ================================================================================

 The campaign is an iterative loop of 7 stages. Each iteration (round) discovers,
 debates, and implements optimizations. The loop repeats until the top bottleneck
 falls below the mechanical stop threshold (see `references/validation-defaults.md`).

 ┌─────────────────────────────────────────────────────────────────────────────────┐
 │                          ROUND N (Stages 1-7)                                  │
 │                                                                                │
 │  Stage 1: Baseline Capture               Stage 2: Bottleneck Mining            │
 │  ┌──────────────────────────┐             ┌───────────────────────────────┐     │
 │  │  Lead scaffolds target    │             │  ammo-researcher subagent     │     │
 │  │  ammo-researcher profiles │────────────>│  Grounded data only:          │     │
 │  │  under production parity  │  task_type: │  - Top-K kernels by GPU time  │     │
 │  │  (CUDA graphs + compile)  │  baseline   │  - Component share f          │     │
 │  │                           │             │  - Bandwidth utilization      │     │
 │  │  Output: constraints.md   │             │  - Physical ceilings + ncu    │     │
 │  └──────────────────────────┘             │  NO estimates, NO projections │     │
 │                                            │                               │     │
 │                                            │  Output: bottleneck_analysis  │     │
 │                                            └──────────────┬────────────────┘     │
 │                                                           │                      │
 │                                                   T5 GATE: artifacts             │
 │                                                                                  │
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
 │  │       │  GATE: Authored-Mechanism Mandate (retuned-constant → reject)   │    │
 │  │  Round 1 (+cond. 2): Evidence ──> Critique ──> Rebuttal              │    │
 │  │       │              │              │                                    │    │
 │  │       └──────────────┼──────────────┘                                   │    │
 │  │                      v                                                   │    │
 │  │  Lead scores via rubric ──> Select 2-3 winners ──> shutdown champions   │    │
 │  │  Output: debate/summary.md  (round team persists for Stages 4-5)       │    │
 │  └──────────────────────────────────────────────────────────┬───────────────┘    │
 │                                                              │                   │
 │                                                              v                   │
 │  Stages 4-5: Parallel Worktree Tracks (Self-Validated Gates)                    │
 │  ┌──────────────────────────────────────────────────────────────────────────┐    │
 │  │                                                                          │    │
 │  │  STEP 1: Spawn impl-champion per track into round team               │    │
 │  │  ┌─────────────────────────┐       ┌─────────────────────────┐          │    │
 │  │  │ Track A (worktree)      │       │ Track B (worktree)      │          │    │
 │  │  │ ammo-impl-champion      │       │ ammo-impl-champion      │          │    │
 │  │  │ - Write kernel code     │       │ - Write kernel code     │          │    │
 │  │  │ - Self-run correctness &│       │ - Self-run correctness &│          │    │
 │  │  │   speedup checks        │       │   speedup checks        │  GPU     │    │
 │  │  │   (own tests/bench)     │       │   (own tests/bench)     │ isolated │    │
 │  │  │ - E2E via sweep script  │       │ - E2E via sweep script  │          │    │
 │  │  └─────────┬───────────────┘       └─────────┬───────────────┘          │    │
 │  │            │                                  │                          │    │
 │  │  STEP 2: Monitor and gate (do NOT stop until ALL complete)              │    │
 │  │  - Gate each impl-champion (T9: compilation check, T10: state update)   │    │
 │  │  - Wait for ALL impl tracks to reach terminal status before Stage 6     │    │
 │  │                                                                          │    │
 │  │  STEP 4: TeamDelete round team after all complete                       │    │
 │  │                                                                          │    │
 │  └──────────────────────────────────────────────────────────┬───────────────┘    │
 │                                                              │                   │
 │                                                              v                   │
 │  Stage 6: Integration Validation                                                 │
 │  ┌──────────────────────────────────────────────────────────────────────────┐    │
 │  │  Disjoint files?  ──yes──>  Cherry-pick both, re-run E2E ──>  SHIP     │    │
 │  │       │                                                                  │    │
 │  │       no (file conflicts)                                                │    │
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
 │  top bottleneck f < threshold?              top bottleneck f < threshold?                 │
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
│   ├── nsys-profiling-guide.md           # nsys Stage 2 workflow, trace-backend matrix, targeted NCU
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
    ├── verify_validation_gates.py        # Stage 5 gate (per-track)
    ├── run_vllm_bench_latency_sweep.py   # E2E benchmarks with GPU lock (flock)
    └── generate_validation_report.py     # Structured reporting

.claude/agents/
├── ammo-researcher.md      # Profiling + bottleneck mining (grounded data only, NO estimates)
├── ammo-champion.md        # Debate: proposes candidates, runs micro-experiments, argues with data
├── ammo-delegate.md        # Research/profiling/benchmarking subagent spawned by champions
├── ammo-impl-champion.md   # Implements kernel in isolated worktree, self-runs kernel correctness & speedup, then E2E sweep
├── ammo-resolver.md        # Resolve merge conflicts when cherry-picking GATED_PASS tracks
└── ammo-transcript-monitor.md  # Monitors agent transcripts for compliance violations
```

## Specialized Agents

| Agent | Role | Key Constraint |
|-------|------|----------------|
| **ammo-researcher** | Profiles baseline, mines bottlenecks | Cannot make feasibility estimates or E2E projections |
| **ammo-champion** | Proposes optimizations, argues in debate | Must back claims with micro-experiments |
| **ammo-delegate** | Research, profiling, benchmarking subagent for champions | Fire-and-forget; returns data, not recommendations |
| **ammo-impl-champion** | Implements kernel in isolated worktree, self-runs kernel correctness & speedup, then the E2E sweep | Works in isolated worktree; writes its own correctness test + CUDA-graph speedup bench; frontmatter Stop hook (DA) enforces validation + Amdahl's sanity |
| **ammo-resolver** | Resolves merge conflicts during Stage 6 cherry-pick | Spawned on conflict; DA reviewer verifies resolution |
| **ammo-transcript-monitor** | Monitors agent transcripts for compliance violations | Flags non-negotiable violations in real-time |

The **lead** (main Claude session) orchestrates all stages, manages `state.json`, owns all gates, and never writes kernel code directly.

## Non-Negotiables

1. **Production parity** - CUDA graphs + torch.compile in ALL measurements. FORBIDDEN: `--enforce-eager`, `TORCH_COMPILE_DISABLE=1`, `VLLM_TORCH_COMPILE_LEVEL=0`
2. **vLLM baseline** - Compare against production kernel, not naive PyTorch
3. **Numerical correctness** - `torch.allclose()` mandatory in every test
4. **GPU sequencing** - E2E benchmarks sequential via flock. Must use `run_vllm_bench_latency_sweep.py` — raw `vllm bench latency` is FORBIDDEN
5. **GPU isolation** - GPU commands MUST use the `gpu_reservation.py reserve` pattern. See `references/gpu-pool.md`.
6. **Full-model E2E** - Download weights, never skip
7. **E2E delta math** - `improvement = f x kernel_speedup` (small `f` = small E2E, not a bug)
8. **Authored-mechanism mandate** (north star: *real engineering work, no flag-flipping*) - A Stage 3 proposal is eligible iff it (a) **authors mechanism logic or host-side structure** — a custom/fused kernel in one of the four authoring classes (**Triton, CuTeDSL, CUTLASS, or CUDA C++**), load-time weight restructuring, or authored scheduling/dispatch/comm/graph-pass host-side code — (b) targets a profiled bottleneck, and (c) produces a measured production-parity E2E win ≥ `min_e2e_improvement_pct`. Retuned constants where the kernel/cubin body is byte-identical (`num_warps`, `num_stages`, `BLOCK_SIZE_*`, `@autotune` tuples, tactic tables, `custom_ops` edits, predicate/env-var flips) are rejected at Phase 0 — *a constant in a `.py` file is still config.* The optimization `category` is a non-binding descriptor, not the gate. Each proposal must include a populated Technology Selection block; the anti-regression rule requires Tier 2+ empirical proof when a proposal replaces a lower-abstraction baseline with higher-abstraction code (library baselines are treated as rank 0). See `SKILL.md` NN#8 and `references/optimization-categories.md`.
9. **Autonomous campaign loop** — The orchestrator MUST NOT ask the user whether to continue. Stop condition is mechanical: `f < min_e2e_improvement_pct` → stop, else continue. No qualitative judgment overrides this.
10. **Stage 2 profiling strategy** -- Stage 1 runs two invocations: clean E2E baseline, then bounded nsys node profiling with `--nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800`. The sweep script profiles short selected-step windows by shifting `input_len`; `--nsys-output-len` remains a horizon override. Hopper/Ampere use default nsys CUDA tracing; Blackwell (B200/B300) always uses `--nsys-trace cuda-sw`. Stage 2 mines nsys traces; targeted NCU is required for occupancy, bandwidth-counter, or physical-ceiling claims.

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

53 scenarios across 4 test files verify that the orchestrator and all subagents correctly understand and follow the AMMO workflow.

| Test File | Agent | Scenarios | Count |
|-----------|-------|-----------|-------|
| [`tests/agents/test-orchestrator.md`](tests/agents/test-orchestrator.md) | Lead orchestrator | 21 (resume, campaign eval, integration, violations, nsys profiling, baseline promotion) | 21 |
| [`tests/agents/test-researcher.md`](tests/agents/test-researcher.md) | ammo-researcher | 12 (grounded data, profiling strategy, production parity, steady-state, nsys profiling, multi-rank analysis) | 12 |
| [`tests/agents/test-champion.md`](tests/agents/test-champion.md) | ammo-champion | 10 (kernel mandate, micro-experiments, CUDA graphs, cache, subagent spawning, debate) | 10 |
| [`tests/agents/test-implementer.md`](tests/agents/test-implementer.md) | ammo-impl-champion | 10 (baseline reuse, sweep script, parity, scope, Amdahl, build, contamination) | 10 |

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

> **Note on the example below**: It predates the **Authored-Mechanism Mandate** (Non-Negotiable #8). Under current rules, OP-001 (Triton config autotuning — config-only, zero code changes) would be rejected at the Phase 0 eligibility gate. Only OP-002 (new fused CUDA kernel) would advance to debate rounds. The example is retained for workflow illustration.

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
- **Profiling method**: nsys with `--cuda-graph-trace=node`

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

**Track OP-001** (config autotuning) — ~~would be rejected under the Authored-Mechanism Mandate~~:
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

### Artifact Directory Layout (v2)

See `references/artifact-layout.md` for the full canonical spec.

```
kernel_opt_artifacts/{target}/
├── state.json                          # Campaign state (orchestrator-only writes)
├── target.json                         # Workload + bench config
├── REPORT.md                           # Terminal deliverable (Stage 7b)
├── report_assets/                      # Charts for REPORT.md
├── rounds/
│   └── {N}/                            # 1-indexed round
│       ├── constraints.md
│       ├── profiling/{nsys,ncu}/
│       ├── sweeps/{baseline,opt/{op_id},integration,golden_capture}/
│       ├── mining/bottleneck_analysis.md
│       ├── debate/{proposals,round_{D},micro_experiments,summary.md}
│       ├── tracks/{op_id}/{validation_results.md,validator_tests,monitor_audits,_scratch}
│       ├── audits/{stage_1.md,stage_45.md,stage_67.md}
│       └── _archive/                   # Superseded sweep runs
└── blockers/                           # Escalation artifacts (cross-round)
```

---

## Full Execution Trace (Per-Round Timeline)

Every script invocation, file created, and agent action in chronological order. All paths relative to `kernel_opt_artifacts/{target}/`.

### Bootstrap (T=-1): `new_target.py`

| Action | Creates | Notes |
|--------|---------|-------|
| `python scripts/new_target.py --artifact-dir ... --model-id ... --hardware ... --dtype ... --tp ...` | `state.json`, `target.json`, `rounds/1/` scaffold, `blockers/` | Pre-populates `rounds[0]` in state.json with full stage skeleton |

---

### Stage 1: Baseline Capture (ammo-researcher, task_type: baseline)

| Step | Script/Action | Creates | Confusion Risk |
|------|---------------|---------|----------------|
| T=1a | `run_vllm_bench_latency_sweep.py --round 1 --slot baseline --labels baseline --capture-golden-refs` | `rounds/1/sweeps/baseline/e2e_latency_results.{json,md}` | **AUTHORITATIVE** baseline (no profiling!) |
| | | `rounds/1/sweeps/baseline/json/golden_refs.json` | For Stage 5.1b correctness |
| | | `rounds/1/sweeps/baseline/json/baseline_bs{BS}.json` | Per-bucket raw latencies |
| | | `rounds/1/sweeps/baseline/logs/*`, `status/*` | Heartbeat/provenance |
| T=1b | `run_vllm_bench_latency_sweep.py --round 1 --slot profiling --labels baseline --nsys-profile --nsys-mode node --nsys-capture-output-steps 2,50%,100% --nsys-num-iters 1 --nsys-timeout-s 1800` | `rounds/1/profiling/nsys/baseline_bs{BS}.nsys-rep` | Stage 2 selected-step ranking traces. Add `--nsys-trace cuda-sw` on Blackwell (B200/B300). |
| | | `rounds/1/sweeps/profiling/e2e_latency_results.json` | Contaminated E2E — NOT used for speedup math |
| T=2 | Researcher writes constraints | `rounds/1/constraints.md` | Baseline truth snapshot |
| T=3 | Researcher updates state.json | `state.json: rounds[0].baseline.e2e_latency = {...}` | `profiling_baseline_path` set |

---

### Stage 2: Bottleneck Mining (ammo-researcher, task_type: mining)

| Step | Script/Action | Creates | Notes |
|------|---------------|---------|-------|
| T=4 | Researcher analyzes nsys traces | `rounds/1/mining/bottleneck_analysis.md` | Component shares + bandwidth utilization (orchestrator extracts top_component, top_f_decode_pct, amdahl_ceiling, decode_frac, component_breakdown into `state.json:.campaign.rounds[N-1].bottleneck_mining` after T2) |
| T=6 | Orchestrator: `verify_stage2_gate.py {artifact_dir} --round 1` | Nothing (read-only check) | Verifies bottleneck_analysis.md + e2e_latency_results.json exist |
| T=7 | Orchestrator: spawn `ammo-auditor` (T_AUDIT_S1) | `rounds/1/audits/stage_1.md` | Audit verdict |

---

### Stage 3: Adversarial Debate (round team)

| Step | Script/Action | Creates | Notes |
|------|---------------|---------|-------|
| T=8 | TeamCreate `ammo-round-{R}-{model}-{hw}` + spawn 2-4 champions | `state.json: rounds[0].team_name, debate.started_at` | Round team persists through Stage 5 |
| T=9 | Each champion: Phase 0 proposal | `rounds/1/debate/proposals/{champion_id}_proposal.md` | Optionally: `rounds/1/debate/micro_experiments/{champion_id}_*.py` |
| T=10 | Orchestrator: eligibility gate (reads proposals, no writes) | — | Rejects config-only proposals |
| T=11..M | Debate rounds (min 1, conditional 2nd if open items declared): argument → critique → rebuttal | `rounds/1/debate/round_{D}/{op_id}_argument.md` | Per phase per champion |
| | | `rounds/1/debate/round_{D}/{op_id}_critique_{target}.md` | |
| | | `rounds/1/debate/round_{D}/{op_id}_rebuttal.md` | |
| T=M | Winner selection + summary render | `state.json: rounds[0].debate.selected_candidates = [...]` | **AUTHORITATIVE** contract |
| | `render_debate_summary.py --state ... --out rounds/1/debate/summary.md` | `rounds/1/debate/summary.md` | **DERIVED** — state.json is source of truth |
| T=M+1 | shutdown_request to champions (team persists) | — | |

---

### Stages 4-5: Parallel Tracks (per winning candidate)

| Step | Script/Action | Creates | Notes |
|------|---------------|---------|-------|
| T=M+1 | Spawn `ammo-impl-champion-{op_id}` + monitor into round team | `state.json: tracks[op_id] = {status: IN_PROGRESS}` | Worktree isolation |
| T=M+2 | Champion implements in worktree, commits | Source files in worktree (not artifact dir) | |
| T=M+3 | Champion self-runs kernel correctness & speedup (writes its own tests/bench) | `rounds/1/tracks/{op_id}/validator_tests/test_correctness.py` | Gate 5.1a script (`validator_tests/` is the retained historical dirname) |
| | | `rounds/1/tracks/{op_id}/validator_tests/bench_gate_5_2.py` | Gate 5.2 script |
| | | `rounds/1/tracks/{op_id}/validator_tests/gate_5_1a_results.json` | (orchestrator copies metrics into `state.json:...tracks[op_id].gate_5_1a_metrics`) |
| | | `rounds/1/tracks/{op_id}/validator_tests/gate_5_2_results.json` | (orchestrator copies metrics into `state.json:...tracks[op_id].gate_5_2_metrics`) |
| | Champion iteration artifacts | `rounds/1/tracks/{op_id}/_scratch/*` | Drafts, debug scripts — non-authoritative |
| T=M+4 | Champion E2E sweep: `run_vllm_bench_latency_sweep.py --round 1 --slot opt/{op_id} --labels opt --baseline-from rounds/1/sweeps/baseline --verify-correctness` | `rounds/1/sweeps/opt/{op_id}/e2e_latency_results.{json,md}` | **AUTHORITATIVE** opt measurement |
| | | `rounds/1/sweeps/opt/{op_id}/json/correctness_verdict.json` | Gate 5.1b |
| | | `rounds/1/sweeps/opt/{op_id}/json/opt_outputs.json` | |
| | | `rounds/1/sweeps/opt/{op_id}/logs/*` | |
| T=M+5 | Champion writes final verdict | `rounds/1/tracks/{op_id}/validation_results.md` | **AUTHORITATIVE** track verdict |
| | Monitors (continuous) | `rounds/1/tracks/{op_id}/monitor_audits/{monitor_id}_observations.md` | |
| T=M+6 | All tracks terminal → T_AUDIT_S45 | `rounds/1/audits/stage_45.md` | Auto-pass if all FAIL |
| | TeamDelete round team | — | |

---

### Stage 6: Integration Validation (orchestrator)

| Step | Script/Action | Creates | Notes |
|------|---------------|---------|-------|
| T=M+7 | (multi-pass) Integration sweep: `run_vllm_bench_latency_sweep.py --round 1 --slot integration --fresh-cache` | `rounds/1/sweeps/integration/e2e_latency_results.{json,md}` | Gate-quality measurement |
| | | `rounds/1/sweeps/integration/json/correctness_verdict.json` | |
| T=M+8 | SHIP decision | `state.json: integration.{final_decision, e2e_latency_combined, commit_sha}` | |
| | | `target.json: bench.baseline_env` updated (env promotion) | |
| | | `git merge --no-ff tracks/{op_id}` | |
| T=M+9 | Post-SHIP golden-refs: `run_vllm_bench_latency_sweep.py --round 1 --slot golden_capture --labels baseline --capture-golden-refs --num-iters 1` | `rounds/1/sweeps/golden_capture/json/golden_refs.json` | Next round's Stage 5.1b reference |
| T=M+10 | T_AUDIT_S67 | `rounds/1/audits/stage_67.md` | Auto-pass on EXHAUSTED |

---

### Stage 7: Campaign Evaluation (orchestrator, autonomous)

| Step | Script/Action | Creates | Notes |
|------|---------------|---------|-------|
| T=M+11 | Read `f`, compare to `min_e2e_improvement_pct` | — | Mechanical — no user interaction |
| | IF `f >= threshold`: append `rounds[1]`, set `current_round=2` | `state.json` updated | Loop back to Stage 2 |
| | IF `f < threshold`: set `campaign.status = "campaign_complete"` or `"campaign_exhausted"` | `state.json` updated | Terminal |

---

### Stage 7b: Report Generation (ammo-report-writer, background)

| Step | Script/Action | Creates | Notes |
|------|---------------|---------|-------|
| T=M+12 | Report writer spawned (background, no wait) | `REPORT.md` | **AUTHORITATIVE** final deliverable |
| | | `report_assets/*.png` (5 charts) | |
| | | `report_assets/gen_*.py` (chart scripts) | |

---

### Disambiguation: Authoritative vs Diagnostic Artifacts

| Path | Status | Consumer |
|------|--------|----------|
| `rounds/{N}/sweeps/baseline/e2e_latency_results.json` | **AUTHORITATIVE** | Debate, impl-champions, integration |
| `rounds/{N}/sweeps/baseline/json/golden_refs.json` | **AUTHORITATIVE** | Stage 5.1b correctness |
| `rounds/{N}/mining/bottleneck_analysis.md` | **AUTHORITATIVE** | Debate champions, routing |
| `rounds/{N}/sweeps/opt/{op_id}/e2e_latency_results.json` | **AUTHORITATIVE** | Gate 5.3b, integration |
| `rounds/{N}/tracks/{op_id}/validation_results.md` | **AUTHORITATIVE** | Orchestrator, auditor, report |
| `state.json` | **AUTHORITATIVE** | All agents |
| `rounds/{N}/profiling/probe/*` | **LEGACY COMPATIBILITY** | Historical probe artifacts only — NOT baseline or Stage 2 input |
| `rounds/{N}/debate/summary.md` | **DERIVED** | Regenerated from state.json — never edit |
| `rounds/{N}/tracks/{op_id}/_scratch/*` | **NON-AUTHORITATIVE** | Champion iteration artifacts |
| `rounds/{N}/_archive/*` | **SUPERSEDED** | Auto-archived by sweep script on re-run |
