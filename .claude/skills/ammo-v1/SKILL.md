---
name: ammo
description: Profile and optimize GPU kernels for vLLM inference on NVIDIA GPUs. Use when targeting specific (kernel component, model, hardware, dtype, TP) deployments to improve latency. Triggers on requests to speed up any vLLM kernel: attention, KV cache, sampling, quantization, MoE, FFN, custom fusions.
---

# AMMO - Automated Model Micro-Optimizer

Profile and optimize **GPU kernels** for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

## Search Anchors

GPU kernel optimization, CUDA graphs, torch.compile, nsys profiling, kernel fusion, attention, KV cache, sampling, quantization, fused_moe, production parity

## Non-Negotiables (BLOCKING)

These are NOT advisory. Violation blocks stage progression.

1. **RUN nsys profiling with production parity** (BLOCKING for Stage 2)
   - MUST run `nsys profile` with vLLM-specific flags (see `references/nsys-profiling-guide.md`)
   - MUST measure vLLM's actual target component kernel times
   - MUST document baseline timings in constraints.md (not just commands)
   - Verify: `python scripts/verify_phase1_baseline.py {artifact_dir}` returns 0

2. **VERIFY component semantics from vLLM source** (BLOCKING)
   - MUST read vLLM source code for the target component
   - MUST trace the forward path through the target component and document correctness invariants
   - MUST record in `{artifact_dir}/constraints.md` using `references/component-constraints-template.md`

3. **SELECT optimization approach using evidence** (BLOCKING for Stage 3)
   - MUST select approach based on profiling evidence and feasibility analysis
   - MUST document rationale with profiling evidence
   - Do NOT default to any approach without data

4. **ENSURE CUDA graphs safety** (BLOCKING for Stage 5)
   - MUST use correct stream, no hidden allocations, stable shapes per bucket
   - See `references/cudagraph-safety.md`

5. **COMPARE against vLLM baseline (not PyTorch)** (BLOCKING for Stage 5)
   - MUST import the production vLLM kernel for the target component as baseline
   - MUST run benchmarks with production settings (torch.compile enabled)
   - MUST include `torch.allclose()` numerical comparison in tests
   - Verify: `python scripts/verify_validation_gates.py {artifact_dir}` returns 0
   - Naive PyTorch loops are NOT valid baselines

6. **KERNEL benchmarks under CUDA graphs** (BLOCKING for Stage 5)
   - MUST capture both baseline AND optimized kernel in CUDA graphs for fair comparison
   - MUST time graph replays, NOT individual kernel launches
   - Triton vs CUDA C++ launch overhead difference is ~50-100 µs - graphs eliminate this
   - See `references/validation-defaults.md` § "Kernel-Level Benchmark Requirements"
   - FORBIDDEN: `TORCH_COMPILE_DISABLE=1` or `VLLM_TORCH_COMPILE_LEVEL=0` in benchmarks

7. **Do not skip full-model E2E because "weights aren't available"**
   - If you need E2E to compute component share `f` (Stage 1) or to validate speedup (Stage 5), and the model isn't cached locally: **download the weights**.
   - Only skip E2E if the **user explicitly waives** the E2E requirement (or the model is gated and the user cannot/does not want to provide access).

## Execution Model: Agent Teams

This skill uses **persistent agent teams** for coordinated multi-stage execution.

**Lead (you)**: Creates team, manages task list, runs verification gates, updates state.json, handles blockers.
**Teammates**: Persistent agents with shared task list, direct messaging, and role-specific expertise.

### Team Structure

| Name | Stages | Rationale |
|------|--------|-----------|
| `lead` | Orchestrator | Gates, state management, blocker resolution, loop mediation |
| `verifier` | 1, 2, 5 | Baseline capture + ALL validation measurement + kill criteria eval. Skeptical mandate: finds flaws, challenges assumptions, derives test methodology from planner's acceptance criteria (not implementer's notes). |
| `planner` | 1, 2, 3 | Source analysis, opportunity ranking, approach selection, plan + acceptance criteria writing |
| `implementer` | 4, 5 | Kernel implementation + initial correctness test script |

All teammates are `general-purpose` (need bash for nsys/ncu/cmake/pytest/vllm commands).

### Implementer-Verifier Loop

When Stage 5 validation fails, the lead mediates a bounded fix cycle:

1. **Verifier** runs T17/T18/T19, finds failure, sends rejection + evidence to lead
2. **Lead** reviews evidence, decides: `retry_stage_4` / `retry_stage_3` / `escalate_human` / `document_proceed`
3. If `retry_stage_4`: Lead creates fix task for implementer (cycle N+1)
4. **Implementer** fixes, lead runs T15 compilation gate, then verifier re-validates
5. **Max 3 cycles** per failure mode. If last 2 cycles show <1% improvement, stop.

See `orchestration/communication-patterns.md` for message templates.

### Opportunity Iteration Loop

When Stage 5 validation results in a **KILL** (optimization doesn't beat baseline), the lead automatically pivots to the next ranked opportunity instead of stopping:

1. **Lead** records the KILL in `state.json` `opportunity_attempts` array
2. **Lead** selects the next untried opportunity from `bottleneck_analysis.md` ranked list
3. **Planner** writes updated `optimization_plan.md` with "Previous Attempts" section
4. **Lead** reviews the new plan (Stage 3 gate)
5. Fresh Stage 4-5 cycle runs for the new opportunity

This loop repeats until:
- An opportunity **SHIPS** (validation passes) → done
- **All attempts exhausted** (`max_attempts` reached, default 3) → team shuts down
- A **BLOCKED** state is reached → follow blocker protocol

See `orchestration/task-dependency-graph.md` § "Opportunity Iteration Loop" for task templates (T23-T26+) and `orchestration/task-prompts.md` § "Lead: Post-T22 Decision Logic" for the decision tree.

### How to Set Up

1. **Create team**: `TeamCreate` with name `ammo-{component}-{model_short}-{hardware}`
2. **Spawn teammates**: Use Task tool with `team_name` and `name` parameters, role briefings from `orchestration/task-prompts.md`
3. **Create tasks**: Use `TaskCreate` with templates from `orchestration/task-dependency-graph.md`
4. **Assign tasks**: Use `TaskUpdate` with `owner` parameter

See `orchestration/team-setup.md` for full startup checklist.

### Gate Enforcement

Gate tasks (T7, T10, T13, T15, T22) are **lead-owned**. Teammates cannot self-unblock past gates.
- Lead runs verification scripts and reviews deliverables
- Lead updates state.json at stage transitions
- If a gate fails, lead creates investigation tasks and notifies affected teammates

### GPU Resource Management

GPU benchmark tasks (T18 kernel perf, T19 E2E latency) MUST run sequentially. Concurrent
GPU benchmarks cause OOM errors and unreliable results due to memory contention.

**Enforcement layers**:
1. **`blockedBy` dependencies (primary)**: T19 is blocked by T18 in the task dependency graph.
   For iteration loops, the lead must set T19' blocked by T18' when creating fresh task chains.
2. **System-wide GPU lock (defense-in-depth)**: `scripts/run_vllm_bench_latency_sweep.py` holds
   an flock in `/tmp/ammo_gpu_locks/` that prevents concurrent sweeps even if task dependencies
   are misconfigured. Agents must use the sweep script for all E2E measurements.
3. **Role briefings (advisory)**: Verifier briefing includes GPU Benchmark Protocol; implementer
   briefing states GPU benchmarks are the verifier's job.

**Why text rules are insufficient**: During the OLMo-3-7B-Instruct verification run on L40S,
text-based HARD RULES in agent briefings were ignored — agents launched concurrent benchmarks
despite broadcast STOP messages. The `blockedBy` system is the only reliable enforcement
mechanism because agents cannot bypass it.

### Communication Patterns

- **Blocker escalation**: Teammate → lead via `SendMessage` (see `orchestration/communication-patterns.md`)
- **Fix instructions**: Lead → teammate via `SendMessage`
- **Critical stop**: Lead → all via `SendMessage` broadcast (use sparingly)
- **Task handoff**: Teammate → teammate via `SendMessage` for findings that feed downstream work

See `orchestration/communication-patterns.md` for message templates.

## 5-Stage Workflow

```
Stage 1: Constraints + Baseline Capture  → {artifact_dir}/constraints.md
Stage 2: Bottleneck Mining and Ranking   → {artifact_dir}/bottleneck_analysis.md
Stage 3: Candidate Selection + Opt Plan  → {artifact_dir}/optimization_plan.md
Stage 4: Implementation                  → kernel/code changes
Stage 5: Validation                      → {artifact_dir}/validation_results.md
         ↓ SHIP → done
         ↓ KILL → loop back to Stage 3 with next opportunity (up to max_attempts)
         ↓ KILL + exhausted → shutdown
```

You **MUST** follow the task dependency graph in `orchestration/task-dependency-graph.md` (canonical)

### State File

Location: `kernel_opt_artifacts/{component}_{model}_{hardware}_{dtype}_{tp}/state.json`

```json
{
  "target": {"model_id": "...", "hardware": "...", "dtype": "...", "tp": 1, "component": "..."},
  "stage": "2_bottleneck_mining",
  "status": "in_progress",
  "last_update": "2026-01-05",
  "notes": ""
}
```

## Helper Scripts

Deterministic measurement and reporting (run, don't modify):

- `scripts/new_target.py` - Scaffold artifact directory
- `scripts/collect_env.py` - Capture environment for reproducibility
- `scripts/run_vllm_bench_latency_sweep.py` - Batch benchmark runner (dry-run by default)
- `scripts/generate_validation_report.py` - Structured reporting

## References (read as needed)

Use `rg`/grep with search anchors to locate details.

| Topic | File |
|-------|------|
| Nsys profiling | `references/nsys-profiling-guide.md` |
| Component constraints | `references/component-constraints-template.md` |
| Bottleneck mining | `references/bottleneck-mining-guide.md` |
| SRAM/tiling | `references/tiling-config.md` |
| Hardware specs | `references/gpu-configs.md` |
| Code patterns | `references/code-templates.md` |
| Validation gates | `references/validation-defaults.md` |
| E2E math | `references/e2e-delta-math.md` |
| Graph safety | `references/cudagraph-safety.md` |
| Investigation | `references/investigation-playbook.md` |
| Feasibility math | `references/fusion-feasibility-heuristics.md` |
| Optimization techniques | `references/optimization-techniques.md` |
| Optimization plan template | `references/optimization-plan-template.md` |
| Scope/support | `references/scope-and-support.md` |
| State schema | `references/state-schema.md` |

## Escalation Protocol

When a teammate encounters a blocker it cannot resolve:

### 1. Teammate: Update state.json and create blocker file

```json
{
  "status": "blocked",
  "blocker": {
    "description": "Description of issue",
    "severity": "critical|major|minor",
    "gate": "gate_5_1_correctness",
    "attempts": 2,
    "escalation_needed": true
  }
}
```

Save detailed blocker info to: `{artifact_dir}/blockers/{stage}_{date}.md`
Use template from `orchestration/blocker-template.md`.

### 2. Teammate: Notify lead via SendMessage

```
SendMessage:
  type: "message"
  recipient: "lead"
  content: "BLOCKER [{severity}]: {description}. Blocker file: {path}. Action needed: {action}"
  summary: "Blocker: {brief description}"
```

See `orchestration/communication-patterns.md` for full message templates.

### 3. Lead action on blockers

| Severity | Action |
|----------|--------|
| **critical** | STOP. Invoke llm-council. Broadcast critical stop if needed. |
| **major** | Adjust constraints, create investigation task for teammate |
| **minor** | Document and continue, send guidance to teammate |

### 4. Verification scripts (BLOCKING — lead-owned)

Before marking any stage complete, the lead runs verification:

```bash
# Stage 1 → Stage 2 gate (task T7)
python scripts/verify_phase1_baseline.py {artifact_dir}
# Exit code must be 0 to proceed

# Validation Stage gate (task T22)
python scripts/verify_validation_gates.py {artifact_dir}
# Exit code must be 0 to proceed
```

**If verification fails**: Do NOT proceed. Update state.json with `"status": "blocked"`. Send failure details to responsible teammate via `SendMessage`.

## Quick Start

### Example 1: Optimize MoE decode latency

```markdown
User: "Use ammo for Qwen3-30B-A3B on L40S TP=1"

Lead:
1. Scaffold artifact directory (scripts/new_target.py)
2. TeamCreate: ammo-moe-qwen3-30b-a3b-l40s
3. Spawn teammates: verifier, planner, implementer
4. Create T1-T22 tasks per orchestration/task-dependency-graph.md
5. Complete T1, teammates pick up unblocked tasks
6. If T22 KILL: automatically pivot to next ranked opportunity (up to max_attempts)
```

### Example 2: Optimize attention kernel

```markdown
User: "Use ammo to optimize attention for Llama-3-70B on H100 TP=8"

Lead:
1. Scaffold artifact directory
2. TeamCreate: ammo-attention-llama3-70b-h100
3. Spawn teammates
4. Create task graph
5. Begin execution
```

### Example 3: Profile and optimize sampling kernels

```markdown
User: "Use ammo for sampling kernels on Mistral on A100"

Lead:
1. Scaffold artifact directory
2. TeamCreate: ammo-sampling-mistral-a100
3. Spawn teammates
4. Create task graph
5. Begin execution
```

### Resume

**Note**: Teammates persist across lead turns. The team and task list survive context compaction.

```
User: "Resume kernel optimization"
Lead:
1. Read team config: ~/.claude/teams/ammo-*/config.json
2. Run TaskList to see current task state
3. Read state.json: ls kernel_opt_artifacts/*/state.json
4. Report status to user (stage, blocked tasks, completed tasks)
5. If blocked: check blocker files, send fix instructions to teammates
6. Send messages to idle teammates to resume work if needed
```

### Resume After Context Compaction

The `PreCompact` and `SessionStart` hooks in `.claude/settings.local.json` automatically:
1. Save checkpoint state before compaction
2. Inject orchestrator role context after compaction

Teammates persist across lead compaction — they have their own context.

If hooks didn't fire or you need a manual reminder:
1. **Read this skill file** (you're doing this now)
2. **Read team config**: `~/.claude/teams/ammo-*/config.json`
3. **Run TaskList** to see task progress
4. **Load state**: `cat kernel_opt_artifacts/*/state.json`
5. **You are the LEAD** — manage tasks and gates, don't implement directly
6. **Send messages** to idle teammates to resume their work

### Status Check

```
User: "What's the optimization status?"
Lead:
1. Run TaskList to see all task statuses
2. Read state.json for stage info
3. Report: "Stage 4 — T14 (kernel implementation) in progress by implementer.
   T2-T13 completed. 8 tasks remaining."
4. If blocked: "T17 blocked — correctness test failed. Verifier investigating."
5. Suggest: "Shall I invoke llm-council for a second opinion?"
```
