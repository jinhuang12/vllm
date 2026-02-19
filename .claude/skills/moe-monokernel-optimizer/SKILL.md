---
name: moe-monokernel-optimizer
description: Design and implement MoE kernel fusion optimizations for vLLM inference. Use when optimizing Mixture-of-Experts layers for specific (model, hardware, dtype, TP/EP) deployments to improve decode latency. Triggers on requests to beat vLLM MoE latency, implement specialized monokernels, or decide fusion boundaries (cooperative monokernel vs hybrid vs split) for MoE architectures (Llama-4, Qwen3-MoE, DeepSeek, Mixtral, etc.).
---

# MoE Monokernel Optimizer

Design MoE kernel-fusion optimizations for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

## Search Anchors

fused_moe, FusedMoE, cooperative monokernel, hybrid fusion, split kernel, CUDA graphs, torch.compile, routing, top_k, SRAM tetris, atomics, production parity

## Non-Negotiables (BLOCKING)

These are NOT advisory. Violation blocks phase progression.

1. **RUN nsys profiling with production parity** (BLOCKING for Phase 2)
   - MUST run `nsys profile` with vLLM-specific flags (see `references/nsys-profiling-guide.md`)
   - MUST measure vLLM's actual `fused_experts`/`fused_moe` kernel times
   - MUST document baseline timings in constraints.md (not just commands)
   - Verify: `python scripts/verify_phase1_baseline.py {artifact_dir}` returns 0

2. **VERIFY routing math from vLLM source** (BLOCKING)
   - MUST read vLLM model file (e.g., `vllm/model_executor/models/qwen2_moe.py`)
   - MUST record in `{artifact_dir}/constraints.md` using `references/moe-parameters-template.md`
   - MUST match exactly: scoring (softmax/sigmoid), renorm, weight placement, accumulation

3. **SELECT fusion boundary using decision tree** (BLOCKING for Phase 3)
   - MUST use `references/route-selection-decision-tree.md` to select:
     - A) Cooperative monokernel
     - B) Hybrid large-grid fusion
     - C) Split-kernel graph-captured
   - MUST document rationale with profiling evidence
   - Do NOT default to "single mega-kernel" without data

4. **ENSURE CUDA graphs safety** (BLOCKING for Phase 5)
   - MUST use correct stream, no hidden allocations, stable shapes per bucket
   - See `references/cudagraph-safety.md`

5. **COMPARE against vLLM baseline (not PyTorch)** (BLOCKING for Phase 4)
   - MUST import `fused_experts` or `fused_moe` from vLLM as baseline
   - MUST run benchmarks with production settings (torch.compile enabled)
   - MUST include `torch.allclose()` numerical comparison in tests
   - Verify: `python scripts/verify_phase4_gates.py {artifact_dir}` returns 0
   - Naive PyTorch loops are NOT valid baselines

6. **RUN verify_phase4_gates.py before Phase 4 completion** (BLOCKING)
   - MUST run: `python scripts/verify_phase4_gates.py {artifact_dir}`
   - MUST return exit code 0
   - MUST record result in state.json: `"verification_run": {"phase4": {"status": "PASS", "date": "..."}}`
   - If script fails: DO NOT proceed to Phase 5, fix blockers first
   - If script not run: Phase 4 is INCOMPLETE regardless of other results

7. **KERNEL benchmarks under CUDA graphs** (BLOCKING for Phase 4)
   - MUST capture both baseline AND monokernel in CUDA graphs for fair comparison
   - MUST time graph replays, NOT individual kernel launches
   - Triton vs CUDA C++ launch overhead difference is ~50-100 µs - graphs eliminate this
   - See `references/validation-defaults.md` § "Kernel-Level Benchmark Requirements"
   - FORBIDDEN: `TORCH_COMPILE_DISABLE=1` or `VLLM_TORCH_COMPILE_LEVEL=0` in benchmarks

8. **Do not skip full-model E2E because "weights aren't available"**
   - If you need E2E to compute MoE share `f` (Phase 1) or to validate speedup (Phase 4.3), and the model isn't cached locally: **download the weights**.
   - Only skip E2E if the **user explicitly waives** the E2E requirement (or the model is gated and the user cannot/does not want to provide access).

## Execution Model: Orchestrator + Task Subagents

This skill uses **orchestrator/subagent separation** for context isolation.

**Orchestrator (you)**: Coordinates phases, manages state, spawns Task subagents.
**Task subagents**: Execute individual phases with fresh context.

### How to Spawn Tasks

Use the **Task tool** with prompt from `orchestration/task-prompts.md`:

```
Task tool parameters:
- description: "Phase N: {brief description}"
- prompt: [Full prompt from task-prompts.md § Phase N]
```

Each Task runs independently. Include in every prompt:
- Path to state file: `{artifact_dir}/state.json`
- All context needed (tasks cannot access prior task memory)

See `orchestration/task-prompts.md` for phase-specific prompts.

## 5-Phase Workflow

```
Phase 1: Constraints     → {artifact_dir}/constraints.md
Phase 2: Planning        → {artifact_dir}/optimization_plan.md
Phase 3: Implementation  → CUDA kernel code
Phase 4: Validation      → {artifact_dir}/validation_results.md
Phase 5: Integration     → vLLM dispatch path
```

You **MUST** follow the stage-gated checklist in `orchestration/task-guide.md` (canonical)

### State File

Location: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/state.json`

```json
{
  "target": {"model_id": "...", "hardware": "...", "dtype": "...", "tp": 1, "ep": 1},
  "phase": "2_planning",
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
| Route decision | `references/route-selection-decision-tree.md` |
| Algorithmic branching | `references/algorithmic-branching.md` |
| SRAM/tiling | `references/tiling-config.md` |
| Hardware specs | `references/gpu-configs.md` |
| Code patterns | `references/code-templates.md` |
| Validation gates | `references/validation-defaults.md` |
| E2E math | `references/e2e-delta-math.md` |
| Graph safety | `references/cudagraph-safety.md` |
| Investigation | `references/investigation-playbook.md` |

### Worked examples
- `examples/MODELS_COMPARISON.md`
- `examples/W1_EPILOGUE_FUSION.md`

## LLM Council (Recommended)

For high-risk decisions, consider invoking the `llm-council` skill:
- Changes to routing math or accumulation semantics
- Major fusion boundary redesigns
- When stuck after 2+ attempts

Not mandatory—use engineering judgment. See `orchestration/llm-council.md`.

## Escalation Protocol

When a Task encounters a blocker it cannot resolve:

### 1. Update state.json with blocker status

```json
{
  "status": "blocked",
  "blocker": {
    "description": "Description of issue",
    "severity": "critical|major|minor",
    "gate": "gate_4_1_correctness",
    "attempts": 2,
    "escalation_needed": true
  }
}
```

### 2. Create blocker file

Save detailed blocker info to: `{artifact_dir}/blockers/{phase}_{stage}_{date}.md`

Use template from `orchestration/blocker-template.md`.

### 3. Orchestrator action on blockers

| Severity | Action |
|----------|--------|
| **critical** | STOP. Invoke llm-council. Do NOT proceed. |
| **major** | Adjust constraints, re-spawn task with modified parameters |
| **minor** | Document and continue, but flag for review |

### 4. Verification scripts (BLOCKING)

Before marking any phase complete, run verification:

```bash
# Phase 1 → Phase 2 gate
python scripts/verify_phase1_baseline.py {artifact_dir}
# Exit code must be 0 to proceed

# Phase 4 → Phase 5 gate
python scripts/verify_phase4_gates.py {artifact_dir}
# Exit code must be 0 to proceed
```

**If verification fails**: Do NOT proceed. Update state.json with `"status": "blocked"`.

## Quick Start

### Example 1: Single Expert (top_k=1)

```markdown
User: "Use moe-monokernel-optimizer for Llama-4-Scout on p5.48xlarge TP=8"
```

### Example 2: Multi Expert (top_k=8)

```markdown
User: "Use moe-monokernel-optimizer for Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 on g6e.24xlarge TP=1"
```

### Resume

**Note**: Tasks have no memory of previous executions. Each task starts fresh.

```
User: "Resume MoE optimization"
Orchestrator:
1. Find state file: ls moe_monokernel_artifacts/*/state.json
2. Read state, identify current phase/stage
3. Report status to user
4. If blocked: check orchestration/failure-handling.md for next action
5. Spawn NEW Task with:
   - Full context from task-prompts.md
   - Path to state.json for reading/updating
   - Any blocker content if retrying
   - subagent_type: "general-purpose"
```

### Resume After Context Compaction

The `PreCompact` and `SessionStart` hooks in `.claude/settings.local.json` automatically:
1. Save checkpoint state before compaction
2. Inject orchestrator role context after compaction

If hooks didn't fire or you need a manual reminder:
1. **Read this skill file** (you're doing this now)
2. **Load state**: `cat moe_monokernel_artifacts/*/state.json`
3. **You are the ORCHESTRATOR** - spawn Tasks, don't implement directly
4. **Spawn next Task** with FULL prompt from `orchestration/task-prompts.md`

### Status Check

```
User: "What's the monokernel status?"
Orchestrator:
1. Load state file
2. Report: "Phase 3, stage up_projection blocked after 3 attempts"
3. Suggest: "Shall I invoke llm-council for a second opinion?"

```
