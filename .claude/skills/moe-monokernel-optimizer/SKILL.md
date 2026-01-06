---
name: moe-monokernel-optimizer
description: Design and implement MoE kernel fusion optimizations for vLLM inference. Use when optimizing Mixture-of-Experts layers for specific (model, hardware, dtype, TP/EP) deployments to improve decode latency. Triggers on requests to beat vLLM MoE latency, implement specialized monokernels, or decide fusion boundaries (cooperative monokernel vs hybrid vs split) for MoE architectures (Llama-4, Qwen3-MoE, DeepSeek, Mixtral, etc.).
---

# MoE Monokernel Optimizer

Design specialized MoE kernels for vLLM that reduce launch overhead and memory traffic by fusing phases when beneficial.

## Search Anchors

fused_moe, FusedMoE, cooperative monokernel, hybrid fusion, split kernel, CUDA graphs, torch.compile, routing, top_k, SRAM tetris, atomics, production parity

## Non-Negotiables

1. **Measure the right baseline (production parity)**
   - Benchmark under CUDA graphs / torch.compile with the same settings as production
   - Measure GPU kernel time, not just kernel count
   - See `references/profiling-launch-vs-kernel.md`

2. **Do not assume model semantics**
   - Verify routing math by reading vLLM model code
   - Record in `{artifact_dir}/constraints.md` using `references/moe-parameters-template.md`
   - Match exactly: scoring (softmax/sigmoid), renorm, weight placement, accumulation

3. **Pick fusion boundary deliberately**
   - Use `references/route-selection-decision-tree.md` to choose:
     - A) Cooperative monokernel
     - B) Hybrid large-grid fusion
     - C) Split-kernel graph-captured
   - Do not default to "single mega-kernel" without profiling evidence

4. **CUDA graphs safety is required**
   - Correct stream, no hidden allocations, stable shapes per bucket
   - See `references/cudagraph-safety.md`

5. **No ungrounded performance claims**
   - Never claim a win without measurement on target buckets with CUDA graphs enabled

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

See `orchestration/checklist.md` for detailed phase gates.

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
| Route decision | `references/route-selection-decision-tree.md` |
| Algorithmic branching | `references/algorithmic-branching.md` |
| SRAM/tiling | `references/tiling-config.md` |
| Hardware specs | `references/gpu-configs.md` |
| Code patterns | `references/code-templates.md` |
| Validation gates | `references/validation-defaults.md` |
| E2E math | `references/e2e-delta-math.md` |
| Graph safety | `references/cudagraph-safety.md` |
| Investigation | `references/investigation-playbook.md` |

## LLM Council (Recommended)

For high-risk decisions, consider invoking the `llm-council` skill:
- Changes to routing math or accumulation semantics
- Major fusion boundary redesigns
- When stuck after 2+ attempts

Not mandatory—use engineering judgment. See `orchestration/llm-council.md`.

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
