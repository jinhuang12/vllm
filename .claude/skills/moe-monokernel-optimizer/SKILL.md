---
name: moe-monokernel-optimizer
description: Design and implement MoE kernel fusion optimizations for vLLM inference. Use when optimizing Mixture-of-Experts layers for specific (model, vLLM config, hardware, dtype) deployments where router + quantization + GEMMs can be fused (single cooperative kernel) or partially fused (split kernels) to improve decode latency, eliminate launch overhead, and reduce memory round-trips. Triggers on requests to optimize MoE decode latency, implement specialized monokernels, or decide fusion boundaries for MoE architectures (e.g., Llama-4, Qwen3-MoE, DeepSeek-style gated MoE).
---

# MoE Monokernel Optimizer

Design specialized MoE kernels for vLLM that reduce launch overhead and memory traffic by fusing phases when beneficial (single cooperative kernel) or splitting into multiple kernels when ownership and occupancy require it.

## When Monokernel Applies

Monokernel optimization is beneficial when:
- Decode batch size is small (typically BS ≤ 32-128 depending on hardware)
- MoE preamble (router + quant + kernel launch gaps) exceeds ~20-30% of layer time
- Static deployment: fixed (model, vLLM config, hardware, dtype) quadruple
- **M_avg and ownership decisions** support a single-kernel path (see Decision 0b/0c)
- Routing distribution is known (or uniform routing is explicitly assumed)
- You will benchmark under CUDA graphs / torch.compile to match production
- Reference profiling of vLLM FusedMoE is available (recommended for planning, required for regressions)

**Barrier budget gate**: If the design requires more than 1-2 grid-wide barriers (`grid.sync`), strongly prefer split-kernel or token-major designs unless M_avg is large and routing is balanced.

**Baseline delta gate**: After combined-graph profiling, compute the µs savings required to beat the baseline; if the required savings are implausible, re-plan or document the limitation.

**Full monokernel default gate** (heuristic):
- Use full monokernel only if **routing/prepare share >= ~15-20%**, **barrier budget <= 1-2**, **cooperative launch feasible**, and **M_avg is high or top_k=1**.
- Otherwise default to split/hybrid and target the expert kernel cost first.

## Supported Data Types

| Type | Element Size | MMA Instruction | Notes |
|------|-------------|-----------------|-------|
| FP8 E4M3 | 1 byte | mma.f32.f8.f8 | Best for inference, requires sm_89+ |
| BF16 | 2 bytes | mma.f32.bf16.bf16 | 2× SMEM cost vs FP8 |
| FP16 | 2 bytes | mma.f32.f16.f16 | Legacy compatibility |
| MXFP4 | 0.5 bytes | Experimental | Future support |

## Supported Models

Reference implementations exist for these model architectures:

| Model | top_k | Hardware | Quantization | Key Patterns |
|-------|-------|----------|--------------|--------------|
| **Llama-4-Scout** | 1 | H100 (sm_90a) | Per-tensor FP8 | Direct write, TMA prefetch |
| **Qwen3-Coder-30B-A3B** | 8 | L40S (sm_89) | 128×128 block FP8 | FP32 accumulator, Split-H, cp.async |

**Pattern Comparison**: See [examples/MODELS_COMPARISON.md](examples/MODELS_COMPARISON.md) for detailed code snippets showing:
- Top-K routing (single vs multi-expert)
- Output accumulation (direct write vs atomicAdd)
- Scale loading (per-tensor vs block quantization)
- MMA instruction selection
- Split-H optimization for small batches

**Future Support Prepared**:
- DeepSeek (shared experts): Decision D patterns documented
- Mixtral/standard MoE: Covered by existing Decision A/C branching

## LLM Council Integration

`llm-council` is a **de-risking tool**, not a hard gate. Use it when the change is high-impact, correctness-sensitive, or you feel uncertain. The ~10-20 minute latency catches issues that would cost hours to debug.

### Risk-Tier Policy

- **Required (high-risk)** — default expectation:
  - Changing **kernel math / accumulation** (FP32 accumulator logic, atomics, Split-H reduction)
  - Changing **memory layout / shared memory / TMA/cp.async staging** that could corrupt results
  - Introducing or modifying **top_k > 1** routing + accumulation behavior
  - Making a **major performance tradeoff** that could regress other batch sizes

- **Recommended (medium-risk)**:
  - After Phase 2 when plan introduces non-trivial architectural choices (tiling, buffering, fusion boundaries)
  - After Phase 4 if investigation proposes non-obvious fix, or if conclusions are based on noisy perf data
  - When stuck after 2+ distinct attempts (fresh eyes help)
  - After Stage 3 (GEMM implementation) - most complex stage

- **Optional (low-risk)**:
  - Mechanical refactors, comments/docs, build-system glue
  - Small edits with strong test coverage and low blast radius

> If you choose to skip a recommended/required council review, proceed — but leave a short rationale in `{artifact_dir}/state.json` under `council_skip_rationale` so future resumption has context.

### High-Risk Stages (Pre-Implementation Review)

For these stages, invoke llm-council BEFORE implementation to review the approach:

| Stage | Risk Factor | Council Review Focus |
|-------|-------------|---------------------|
| `gemm_implementation` | Most complex: triple buffering, MMA loops, K-chunking | Verify tile sizes, buffer strategy, MMA instruction selection |
| Down-projection (top_k > 1) | Atomic contention with multi-expert accumulation | Review accumulation strategy, Split-H decision |

### How to Invoke llm-council

The llm-council skill is **model-invoked**. When you need a second opinion, use the `Skill` tool to invoke `llm-council`. The llm-council skill has its own instructions for preparing context and running the review.

See [orchestration/failure-handling.md](orchestration/failure-handling.md) for the 6-level escalation ladder.

## Execution Model: Orchestrator + Task Subagents

**CRITICAL**: This skill uses a **strict orchestrator/subagent separation**. The orchestrator (you, the main Claude instance) MUST NOT implement phases directly. Instead, you coordinate by spawning Task subagents.

### Orchestrator Role (YOU)

You are the **orchestrator**. Your responsibilities:
- Read this skill and understand the workflow
- Create/load the state file
- Spawn Task subagents for each phase using the Task tool
- Update state based on Task outcomes
- Handle failures per `orchestration/failure-handling.md`

**YOU DO NOT**:
- Write CUDA code directly
- Implement any phase yourself
- Skip Task spawning "for efficiency"
- Paraphrase or summarize the task prompts

### How to Spawn a Task

Use the **Task tool** with these required parameters:

| Parameter | Required | Description |
|-----------|----------|-------------|
| `description` | Yes | Short description (3-5 words) |
| `prompt` | Yes | Full task instructions |
| `subagent_type` | Yes | Agent type (see table below) |

**Subagent Types by Phase**:

| Phase | Subagent Type | Rationale |
|-------|---------------|-----------|
| Phase 1: Constraints | `general-purpose` | Primarily code analysis and reading |
| Phase 2: Planning | `general-purpose` | Design and decision-making |
| Phase 3: Implementation | `general-purpose` | Writing CUDA code, running compiles |
| Phase 4: Validation | `general-purpose` | Running tests and benchmarks |
| Phase 5: Integration | `general-purpose` | Modifying build system and Python |

**Task Invocation Example**:
```
Use the Task tool with:
- description: "Phase 1: Gather MoE constraints"
- subagent_type: "general-purpose"
- prompt: [Copy FULL prompt from orchestration/task-prompts.md § Phase 1]
```

**CRITICAL - State Management**:
Each Task runs **independently with no shared memory**. Every task prompt MUST include:
1. Path to read state: `{artifact_dir}/state.json`
2. Instructions to update state before completing
3. All context needed (tasks cannot access prior task memory)

**IMPORTANT**:
- Copy the FULL task prompt from `orchestration/task-prompts.md`
- Fill in the template variables ({model_id}, {hardware}, {tp}, {dtype}, etc.)
- Wait for Task completion before proceeding
- Do NOT paraphrase or summarize - subagents need complete context

**CRITICAL - Expand `{common_behavioral_footer}`**:
Each phase prompt ends with `{common_behavioral_footer}`. You MUST replace this placeholder with the **FULL content** from the "Common Behavioral Footer" section at the top of `task-prompts.md`. This is approximately 50 lines including:
- LLM Council Consultation (with "To invoke" instructions)
- Behavioral Expectations (ALL 6 items):
  1. Read before write
  2. Review plan with llm-council (revise until accepted)
  3. Compile often (with error handling)
  4. Stuck handling (document blockers, exit blocked)
  5. Review implementation with llm-council
  6. Stay goal-aligned
- State Management (all 3 items)
- Context Preservation section

**DO NOT abbreviate the footer**. Items 2-5 (llm-council review loops, compile errors, stuck handling, implementation review) are critical for quality control. Skipping them causes subagents to produce incomplete work without proper review cycles.

### Complex Phase Planning (EnterPlanMode)

For complex phases, the orchestrator should use **EnterPlanMode** before spawning Tasks to ensure proper review of critical decisions:

| Phase/Stage | Use EnterPlanMode? | Reason |
|-------------|-------------------|--------|
| Phase 2 (Planning) | **Yes** | Design optimization approach with user review |
| Phase 3 GEMM stage | **Yes** | Most complex stage - verify approach before implementation |
| Other phases/stages | No | Standard Task spawning sufficient |

This ensures architecturally significant decisions get explicit user approval before implementation begins.

### State File

Location: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/state.json`

Example for Qwen3-30B-A3B-FP8 on L40S with TP=1:
`moe_monokernel_artifacts/qwen3-30b-a3b_l40s_fp8_tp1/state.json`

```json
{
  "version": "1.0",
  "model_id": "Qwen/Qwen3-30B-A3B-FP8",
  "model_short": "qwen3-30b-a3b",
  "hardware": "L40S",
  "dtype": "fp8",
  "tp": 1,
  "artifact_dir": "moe_monokernel_artifacts/qwen3-30b-a3b_l40s_fp8_tp1",
  "cuda_dir": "csrc/moe/moe_monokernel_qwen3-30b-a3b_l40s_fp8_tp1",
  "phases": {
    "1_constraints": {"status": "complete"},
    "2_planning": {"status": "complete"},
    "3_implementation": {
      "status": "in_progress",
      "current_stage": "up_projection",
      "stages": {
        "router": {"status": "complete"},
        "prepare": {"status": "complete"},
        "scale_inputs": {"status": "complete"},
        "up_projection": {"status": "blocked", "attempts": 3}
      }
    }
  },
  "constraints": {
    "K": 2048, "N": 768, "E": 128, "top_k": 8,
    "num_shared_experts": 0,
    "smem_limit_kb": 100, "sm_arch": "sm_89",
    "weight_dtype": "fp8_e4m3", "activation_dtype": "bf16"
  }
}
```

## 5-Phase Workflow

Artifact directory: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/`
CUDA directory: `csrc/moe/moe_monokernel_{model}_{hardware}_{dtype}_{tp}/`

```
Phase 1: Gather Constraints     → {artifact_dir}/constraints.md
Phase 2: Optimization Planning  → {artifact_dir}/optimization_plan.md  
Phase 3: Implementation         → {cuda_dir}/*.cu
Phase 4: Validation             → {artifact_dir}/validation_results.md
Phase 5: Integration            → vLLM dispatch path
```

### Phase 1: Gather Constraints

**Subagent Type**: `general-purpose`

**Spawn Task** with prompt from `orchestration/task-prompts.md` § Phase 1.

```
Task tool parameters:
- description: "Phase 1: Gather MoE constraints for {model_id}"
- subagent_type: "general-purpose"
- prompt: [Full prompt from task-prompts.md § Phase 1]
```

Collects:
- Locates the model's MoE implementation in vLLM source
- Traces the forward path (routing → expert execution → accumulation)
- Compares against Llama 4 reference semantics
- Model geometry (K, N, E, top_k, num_shared_experts)
- Hardware specs from `references/gpu-configs.md`
- vLLM parallelism (TP, EP)
- **Data types** (weight_dtype, activation_dtype, scale_format)

### Phase 2: Optimization Planning

**Subagent Type**: `general-purpose`

**Spawn Task** with prompt from `orchestration/task-prompts.md` § Phase 2.

```
Task tool parameters:
- description: "Phase 2: Plan {model_id} optimization"
- subagent_type: "general-purpose"
- prompt: [Full prompt from task-prompts.md § Phase 2]
```

Makes decisions:
- **Decision 0**: Saturation score (from `references/algorithmic-branching.md`)
- **Decision 0b**: Per-expert M under uniform routing (M_avg = BS * top_k / E_global) or measured histogram
- **Decision 0c**: Ownership model + fusion boundary (token-major vs expert-major; cooperative vs split)
- **Decision 0d**: Baseline reference profiling summary (optional but recommended)
- **Decision 0e**: Baseline delta requirements (target savings vs combined-graph baseline)
- **Decision 1**: Output path (atomics vs direct write, depends on ownership)
- **Decision 2**: Shared expert strategy (from `references/architecture-pattern.md`)
- **Decision 3**: GEMM Strategy (Per-pair GEMV vs Expert-grouped GEMM)
- **Decision 4**: SRAM Tetris (from `references/tiling-config.md`) - **dtype affects buffer sizes**
- **Decision 5**: Warp configuration
- **Decision 6**: MMA instruction selection (based on dtype)

**Early Exit Conditions**: If Phase 2 concludes:
- Monokernel is not applicable → document in `optimization_plan.md`, stop before Phase 3
- Split kernels selected → Phase 3 implements split-kernel path instead
- Baseline delta implausible → re-plan or document limitation

### Phase 3: Implementation

**Subagent Type**: `general-purpose` (for all stages)

Phase 3 is structured into **4 stages** to keep GEMM work together.

**Spawn Task for EACH stage sequentially**. DO NOT implement multiple stages in one Task.

| Stage | Components | Rationale |
|-------|------------|-----------|
| routing_and_prepare | router + prepare | Tightly coupled, non-GEMM |
| activation_quantization | scale_inputs | Conditional (FP8 only) |
| **gemm_implementation** | **up_proj + down_proj** | Share structure (see note below) |
| kernel_assembly | output + main kernel | Wire everything together |

**GEMM Stage Flexibility**: `up_projection` and `down_projection` are implemented together **only** when:
- Single cooperative monokernel is chosen (Decision 0c) AND
- Both projections share the same decomposition

If ownership is token-major or hybrid, or if fusion boundary is split, treat up/down as separate kernels and allow different tiling/ownership per phase.

**For each stage**, spawn Task with:
```
Task tool parameters:
- description: "Phase 3 {stage_name}: {model} monokernel"
- subagent_type: "general-purpose"
- prompt: [COPY FULL prompt from task-prompts.md § Phase 3 → {stage_name}]
```

**Activation Function Handling**:
- Common activations (SiLU, GELU, ReLU): Use templates from [references/code-templates.md](references/code-templates.md)
- Custom/unknown: Use explore subagent to investigate, document findings

**GEMM Stage Self-Verification**: Tasks must verify no TODOs in MMA loops before completing. See [orchestration/task-prompts.md](orchestration/task-prompts.md).

### Phase 4: Validation & Investigation

Validate monokernel against stock `fused_moe` in 3 stages:

| Stage | Check | Success Criteria |
|-------|-------|------------------|
| 4.1 Correctness | Numerical accuracy | `max_diff < tolerance` (dtype-specific) |
| 4.2 Kernel-Level | Performance under CUDA graphs | `speedup >= 1.0x` at all batch sizes |
| 4.3 E2E Latency | Real inference impact | `>5%` improvement (BS≤8), `>0%` (BS>8) |

**Goal**: Beat the **combined routing+experts CUDA-graph baseline**, not just reduce kernel count.

For kernel regressions (4.2), investigations must include **reference FusedMoE profiling** under identical CUDA graph / torch.compile settings.

**On failure**: Orchestrator spawns investigation task (1 bounded cycle).
Investigation diagnoses root cause → council reviews fix proposal → decision:

| Decision | Action |
|----------|--------|
| `phase_3` | Back to Phase 3 (specific stage with fix context) |
| `phase_2` | Back to Phase 2 (re-plan with new constraints) |
| `document_proceed` | Document limitation, proceed to Phase 5 |
| `rerun_validation` | Re-spawn the failing validation stage (measurement/env issue) |
| `escalate_human` | Pause for human review |

See [validation/README.md](validation/README.md) and [orchestration/investigation-prompts.md](orchestration/investigation-prompts.md).

### Phase 5: Integration

Wire monokernel into vLLM:
- CMakeLists.txt modifications
- Torch bindings
- Python wrapper
- MoE layer fast-path dispatch


## Reference Materials

| Document | Purpose |
|----------|---------|
| [MODELS_COMPARISON.md](examples/MODELS_COMPARISON.md) | Detailed code snippets comparing Llama 4 vs Qwen3 patterns |
| [algorithmic-branching.md](references/algorithmic-branching.md) | Decisions A-G: output path, sorter, weight timing, shared experts, kernel arch, GEMM strategy, multi-expert accumulation |
| [gpu-configs.md](references/gpu-configs.md) | Hardware specs, monokernel thresholds, TMA support |
| [tiling-config.md](references/tiling-config.md) | SRAM Tetris formulas, dtype-aware tile size search |
| [router-design.md](references/router-design.md) | Top-k selection implementation patterns |
| [expert-grouping.md](references/expert-grouping.md) | Bitfield vs histogram token sorting |
| [optimization-techniques.md](references/optimization-techniques.md) | T1-T14 optimization patterns (includes Split-H for small batches) |
| [code-templates.md](references/code-templates.md) | C++/Python scaffolds, activation templates, block quant, timing |
| [architecture-pattern.md](references/architecture-pattern.md) | Controller-worker, split-H latency kernel patterns |

## Assets

| File | Purpose |
|------|---------|
| [LLAMA4_MONOKERNEL_PATCH.md](assets/LLAMA4_MONOKERNEL_PATCH.md) | Reference implementation (168KB) |
| [MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md](assets/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md) | 13 optimization techniques |

## Orchestration

| Document | Purpose |
|----------|---------|
| [task-prompts.md](orchestration/task-prompts.md) | Full behavioral specs for all phases/stages |
| [workflow.md](orchestration/workflow.md) | Phase state machine, stage grouping, compile cadence |
| [failure-handling.md](orchestration/failure-handling.md) | 6-level escalation ladder, blocker format, council protocol |
| [investigation-prompts.md](orchestration/investigation-prompts.md) | Phase 4 investigation task prompts (correctness, kernel-perf, e2e-perf) |

## Validation

| Document | Purpose |
|----------|---------|
| [validation/README.md](validation/README.md) | Details on accuracy, benchmarking, profiling verification |
| [validation/QWEN3_BASELINE.md](validation/QWEN3_BASELINE.md) | Qwen3 FP8 block-quant tolerances and performance targets |

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

## Validation Checklist

### Implementation (Phase 3)
- [ ] vLLM code analysis completed (Phase 1)
- [ ] Semantic differences from Llama 4 documented
- [ ] Weight application order correct (Decision C)
- [ ] Activation function handled (template or explored)
- [ ] **No TODOs in GEMM kernels** (Stage 3 verification)
- [ ] **MMA calls present in both up and down projection**
- [ ] `static_assert` all dimension divisibility constraints
- [ ] SMEM usage < SMEM_limit - 8KB margin
- [ ] Grid size ≤ SM count (for cooperative launch)

### Validation (Phase 4)
- [ ] **4.1 Correctness**: `max_diff < tolerance` for all batch sizes [1, 8, 64]
- [ ] **4.2 Kernel Performance**: `speedup >= 1.0x` at all batch sizes (no regressions)
- [ ] **4.3 E2E Latency**: `>5%` improvement at BS≤8, `>0%` at BS>8
- [ ] If any validation failed → investigation completed with decision
- [ ] If `document_proceed` → limitation documented in validation_results.md
