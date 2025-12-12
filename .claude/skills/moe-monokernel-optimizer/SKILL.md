---
name: moe-monokernel-optimizer
description: Design and implement fused MoE monokernel optimizations for vLLM inference. Use when optimizing Mixture-of-Experts layers for specific (model, vLLM config, hardware, dtype) deployments where router + quantization + GEMMs can be fused into a single cooperative CUDA kernel. Triggers on requests to optimize MoE decode latency, eliminate kernel launch overhead, reduce global memory round-trips, or implement specialized monokernels for models like Llama-4, DeepSeek, Qwen3-MoE, or any vLLM-supported MoE architecture. Supports FP8 W8A8, FP16, BF16, and other quantization formats on Hopper/Ada GPUs (H100, H200, L40S).
---

# MoE Monokernel Optimizer

Design specialized fused MoE kernels for vLLM that eliminate kernel launch overhead by combining router, quantization, and GEMMs into a single cooperative kernel.

## When Monokernel Applies

Monokernel optimization is beneficial when:
- Decode batch size is small (typically BS ≤ 32-128 depending on hardware)
- MoE preamble (router + quant + kernel launch gaps) exceeds 20-30% of layer time
- Static deployment: fixed (model, vLLM config, hardware, dtype) quadruple

## Supported Data Types

| Type | Element Size | MMA Instruction | Notes |
|------|-------------|-----------------|-------|
| FP8 E4M3 | 1 byte | mma.f32.f8.f8 | Best for inference, requires sm_89+ |
| BF16 | 2 bytes | mma.f32.bf16.bf16 | 2× SMEM cost vs FP8 |
| FP16 | 2 bytes | mma.f32.f16.f16 | Legacy compatibility |
| MXFP4 | 0.5 bytes | Experimental | Future support |

## Environment Requirement

This skill uses **Task subagents** for parallel execution. The Task tool is only available in **Claude Code CLI/IDE** environments.

**To verify**: Check if `Task(...)` appears in your available tools.

**If Task unavailable**: The orchestrator will implement phases sequentially (suboptimal but functional).

## 5-Phase Workflow

### Phase 1: Gather Constraints (with vLLM Code Analysis)

**NEW**: Phase 1 now analyzes the actual vLLM implementation code for the target model, not just config.json. This catches semantic differences that config parameters don't capture.

The task:
1. Locates the model's MoE implementation in vLLM source
2. Traces the forward path (routing → expert execution → accumulation)
3. Compares against Llama 4 reference semantics
4. Extracts config.json parameters
5. Synthesizes into constraints document

**Output**: `{artifact_dir}/constraints.md`

See [orchestration/task-prompts.md](orchestration/task-prompts.md) for full task specification.

### Phase 2: Optimization Planning

Apply algorithmic branching decisions:

**Decision A: Output Path**
```
IF top_k == 1:  USE_ATOMICS = false
IF top_k > 1:   USE_ATOMICS = true
```

**Decision B: Sorter Strategy**
```
coalesce_size = E_local × dtype_bytes
TOKENS_PER_WARP = 128 / coalesce_size if < 128 else 1
```

**Decision C: Weight Application Order** (CRITICAL for correctness)
```
IF top_k == 1:  APPLY_WEIGHT = before_activation  (can fold into scale)
IF top_k > 1:   APPLY_WEIGHT = after_activation   (MUST apply after SiLU)
```

**Decision F: GEMM Strategy** (Per-pair GEMV vs Expert-grouped GEMM)
```
λ = (BS_max × top_k) / E
r_max = λ / (1 - e^{-λ})
IF r_max >= 2.0:  USE_EXPERT_GROUPING = true   (Grouped-GEMM)
IF r_max < 2.0:   USE_EXPERT_GROUPING = false  (Per-pair GEMV)
```

Solve SRAM Tetris for tile sizes. See [references/tiling-config.md](references/tiling-config.md).

**Output**: `{artifact_dir}/optimization_plan.md`

### Phase 3: Implementation (4 Stages)

Phase 3 is structured into **4 stages** (not 6) to keep GEMM work together:

| Stage | Components | Rationale |
|-------|------------|-----------|
| routing_and_prepare | router + prepare | Tightly coupled, non-GEMM |
| activation_quantization | scale_inputs | Conditional (FP8 only) |
| **gemm_implementation** | **up_proj + down_proj** | Share 90% of structure |
| kernel_assembly | output + main kernel | Wire everything together |

**Critical change**: `up_projection` and `down_projection` are implemented **together** because they share:
- MMA instruction patterns
- Warp specialization (8 calc, 4 prefetch)
- Double-buffering logic
- K-chunk iteration structure

This prevents the "MMA loop TODO" problem where a task generates infrastructure but runs out of context before completing the compute loops.

**Activation Function Handling**:
- Common activations (SiLU, GELU, ReLU): Use templates from [references/code-templates.md](references/code-templates.md)
- Custom/unknown: Use explore subagent to investigate, document findings

**GEMM Stage Self-Verification**: Tasks must verify no TODOs in MMA loops before completing. See [orchestration/task-prompts.md](orchestration/task-prompts.md).

**Output**: CUDA files in `csrc/moe/moe_monokernel_{config}/`

### Phase 4: Validation

Compare monokernel output against stock `fused_moe`:
- Numerical correctness (max diff < 1e-2)
- Performance benchmarking (MoE layer & end to end) across batch sizes
- Use [validation/README.md](validation/README.md) for more details

**Output**: `{artifact_dir}/validation_results.md`

### Phase 5: Integration

Wire monokernel into vLLM:
- CMakeLists.txt modifications
- Torch bindings
- Python wrapper
- MoE layer fast-path dispatch

**Output**: Git patch and integration instructions

## State File

Location: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/state.json`

```json
{
  "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8",
  "hardware": "l40s",
  "dtype": "fp8",
  "tp": 1,
  "current_phase": "3_implementation",
  "phases": {
    "1_constraints": {"status": "complete"},
    "2_planning": {"status": "complete"},
    "3_implementation": {"status": "in_progress"},
    "4_validation": {"status": "pending"},
    "5_integration": {"status": "pending"}
  },
  "stages": {
    "routing_and_prepare": {"status": "complete"},
    "activation_quantization": {"status": "complete"},
    "gemm_implementation": {"status": "in_progress"},
    "kernel_assembly": {"status": "pending"}
  }
}
```

## Reference Materials

| Document | Purpose |
|----------|---------|
| [algorithmic-branching.md](references/algorithmic-branching.md) | Decisions A-F: output path, sorter, weight timing, shared experts, kernel arch, GEMM strategy |
| [gpu-configs.md](references/gpu-configs.md) | Hardware specs, monokernel thresholds, TMA support |
| [tiling-config.md](references/tiling-config.md) | SRAM Tetris formulas, dtype-aware tile size search |
| [router-design.md](references/router-design.md) | Top-k selection implementation patterns |
| [expert-grouping.md](references/expert-grouping.md) | Bitfield vs histogram token sorting |
| [optimization-techniques.md](references/optimization-techniques.md) | T1-T13 patterns from Llama-4 patch |
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

## Validation

| Document | Purpose |
|----------|---------|
| [validation/README.md](validation/README.md) | Details on accuracy, benchmarking, profiling verification |

## Quick Start

```markdown
User: "Use moe-monokernel-optimizer for Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 on g6e.24xlarge TP=1"

Orchestrator:
1. Reads SKILL.md
2. Spawns: Task("Phase 1: Gather constraints for Qwen3-Coder-30B-A3B-Instruct-FP8 on L40S...")
   ⏺ Task(Phase 1...) ⎿ Done (constraints.md created)
3. Spawns: Task("Phase 2: Create optimization plan...")
   ⏺ Task(Phase 2...) ⎿ Done (optimization_plan.md created)
4. Spawns implementation stages in sequence:
   ⏺ Task(router) ⎿ Done
   ⏺ Task(prepare) ⎿ Done
   ⏺ Task(up_projection) ⎿ Done
   ...
5. Spawns: Task("Phase 4: Validate correctness and performance...")
   ⏺ Task(Phase 4...) ⎿ Done (validation_results.md created)
6. Spawns: Task("Phase 5: Integrate into vLLM...")
   ⏺ Task(Phase 5...) ⎿ Done (integration patch created)
```

## Validation Checklist

- [ ] vLLM code analysis completed (Phase 1)
- [ ] Semantic differences from Llama 4 documented
- [ ] Weight application order correct (Decision C)
- [ ] Activation function handled (template or explored)
- [ ] **No TODOs in GEMM kernels** (Stage 3 verification)
- [ ] **MMA calls present in both up and down projection**
- [ ] `static_assert` all dimension divisibility constraints
- [ ] SMEM usage < SMEM_limit - 8KB margin
- [ ] Grid size ≤ SM count (for cooperative launch)
- [ ] Correctness test vs stock `fused_moe` output (max diff < 1e-2)
- [ ] Performance improvement documented across batch sizes
