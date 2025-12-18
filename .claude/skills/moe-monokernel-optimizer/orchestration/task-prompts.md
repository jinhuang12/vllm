# MoE Monokernel Task Prompts

This document contains the full behavioral spec task prompts for each phase and stage.

## Directory Convention

All artifacts go in: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/`
CUDA files go in: `csrc/moe/moe_monokernel_{model}_{hardware}_{dtype}_{tp}/`

Example: For Qwen3-30B-A3B-FP8 on L40S with TP=1:
- Artifacts: `moe_monokernel_artifacts/qwen3-30b-a3b_l40s_fp8_tp1/`
- CUDA: `csrc/moe/moe_monokernel_qwen3-30b-a3b_l40s_fp8_tp1/`

## Planning Phases Behavioral Footer

**Use for**: Phase 1 (Constraints), Phase 2 (Optimization Planning)

**Rationale**: Planning phases produce the plan/document itself - there's no separate "approach" to review before writing. Only final output review is required.

Append to Phase 1 and Phase 2 task prompts:

```markdown
## MANDATORY: LLM Council Review (BLOCKING)

**THIS IS NOT OPTIONAL. ALL TASKS MUST GO THROUGH LLM-COUNCIL REVIEW BEFORE COMPLETION.**

You MUST invoke the `llm-council` skill to review your **output** before marking this task complete.
DO NOT skip this step. DO NOT mark the task as complete without council approval.

### When to Invoke llm-council:
1. **AFTER completing the output** - Review your generated document (constraints.md, optimization_plan.md)
2. **When stuck** - After 2+ failed attempts at the same problem
3. **When uncertain** - About completeness, correctness, or implications

### How to Invoke:
Use the Skill tool with skill name `llm-council`. The skill has its own context preparation instructions.

### Review Loop (MANDATORY):
1. Complete your draft work (document or plan)
2. Invoke `llm-council` to review the output
3. If REJECTED with valid reasons → Revise and re-submit for another review
4. If ACCEPTED → You may proceed to mark task complete
5. **DO NOT mark task complete if review was skipped or rejected**

### Completion Checklist:
Before marking this task complete, verify:
- [ ] llm-council was invoked to review the output
- [ ] Review feedback was addressed
- [ ] Final output was ACCEPTED by council (or rejection reasons were invalid)

---

**Behavioral Expectations**:
1. **Read before write**: Read all referenced documents before drafting
2. **BLOCKING: Final review with llm-council**: Review your output AFTER completing it
   - This is MANDATORY - do not skip
   - Revise until accepted: If REJECTED with valid reasons, revise and re-submit
   - Only mark complete after approval
3. **Stuck** (error 2+ times even after consulting `llm-council`):
   - Document in `{artifact_dir}/blockers/{component}_blocker.md`:
     - Error message (full)
     - Attempts made with different approaches
     - Hypotheses for root cause
     - What you tried and why it didn't work
   - Exit with status "blocked"
4. **Stay goal-aligned**: This task contributes to: {ultimate_goal}

**State Management** (CRITICAL - Tasks have no shared memory):
1. **Read state first**: `cat {artifact_dir}/state.json` to understand current progress
2. **Update state when done**: Before completing, update the appropriate phase/stage status
3. **Track llm-council review**: Record review status in state (MANDATORY)
4. **Write state back**: Save updated state.json with new status, timestamp, council review, AND orchestrator metadata:
```json
{
  "phases": {
    "{current_phase}": {
      "status": "complete",
      "completed_at": "{timestamp}",
      "llm_council_review": {
        "invoked": true,
        "outcome": "ACCEPTED",
        "revision_count": 0,
        "summary": "Brief summary of council feedback"
      }
    }
  },
  "orchestrator": {
    "last_completed_task": "{task_description}",
    "next_action": "{next_phase_or_stage}",
    "resume_hint": "Read state.json, then spawn Task for {next_phase_or_stage}"
  }
}
```

**IMPORTANT**: If `llm_council_review.invoked` is false or missing, the task is NOT properly complete.

**Context Preservation** (for orchestrator compaction recovery):
- Current phase: {current_phase}
- Completed stages: {completed_stages}
- Remaining stages: {remaining_stages}
- State file: {artifact_dir}/state.json
- CUDA directory: {cuda_dir}
- Skill file: `.claude/skills/moe-monokernel-optimizer/SKILL.md`
- Task prompts: `.claude/skills/moe-monokernel-optimizer/orchestration/task-prompts.md`

**After Context Compaction**:
If the orchestrator's context is compacted and you need to resume:
1. Re-read state.json to understand current progress
2. Re-read SKILL.md to understand the workflow
3. The orchestrator spawns Tasks - it does not implement directly
4. Check `orchestrator.resume_hint` in state.json for next action
```

---

## Implementation Phases Behavioral Footer

**Use for**: Phase 3 (Implementation stages), Phase 4 (Validation stages), Phase 5 (Integration)

**Rationale**: Implementation phases benefit from TWO review checkpoints - one to validate the approach BEFORE coding, and one to validate the final implementation AFTER coding.

Append to Phase 3, Phase 4, and Phase 5 task prompts:

```markdown
## MANDATORY: LLM Council Review (BLOCKING) - TWO CHECKPOINTS

**THIS IS NOT OPTIONAL. IMPLEMENTATION TASKS REQUIRE TWO REVIEW CHECKPOINTS.**

### Checkpoint 1: BEFORE Implementation (Approach Review)
After reading references and designing your approach, invoke `llm-council` to review your **planned approach** BEFORE writing implementation code.

### Checkpoint 2: AFTER Implementation (Final Review)
After completing code/validation, invoke `llm-council` to review your **final implementation** before marking task complete.

### When to Invoke llm-council:
1. **Checkpoint 1 (BEFORE coding)**: After designing approach, before writing code
2. **Checkpoint 2 (AFTER coding)**: Before marking task complete
3. **When stuck** - After 2+ failed attempts at the same problem
4. **When uncertain** - About approach, correctness, or implementation decisions

### How to Invoke:
Use the Skill tool with skill name `llm-council`. The skill has its own context preparation instructions.

### Review Loop (MANDATORY - Both checkpoints required):
1. Draft your approach/plan
2. **Checkpoint 1**: Invoke `llm-council` to review approach
3. If REJECTED → Revise approach and re-submit until ACCEPTED
4. Implement based on approved approach
5. **Checkpoint 2**: Invoke `llm-council` to review final implementation
6. If REJECTED → Revise and re-submit until ACCEPTED
7. **Only mark task complete after BOTH checkpoints pass**

### Completion Checklist:
Before marking this task complete, verify:
- [ ] llm-council Checkpoint 1 (approach) was invoked and ACCEPTED
- [ ] Implementation completed based on approved approach
- [ ] llm-council Checkpoint 2 (implementation) was invoked and ACCEPTED
- [ ] All validation tests passed (if applicable)

---

**Behavioral Expectations**:
1. **Read before write**: Read all referenced documents before drafting an implementation plan
2. **BLOCKING: Checkpoint 1 - Review approach with llm-council**: Before coding
   - This is MANDATORY - do not skip
   - Revise until accepted: If REJECTED with valid reasons, revise and re-submit
   - Only proceed to implementation after approach is approved
3. **Compile often**: Run `cmake --build --preset release --target install` after each major change:
   - On compile error:
    - Read the full error message
    - Identify root cause (typo, missing include, dimension mismatch, type error)
    - Fix and retry (up to 2 attempts per distinct error)
    - Consult with `llm-council` if unable to fix after 2 attempts
4. **Stuck** (error 2+ times even after consulting `llm-council`):
   - Document in `{artifact_dir}/blockers/{component}_blocker.md`:
     - Error message (full)
     - Attempts made with different approaches
     - Hypotheses for root cause
     - What you tried and why it didn't work
   - Exit with status "blocked"
5. **BLOCKING: Checkpoint 2 - Review implementation with llm-council**: Before completing
   - This is MANDATORY - do not skip
   - Revise until accepted: If REJECTED with valid reasons, revise and re-submit
   - Only mark complete after both checkpoints pass
6. **Stay goal-aligned**: This task contributes to: {ultimate_goal}

**State Management** (CRITICAL - Tasks have no shared memory):
1. **Read state first**: `cat {artifact_dir}/state.json` to understand current progress
2. **Update state when done**: Before completing, update the appropriate phase/stage status
3. **Track llm-council reviews**: Record BOTH checkpoint statuses in state (MANDATORY)
4. **Write state back**: Save updated state.json with new status, timestamp, council reviews, AND orchestrator metadata:
```json
{
  "phases": {
    "{current_phase}": {
      "status": "complete",
      "completed_at": "{timestamp}",
      "llm_council_review": {
        "checkpoint_1_approach": {
          "invoked": true,
          "outcome": "ACCEPTED",
          "revision_count": 0,
          "summary": "Brief summary of approach review feedback"
        },
        "checkpoint_2_implementation": {
          "invoked": true,
          "outcome": "ACCEPTED",
          "revision_count": 0,
          "summary": "Brief summary of implementation review feedback"
        }
      }
    }
  },
  "orchestrator": {
    "last_completed_task": "{task_description}",
    "next_action": "{next_phase_or_stage}",
    "resume_hint": "Read state.json, then spawn Task for {next_phase_or_stage}"
  }
}
```

**IMPORTANT**: If either checkpoint's `invoked` is false or missing, the task is NOT properly complete.

**Context Preservation** (for orchestrator compaction recovery):
- Current phase: {current_phase}
- Completed stages: {completed_stages}
- Remaining stages: {remaining_stages}
- State file: {artifact_dir}/state.json
- CUDA directory: {cuda_dir}
- Skill file: `.claude/skills/moe-monokernel-optimizer/SKILL.md`
- Task prompts: `.claude/skills/moe-monokernel-optimizer/orchestration/task-prompts.md`

**After Context Compaction**:
If the orchestrator's context is compacted and you need to resume:
1. Re-read state.json to understand current progress
2. Re-read SKILL.md to understand the workflow
3. The orchestrator spawns Tasks - it does not implement directly
4. Check `orchestrator.resume_hint` in state.json for next action
```

---

## Phase 1: Gather Constraints (with vLLM Code Analysis)

**Task Tool Parameters**:
- `description`: "Phase 1: Gather MoE constraints for {model_id}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Phase 1: Gather MoE monokernel constraints for {model_id} on {hardware} with TP={tp}, dtype={dtype}.

**Ultimate Goal**: Successfully integrate optimized MoE monokernel for {model_id} on {hardware}, targeting decode batch sizes ≤64.

**Artifact Directory**: moe_monokernel_artifacts/{model_short}_{hardware_lower}_{dtype}_{tp}/
**CUDA Directory**: csrc/moe/moe_monokernel_{model_short}_{hardware_lower}_{dtype}_{tp}/

---

## Part A: Locate and Analyze vLLM MoE Implementation (CRITICAL)

Before extracting config parameters, you MUST understand how vLLM actually implements MoE for this model.
This catches implementation details that config.json doesn't capture.

### A1. Find the model's MoE implementation path

Search for the model class in vLLM:
```bash
# Find model file
find vllm/model_executor/models -name '*.py' | xargs grep -l '{model_family}' | head -5

# Once found, look for MoE layer
grep -n 'FusedMoE\|MixtureOfExperts\|MoELayer\|fused_moe\|fused_experts' vllm/model_executor/models/{model_file}.py
```

Trace the MoE forward path:
1. Where is the router output computed? (usually `gate_proj` or `router`)
2. What function processes router logits? (`fused_topk`, custom routing?)
3. What expert execution function is called? (`fused_experts`, `fused_moe`, custom?)
4. How are expert outputs accumulated?

### A2. Analyze the MoE forward() implementation

Find and read the forward pass. Look for patterns like:
```python
def forward(self, hidden_states, router_logits):
    topk_weights, topk_ids = fused_topk(...)  # or custom
    # ... expert processing ...
    return final_output
```

Document each step:
1. **Routing computation**: softmax? sigmoid? renormalize after top-k?
2. **Weight application timing**: when is topk_weight multiplied?
3. **Activation function**: silu? gelu? custom?
4. **Scale handling**: where are quantization scales applied?
5. **Output accumulation**: simple sum? weighted? in-place?

### A3. Compare against Llama 4 semantics

For each semantic difference found, document in the delta report:

| Stage | Llama 4 Behavior | {model} Behavior | Code Location | Monokernel Impact |
|-------|------------------|------------------|---------------|-------------------|
| Routing | sigmoid scoring | {softmax/sigmoid/custom} | {file:line} | {affects router kernel} |
| Weight timing | apply before SiLU (top_k=1) | {when applied?} | {file:line} | {affects up_proj reduction} |
| Activation | SiLU | {activation} | {file:line} | {fused kernel formula} |
| Accumulation | direct write (top_k=1) | {atomic? sum?} | {file:line} | {output path strategy} |
| Shared experts | separate sidecar | {method} | {file:line} | {additional compute path} |

### A4. Check quantization implementation

If model uses FP8/INT8:
```bash
grep -n 'quantization\|scale\|quant_config\|block_size' vllm/model_executor/models/{model_file}.py
```

Document:
- Scale format: per-tensor, per-channel, or block-wise?
- Block size if block-quantized: typically [128, 128]
- When scales are applied (during GEMM accumulation? post-GEMM?)

---

## Part B: Extract Model Configuration

Now that you understand the implementation, extract the parameters.

### B1. Fetch config.json
```bash
curl -s 'https://huggingface.co/{model_id}/raw/main/config.json' | python -m json.tool
```

### B2. Extract MoE parameters

Standard fields (try multiple possible names):
```python
moe_fields = {
    'hidden_size': ['hidden_size'],
    'intermediate_size': ['moe_intermediate_size', 'intermediate_size'],
    'num_experts': ['num_local_experts', 'num_experts', 'n_routed_experts'],
    'num_experts_per_tok': ['num_experts_per_tok', 'top_k'],
    'num_shared_experts': ['num_shared_experts', 'n_shared_experts'],  # default 0
    'hidden_act': ['hidden_act', 'activation_function'],  # default 'silu'
    'norm_topk_prob': ['norm_topk_prob'],  # renormalize?
    'scoring_func': ['scoring_func'],  # 'softmax' or 'sigmoid'
    'routed_scaling_factor': ['routed_scaling_factor'],  # DeepSeek uses this
}
```

### B3. Determine data types
- Check model name for hints (-FP8, -AWQ, -GPTQ)
- Check `quantization_config` in config.json
- For block-quantized: note `weight_block_size`

---

## Part C: Reference Study

Read these documents for optimization patterns:

1. `assets/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md` - 13 techniques
2. Skim `assets/LLAMA4_MONOKERNEL_PATCH.md` - structure and patterns
3. `references/gpu-configs.md` - hardware specs
4. `references/algorithmic-branching.md` - Decision C for weight timing

---

## Part D: Synthesize Constraints Document

**Output**: Write to `{artifact_dir}/constraints.md`

```markdown
# MoE Monokernel Constraints

## Target
- Model: {model_id}
- Model short: {model_short}
- Hardware: {hardware}
- Parallelism: TP={tp}, EP={ep}
- Artifact dir: {artifact_dir}
- CUDA dir: {cuda_dir}

## Implementation Analysis (from vLLM code review)

### MoE Code Path
- Model file: `vllm/model_executor/models/{file}.py`
- MoE layer class: {class name}
- Router function: {function} at line {N}
- Expert function: {function} at line {N}

### Semantic Differences from Llama 4

| Aspect | Llama 4 | This Model | Monokernel Implication |
|--------|---------|------------|------------------------|
| Scoring | sigmoid | {actual} | Router kernel changes |
| Top-k weight timing | before SiLU | {after/before/during} | Up-proj reduction order (see Decision C) |
| Activation | SiLU | {actual} | Fused activation formula |
| Renormalize weights | no | {yes/no} | Routing kernel |
| Shared experts | sidecar | {method} | Additional kernel path |
| Scale application | per-tensor | {method} | GEMM accumulation |

### CRITICAL Implementation Notes
{List any gotchas discovered from code analysis}

## Data Types
| Component | Type | Size | Notes |
|-----------|------|------|-------|
| Weight | {weight_dtype} | {1 or 2} byte | {block/per-tensor scale?} |
| Activation | {activation_dtype} | {1 or 2} byte | |
| Scale | {scale_format} | 4 bytes | {shape if block-wise} |
| Accumulator | FP32 | 4 bytes | Always FP32 |

## Activation Function
- Type: {silu/gelu/geglu/relu/custom}
- Formula: {mathematical formula}
- Template available: {yes/no - see references/code-templates.md}
- If custom/unsupported: Flag for exploration in Phase 3

## Reference Comparison (vs Llama 4)
| Aspect | Llama 4 | {model_id} | Implication |
|--------|---------|------------|-------------|
| top_k | 1 | {top_k} | {atomics needed?} |
| Shared experts | 1 | {shared} | {sidecar needed?} |
| E_local | 16/128 | {E_local} | {bitfield size?} |
| N_local | 1024 | {N_local} | {SMEM pressure?} |
| Weight dtype | FP8 E4M3 | {weight_dtype} | {MMA instruction?} |

## Model Geometry
| Parameter | Global | Local |
|-----------|--------|-------|
| Hidden size (K) | {K} | {K} |
| Intermediate (N) | {N_global} | {N_local} |
| Experts (E) | {E_global} | {E_local} |
| Top-K | {top_k} | {top_k} |
| Shared experts | {shared} | {shared} |

## Hardware: {hardware}
- Architecture: {sm_xx}
- Shared memory/SM: {total} KB
- Usable SMEM: {usable} KB
- SM count: {count}
- TMA support: {yes/no}
- Max cooperative grid: {sm_count} blocks

## Weight Shapes (after TP split)
- W_up (w13): [{E_local}, {2*N_local}, {K}] {weight_dtype}
- W_down (w2): [{E_local}, {K}, {N_local}] {weight_dtype}
- Scales: {scale_format} {shape if block-wise}

## Activation Flow
- Input: [{BS}, {K}] BF16
- After quantize (if needed): [{BS}, {K}] {activation_dtype}
- After router: topk_ids[{BS}], topk_weights[{BS}]
- After up-proj: [{BS}, {2*N_local}] FP32 → {activation} → [{BS}, {N_local}]
- After down-proj: [{BS}, {K}] FP32
- Output: [{BS}, {K}] BF16

## Applicable Techniques from Guide
- [ ] T1: Batch-size specialization (BS≤8 vs BS≤64)
- [ ] T2: Warp specialization (8:4 compute:prefetch)
- [ ] T3: Triple buffering
- [ ] T4: Bank conflict mitigation (swizzle vs padding)
- [ ] T5: Custom FP8 MMA (FP8 only - adapt for BF16/FP16)
- [ ] T6: Bitfield expert dedup (if E≤128)
- [ ] T7: Scale folding into routing weights
- [ ] T8: Speculative compute + mask filter
- [ ] T9: Branchless expert selection
- [ ] T10: Prefix sum token sorting
- [ ] T11: Activation fusion with GEMM ({activation} - see templates)
- [ ] T12: Vectorized dtype conversion with NaN handling
- [ ] T13: Cooperative grid sync
```

{planning_behavioral_footer}
```

---

## Phase 2: Optimization Planning

**Task Tool Parameters**:
- `description`: "Phase 2: Plan {model_id} optimization"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Phase 2: Create optimization plan for {model_id} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Prerequisites**: Read `{artifact_dir}/constraints.md`

**Objective**: Apply algorithmic branching decisions and SRAM Tetris to produce a concrete optimization plan.

**Steps**:

1. Read all prerequisite docs:
   - `{artifact_dir}/constraints.md` (from Phase 1)
   - `references/algorithmic-branching.md`
   - `references/tiling-config.md`
   - `references/code-templates.md` (especially activation templates)

2. Apply Decision A (Output Path):
   ```
   top_k = {top_k from constraints}
   USE_ATOMICS = (top_k > 1)
   ```

3. Apply Decision B (Sorter Strategy):
   ```
   coalesce_size = E_local × dtype_bytes
   TOKENS_PER_WARP = 128 / coalesce_size if coalesce_size < 128 else 1
   ```

4. Apply Decision C (Weight Application Order):
   ```
   if top_k == 1:
       APPLY_WEIGHT = 'before_silu'  # Can fold into scale
   else:
       APPLY_WEIGHT = 'after_silu'   # Must apply after activation
   ```

5. Solve SRAM Tetris:
   - Use tiling-config.md formulas with dtype from constraints
   - Search for (M_tile, N_tile, K_chunks, buffers) that fits
   - Document the search process and why final choice was made

6. Determine activation function approach:
   - If silu: Use standard template from code-templates.md
   - If gelu/geglu/relu: Use template from code-templates.md
   - If custom/unknown: Flag for exploration subagent in Phase 3

**Output**: Write to `{artifact_dir}/optimization_plan.md`

Template:
```markdown
# Optimization Plan: {model_short} on {hardware}

## Algorithmic Decisions

### Decision A: Output Path
- top_k = {N}
- USE_ATOMICS = {true/false}
- Rationale: {why}

### Decision B: Sorter Strategy  
- E_local = {N}
- dtype_bytes = {1 or 2}
- coalesce_size = {N}
- TOKENS_PER_WARP = {N}
- Rationale: {why}

### Decision C: Weight Application Order
- top_k = {N}
- APPLY_WEIGHT = {before_silu/after_silu}
- Rationale: {why this matters for correctness}

## SRAM Tetris Solution

### Search Process
{Show the cascade of attempts and why each failed/succeeded}

### Final Configuration
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| M_tile | {8 or 16} | {why} |
| N_tile | {16/32/64} | {why} |
| K_chunks | {1/2/4} | {why} |
| Buffers (up) | {2 or 3} | {why} |
| Buffers (down) | 2 | standard |

### SMEM Budget
- Available: {N} KB
- Used: {N} KB ({breakdown})
- Margin: {N} KB

## Activation Function

### Type: {silu/gelu/geglu/relu/custom}
### Template: {available/needs_exploration}
### CUDA Formula:
```cpp
// {formula}
```

## Kernel Configuration

```cpp
struct Config_{model_short} {{
    // Dimensions
    static constexpr uint32_t K = {K};
    static constexpr uint32_t N = {N_local};
    static constexpr uint32_t E = {E_local};
    static constexpr uint32_t TOP_K = {top_k};
    
    // Decisions
    static constexpr bool USE_ATOMICS = {true/false};
    static constexpr uint32_t TOKENS_PER_WARP = {N};
    static constexpr bool WEIGHT_AFTER_ACTIVATION = {true/false};
    
    // Tiles
    static constexpr uint32_t M_TILE = {N};
    static constexpr uint32_t N_TILE = {N};
    static constexpr uint32_t K_CHUNKS = {N};
    static constexpr uint32_t UP_BUFFERS = {N};
    
    // Grid
    static constexpr uint32_t CALC_WARPS = 8;
    static constexpr uint32_t PREFETCH_WARPS = 4;
    static constexpr uint32_t BLOCK_SIZE = 384;
    static constexpr uint32_t GRID_SIZE = {SM_count};
}};
```

## Stage Implementation Notes

{For each stage, note any model-specific adaptations needed}
```

{planning_behavioral_footer}
```

---

## Phase 3: Implementation Stages

Phase 3 is restructured into 4 stages (not 6) to keep related work together:

| Stage | Components | Rationale |
|-------|------------|-----------|
| 1. routing_and_prepare | router + prepare | Tightly coupled, non-GEMM |
| 2. activation_quantization | scale_inputs | Conditional (FP8 only) |
| 3. gemm_implementation | up_proj + down_proj | **CRITICAL**: Share MMA patterns |
| 4. kernel_assembly | output + main kernel | Wire everything together |

### Stage 1: Routing and Prepare

**Task Tool Parameters**:
- `description`: "Phase 3 routing_and_prepare: {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Implement routing and token preparation for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md` - especially 'Semantic Differences' section
- `{artifact_dir}/optimization_plan.md` - Decisions A, B, F
- `references/router-design.md`
- `references/expert-grouping.md` - **note the Decision F cross-reference**
- `references/algorithmic-branching.md` - Decision F (GEMM strategy)
- `assets/LLAMA4_MONOKERNEL_PATCH.md` & `examples/MODELS_COMPARISON.md` - search for 'moe_routing.cu' and 'moe_prepare.cu'

**Objective**: Implement top-k routing AND token-by-expert sorting in one cohesive unit.

**FIRST: Calculate Decision F (GEMM Strategy)**:

Before implementing sorting, calculate whether expert grouping is worthwhile:

```python
import math
E = {num_experts}  # From constraints
k = {top_k}        # From constraints
B_max = {bs_max}   # Target max batch size (typically 32 or 64)

λ = (B_max * k) / E
r_max = λ / (1 - math.exp(-λ)) if λ >= 0.01 else 1.0

print(f"λ = {λ:.2f}, r_max = {r_max:.2f}")
if r_max >= 2.0:
    print("Decision F: USE_EXPERT_GROUPING = True (Grouped-GEMM)")
else:
    print("Decision F: USE_EXPERT_GROUPING = False (Per-pair GEMV)")
```

**Document in constraints.md**:
- Add λ and r_max values
- Add Decision F outcome
- If r_max < 2.0, note that per-pair GEMV is preferred and expert grouping overhead may be wasteful

**Key Considerations from Constraints**:
- Scoring function: {from constraints - softmax/sigmoid}
- Renormalize: {from constraints - yes/no}
- top_k: {from constraints}
- TOKENS_PER_WARP: {from plan - Decision B}
- USE_EXPERT_GROUPING: {from Decision F calculation above}

**Implementation**:
1. Create `{cuda_dir}/moe_routing.cu`:
   - Top-k selection matching the model's scoring function
   - If renormalize=true, normalize weights after selection
   - Store topk_ids and topk_weights in shared memory

2. Create `{cuda_dir}/moe_prepare.cu`:
   - Token sorting by expert (bitfield or histogram based on E count)
   - Expert reference struct population
   - Pair index computation for GEMM stages

**Validation (BLOCKING - REQUIRED before completing)**:

1. **Build**:
```bash
cmake --build --preset release --target install
```

2. **Correctness Test** - Run this validation inline:
```python
import torch
from vllm.model_executor.layers.fused_moe import fused_topk

# Test parameters from constraints
batch_size, E, top_k = 8, {E}, {top_k}
router_logits = torch.randn(batch_size, E, device='cuda', dtype=torch.bfloat16)

# Reference: vLLM fused_topk
ref_weights, ref_ids = fused_topk(router_logits, topk=top_k, renormalize={renormalize})

# Your routing implementation (adapt import to your kernel name)
from vllm._custom_ops import moe_monokernel_{model_short}_routing
mono_weights, mono_ids = moe_monokernel_{model_short}_routing(router_logits, top_k)

# Validate - IDs must match exactly (same top-k selection)
assert torch.equal(mono_ids, ref_ids), f"TopK IDs mismatch: mono={mono_ids[:2]} vs ref={ref_ids[:2]}"

# Weights should be close (floating point tolerance)
torch.testing.assert_close(mono_weights, ref_weights, atol=1e-5, rtol=1e-5)
print("Stage 1 routing_and_prepare validation: PASS")
```

3. **If validation fails** → Exit with status "blocked" and include:
   - The assertion error message
   - Sample mismatched values
   - Your hypothesis for the discrepancy

{implementation_behavioral_footer}
```

### Stage 2: Activation Quantization (Conditional)

**Task Tool Parameters**:
- `description`: "Phase 3 activation_quantization: {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Implement activation quantization for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**CONDITIONAL**: This stage is only needed if weight_dtype is FP8 or INT8.
If weight_dtype is BF16/FP16, this stage can be minimal (just copy activations).

**Read First**:
- `{artifact_dir}/constraints.md` - check Data Types section
- `{artifact_dir}/optimization_plan.md`
- `assets/LLAMA4_MONOKERNEL_PATCH.md` & `examples/MODELS_COMPARISON.md` - search for 'moe_scale_inputs.cu'

**Implementation**:
1. Create `{cuda_dir}/moe_scale_inputs.cu`

If FP8/INT8:
- Compute per-token max absolute value
- Compute scale = max_abs / FP8_MAX
- Quantize activations: fp8_val = bf16_val / scale
- Fold scale into topk_weights for later compensation

If BF16/FP16 (no quantization needed):
- Simple passthrough or direct copy to SMEM
- No scale folding needed

**Validation (BLOCKING - REQUIRED before completing)**:

1. **Build**:
```bash
cmake --build --preset release --target install
```

2. **Correctness Test** (FP8/INT8 only - skip for BF16/FP16):
```python
import torch

# Test parameters from constraints
batch_size, K = 8, {K}
activations = torch.randn(batch_size, K, device='cuda', dtype=torch.bfloat16) / 10

# Reference: Torch dynamic quantization
FP8_MAX = 448.0  # E4M3 max value
ref_scale = activations.abs().amax(dim=-1, keepdim=True) / FP8_MAX
ref_quantized = (activations / ref_scale).clamp(-FP8_MAX, FP8_MAX)

# Your quantization implementation
from vllm._custom_ops import moe_monokernel_{model_short}_quantize
mono_quantized, mono_scale = moe_monokernel_{model_short}_quantize(activations)

# Validate scales match (critical for GEMM correctness)
torch.testing.assert_close(mono_scale, ref_scale, atol=1e-3, rtol=1e-3)

# Validate dequantized values (roundtrip)
ref_dequant = ref_quantized * ref_scale
mono_dequant = mono_quantized.float() * mono_scale
torch.testing.assert_close(mono_dequant, ref_dequant.float(), atol=0.5, rtol=0.1)
print("Stage 2 activation_quantization validation: PASS")
```

3. **If validation fails** → Exit with status "blocked" and include:
   - Scale comparison values
   - Maximum dequantization error
   - Your hypothesis for the discrepancy

{implementation_behavioral_footer}
```

### Stage 3: GEMM Implementation (CRITICAL - Up + Down Together)

This is the most complex stage. **Up-projection and down-projection are implemented together**
because they share 90% of their structure (MMA patterns, warp specialization, double buffering).

**Task Tool Parameters**:
- `description`: "Phase 3 gemm_implementation: {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Implement GEMM kernels (up-projection AND down-projection) for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**CRITICAL**: This task implements BOTH up-projection and down-projection together.
They share the same MMA infrastructure - implement common helpers once, apply to both.

**Read First** (IN THIS ORDER):
1. `{artifact_dir}/optimization_plan.md` - Decisions C, F (weight timing, MMA details)
2. `{artifact_dir}/constraints.md` - dimensions, dtype
3. `assets/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md` - **13 optimization techniques** (warp specialization, triple buffering, bank conflict mitigation, custom MMA patterns, bitfield deduplication)
4. `references/code-templates.md` - MMA templates section (FP8 AND BF16)
5. `references/tiling-config.md` - SMEM layout
6. `assets/LLAMA4_MONOKERNEL_PATCH.md` & `examples/MODELS_COMPARISON.md`  - search for 'moe_up_projection.cu', 'moe_down_projection.cu'

**Key Configuration**:
- weight_dtype: {fp8/bf16/fp16}
- APPLY_WEIGHT: {before_activation/after_activation} (Decision C)
- K_CHUNKS_UP: ceil(K / K_TILE)
- K_CHUNKS_DOWN: ceil(N / K_TILE)
- USE_BLOCK_QUANT: {true/false}

---

## PART A: Shared MMA Infrastructure

Create `{cuda_dir}/moe_mma_common.cuh` with:

1. **MMA wrapper function** (dtype-specific):
```cpp
// For BF16:
__device__ __forceinline__ void mma_bf16_bf16(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1,
    float c0, float c1, float c2, float c3);

// For FP8:
__device__ __forceinline__ void mma_fp8_fp8(...);
```

2. **Fragment loaders**:
```cpp
template <uint32_t BS_MAX>
__device__ void load_A_fragment(const dtype* smem, uint32_t row, uint32_t k_offset,
                                 uint32_t& a0, uint32_t& a1, uint32_t& a2, uint32_t& a3);

template <uint32_t BS_MAX>
__device__ void load_B_fragment(const dtype* smem, uint32_t n_row, uint32_t k_offset,
                                 uint32_t& b0, uint32_t& b1);
```

3. **K-chunk iteration template**:
```cpp
template <uint32_t BS_MAX, uint32_t K_CHUNKS, typename ComputeFunc>
__device__ void iterate_k_chunks(
    SharedMem* smem,
    uint32_t read_buf,
    ComputeFunc compute);
```

---

## PART B: Up-Projection Implementation

Create `{cuda_dir}/moe_up_projection.cu`:

1. SMEM async prefetch for weights [2*N_TILE, K_TILE]
2. SMEM async prefetch for activations [M_TILE, K_TILE]
3. **THE MMA LOOP** (MUST BE COMPLETE - NO TODOs):

```cpp
// This loop MUST have actual MMA calls - not placeholders
for (uint32_t k_chunk = 0; k_chunk < K_CHUNKS_UP; k_chunk++) {
    // Wait for async loads
    asm volatile("cp.async.wait_group 0;");
    __syncthreads();
    
    if (is_calc_warp<BS_MAX>()) {
        // Inner K iteration (K_TILE / 16 for m16n8k16)
        for (uint32_t k_inner = 0; k_inner < K_TILE; k_inner += 16) {
            // Load A fragment
            load_A_fragment<BS_MAX>(gemm1.a[read_buf], row, k_inner, a0, a1, a2, a3);
            
            // Load B fragments (x and gate rows)
            load_B_fragment<BS_MAX>(gemm1.w[read_buf], n_row, k_inner, b0_x, b1_x);
            load_B_fragment<BS_MAX>(gemm1.w[read_buf], n_row + N_TILE, k_inner, b0_gate, b1_gate);
            
            // MMA for x path
            mma_{dtype}_{dtype}(d0_x, d1_x, d2_x, d3_x, a0, a1, a2, a3, b0_x, b1_x, d0_x, d1_x, d2_x, d3_x);
            
            // MMA for gate path
            mma_{dtype}_{dtype}(d0_g, d1_g, d2_g, d3_g, a0, a1, a2, a3, b0_gate, b1_gate, d0_g, d1_g, d2_g, d3_g);
        }
    }
    
    // Swap buffers, issue next prefetch
    read_buf = 1 - read_buf;
    __syncthreads();
}
```

4. **Activation fusion** (after MMA loop completes):
   - Apply SiLU/GELU/etc per Decision C
   - Apply topk_weight at correct stage
   - Store to intermediate buffer

---

## PART C: Down-Projection Implementation

Create `{cuda_dir}/moe_down_projection.cu`:

Nearly identical structure to up-projection, but:
- Input: intermediate buffer [BS*TOP_K, N_LOCAL]
- Output: accumulator [BS, K]
- K_CHUNKS_DOWN = ceil(N_LOCAL / K_TILE) instead of ceil(K / K_TILE)
- Output uses atomicAdd if USE_ATOMICS=true (top_k > 1)

**THE MMA LOOP** (same structure, different dimensions):
```cpp
for (uint32_t k_chunk = 0; k_chunk < K_CHUNKS_DOWN; k_chunk++) {
    // ... same pattern as up-projection ...
    
    for (uint32_t k_inner = 0; k_inner < K_TILE; k_inner += 16) {
        load_A_fragment<BS_MAX>(...);
        load_B_fragment<BS_MAX>(...);
        mma_{dtype}_{dtype}(d0, d1, d2, d3, ...);
    }
}
```

---

## VALIDATION (BLOCKING - REQUIRED before completing)

Before completing this task, you MUST pass ALL of these validations:

### 1. Build Check
```bash
cmake --build --preset release --target install
```

### 2. Code Completeness Check
```bash
# Search for incomplete code markers
grep -n 'TODO\|FIXME\|XXX\|unimplemented' {cuda_dir}/moe_*_projection.cu
```
**If ANY results** in MMA loops or compute paths → Exit BLOCKED. Do not proceed.

### 3. MMA Implementation Check
```bash
# Verify MMA calls exist in both projections
UP_MMA=$(grep -c 'mma_' {cuda_dir}/moe_up_projection.cu)
DOWN_MMA=$(grep -c 'mma_' {cuda_dir}/moe_down_projection.cu)
echo "Up projection MMA calls: $UP_MMA"
echo "Down projection MMA calls: $DOWN_MMA"
```
**If either count is 0** → MMA loop not implemented. Exit BLOCKED.

### 4. Correctness Test (CRITICAL)
Run this validation to compare against torch.matmul reference:
```python
import torch
import time

# Test parameters from constraints
batch_size, K, N, E, top_k = 8, {K}, {N}, {E}, {top_k}

# Create test inputs
torch.manual_seed(42)  # Reproducibility
activations = torch.randn(batch_size, K, device='cuda', dtype=torch.bfloat16) / 10
topk_ids = torch.randint(0, E, (batch_size, top_k), device='cuda')
topk_weights = torch.softmax(torch.randn(batch_size, top_k, device='cuda'), dim=-1)
w_up = torch.randn(E, 2*N, K, device='cuda', dtype={weight_dtype}) / 10
w_down = torch.randn(E, K, N, device='cuda', dtype={weight_dtype}) / 10

# Reference: Naive torch implementation (known correct)
ref_output = torch.zeros(batch_size, K, device='cuda', dtype=torch.float32)
for b in range(batch_size):
    for i in range(top_k):
        expert_id = topk_ids[b, i].item()
        weight = topk_weights[b, i].item()
        # Up projection: [K] -> [2*N]
        up_out = torch.matmul(activations[b].float(), w_up[expert_id].float().T)
        gate, x = up_out[:N], up_out[N:]
        # Activation: SiLU-gated (adjust if model uses different activation)
        activated = torch.nn.functional.silu(gate) * x * weight
        # Down projection: [N] -> [K]
        down_out = torch.matmul(activated, w_down[expert_id].float().T)
        ref_output[b] += down_out

# Your GEMM implementation (adapt import to your kernel)
from vllm._custom_ops import moe_monokernel_{model_short}_gemm
mono_output = moe_monokernel_{model_short}_gemm(
    activations, topk_ids, topk_weights, w_up, w_down
)

# Correctness validation with dtype-appropriate tolerance
# BF16: atol=1e-2, rtol=1e-2 | FP8: atol=1.0, rtol=0.1
torch.testing.assert_close(mono_output, ref_output, atol={atol}, rtol={rtol})
print("GEMM correctness: PASS")
```
**If assertion fails** → Exit BLOCKED with the error message and max diff values.

### 5. Performance Sanity Check (CRITICAL - catches naive implementations)
```python
# This catches "naive GEMM" that is correct but 100x+ slower
torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(100):
    moe_monokernel_{model_short}_gemm(activations, topk_ids, topk_weights, w_up, w_down)
torch.cuda.synchronize()
elapsed_ms = (time.perf_counter() - start) / 100 * 1000

print(f"GEMM latency: {elapsed_ms:.2f}ms for BS=8")

# Sanity threshold: < 10ms for small batch sizes
# Naive loop implementation would be 50-200ms
if elapsed_ms > 10.0:
    print(f"WARNING: GEMM too slow ({elapsed_ms:.2f}ms)")
    print("This likely indicates:")
    print("  - MMA instructions not being used (naive for-loop GEMM)")
    print("  - Missing tensor core utilization")
    print("  - Incorrect memory access patterns")
    raise AssertionError(f"GEMM performance unacceptable: {elapsed_ms:.2f}ms > 10ms threshold")

print(f"GEMM performance sanity check: PASS ({elapsed_ms:.2f}ms)")
```
**If performance fails** → Exit BLOCKED. This indicates MMA loops are not properly implemented.

### If ANY validation fails:
- Do NOT exit with status "complete"
- Exit with status "blocked"
- Document in `{artifact_dir}/blockers/gemm_implementation_blocker.md`:
  - Which specific validation failed
  - The error message or performance numbers
  - What you tried and why it didn't work
  - Your hypothesis for the root cause

{implementation_behavioral_footer}
```

### Stage 4: Kernel Assembly

**Task Tool Parameters**:
- `description`: "Phase 3 kernel_assembly: {model_short}"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Assemble main kernel and output conversion for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/optimization_plan.md`
- `{cuda_dir}/moe_*.cu` - all previously implemented stages
- `assets/LLAMA4_MONOKERNEL_PATCH.md` & `examples/MODELS_COMPARISON.md` - search for 'moe.cu' (main kernel)

**Objective**: Wire all stages together into the main cooperative kernel.

**Implementation**:

1. Create `{cuda_dir}/moe.cu` - Main kernel entry:
```cpp
template <typename Dims>
__global__ void moe_kernel(...) {
    extern __shared__ char smem[];
    auto grid = cooperative_groups::this_grid();
    
    // Phase 1: Routing
    topk_route<Dims>(...);
    __syncthreads();
    
    // Phase 2: Prepare
    prepare_moe<Dims>(...);
    __syncthreads();
    
    // Phase 3: Quantize (if needed)
    if constexpr (Dims::NEEDS_QUANTIZATION) {
        quantize_activations<Dims>(...);
    }
    grid.sync();
    
    // Phase 4: Up-projection
    moe_up_projection<Dims>(...);
    grid.sync();
    
    // Phase 5: Down-projection
    moe_down_projection<Dims>(...);
    grid.sync();
}
```

2. Create `{cuda_dir}/moe_output.cu` - Output conversion:
   - FP32 accumulator → BF16 output
   - Handle NaN/Inf clamping
   - Vectorized conversion if possible

3. Create `{cuda_dir}/moe_interface.h` - Public interface:
   - Dimension structs for BS8 and BS64 paths
   - Kernel configuration
   - Launch parameter computation

**Validation (BLOCKING - REQUIRED before completing)**:

### 1. Build Check
```bash
cmake --build --preset release --target install
```

### 2. Full Kernel Integration Test
Run this validation to test the complete assembled kernel against vLLM's fused_moe:
```python
import torch
from vllm.model_executor.layers.fused_moe import fused_moe

# Test parameters from constraints
batch_size, K, N, E, top_k = 8, {K}, {N}, {E}, {top_k}

# Create test inputs
torch.manual_seed(42)  # Reproducibility
activations = torch.randn(batch_size, K, device='cuda', dtype=torch.bfloat16) / 10
router_logits = torch.randn(batch_size, E, device='cuda', dtype=torch.bfloat16)
w_up = torch.randn(E, 2*N, K, device='cuda', dtype={weight_dtype}) / 10
w_down = torch.randn(E, K, N, device='cuda', dtype={weight_dtype}) / 10

# Reference: vLLM fused_moe (the production baseline)
ref_output = fused_moe(
    activations, w_up, w_down, router_logits,
    topk=top_k, renormalize={renormalize}
)

# Your full assembled monokernel
from vllm._custom_ops import moe_monokernel_{model_short}
mono_output = moe_monokernel_{model_short}(
    activations, router_logits, w_up, w_down
)

# Validate with appropriate tolerance
# This tests the full pipeline: routing -> quant -> GEMM -> output
torch.testing.assert_close(mono_output, ref_output, atol=1e-2, rtol=1e-2)
print("Full kernel integration test: PASS")

# Test across multiple batch sizes
for bs in [1, 4, 16, 32, 64]:
    test_act = torch.randn(bs, K, device='cuda', dtype=torch.bfloat16) / 10
    test_router = torch.randn(bs, E, device='cuda', dtype=torch.bfloat16)

    ref = fused_moe(test_act, w_up, w_down, test_router, topk=top_k, renormalize={renormalize})
    mono = moe_monokernel_{model_short}(test_act, test_router, w_up, w_down)

    torch.testing.assert_close(mono, ref, atol=1e-2, rtol=1e-2)
    print(f"  BS={bs}: PASS")

print("All batch sizes validated: PASS")
```

### 3. Smoke Test (Kernel Launches Without Crash)
```python
# Quick sanity check that kernel launches work
for bs in [1, 8, 64]:
    x = torch.randn(bs, {K}, device='cuda', dtype=torch.bfloat16)
    r = torch.randn(bs, {E}, device='cuda', dtype=torch.bfloat16)
    out = moe_monokernel_{model_short}(x, r, w_up, w_down)
    assert out.shape == (bs, {K}), f"Wrong output shape: {out.shape}"
print("Smoke test: PASS")
```

### If ANY validation fails:
- Exit with status "blocked"
- Document in `{artifact_dir}/blockers/kernel_assembly_blocker.md`:
  - Which batch size failed
  - The error message or shape mismatch
  - Maximum difference from reference

{implementation_behavioral_footer}
```

---

## Phase 4: Validation (3 Stages)

Phase 4 validates the monokernel through three complementary approaches:

| Stage | Purpose | Tool |
|-------|---------|------|
| 4.1 Correctness | Verify numerical accuracy | pytest / manual test |
| 4.2 Kernel-Level | Pure kernel performance (CUDA graphs) | benchmark script |
| 4.3 E2E Latency | Real inference impact | `vllm bench latency` |

**Key Insight**: The monokernel is a **decode-phase optimization** that only activates when batch_size ≤ 64 tokens pass through the MoE layer. Use decode-heavy workloads to showcase impact.

---

### Stage 4.1: Correctness Verification

**Task Tool Parameters**:
- `description`: "Phase 4.1: Correctness test for {model_short} monokernel"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Verify {model_short} monokernel numerical correctness.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md` - model dimensions and dtype
- `validation/QWEN3_BASELINE.md` - tolerance reference

**Tolerance by dtype**:
- FP32: atol=1e-3, rtol=1e-3
- BF16/FP16: atol=1e-2, rtol=1e-2
- FP8 block-quant: atol=300, rtol=0.5

**Test Implementation**:
```python
import torch
from vllm._custom_ops import moe_monokernel_{model_short}
from vllm.model_executor.layers.fused_moe import fused_moe

# Test across batch sizes
for BS in [1, 8, 64]:
    # Create inputs (dimensions from constraints.md)
    x = torch.randn(BS, {K}, dtype=torch.bfloat16, device='cuda') / 10
    router = torch.randn(BS, {E}, dtype=torch.bfloat16, device='cuda')
    # ... setup weights from constraints ...

    # Run both paths
    mono_out = moe_monokernel_{model_short}(x, router, ...)
    stock_out = fused_moe(x, router, ...)

    # Compare with dtype-appropriate tolerance
    torch.testing.assert_close(mono_out, stock_out, atol={atol}, rtol={rtol})
    print(f'BS={BS}: Max diff={diff:.4f} - PASS')
```

**Output**: Update `{artifact_dir}/state.json`:

**If PASS** (all batch sizes within tolerance):
```json
"phases": {
  "4_validation": {
    "stages": {
      "4_1_correctness": {
        "status": "complete",
        "max_abs_diff": 0.00X,
        "batch_sizes_tested": [1, 8, 64]
      }
    }
  }
}
```

**If FAIL** (any batch size exceeds tolerance):
```json
"phases": {
  "4_validation": {
    "stages": {
      "4_1_correctness": {
        "status": "needs_investigation",
        "max_abs_diff": 0.XXX,
        "tolerance": {atol},
        "failing_batch_sizes": [8, 64],
        "failure_details": "Description of where divergence occurs (e.g., 'Output diverges after down-projection, error grows with batch size')"
      }
    }
  }
}
```
Exit with status `"needs_investigation"` (NOT `"blocked"`).

The orchestrator will spawn an investigation task. See `orchestration/investigation-prompts.md`.

{implementation_behavioral_footer}
```

---

### Stage 4.2: Kernel-Level Benchmark (CUDA Graphs)

**Task Tool Parameters**:
- `description`: "Phase 4.2: Kernel benchmark for {model_short} monokernel"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Benchmark {model_short} monokernel kernel-level performance under CUDA graphs.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md` - model dimensions
- `benchmarks/kernels/benchmark_moe_monokernel_qwen3.py` - reference implementation

**Why CUDA Graphs**: Eliminates kernel launch overhead for apples-to-apples comparison between monokernel (1 kernel) and baseline (5-7 kernels).

**Run Benchmark**:
```bash
# Adapt existing benchmark or create new one
python benchmarks/kernels/benchmark_moe_monokernel_{model_short}.py \
    --batch-sizes 1 4 8 16 32 64 \
    --use-cuda-graph \
    --baseline-scope e2e \
    --num-iters 100 \
    --output /tmp/{model_short}_kernel_benchmark
```

**Baseline Scope = e2e**: Includes routing (topk_softmax) + expert computation (fused_experts) in baseline timing, matching what monokernel fuses.

**Expected Output Format**:
```
BS=1: monokernel=X.XXms, baseline=Y.YYms, speedup=Z.ZZx
BS=8: monokernel=X.XXms, baseline=Y.YYms, speedup=Z.ZZx
...
```

**Success Criteria (STRICT)**:
- Monokernel must be **faster than or equal to** baseline at ALL tested batch sizes
- Required: `speedup >= 1.0x` at every batch size
- NO regressions allowed under CUDA graphs (launch overhead is eliminated)
- If ANY batch size shows `speedup < 1.0x` → FAIL

**Output**: Update `{artifact_dir}/state.json`:

**If PASS** (all batch sizes show speedup >= 1.0x):
```json
"phases": {
  "4_validation": {
    "stages": {
      "4_2_kernel_perf": {
        "status": "complete",
        "results": {
          "bs_1": {"monokernel_ms": 0.XX, "baseline_ms": 0.YY, "speedup": 1.ZZ},
          "bs_4": {"monokernel_ms": 0.XX, "baseline_ms": 0.YY, "speedup": 1.ZZ},
          "bs_8": {"monokernel_ms": 0.XX, "baseline_ms": 0.YY, "speedup": 1.ZZ},
          "bs_16": {"monokernel_ms": 0.XX, "baseline_ms": 0.YY, "speedup": 1.ZZ},
          "bs_32": {"monokernel_ms": 0.XX, "baseline_ms": 0.YY, "speedup": 1.ZZ},
          "bs_64": {"monokernel_ms": 0.XX, "baseline_ms": 0.YY, "speedup": 1.ZZ}
        },
        "min_speedup": 1.XX,
        "max_speedup": X.XX
      }
    }
  }
}
```

**If FAIL** (any batch size shows speedup < 1.0x):
```json
"phases": {
  "4_validation": {
    "stages": {
      "4_2_kernel_perf": {
        "status": "needs_investigation",
        "results": { ... },
        "failing_batch_sizes": [1, 4],
        "worst_speedup": 0.XXx,
        "worst_batch_size": N
      }
    }
  }
}
```
Exit with status `"needs_investigation"` (NOT `"blocked"`).

The orchestrator will spawn a kernel performance investigation task. See `orchestration/investigation-prompts.md`.

{implementation_behavioral_footer}
```

---

### Stage 4.3: E2E Latency Benchmark

**Task Tool Parameters**:
- `description`: "Phase 4.3: E2E latency benchmark for {model_short} monokernel"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Benchmark {model_short} monokernel E2E inference latency using vLLM CLI.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md` - model ID and TP setting
- `validation/E2E_LATENCY_GUIDE.md` - detailed guide

**Key Understanding**:
- Monokernel is a **decode optimization** (batch_size ≤ 64 tokens through MoE)
- During prefill with input_len=1024: monokernel NOT used (>64 tokens)
- During decode with batch_size=8: monokernel IS used (8 tokens)
- Use decode-heavy workloads: `--input-len 64 --output-len 512`

**Run E2E Benchmarks**:
```bash
# Test representative batch sizes: 4 (best), 8 (typical), 32 (marginal)
for BS in 4 8 32; do
    echo "=== Testing batch_size=$BS ==="

    # Baseline
    vllm bench latency \
        --model {model_id} \
        --tensor-parallel-size {tp} \
        --max-model-len 4096 \
        --input-len 64 --output-len 512 \
        --batch-size $BS \
        --num-iters 10 \
        --output-json /tmp/baseline_bs${BS}.json 2>&1 | grep "Avg latency"

    # Monokernel
    VLLM_USE_MOE_MONOKERNEL=1 vllm bench latency \
        --model {model_id} \
        --tensor-parallel-size {tp} \
        --max-model-len 4096 \
        --input-len 64 --output-len 512 \
        --batch-size $BS \
        --num-iters 10 \
        --output-json /tmp/monokernel_bs${BS}.json 2>&1 | grep "Avg latency"
done
```

**Expected Results** (Qwen3-FP8 on L40S reference):
| Batch Size | Expected Improvement |
|------------|---------------------|
| 4 | ~11% (best) |
| 8 | ~7% |
| 32 | ~4-5% |

**Success Criteria**:
| Batch Size | Required Improvement |
|------------|---------------------|
| 1, 4, 8 | > 5% |
| 16, 32, 64 | > 0% (not slower) |

Calculate improvement:
```python
improvement_pct = (baseline_latency - monokernel_latency) / baseline_latency * 100
```

**Output**: Update `{artifact_dir}/state.json`:

**If PASS** (all batch sizes meet threshold):
```json
"phases": {
  "4_validation": {
    "stages": {
      "4_3_e2e_latency": {
        "status": "complete",
        "results": {
          "bs_4": {"baseline_s": X.XX, "monokernel_s": Y.YY, "improvement_pct": Z.Z},
          "bs_8": {"baseline_s": X.XX, "monokernel_s": Y.YY, "improvement_pct": Z.Z},
          "bs_32": {"baseline_s": X.XX, "monokernel_s": Y.YY, "improvement_pct": Z.Z}
        }
      }
    }
  }
}
```

**If FAIL** (any batch size below threshold):
```json
"phases": {
  "4_validation": {
    "stages": {
      "4_3_e2e_latency": {
        "status": "needs_investigation",
        "results": { ... },
        "failing_batch_sizes": [4, 8],
        "failure_reason": "BS=4 improvement 3.2% < required 5%"
      }
    }
  }
}
```
Exit with status `"needs_investigation"` (NOT `"blocked"`).

The orchestrator will spawn an E2E performance investigation task. See `orchestration/investigation-prompts.md`.

**Document Results**: Write to `{artifact_dir}/validation_results.md`:
```markdown
# Validation Results

## Correctness
- Max absolute diff: {value}
- Status: PASS

## Kernel-Level Performance (CUDA Graphs)
| Batch Size | Baseline (ms) | Monokernel (ms) | Speedup |
|------------|--------------|-----------------|---------|
| 1 | X.XX | Y.YY | Z.ZZx |
| ... | ... | ... | ... |

## E2E Latency (vllm bench latency)
Config: input_len=64, output_len=512, {hardware}

| Batch Size | Baseline (s) | Monokernel (s) | Speedup | Improvement |
|------------|-------------|----------------|---------|-------------|
| 4 | X.XX | Y.YY | Z.ZZx | N% |
| 8 | X.XX | Y.YY | Z.ZZx | N% |
| 32 | X.XX | Y.YY | Z.ZZx | N% |

## Recommendation
- Enable monokernel for batch_size ≤ {threshold}
- Expected improvement: {X}% for typical serving load
- Best use case: Low-concurrency serving (chat, code completion)
```

{implementation_behavioral_footer}
```

---

## Phase 5: Integration

**Task Tool Parameters**:
- `description`: "Phase 5: Integrate {model_short} monokernel into vLLM"
- `subagent_type`: "general-purpose"
- `prompt`: [Copy the full prompt below]

```markdown
Integrate {model_short} monokernel into vLLM.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md` - model file location
- `{artifact_dir}/validation_results.md` - which batch sizes to enable

**Objective**: Wire monokernel into vLLM's MoE path with fast-path dispatch.

**Steps**:

1. Add to CMakeLists.txt:
   - Add kernel source files
   - Set architecture flags

2. Add torch bindings in `csrc/moe/torch_bindings.cpp`

3. Add Python wrapper in `vllm/_custom_ops.py`

4. Modify MoE layer to dispatch to monokernel:
   - File: {from constraints - model file}
   - Add batch size check
   - Call monokernel for BS ≤ threshold

**Output**: 
- Git patch or diff showing all changes
- Integration instructions in `{artifact_dir}/integration.md`

{implementation_behavioral_footer}
```
