# MoE Monokernel Phase Prompts (Codex)

> This file is an optional library of phase-level prompts you can paste into a Codex interactive session or pass to `codex exec`.
> It was adapted from a Claude Code workflow; any Claude-specific "Task(...)" wording has been removed.

This document contains the full behavioral spec phase prompts for each phase and stage.

## Directory Convention

All artifacts go in: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/`
CUDA files go in: `csrc/moe/moe_monokernel_{model}_{hardware}_{dtype}_{tp}/`

Example: For Qwen3-30B-A3B-FP8 on L40S with TP=1:
- Artifacts: `moe_monokernel_artifacts/qwen3-30b-a3b_l40s_fp8_tp1/`
- CUDA: `csrc/moe/moe_monokernel_qwen3-30b-a3b_l40s_fp8_tp1/`

## Common Behavioral Footer

Append to ALL phase prompts:

```markdown
**Behavioral Expectations**:
1. **Read before write**: Read all referenced documents before implementing
2. **Compile often**: Run `cmake --build --preset release --target install` after each major change
3. **On compile error**:
   - Read the full error message
   - Identify root cause (typo, missing include, dimension mismatch, type error)
   - Fix and retry (up to 3 attempts per distinct error)
4. **On stuck** (same error 3+ times):
   - Document in `{artifact_dir}/blockers/{component}_blocker.md`:
     - Error message (full)
     - Attempts made with different approaches
     - Hypotheses for root cause
     - What you tried and why it didn't work
   - Exit with status "blocked"
5. **Stay goal-aligned**: This task contributes to: {ultimate_goal}

**Context Preservation**:
- Current phase: {current_phase}
- Completed stages: {completed_stages}
- Remaining stages: {remaining_stages}
- State file: {artifact_dir}/state.json
- CUDA directory: {cuda_dir}
```

---

## Phase 1: Gather Constraints (with vLLM Code Analysis)

```markdown
(phase task: "Phase 1: Gather MoE monokernel constraints for {model_id} on {hardware} with TP={tp}, dtype={dtype}.

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

{common_behavioral_footer}
")
```

---

## Phase 2: Optimization Planning

```markdown
(phase task: "Phase 2: Create optimization plan for {model_id} monokernel.

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
   - If custom/unknown: Flag for exploration run in Phase 3

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

{common_behavioral_footer}
")
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

```markdown
(phase task: "Implement routing and token preparation for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md` - especially 'Semantic Differences' section
- `{artifact_dir}/optimization_plan.md` - Decisions A, B
- `references/router-design.md`
- `references/expert-grouping.md`
- `assets/LLAMA4_MONOKERNEL_PATCH.md` - search for 'moe_routing.cu' and 'moe_prepare.cu'

**Objective**: Implement top-k routing AND token-by-expert sorting in one cohesive unit.

**Key Considerations from Constraints**:
- Scoring function: {from constraints - softmax/sigmoid}
- Renormalize: {from constraints - yes/no}
- top_k: {from constraints}
- TOKENS_PER_WARP: {from plan - Decision B}

**Implementation**:
1. Create `{cuda_dir}/moe_routing.cu`:
   - Top-k selection matching the model's scoring function
   - If renormalize=true, normalize weights after selection
   - Store topk_ids and topk_weights in shared memory

2. Create `{cuda_dir}/moe_prepare.cu`:
   - Token sorting by expert (bitfield or histogram based on E count)
   - Expert reference struct population
   - Pair index computation for GEMM stages

**Validation**:
```bash
cmake --build --preset release --target install
```

{common_behavioral_footer}
")
```

### Stage 2: Activation Quantization (Conditional)

```markdown
(phase task: "Implement activation quantization for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**CONDITIONAL**: This stage is only needed if weight_dtype is FP8 or INT8.
If weight_dtype is BF16/FP16, this stage can be minimal (just copy activations).

**Read First**:
- `{artifact_dir}/constraints.md` - check Data Types section
- `{artifact_dir}/optimization_plan.md`
- `assets/LLAMA4_MONOKERNEL_PATCH.md` - search for 'moe_scale_inputs.cu'

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

{common_behavioral_footer}
")
```

### Stage 3: GEMM Implementation (CRITICAL - Up + Down Together)

This is the most complex stage. **Up-projection and down-projection are implemented together**
because they share 90% of their structure (MMA patterns, warp specialization, double buffering).

**CRITICAL**: This stage produces the core compute kernels. A naive/slow implementation is WORSE than failure because the user already has a working fused_moe. The purpose of this skill is SPEEDUP.

```markdown
(phase task: "Implement GEMM kernels (up-projection AND down-projection) for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**CRITICAL**: This task implements BOTH up-projection and down-projection together.
They share the same MMA infrastructure - implement common helpers once, apply to both.

**Read First** (IN THIS ORDER):
1. `{artifact_dir}/optimization_plan.md` - Decisions C, F (weight timing, MMA details)
2. `{artifact_dir}/constraints.md` - dimensions, dtype
3. `references/code-templates.md` - MMA templates section (FP8 AND BF16)
4. `references/tiling-config.md` - SMEM layout
5. `assets/LLAMA4_MONOKERNEL_PATCH.md` - search for 'moe_up_projection.cu', 'moe_down_projection.cu'

**Key Configuration**:
- weight_dtype: {fp8/bf16/fp16}
- APPLY_WEIGHT: {before_activation/after_activation} (Decision C)
- K_CHUNKS_UP: ceil(K / K_TILE)
- K_CHUNKS_DOWN: ceil(N / K_TILE)
- USE_BLOCK_QUANT: {true/false}

---

## FORBIDDEN PATTERNS (READ THIS FIRST)

**The purpose of this skill is SPEEDUP.** A slow kernel is WORSE than no kernel because the user already has fused_moe that works. If you cannot implement optimized GEMM, exit BLOCKED - do NOT implement naive fallback.

**NEVER implement any of these patterns**:
```cpp
// FORBIDDEN: Serial triple-nested GEMM loop
for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < K; k++)
            C[i][j] += A[i][k] * B[k][j];  // NO!

// FORBIDDEN: Per-element accumulation without MMA
float sum = 0;
for (int k = 0; k < K; k++)
    sum += a[k] * b[k];  // NO! Use mma_ instructions

// FORBIDDEN: Naive function names
void up_projection_naive(...);   // NO!
void gemm_simple(...);           // NO!
void fallback_matmul(...);       // NO!
```

**REQUIRED patterns**:
```cpp
// REQUIRED: MMA instruction calls
mma_bf16_bf16(d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, ...);  // YES

// REQUIRED: K-chunk tiling iteration
for (uint32_t k_chunk = 0; k_chunk < K_CHUNKS; k_chunk++) {  // YES
    // ... MMA inside ...
}

// REQUIRED: Warp-level fragment loads
load_A_fragment<BS_MAX>(smem, row, k_offset, a0, a1, a2, a3);  // YES
```

**If you cannot implement MMA-based GEMM**:
- Do NOT write a naive fallback "just to make it work"
- Exit with status BLOCKED
- Document specifically what you couldn't implement
- A 220x slowdown is UNACCEPTABLE - that's what happened when naive was used

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

## MANDATORY VERIFICATION (DO NOT SKIP)

Before completing this task, you MUST verify ALL of the following:

### Check 1: Search for incomplete code
```bash
grep -n 'TODO\|FIXME\|XXX\|unimplemented' {cuda_dir}/moe_*_projection.cu
```
If ANY results in MMA loops or compute paths → You are NOT done. Exit BLOCKED.

### Check 2: Verify MMA calls exist
```bash
UP_MMA=$(grep -c 'mma_' {cuda_dir}/moe_up_projection.cu)
DOWN_MMA=$(grep -c 'mma_' {cuda_dir}/moe_down_projection.cu)
echo "MMA calls: up=$UP_MMA, down=$DOWN_MMA"
```
**If either count is 0 → You implemented naive GEMM. This is UNACCEPTABLE. Exit BLOCKED.**

### Check 3: Verify NO naive patterns exist
```bash
grep -n 'naive\|simple\|fallback\|serial' {cuda_dir}/moe_*_projection.cu
```
**If ANY matches → You implemented naive code. DELETE IT and implement proper MMA. Exit BLOCKED if you cannot.**

### Check 4: Verify K-chunk tiling exists
```bash
grep -c 'k_chunk\|K_CHUNK' {cuda_dir}/moe_*_projection.cu
```
**If count is 0 → No tiling implemented. This will be extremely slow. Exit BLOCKED.**

### Check 5: Verify fragment loaders exist
```bash
grep -c 'load_.*_fragment\|load_A\|load_B' {cuda_dir}/moe_*_projection.cu
```
**If count is 0 → No proper shared memory usage. Exit BLOCKED.**

---

## COMPLETION CRITERIA

You may ONLY mark this task as "complete" if ALL of these are true:
- [ ] MMA instruction calls exist in both projection files (mma_bf16_bf16 or mma_fp8_fp8)
- [ ] K-chunk iteration loop exists with MMA inside
- [ ] Fragment loaders implemented for A and B matrices
- [ ] NO naive/simple/fallback code anywhere
- [ ] NO TODOs in compute paths
- [ ] Code compiles without errors

**If ANY check fails**:
- Do NOT exit with status "complete"
- Do NOT implement a naive fallback
- Exit with status "blocked"
- Document in blocker file:
  - What specific part you couldn't implement
  - What reference you tried to follow
  - What went wrong
  - What help you need from council

Remember: A 220x slowdown happened when naive GEMM was used. That's worse than no kernel at all.

{common_behavioral_footer}
")
```

### Stage 4: Kernel Assembly

```markdown
(phase task: "Assemble main kernel and output conversion for {model_short} monokernel.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/optimization_plan.md`
- `{cuda_dir}/moe_*.cu` - all previously implemented stages
- `assets/LLAMA4_MONOKERNEL_PATCH.md` - search for 'moe.cu' (main kernel)

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

**Validation**:
```bash
cmake --build --preset release --target install
```

{common_behavioral_footer}
")
```

---

## Phase 4: Validation

**IMPORTANT**: This phase validates BOTH correctness AND performance. A kernel that is correct but slower than baseline is a FAILURE. The entire purpose of this skill is speedup.

```markdown
(phase task: "Validate {model_short} monokernel correctness and performance.

**Ultimate Goal**: {ultimate_goal}

**Read First**:
- `{artifact_dir}/constraints.md`
- `assets/benchmark_template.py` - adapt for this model

**Objective**: Verify numerical correctness AND confirm performance improvement.

---

## Step 1: Correctness Test

```python
# Compare monokernel vs stock fused_moe
import torch
from vllm._custom_ops import moe_monokernel_{model_short}
from vllm.model_executor.layers.fused_moe import fused_moe

# Create test inputs
x = torch.randn(BS, K, dtype=torch.bfloat16, device='cuda')
router = torch.randn(BS, E, dtype=torch.bfloat16, device='cuda')
# ... setup weights ...

# Run both
mono_out = moe_monokernel_{model_short}(x, router, ...)
stock_out = fused_moe(x, router, ...)

# Compare
diff = (mono_out - stock_out).abs().max()
assert diff < 1e-2, f'Numerical mismatch: {diff}'
print(f'Max diff: {diff} - PASS')
```

---

## Step 2: Performance Benchmark

Adapt `assets/benchmark_template.py` for this model:
- Update dimensions (K, N, E, top_k)
- Update batch sizes to test (1, 4, 8, 16, 32, 64)
- Run and collect results

---

## Step 3: Performance Threshold Validation (MANDATORY)

**These thresholds determine SUCCESS or FAILURE:**

| Batch Size | Maximum Acceptable Ratio | Reason |
|------------|--------------------------|--------|
| BS=1 | monokernel ≤ 2.0× baseline | Small BS has overhead, some slowdown OK |
| BS=4-8 | monokernel ≤ 1.2× baseline | Should roughly break even |
| BS=16+ | monokernel < 1.0× baseline | **MUST show speedup** |
| BS=32+ | monokernel < 0.85× baseline | Should show significant speedup |

**Evaluation:**
```python
def evaluate_performance(results):
    failures = []
    
    for bs, (baseline_ms, mono_ms) in results.items():
        ratio = mono_ms / baseline_ms
        
        if bs == 1 and ratio > 2.0:
            failures.append(f'BS={bs}: {ratio:.2f}x slower (max 2.0x)')
        elif bs in [4, 8] and ratio > 1.2:
            failures.append(f'BS={bs}: {ratio:.2f}x slower (max 1.2x)')
        elif bs >= 16 and ratio >= 1.0:
            failures.append(f'BS={bs}: {ratio:.2f}x - NO SPEEDUP (must be <1.0x)')
        elif bs >= 32 and ratio >= 0.85:
            failures.append(f'BS={bs}: {ratio:.2f}x - insufficient speedup (must be <0.85x)')
    
    return failures

failures = evaluate_performance(benchmark_results)
if failures:
    print('PERFORMANCE VALIDATION FAILED:')
    for f in failures:
        print(f'  - {f}')
    print('')
    print('This indicates suboptimal implementation (possibly naive GEMM).')
    print('Phase 4 status: FAILED')
    exit(1)
else:
    print('Performance validation PASSED')
```

---

## Step 4: Document Results

Write to `{artifact_dir}/validation_results.md`:

```markdown
# Validation Results

## Correctness
- Max absolute diff: {value}
- Status: PASS/FAIL

## Performance

| Batch Size | Stock (ms) | Monokernel (ms) | Speedup | Threshold | Status |
|------------|------------|-----------------|---------|-----------|--------|
| 1 | {x} | {y} | {z}x | ≤2.0x | PASS/FAIL |
| 4 | ... | ... | ... | ≤1.2x | PASS/FAIL |
| 8 | ... | ... | ... | ≤1.2x | PASS/FAIL |
| 16 | ... | ... | ... | <1.0x | PASS/FAIL |
| 32 | ... | ... | ... | <0.85x | PASS/FAIL |
| 64 | ... | ... | ... | <0.85x | PASS/FAIL |

## Overall Status
- Correctness: PASS/FAIL
- Performance: PASS/FAIL
- **Final**: PASS/FAIL

## If FAILED
{Document which batch sizes failed and by how much}
{Likely cause: naive GEMM, missing MMA, untiled implementation}
```

---

## Completion Criteria

**Phase 4 is COMPLETE only if BOTH**:
1. ✅ Correctness: max_diff < 1e-2
2. ✅ Performance: ALL batch size thresholds met

**If performance fails**:
- Exit with status BLOCKED (not COMPLETE)
- Document which thresholds failed
- This likely indicates naive GEMM in Phase 3 - may need to redo

{common_behavioral_footer}
")
```

---

## Phase 5: Integration

```markdown
(phase task: "Integrate {model_short} monokernel into vLLM.

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

{common_behavioral_footer}
")
```