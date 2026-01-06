# MoE Monokernel Task Prompts

Task prompts for spawning subagents. Copy the appropriate phase prompt and fill template variables.

## Template Variables

```
{model_id}      - Full model ID (e.g., "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
{model_short}   - Short name (e.g., "qwen3-30b-a3b")
{hardware}      - GPU type (e.g., "L40S", "H100")
{dtype}         - Data type (e.g., "fp8", "bf16")
{tp}            - Tensor parallelism
{ep}            - Expert parallelism
{artifact_dir}  - e.g., "moe_monokernel_artifacts/qwen3-30b-a3b_l40s_fp8_tp1"
{cuda_dir}      - e.g., "csrc/moe/moe_monokernel_qwen3-30b-a3b_l40s_fp8_tp1"
{batch_buckets} - e.g., [1, 4, 8, 16, 32, 64]
```

---

## Phase 1: Gather Constraints

```markdown
# Task: Gather MoE Constraints for {model_id}

## Goal
Capture model semantics and baseline truth snapshot for {model_id} on {hardware}.

## Context
- Artifact directory: {artifact_dir}
- State file: {artifact_dir}/state.json
- Target: {model_id}, {hardware}, {dtype}, TP={tp}, EP={ep}
- Batch buckets: {batch_buckets}

## Instructions

1. **Read first**:
   - `.claude/skills/moe-monokernel-optimizer/references/moe-parameters-template.md`
   - `.claude/skills/moe-monokernel-optimizer/references/profiling-launch-vs-kernel.md`

2. **Locate vLLM model code**:
   - Find MoE implementation in `vllm/model_executor/models/`
   - Trace forward path: routing → expert execution → accumulation

3. **Document semantics** (CRITICAL for correctness):
   - Router scoring: softmax vs sigmoid
   - Renormalization behavior
   - Weight application: before vs after activation
   - Shared experts (if any)
   - Accumulation: direct write vs atomics vs reduce

4. **Run baseline profiling** (production parity):
   - Enable CUDA graphs and torch.compile
   - Profile for batch sizes: {batch_buckets}
   - Separate GPU kernel time from launch overhead
   - Record per-bucket timings

5. **Write constraints.md** using template from references/moe-parameters-template.md

## Output
Create `{artifact_dir}/constraints.md` with:
- Target envelope section
- Model semantics section (routing, activation, weight placement)
- Baseline Truth Snapshot (bucket timings under CUDA graphs)
- "Already fused?" checklist

## State Update
When complete, update `{artifact_dir}/state.json`:
```json
{"phase": "2_planning", "status": "pending", "last_update": "YYYY-MM-DD"}
```
```

---

## Phase 2: Optimization Planning

```markdown
# Task: Plan {model_short} MoE Optimization

## Goal
Choose fusion route, make algorithmic decisions, produce implementation plan.

## Context
- Artifact directory: {artifact_dir}
- Constraints: {artifact_dir}/constraints.md (READ THIS FIRST)
- Target: {model_id}, {hardware}, TP={tp}

## Instructions

1. **Read first**:
   - `{artifact_dir}/constraints.md` (your baseline truth)
   - `.claude/skills/moe-monokernel-optimizer/references/route-selection-decision-tree.md`
   - `.claude/skills/moe-monokernel-optimizer/references/algorithmic-branching.md`
   - `.claude/skills/moe-monokernel-optimizer/references/fusion-feasibility-heuristics.md`

2. **Compute derived values**:
   - P = BS × top_k (token-expert pairs)
   - M_avg = BS × top_k / E_local (per-expert tokens, uniform routing)
   - Saturation = P / SM_count

3. **Pick route (A/B/C)** using decision tree:
   - A) Cooperative monokernel
   - B) Hybrid large-grid fusion
   - C) Split-kernel graph-captured

4. **Make algorithmic decisions** (record rationale for each):
   - Decision A: Output path (atomics vs direct write)
   - Decision B: Sorter strategy (TOKENS_PER_WARP)
   - Decision C: Weight placement (BEFORE/AFTER activation) - CRITICAL
   - Decision D: Shared expert strategy (none/sidecar/sequential)
   - Decision E: GEMM strategy (per-pair GEMV vs grouped)

5. **SRAM Tetris** (if cooperative route):
   - Read `.claude/skills/moe-monokernel-optimizer/references/tiling-config.md`
   - Compute tile sizes, buffer counts, SMEM budget

6. **Define kill criteria** (when to pivot routes):
   - List 2-3 measurable "stop" conditions

## Output
Create `{artifact_dir}/optimization_plan.md` with:
- Route decision + rationale
- Feasibility math (required savings vs upper bound)
- Algorithmic decisions table
- SRAM configuration (if applicable)
- Kill criteria
- Implementation stages plan

## State Update
When complete, update `{artifact_dir}/state.json`:
```json
{"phase": "3_implementation", "status": "pending", "last_update": "YYYY-MM-DD"}
```

## Early Exit
If monokernel is not applicable (baseline already optimal, infeasible savings):
- Document in optimization_plan.md
- Set phase to "complete" with status "not_applicable"
```

---

## Phase 3: Implementation Stages

### Stage 3.1: Routing and Prepare

```markdown
# Task: Implement Routing and Prepare for {model_short}

## Goal
Implement router and token preparation stage.

## Context
- Artifact directory: {artifact_dir}
- CUDA directory: {cuda_dir}
- Constraints: {artifact_dir}/constraints.md
- Plan: {artifact_dir}/optimization_plan.md (READ THIS FIRST)

## Instructions

1. **Read first**:
   - `{artifact_dir}/optimization_plan.md` (your algorithmic decisions)
   - `.claude/skills/moe-monokernel-optimizer/references/router-design.md`
   - `.claude/skills/moe-monokernel-optimizer/references/code-templates.md`

2. **Implement routing kernel**:
   - Match model's scoring function (softmax/sigmoid)
   - Implement top-k selection
   - Apply renormalization if required
   - Store routing weights for later use

3. **Implement prepare stage**:
   - Sort tokens by expert (use Decision B: TOKENS_PER_WARP)
   - Build expert dispatch structures
   - Ensure coalesced memory access patterns

4. **Ensure CUDA graphs safety**:
   - No allocations in hot path
   - Stable shapes per batch bucket
   - Correct stream usage

5. **Compile and test**:
   - Run `make` or cmake build
   - Fix all compile errors before marking complete
   - Smoke test with simple input

## Output
- `{cuda_dir}/moe_routing.cu`
- `{cuda_dir}/moe_prepare.cu`
- Update `{artifact_dir}/implementation_notes.md`

## State Update
Update state.json with stage completion.
```

### Stage 3.2: Activation and Quantization

```markdown
# Task: Implement Activation/Quantization for {model_short}

## Goal
Implement activation function and FP8 quantization (if applicable).

## Context
- CUDA directory: {cuda_dir}
- Plan: {artifact_dir}/optimization_plan.md
- Dtype: {dtype}

## Instructions

1. **Read first**:
   - `{artifact_dir}/optimization_plan.md` (Decision C: weight placement)
   - `.claude/skills/moe-monokernel-optimizer/references/code-templates.md` § Activation Templates

2. **Implement activation**:
   - SiLU: `x * sigmoid(x)` or `x / (1 + exp(-x))`
   - GELU/ReLU: use standard formulas
   - Apply routing weight at correct stage (Decision C)

3. **Implement quantization** (FP8 only):
   - Block quantization with per-block scales
   - Scale layout from constraints.md
   - Verify scale indexing

4. **Compile and test**

## Output
- `{cuda_dir}/moe_scale_inputs.cu` (if FP8)
- Update implementation_notes.md
```

### Stage 3.3: GEMM Implementation

```markdown
# Task: Implement GEMM Kernels for {model_short}

## Goal
Implement up-projection and down-projection GEMMs.

## Context
- CUDA directory: {cuda_dir}
- Plan: {artifact_dir}/optimization_plan.md (SRAM config, decisions)
- Constraints: {artifact_dir}/constraints.md (dimensions, dtypes)

## Instructions

1. **Read first**:
   - `{artifact_dir}/optimization_plan.md` (SRAM Tetris solution)
   - `.claude/skills/moe-monokernel-optimizer/references/tiling-config.md`
   - `.claude/skills/moe-monokernel-optimizer/references/code-templates.md` § MMA Templates
   - `.claude/skills/moe-monokernel-optimizer/assets/LLAMA4_MONOKERNEL_PATCH.md` (reference implementation)

2. **Implement up-projection**:
   - Load activation tiles with prefetching
   - Load weight tiles (gate + up weights)
   - MMA operations with correct tile sizes
   - Apply activation + routing weight (per Decision C)

3. **Implement down-projection**:
   - Load intermediate tiles
   - MMA with down weights
   - Accumulate to output (per Decision A: atomics vs direct)
   - Handle top_k > 1 pair-to-token mapping

4. **Verify GEMM correctness**:
   - Check MMA calls are present (not just placeholder)
   - Verify no TODOs remain in compute loops
   - Static assert dimension divisibility

5. **Compile and test**

## Critical Verification
Before marking complete, verify:
- [ ] MMA calls present in both up and down projection
- [ ] No TODOs in GEMM compute loops
- [ ] SMEM usage within budget (check implementation_notes.md)

## Output
- `{cuda_dir}/moe_up_projection.cu`
- `{cuda_dir}/moe_down_projection.cu`
- Update implementation_notes.md with SMEM usage
```

### Stage 3.4: Kernel Assembly

```markdown
# Task: Assemble {model_short} Monokernel

## Goal
Wire all stages together into main kernel entry point.

## Context
- CUDA directory: {cuda_dir}
- All prior stages complete

## Instructions

1. **Read first**:
   - `.claude/skills/moe-monokernel-optimizer/references/cudagraph-safety.md`
   - `.claude/skills/moe-monokernel-optimizer/references/architecture-pattern.md`

2. **Create main kernel**:
   - Include all stage files
   - Implement kernel entry point
   - Add grid synchronization points (minimize count)
   - Handle batch size dispatch

3. **Create interface**:
   - Torch binding (pybind11)
   - Python wrapper with envelope guard
   - Fallback to baseline outside envelope

4. **Final verification**:
   - Full compile
   - Smoke test all batch buckets
   - Verify CUDA graphs capture works

## Output
- `{cuda_dir}/moe_kernel.cu` (main entry)
- `{cuda_dir}/moe_interface.h`
- Torch bindings
- Update implementation_notes.md

## State Update
When complete:
```json
{"phase": "4_validation", "status": "pending", "last_update": "YYYY-MM-DD"}
```
```

---

## Phase 4: Validation

```markdown
# Task: Validate {model_short} Monokernel

## Goal
Prove correctness, kernel performance, and e2e latency improvement.

## Context
- Artifact directory: {artifact_dir}
- CUDA directory: {cuda_dir}
- Batch buckets: {batch_buckets}

## Instructions

1. **Read first**:
   - `.claude/skills/moe-monokernel-optimizer/references/validation-defaults.md`
   - `.claude/skills/moe-monokernel-optimizer/validation/E2E_LATENCY_GUIDE.md`

2. **Gate 4.1: Correctness**
   - Compare outputs vs baseline for all batch buckets
   - Use tolerances from validation-defaults.md
   - If top_k > 1: test reduction edge cases
   - **If fails**: Stop, use investigation-playbook.md § Correctness

3. **Gate 4.2: Kernel Performance**
   - Measure GPU kernel time under CUDA graphs
   - Gate: optimized ≤ baseline for ALL buckets
   - Run NCU to verify no occupancy regression
   - **If fails**: Stop, use investigation-playbook.md § Kernel Perf

4. **Gate 4.3: E2E Latency**
   - Run `vllm bench latency` with identical settings
   - Compare baseline vs monokernel
   - Calculate improvement percentage

## Output
Create `{artifact_dir}/validation_results.md` with:
- Correctness: pass/fail, max diff, tolerance used
- Kernel perf: per-bucket timings, speedup
- E2E: latency comparison, improvement %

## State Update
When complete:
```json
{"phase": "5_integration", "status": "pending", "last_update": "YYYY-MM-DD"}
```

## On Failure
If any gate fails, update state:
```json
{"phase": "4_validation", "status": "needs_investigation", "notes": "Gate 4.X failed: <reason>"}
```
Follow investigation-playbook.md with bounded cycles.
```

---

## Phase 5: Integration

```markdown
# Task: Integrate {model_short} Monokernel into vLLM

## Goal
Land the monokernel as a bounded, safe fast-path.

## Context
- Artifact directory: {artifact_dir}
- Validation: {artifact_dir}/validation_results.md (must pass all gates)

## Instructions

1. **Add fast-path dispatch**:
   - Guard for exact validated envelope:
     - Model ID: {model_id}
     - Dtype: {dtype}
     - TP/EP: {tp}/{ep}
     - Batch size: ≤ max validated bucket
   - Env var toggle: `VLLM_USE_MOE_MONOKERNEL`

2. **Ensure fallback**:
   - Outside envelope → use baseline fused_moe
   - On any error → graceful fallback

3. **Add tests**:
   - Correctness test matching validation
   - Add to CI

4. **Document**:
   - Validated envelope
   - How to enable/disable
   - How to reproduce perf numbers

## Output
- Integration code in vLLM
- `{artifact_dir}/integration.md`
- CI tests

## State Update
When complete:
```json
{"phase": "complete", "status": "shipped", "last_update": "YYYY-MM-DD"}
```
```
