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

## Fast Phase-1 Bucket Set (Time-Bounded Default)

Unless you have a stronger reason, constrain Phase 1 profiling to **two buckets** to keep iteration time low:
- `batch_size ∈ {8, 64}`
- `input_len = 1024`, `output_len = 32`

Expand to more buckets only after Phase 2 has a clear win hypothesis.

## BLOCKING REQUIREMENTS (ALL must be completed)

These are NOT optional. Phase 2 CANNOT start until ALL are met.

### Requirement 1: nsys Profile Files MUST Exist
```bash
# RUN this command (not just document it):
VLLM_WORKER_MULTIPROC_METHOD=spawn \
VLLM_TORCH_COMPILE_LEVEL=3 \
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o {artifact_dir}/runs/baseline \
  vllm bench latency \
    --model {model_id} \
    --batch-size 8 \
    --input-len 1024 --output-len 2 \
    --num-iters-warmup 5 \
    --num-iters 1

# VERIFY file exists:
ls {artifact_dir}/runs/baseline.nsys-rep
# If no file, you have NOT completed this requirement

# Extract MoE kernel summary:
nsys stats --report cuda_gpu_kern_sum {artifact_dir}/runs/baseline.nsys-rep \
  | grep -E "fused_moe|fused_experts|topk"
```

### Requirement 2: Baseline MUST Be vLLM Production Kernels
Document actual kernel timings from nsys, specifically:
- `fused_moe` or `fused_experts` kernel times (NOT naive PyTorch loops)
- CUDA graphs MUST be enabled during profiling
- torch.compile MUST be at production level (VLLM_TORCH_COMPILE_LEVEL=3)

### Requirement 3: Timing Data MUST Be Numbers (Not Commands)
constraints.md MUST contain actual measured values like:
```
## Baseline Truth Snapshot (BS=8)
- fused_moe kernel time = 523.4 µs
- topk_softmax kernel time = 45.2 µs
- Total MoE time = 671.1 µs
```
Documenting "run this command" is NOT sufficient.

### Requirement 4: Baseline Normalization (CRITICAL if applicable)
If baseline MoE kernels fall back to a default/tuner-missing config (common for Triton MoE kernels):
- Generate a tuned baseline config (bounded tuner is acceptable)
- **Re-run the baseline snapshot after normalization** (do NOT use the un-tuned timings)
- Use the *normalized* baseline for route selection in Phase 2
- Otherwise you may optimize the wrong target (the un-tuned baseline is artificially slow)

### Requirement 5: E2E Baseline MUST Exist
You must record at least one **full-model** E2E latency baseline in Phase 1:
- Run `vllm bench latency` for at least one representative bucket
- If model weights are not cached locally: **download them** (do not skip)
- If download is blocked (gated/auth/terms/network/disk): set `state.json` status to `blocked` and ask user for waiver
- This is required to compute MoE share `f = T_moe / T_total`

## Instructions

1. **Read first**:
   - `.claude/skills/moe-monokernel-optimizer/references/nsys-profiling-guide.md` (practical commands)
   - `.claude/skills/moe-monokernel-optimizer/references/moe-parameters-template.md`
   - `.claude/skills/moe-monokernel-optimizer/references/profiling-launch-vs-kernel.md` (conceptual)

2. **Locate vLLM model code**:
   - Find MoE implementation in `vllm/model_executor/models/`
   - Trace forward path: routing → expert execution → accumulation

3. **Document semantics** (CRITICAL for correctness):
   - Router scoring: softmax vs sigmoid
   - Renormalization behavior
   - Weight application: before vs after activation
   - Shared experts (if any)
   - Accumulation: direct write vs atomics vs reduce

4. **RUN baseline profiling** (DO NOT SKIP):
   - Use commands from Requirement 1 above (single BS=8 baseline)
   - Extract MoE kernel times using `nsys stats`
   - Record timings WITH ACTUAL NUMBERS in constraints.md

5. **Write constraints.md** using template from references/moe-parameters-template.md

6. **Hopper IO sanity-check (sm_90a only)**:
   - If targeting H100/H200, check whether the MoE critical path is limited by **output atomics / scatter-like stores / heavy epilogues** (vs pure GEMM math)
   - If yes, prefer designs that keep GEMM stores **regular/contiguous** and move irregularity into a separate **token-major aggregation** step (instead of fusing scatter/atomics into a GEMM epilogue)
   - Treat as a hypothesis and gate by CUDA-graph kernel-time wins

## Output
Create `{artifact_dir}/constraints.md` with:
- Target envelope section
- Model semantics section (routing, activation, weight placement)
- Baseline Truth Snapshot (BS=8 timings under CUDA graphs) WITH NUMBERS
- "Already fused?" checklist

## VERIFICATION (MUST RUN BEFORE COMPLETION)
```bash
python .claude/skills/moe-monokernel-optimizer/scripts/verify_phase1_baseline.py {artifact_dir}
# Exit code MUST be 0 to proceed
# If non-zero: DO NOT mark phase complete, fix blockers first
```

## State Update
ONLY after verification passes, update `{artifact_dir}/state.json`:
```json
{"phase": "2_planning", "status": "pending", "last_update": "YYYY-MM-DD"}
```

## On Blocker
If you cannot complete a BLOCKING REQUIREMENT:
1. Update state.json: `"status": "blocked", "blocker": {"description": "...", "severity": "critical"}`
2. Create blocker file: `{artifact_dir}/blockers/phase1_{date}.md`
3. Do NOT mark phase complete
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

## BLOCKING REQUIREMENTS (Phase 2)

### Requirement 1: Use Optimization Plan Template
Write `{artifact_dir}/optimization_plan.md` using `references/optimization-plan-template.md`:
- Copy/paste the template structure
- Reject the plan if it contains placeholders ("1. 2. 3." or empty sections)
- Must include ranked top-10 opportunity list (section 3A) with evidence
- Must include 2-3 active hypotheses tied to opportunity IDs (section 3B)

### Requirement 2: Fusion Enumeration
Enumerate at least 2 concrete fusion opportunities (or prove none exist):
- Each opportunity must cite Phase 1 evidence (kernel name, time)
- Compute required savings vs baseline and tie to dominant kernel metrics

### Requirement 3: Mid-Stage Replanning Rule
Do NOT change the plan mid-implementation:
- If profiling during Phase 3 indicates the plan is wrong, STOP
- Trigger Phase 2 re-plan with updated evidence
- Do NOT thrash by repeatedly modifying approach during implementation

## Instructions

1. **Read first**:
   - `{artifact_dir}/constraints.md` (your baseline truth)
   - `.claude/skills/moe-monokernel-optimizer/references/route-selection-decision-tree.md`
   - `.claude/skills/moe-monokernel-optimizer/references/algorithmic-branching.md`
   - `.claude/skills/moe-monokernel-optimizer/references/fusion-feasibility-heuristics.md`
   - `.claude/skills/moe-monokernel-optimizer/references/optimization-plan-template.md`

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

7. **Optional: Council review**:
   - If risk tier is high (semantics changes, top_k>1, major fusion boundary change), consider invoking `llm-council` per `orchestration/llm-council.md`

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

## Phase 3 General Requirements (apply to all stages)

### Grid Barrier Limits (Cooperative Route)
Minimize global barriers (`grid.sync`) to ≤ 1–2 unless M_avg is large enough to amortize.
Each additional barrier is a tax that must be justified by measured savings.

### Split Kernels Warning
If the "win" comes only from CUDA graph replay (vs baseline also using graphs), that is NOT sufficient:
- Baseline also benefits from CUDA graphs
- Must show kernel-time improvement with both under graphs

### Do NOT Change Plan Mid-Stage
If profiling during implementation indicates the plan is wrong:
1. STOP implementation
2. Update state.json with findings
3. Return to Phase 2 to re-plan with new evidence
4. Do NOT thrash by repeatedly modifying approach

### Route-Specific Checklists

**A) Cooperative monokernel**:
- [ ] Use `references/code-templates.md` and `references/tiling-config.md` for kernel structure and SRAM budgeting
- [ ] Minimize `grid.sync` barriers; treat each barrier as a tax that must be amortized
- [ ] Ensure accumulation semantics match baseline (especially for top_k>1)

**B) Hybrid large-grid fusion**:
- [ ] Keep baseline large-grid GEMM(s); fuse *around* them (see `references/hybrid-large-grid-fusion.md`)
- [ ] Target "material fusion" deliverables (e.g., W1 epilogue fusion, routing+prepare fusion)
- [ ] Do not double-count: verify what baseline already fuses before claiming a win

**C) Split-kernel graph-captured route**:
- [ ] Prefer fewer, simpler kernels that each keep a large grid and high occupancy
- [ ] Make workspace allocation explicit and reusable; avoid per-iteration allocations

---

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

## BLOCKING REQUIREMENTS (ALL must be completed)

These are NOT optional. Phase 5 CANNOT start until ALL are met.

### Requirement 1: Baseline MUST Be vLLM (NOT Naive PyTorch)
```python
# CORRECT - import vLLM's actual production kernel:
from vllm.model_executor.layers.fused_moe import fused_experts
# or
from vllm.model_executor.layers.fused_moe import fused_moe

# WRONG - do NOT use naive PyTorch loops as baseline:
# for expert_idx in range(num_experts):  # WRONG
#     out += torch.matmul(x, expert_weights[expert_idx])  # WRONG
```

### Requirement 2: Numerical Comparison MUST Use torch.allclose
```python
# Test file MUST contain numerical comparison:
baseline_output = fused_experts(x, w1, w2, topk_weights, topk_ids, ...)
monokernel_output = monokernel_forward(x, w1, w2, topk_weights, topk_ids, ...)

assert torch.allclose(
    monokernel_output,
    baseline_output,
    atol=TOLERANCE,
    rtol=TOLERANCE
), f"FAIL: max diff = {(monokernel_output - baseline_output).abs().max()}"

# Smoke tests (shape, NaN checks) alone are NOT sufficient
```

### Requirement 3: Production Parity Environment
Benchmarks MUST run with production settings:
```python
# REQUIRED - do NOT disable these:
os.environ["VLLM_TORCH_COMPILE_LEVEL"] = "3"  # Production level
# CUDA graphs enabled by default

# FORBIDDEN - do NOT use:
# os.environ["TORCH_COMPILE_DISABLE"] = "1"  # WRONG
# enforce_eager=True  # WRONG
```

### Requirement 4: Kernel Benchmarks Under CUDA Graphs (BLOCKING)
For kernel-level (isolated) comparisons between Triton and CUDA C++:
```python
# REQUIRED: Wrap both baseline and monokernel in CUDA graphs
import torch.cuda

# Capture baseline graph
g_baseline = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_baseline):
    baseline_out = fused_experts(x, w1, w2, ...)

# Capture monokernel graph
g_mono = torch.cuda.CUDAGraph()
with torch.cuda.graph(g_mono):
    mono_out = token_major_down_projection(...)

# Time graph replays (NOT individual kernel launches)
start.record()
for _ in range(iters):
    g_baseline.replay()  # Graph replay, not kernel launch
end.record()
```

**Rationale**: Triton kernels have ~50-100 µs launch overhead per invocation.
CUDA C++ kernels have ~5-10 µs launch overhead. Without CUDA graphs, this
difference dominates timing and makes comparisons meaningless.

**INVALID** (timing without graph capture):
```python
# WRONG - unfair due to launch overhead difference:
start.record()
baseline_out = fused_experts(...)  # Triton: ~50-100 µs overhead
end.record()
# vs
start.record()
mono_out = monokernel(...)  # CUDA C++: ~5-10 µs overhead
end.record()
```

### Requirement 5: ALL Kill Criteria MUST Be Evaluated
Every kill criterion from Phase 2 MUST have a result (PASS/FAIL).
"TODO", "optional", or "skip" are NOT valid results.

Update state.json with:
```json
"kill_criteria_results": {
    "criterion_1_xxx": "PASS - measured X vs Y",
    "criterion_2_xxx": "PASS - no regression",
    "criterion_3_xxx": "PASS - occupancy verified via NCU",
    "criterion_4_xxx": "PASS - MoE is 25% of E2E time"
}
```

## Instructions

1. **Read first**:
   - `.claude/skills/moe-monokernel-optimizer/references/validation-defaults.md`
   - `.claude/skills/moe-monokernel-optimizer/validation/E2E_LATENCY_GUIDE.md`

2. **Gate 4.1: Correctness** (BLOCKING)
   - Import vLLM's `fused_experts` or `fused_moe` as baseline
   - Use `torch.allclose()` with tolerances from validation-defaults.md
   - Test ALL batch buckets
   - **If fails**: STOP. Use investigation-playbook.md § Correctness

3. **Gate 4.2: Kernel Performance** (BLOCKING)
   - Baseline = vLLM's fused_moe kernel (NOT naive PyTorch)
   - Measure under CUDA graphs + torch.compile enabled
   - Gate: optimized ≤ baseline for ALL buckets
   - Run NCU to verify occupancy
   - **If fails**: STOP. Use investigation-playbook.md § Kernel Perf

4. **Gate 4.3: E2E Latency** (BLOCKING)
   - Run `vllm bench latency` with production settings
   - Both baseline and optimized must use same environment
   - Calculate improvement percentage
   - Document actual numbers (not estimates)

## Output
Create `{artifact_dir}/validation_results.md` with:
- Correctness: pass/fail, max diff, tolerance used
- Kernel perf: per-bucket timings vs vLLM baseline (NOT PyTorch)
- E2E: latency comparison, improvement %

Create/update `{artifact_dir}/benchmark_validation.py` with:
- Import from vllm.model_executor.layers.fused_moe (NOT naive loops)
- torch.allclose numerical comparison
- Production parity environment (no TORCH_COMPILE_DISABLE)

## VERIFICATION (MUST RUN BEFORE COMPLETION)
```bash
python .claude/skills/moe-monokernel-optimizer/scripts/verify_phase4_gates.py {artifact_dir}
# Exit code MUST be 0 to proceed
# If non-zero: DO NOT mark phase complete, fix blockers first
```

## State Update
ONLY after verification passes, update `{artifact_dir}/state.json`:
```json
{"phase": "5_integration", "status": "pending", "last_update": "YYYY-MM-DD"}
```

## On Failure
If any gate fails OR verification script returns non-zero:
1. Update state.json: `"status": "blocked", "blocker": {"description": "...", "severity": "critical"}`
2. Create blocker file: `{artifact_dir}/blockers/phase4_{date}.md`
3. Do NOT declare "SHIP"
4. Follow investigation-playbook.md with bounded cycles
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
