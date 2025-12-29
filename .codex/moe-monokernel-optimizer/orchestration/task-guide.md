# MoE Monokernel Task Guide (Single‑Agent)

This guide is the **single‑agent execution checklist** for Codex CLI. Use it instead of `task-prompts.md`.

## How to Use

1. Read `SKILL.md` for non‑negotiables (baseline profiling, GEMM hot‑path constraints, ownership decisions).
2. Read `{artifact_dir}/state.json` to determine current phase/stage.
3. For the target phase/stage below:
   - Write a 3–7 step **micro‑plan** at the top of the phase artifact or in `state.json`.
   - Execute the checklist steps in order.
   - Update `state.json` with status + summary.
4. If validation fails, follow `orchestration/investigation-prompts.md`.
5. Use `orchestration/llm-council.md` for council policy when appropriate.

---

## Phase 1: Constraints

**Purpose**: Capture model semantics, geometry, hardware, and baseline profiling (combined routing+experts) that the plan must beat.

**Inputs**
- Model code in `vllm/model_executor/models/`
- `references/algorithmic-branching.md`
- `references/router-design.md`
- `references/gpu-configs.md`

**Outputs**
- `{artifact_dir}/constraints.md`
- `{artifact_dir}/state.json`

**Steps**
1. Identify model MoE path, routing semantics, weight timing, activation, accumulation.
2. Record geometry (K, N, E, top_k, shared experts) + TP/EP (E_local).
3. Record quantization/scales and dtype.
4. **Run baseline profiling** (single GPU, combined routing+experts under CUDA graphs). Capture totals + kernel breakdown.
5. Run Nsight Systems once to separate CUDA API time vs GPU kernel time for the MoE subgraph (advisory but strongly recommended).
6. Run NCU on the combined‑graph baseline and record key metrics (occupancy, SM/TC utilization, DRAM/L2 traffic).
7. Write the **Baseline Truth Snapshot** section into `{artifact_dir}/constraints.md` (template: `references/route-selection-decision-tree.md`).
8. Note any warnings (missing tuned config) or unavailable hardware (if so, document the reason explicitly).

**Validation**
- Constraints file includes baseline timings + NCU metrics + Baseline Truth Snapshot.

**Stop/Retry**
- If hardware/NCU unavailable, document Inputs Required and pause.

**State Update**
- `phase=1_constraints`, `status=complete`, `summary=...`

---

## Phase 2: Optimization Planning

**Purpose**: Decide ownership + fusion boundary based on constraints and baseline delta required to win.

**Inputs**
- `{artifact_dir}/constraints.md`
- `references/algorithmic-branching.md`
- `references/tiling-config.md`
- `references/code-templates.md`

**Outputs**
- `{artifact_dir}/optimization_plan.md`
- `{artifact_dir}/state.json`

**Steps**
1. Compute M_avg using E_local (or E_global if EP not pre‑dispatch).
2. Decide ownership (token‑major/expert‑major/hybrid).
3. Decide route (cooperative monokernel vs hybrid large‑grid fusion vs split kernels) using the decision tree.
4. Write the **Route Decision** section (required) in `{artifact_dir}/optimization_plan.md` (template: `references/route-selection-decision-tree.md`), including “why not” and kill criteria.
5. Enumerate at least 2 concrete “fusion opportunities” (or prove none exist) using the baseline kernel breakdown:
   - Example opportunities: W1 epilogue fusion (activation+quant), routing+prepare fusion, reduce fusion (usually not).
   - If you choose “no fusion”: include evidence (e.g., nsys shows stages already fused / negligible).
6. Decide output accumulation path (atomics only if output is not uniquely owned).
7. Decide weight placement based on model semantics.
8. Solve SRAM tiling and warp config (if applicable to the chosen route).
9. **Baseline delta requirements**: compute required savings vs combined‑graph baseline and tie to dominant kernel metrics.

**Validation**
- Plan includes baseline summary, NCU highlights, Route Decision, delta‑to‑baseline targets, feasibility call, and either (a) at least one fusion target for Phase 3 or (b) a documented “no fusion” proof.

**Stop/Retry**
- If required savings are implausible, re‑plan or document limitation.

**State Update**
- `phase=2_planning`, `status=complete`, `summary=...`

---

## Phase 3: Implementation

### Stage 1: routing_and_prepare
**Purpose**: Implement/validate routing + prepare stage (if not already provided).

**Inputs**: constraints + plan, `references/router-design.md`
**Outputs**: `{cuda_dir}` sources, `validation_results.md`, `state.json`
**Steps**
1. Implement routing/prepare per plan.
2. Validate correctness vs reference.
3. Optional parity check under CUDA graphs (advisory).
**State Update**: mark stage complete or needs_investigation.

### Stage 2: activation_quantization (if FP8/INT8)
**Purpose**: Implement quantization stage per plan.

**Inputs**: constraints + plan, `references/code-templates.md`
**Outputs**: `{cuda_dir}` sources, `validation_results.md`, `state.json`
**Steps**
1. Implement quantization and scale handling.
2. Validate correctness vs reference scales.
3. Optional parity check under CUDA graphs (advisory).
**State Update**: mark stage complete or needs_investigation.

### Stage 3: gemm_implementation (CRITICAL)
**Purpose**: Implement the hot path per the chosen route.

**Inputs**: constraints + plan, `references/code-templates.md`, `references/tiling-config.md`
**Outputs**: `{cuda_dir}` sources, `validation_results.md`, `state.json`
**Steps**
1. **Cooperative monokernel route**: implement MMA‑based up/down projections (CUDA/CuTe/CUTLASS only).
2. **Hybrid large‑grid fusion route**: implement at least one material fusion around baseline GEMM(s), e.g.:
   - W1 epilogue fusion (activation + quantization into W1 GEMM kernel), or
   - routing+prepare fusion kernel (if it is material under CUDA graphs).
3. Ensure weight timing and accumulation match constraints.
4. Validate correctness vs reference.
5. Profile under CUDA graphs to check perf.

**Non‑negotiables (see SKILL.md)**
- Cooperative route: no reference GEMM calls for Stage 3 completion; Triton is not allowed for GEMM hot path.
- Hybrid route: keeping baseline GEMM(s) is allowed, but Stage 3 must not be “tuning only” unless you have a documented “no fusion” proof.

**State Update**:
- Cooperative route: mark stage complete only if the new GEMM is the hot path.
- Hybrid route: mark stage complete only if the planned fusion target(s) are implemented and validated under CUDA graphs (or a “no fusion opportunities” proof is documented).

### Stage 4: kernel_assembly
**Purpose**: Wire stages into the dispatch path.

**Inputs**: `{cuda_dir}` kernels, plan, `references/architecture-pattern.md`
**Outputs**: `{cuda_dir}/moe.cu`, bindings, `validation_results.md`, `state.json`
**Steps**
1. Cooperative route: assemble main kernel (cooperative or split per plan).
2. Hybrid route: wire the new routing op and/or fused epilogue kernel into the existing expert path (guarded by env vars as needed).
3. Ensure the default hot path matches the plan (new GEMM for cooperative; fused epilogue / routing op for hybrid).
4. Validate correctness/perf under CUDA graphs.

---

## Phase 4: Validation & Investigation

**Purpose**: Prove correctness and beat baseline in the enabled region.

**Inputs**: `validation/validation.md`, baseline in constraints.

**Steps**
1. Run Stage 4.1 correctness tests.
2. Run Stage 4.2 kernel‑level perf under CUDA graphs.
3. Run Stage 4.3 E2E latency.
4. If any stage fails, set `needs_investigation` and follow `orchestration/investigation-prompts.md`.

**State Update**: mark stage complete only when success criteria are met.

---

## Phase 5: Integration

**Purpose**: Wire fast‑path dispatch and ensure enablement envelope is correct.

**Inputs**: plan + validation results.

**Steps**
1. Add build integration + bindings.
2. Add Python dispatch with enablement envelope (guarded fast path).
3. Validate fallback parity outside envelope.

**State Update**: `phase=5_integration`, `status=complete`.
