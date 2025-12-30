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

**Freedom**: LOW (do not improvise; produce the required artifacts)

**Steps**
1. Create `{artifact_dir}` and record baseline environment:
   - vLLM commit hash, GPU/driver/CUDA, production knobs (`-O`, `-cc`) used for benchmarks.
2. Identify model MoE path and routing semantics in `vllm/model_executor/models/` (file + class + forward path).
3. Fill **MoE Parameters & Semantics** using:
   - `references/moe-parameters-template.md` (required).
4. Record geometry (K, N, E_global/E_local, top_k, shared experts) and TP/EP behavior.
5. Record quantization/scales and dtype assumptions.
6. Run **combined routing+experts baseline profiling** under CUDA graphs (single GPU) and capture:
   - totals for BS sweep
   - kernel breakdown
   - NCU highlights (occupancy, SM/TC util, DRAM/L2 traffic)
7. Write the **Baseline Truth Snapshot** into `{artifact_dir}/constraints.md`
   (template: `references/route-selection-decision-tree.md`).
8. Save/commit `{artifact_dir}/state.json` (phase=1 complete).

**Validation**
- `{artifact_dir}/constraints.md` includes:
  - production benchmark knobs
  - MoE semantics template filled
  - CUDA-graph combined baseline + kernel breakdown + NCU highlights
  - Baseline Truth Snapshot

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

This is the core performance stage. Implement the **route‑specific hot path** chosen in Phase 2.

**Hard rule**: Do not change the plan mid‑stage. If profiling indicates the plan is wrong, stop and trigger a Phase 2 re‑plan.

#### Deliverables (all routes)
- A working CUDA implementation (kernel(s) + C++/PyTorch binding) that runs under CUDA graphs.
- A minimal benchmark that isolates the hot path and reports GPU time (CUDA events), not wall-clock.
- A brief note in `{artifact_dir}/implementation_notes.md` explaining the intended win mechanism (e.g., “epilogue fusion removes scale_inputs + activation kernels and 1 global round‑trip”).

#### Route‑specific requirements

**A) Cooperative monokernel route**
- Implement MMA-based expert projections **in CUDA** (CuTe/CUTLASS allowed).
- No “call cuBLAS” or “call Triton” for the expert GEMM(s).
- Minimize global barriers (`grid.sync`) to ≤ 1–2 unless Phase 1 showed M_avg is large enough to amortize.

**B) Hybrid large‑grid fusion route**
- Baseline GEMM(s) allowed.
- Implement ≥1 *material* fusion around them (examples: W1 epilogue fusion = activation + quantize; routing+prepare fusion feeding baseline GEMMs; fused reduction).
- Demonstrate that the fusion yields a measurable GPU‑time win under CUDA graphs in the kernel benchmark.

**C) Split kernels + CUDA graphs route (production parity)**
- Baseline GEMM(s) allowed.
- You may keep multiple kernels, but you must:
  - Remove/fuse at least one non‑GEMM stage or reduce its memory traffic (routing+prepare and/or epilogue are typical targets), AND
  - Demonstrate that **captured graph replay** is faster than baseline across the benchmark bucket set.
- If the “win” comes only from graph replay, that is not sufficient; baseline also benefits from graphs. The win must be attributable to reduced GPU kernel work or fewer global round‑trips.

#### Exit Criteria
- Correctness passes for this stage’s outputs on the benchmark’s tested batch sizes.
- GPU time is **not worse** than baseline at every tested batch size under CUDA graphs; otherwise mark as `needs_investigation` and proceed to Phase 4 investigation prompts.

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
