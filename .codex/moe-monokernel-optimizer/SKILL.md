---
name: moe-monokernel-optimizer
description: Design and implement MoE kernel fusion optimizations for vLLM inference. Use when optimizing Mixture-of-Experts layers for specific (model, vLLM config, hardware, dtype) deployments where router + quantization + GEMMs can be fused (single cooperative kernel) or partially fused (split kernels) to improve decode latency, eliminate launch overhead, and reduce memory round-trips. Triggers on requests to optimize MoE decode latency, implement specialized monokernels, or decide fusion boundaries for MoE architectures (e.g., Llama-4, Qwen3-MoE, DeepSeek-style gated MoE).
---

# MoE Monokernel Optimizer (Codex)

Design specialized MoE kernels for vLLM that reduce launch overhead and memory traffic by fusing phases when beneficial (single cooperative kernel) or splitting into multiple kernels when ownership and occupancy require it.

## When Monokernel Applies

Monokernel optimization is beneficial when:
- Decode batch size is small (typically BS ≤ 32–128 depending on hardware)
- MoE preamble (router + quant + kernel launch gaps) exceeds ~20–30% of layer time
- Static deployment: fixed (model, vLLM config, hardware, dtype) quadruple
- **M_avg and ownership decisions** support a single-kernel path (see Decision 0b/0c)
- Routing distribution is known (or uniform routing is explicitly assumed)
- You will benchmark under CUDA graphs / torch.compile to match production
- Reference profiling of vLLM FusedMoE is available (required for planning unless hardware/NCU is unavailable)

**Barrier budget gate**: If the design requires more than 1–2 grid-wide barriers (`grid.sync`), strongly prefer split‑kernel or token‑major designs unless M_avg is large and routing is balanced.

**Baseline delta gate**: After combined‑graph profiling, compute the µs savings required to beat the baseline; if the required savings are implausible, re‑plan or document the limitation. Under CUDA graphs, treat “launch overhead” as mostly amortized and focus on GPU kernel time; use Nsight Systems to separate CUDA API time vs kernel execution time (see `references/profiling-launch-vs-kernel.md`). Always sanity‑check expected end‑to‑end impact via MoE share math (see `references/e2e-delta-math.md`).

**Full monokernel default gate** (heuristic):
- Use full monokernel only if **routing/prepare share ≥ ~15–20%**, **barrier budget ≤ 1–2**, **cooperative launch feasible**, and **M_avg is high or top_k=1**.
- Otherwise default to split/hybrid and target the expert kernel cost first.

## Solution Modes (Pick One)

Three practical modes exist; choose based on your Phase 1 Baseline Truth Snapshot (combined routing+experts, production parity under CUDA graphs / torch.compile where applicable):

- **Full cooperative monokernel**: one kernel fuses routing→prepare→(quant)→W1→act→(quant)→W2 (best when barriers are cheap and SMEM fits).
- **Hybrid large‑grid fusion (recommended default)**: keep baseline large‑grid expert GEMM(s) and fuse *around* them (routing/prepare kernel(s) + GEMM epilogues), avoiding `grid.sync` costs and SMEM bloat. See `references/hybrid-large-grid-fusion.md`.
- **Split kernels + CUDA graphs (production parity)**: 2–3 graph‑captured kernels (e.g., Router/Prepare → W1+Act(+Quant) → W2). Use when cooperative barriers/SMEM would reduce occupancy, or when graphs already amortize launch overhead but you still want to remove intermediate global‑memory round‑trips. See `assets/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md` (Pattern D) for the split‑kernel flow.

Always enforce CUDA‑graphs safety for any new CUDA ops (stream correctness, capture rules): see `references/cudagraph-safety.md`.

For a worked, model‑agnostic hybrid example (k‑way merge routing + W1 epilogue fusion), see `examples/HYBRID_FUSION_KWAYMERGE_W1_EPILOGUE.md`.
For an advanced routing+prepare port plan (SonicMoE-style runtime dispatch + specializations), see `examples/ADVANCED_SONICMOE_ROUTING_PREPARE_PORT.md`.

## Route Selection (Required)

Before Phase 3, you must:
- Write a **Baseline Truth Snapshot** in `{artifact_dir}/constraints.md` (template + required fields: `references/route-selection-decision-tree.md`).
- Write a **Route Decision** in `{artifact_dir}/optimization_plan.md` (pick one route, justify with snapshot numbers, include kill criteria: `references/route-selection-decision-tree.md`).

## Baseline Normalization (Required)

If the baseline warns that it is using a **default / missing tuned config** (common for Triton MoE), treat “generate a tuned baseline config” as a **normalization step**, not the main optimization outcome:
- Generate the tuned config (bounded search is fine).
- Re-run the Baseline Truth Snapshot under CUDA graphs.
- Only then finalize the Route Decision and pick fusion targets.

For the **hybrid large‑grid fusion** route, “tuning-only” is not sufficient unless you provide a documented “no fusion opportunities” proof (see `references/hybrid-large-grid-fusion.md`).

## Supported Data Types

| Type | Element Size | MMA Instruction | Notes |
|------|-------------|-----------------|-------|
| FP8 E4M3 | 1 byte | FP8 MMA on Tensor Cores (sm_89+) | Best for inference, requires sm_89+ |
| BF16 | 2 bytes | BF16 MMA on Tensor Cores (sm_80+) | 2× SMEM cost vs FP8 |
| FP16 | 2 bytes | FP16 MMA on Tensor Cores (sm_80+) | Legacy compatibility |
| MXFP4 | 0.5 bytes | Experimental | Future support |

For exact intrinsics/PTX patterns, see `references/code-templates.md` (MMA templates section).

## Validated Examples

Validated examples exist for these model architectures (the workflow targets any vLLM MoE that matches the fused_moe interface):

| Model | top_k | Hardware | Quantization | Key Patterns |
|-------|-------|----------|--------------|--------------|
| **Llama-4-Scout** | 1 | H100 (sm_90a) | Per-tensor FP8 | Direct write, TMA prefetch |
| **Qwen3-Coder-30B-A3B** | 8 | L40S (sm_89) | 128×128 block FP8 | FP32 accumulator, Split-H, cp.async |

See `examples/MODELS_COMPARISON.md` for detailed pattern notes.

## LLM Council Integration

Use `llm-council` as a de‑risking tool for correctness‑sensitive or high‑impact changes.
The single source of truth for policy, checkpoints, and invocation steps is:
`orchestration/llm-council.md`.

## Execution model (Codex CLI)

Codex CLI skills typically run **single-agent** (no separate Task subagents). Use a **stage-gated workflow**:

- Follow the **5 phases** below sequentially.
- Track progress in a **state file** so you can resume safely:
  - `{artifact_dir}/state.json` (recommended)
- If you need isolation for a tricky stage, you may run that stage in a fresh session via `codex exec` from your repo root, but you must still read/update `{artifact_dir}/state.json`.

The detailed phase/stage checklists live in:
- `orchestration/task-guide.md`

The workflow logic and stage structure live in:
- `orchestration/workflow.md`

## Phase/Stage Kickoff Micro‑Plan (Required)

At the start of **every phase and stage**, write a **micro‑plan (3–7 concrete steps)** and include it in the phase artifact or `{artifact_dir}/state.json`.

Rules:
- **No open questions by default**: convert unknowns into **action items** (measure, inspect code, run profiler).
- If blocked by **user input or missing hardware**, add a short **Inputs Required** section and pause.
- Keep micro‑plans concise; they are meant to guide execution, not replace the phase artifacts.
- After writing the micro‑plan, copy it into the phase artifact (e.g., top of `constraints.md`, `optimization_plan.md`, `validation_results.md`) or record it in `{artifact_dir}/state.json`.

## 5-Phase Workflow

Artifact directory: `moe_monokernel_artifacts/{model}_{hardware}_{dtype}_{tp}/`  
CUDA directory: `csrc/moe/moe_monokernel_{model}_{hardware}_{dtype}_{tp}/`

```
Phase 1: Gather Constraints     → {artifact_dir}/constraints.md
Phase 2: Optimization Planning  → {artifact_dir}/optimization_plan.md
Phase 3: Implementation         → {cuda_dir}/*.cu
Phase 4: Validation & Investigation → {artifact_dir}/validation_results.md (+ {artifact_dir}/investigation/* if needed)
Phase 5: Integration            → vLLM dispatch path
```

### Phase 1: Gather Constraints

Produce `{artifact_dir}/constraints.md`:
- Locate the model's MoE implementation in vLLM
- Trace forward semantics (routing → expert exec → accumulation)
- Compare against Llama 4 reference semantics
- Record model geometry (K, N, E, top_k, shared experts)
- Record hardware constraints (SM arch, SMEM limit, warp size, etc.)
- Record vLLM parallelism (TP/EP) and data types (weights/acts/scales)
- If available, capture per‑expert routing distribution; otherwise note uniform assumption
- Capture **combined routing+experts** CUDA‑graph baseline (single GPU) and record as a **required constraint** (e.g., `benchmarks/kernels/benchmark_moe_baseline_qwen3.py`).
- Record which baseline features are already fused (e.g., fused grouped_topk, routed-weight multiplication flags, activation/quant kernels).
- Run Nsight Systems once to split **CUDA API time vs GPU kernel time** for the baseline (see `references/profiling-launch-vs-kernel.md`).
- Run NCU on the combined‑graph baseline and record key device metrics (achieved occupancy, SM/Tensor‑Core utilization, DRAM bytes, L2/TEX traffic).
- Write the **Baseline Truth Snapshot** (required) to make route selection auditable: `references/route-selection-decision-tree.md`.
- If hardware/NCU is unavailable, document the reason explicitly in constraints.

After Phase 1: consider invoking `llm-council` to review constraints (recommended if model semantics are unclear or high-risk).

### Phase 2: Optimization Planning

Produce `{artifact_dir}/optimization_plan.md` with key decisions:

- **Decision 0**: Saturation score + routing distribution (from `references/algorithmic-branching.md`)
- **Decision 0b**: Per-expert M under uniform routing (M_avg = BS * top_k / E_local when EP pre‑dispatch; otherwise E_global) or measured histogram
- **Decision 0c**: Ownership model + fusion boundary (token-major vs expert-major; cooperative vs split)
- **Decision 0d**: Baseline profiling summary (required; combined routing+experts CUDA‑graph + NCU device metrics)
- **Decision 0e**: Baseline delta requirements (target savings vs combined‑graph baseline; tie to dominant kernel metrics)
- **Decision 0f**: Route decision (required): cooperative vs hybrid large‑grid fusion vs split kernels, with “why not” and kill criteria (`references/route-selection-decision-tree.md`).
- **Decision 1**: Output path (atomics vs direct write, depends on ownership)
- **Decision 2**: Shared expert strategy (from `references/architecture-pattern.md`)
- **Decision 3**: GEMM strategy (per-pair GEMV vs expert-grouped GEMM)
- **Decision 4**: SRAM Tetris (from `references/tiling-config.md`) — dtype affects buffer sizes
- **Decision 5**: Warp configuration
- **Decision 6**: MMA instruction selection (based on dtype + SM)

After Phase 2: invoke `llm-council` to review the plan.

### Phase 3: Implementation (4 stages)

| Stage | Name | Output |
|------:|------|--------|
| 3.1 | `routing_and_prepare` | token→expert mapping buffers (+ any packed/aligned layout) |
| 3.2 | `activation_quantization` (conditional) | activation/quant semantics (may be fused into Stage 3.3) |
| 3.3 | `gemm_implementation` (**CRITICAL**) | route‑specific hot path (expert GEMM(s) and/or material fusions) |
| 3.4 | `kernel_assembly` | final kernel(s) + wrapper + runtime dispatch |

**Hot‑Path Constraints** (non‑negotiable):
- **Cooperative monokernel route**: implement MMA‑based up/down projections **in CUDA** (CuTe/CUTLASS allowed). No “call cuBLAS/Triton” for the expert GEMM(s).
- **Hybrid large‑grid fusion route**: baseline GEMM(s) allowed, but implement ≥1 *material* fusion around them (e.g., W1 epilogue fusion), and validate speedup under CUDA graphs.
- **Split kernels + CUDA graphs route**: baseline GEMM(s) allowed; keep multiple kernels if needed, but the **captured graph replay** must beat baseline across the benchmark bucket set. At least one non‑GEMM stage must be removed/fused or its memory traffic reduced (routing+prepare and/or epilogue are typical targets).

See: `orchestration/task-guide.md` for step‑by‑step checklists.

Activation handling:
- Common activations (SiLU/GELU/ReLU): use templates in `references/code-templates.md`
- Unknown/custom: follow the “unknown activation” exploration recipe in `references/code-templates.md`

### Phase 4: Validation & Investigation

Validate monokernel against stock `fused_moe` in **3 stages**:

| Stage | Check | Success Criteria |
|-------|-------|------------------|
| 4.1 Correctness | Numerical accuracy | `max_diff < tolerance` (dtype-specific) |
| 4.2 Kernel-Level | Performance under CUDA graphs | `speedup >= 1.0x` at **all** tested batch sizes (no regressions) |
| 4.3 E2E Latency | Real inference impact | `>5%` improvement (BS≤8), `>0%` (BS>8) |

**Goal**: beat the **combined routing+experts CUDA‑graph baseline**, not just reduce kernel count.

**On failure**: treat Phase 4 failures as *investigation problems*, not generic "blocked".

- Set the failing validation stage to `status: "needs_investigation"` in `{artifact_dir}/state.json`
- Run **one bounded investigation cycle** using `orchestration/investigation-prompts.md`
- Produce investigation artifacts under `{artifact_dir}/investigation/` and a council-reviewed `fix_proposal.md`
- Choose one of the explicit outcomes:

| Decision | Action |
|----------|--------|
| `phase_3` | Back to Phase 3 (specific stage with fix context) |
| `phase_2` | Back to Phase 2 (re-plan with new constraints) |
| `document_proceed` | Document limitation, proceed to Phase 5 |
| `rerun_validation` | Re-run the failing validation stage (measurement/env issue) |
| `escalate_human` | Pause for human review |

See `validation/validation.md` and `orchestration/investigation-prompts.md`.

After Phase 4: invoke `llm-council` to sanity check results and conclusions.

### Phase 5: Integration

Wire monokernel into vLLM:
- CMakeLists / build integration
- Torch bindings
- Python wrapper
- Dispatch fast-path selection

## Resume

1. Find state: `ls moe_monokernel_artifacts/*/state.json`
2. Read it and identify current phase/stage
3. Continue with the next phase/stage checklist from `orchestration/task-guide.md`
4. If blocked: follow `orchestration/failure-handling.md` and consider invoking `llm-council`

## Validation checklist (high-signal)

### Implementation (Phase 3)

- [ ] Phase 1 constraints complete and reviewed
- [ ] Phase 2 plan complete and reviewed
- [ ] Activation function handled (template or explored)
- [ ] **No TODOs in GEMM kernels**
- [ ] **MMA calls present in both up and down projection**

### Validation (Phase 4)

- [ ] **4.1 Correctness**: `max_diff < tolerance` for batch sizes `[1, 8, 64]`
- [ ] **4.2 Kernel Performance**: `speedup >= 1.0x` at **all** tested batch sizes (no regressions)
- [ ] **4.3 E2E Latency**: `>5%` improvement at BS≤8, `>0%` at BS>8
- [ ] If any validation failed → investigation completed with an explicit decision
- [ ] If `document_proceed` → limitation documented in `validation_results.md`
