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
- Reference profiling of vLLM FusedMoE is available (recommended for planning, required for regressions)

**Barrier budget gate**: If the design requires more than 1–2 grid-wide barriers (`grid.sync`), strongly prefer split‑kernel or token‑major designs unless M_avg is large and routing is balanced.

**Baseline delta gate**: After combined‑graph profiling, compute the µs savings required to beat the baseline; if the required savings are implausible, re‑plan or document the limitation.

**Full monokernel default gate** (heuristic):
- Use full monokernel only if **routing/prepare share ≥ ~15–20%**, **barrier budget ≤ 1–2**, **cooperative launch feasible**, and **M_avg is high or top_k=1**.
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

See `examples/MODELS_COMPARISON.md` for detailed pattern notes.

## LLM Council Integration

`llm-council` is a **de-risking tool**, not a hard gate. Use it when the change is high-impact, correctness-sensitive, or you feel uncertain.

### Risk-tier policy (C)

- **Required (high-risk)** — default expectation:
  - You are changing **kernel math / accumulation** (e.g., FP32 accumulator logic, atomics, Split‑H reduction)
  - You are changing **memory layout / shared memory / TMA/cp.async staging** in a way that could silently corrupt results
  - You are introducing or modifying **top_k > 1** routing + accumulation behavior
  - You are making a **major performance tradeoff** that could regress other batch sizes or architectures (e.g., new tiling strategy)

- **Recommended (medium-risk)**:
  - After Phase 2 when the plan introduces non-trivial architectural choices (tiling, buffering, fusion boundaries)
  - After Phase 4 if an investigation proposes a non-obvious fix, or if conclusions are based on noisy perf data
  - When you are stuck after 2+ distinct attempts (fresh eyes help)

- **Optional (low-risk)**:
  - Mechanical refactors, comments/docs, build-system glue, formatting-only changes
  - Small edits with strong test coverage and low blast radius

> If you choose to skip a recommended/required council review, proceed — but consider leaving a short rationale in `{artifact_dir}/state.json` so future resumption has context.

### How to invoke llm-council in Codex

This repo contains a separate `llm-council` skill. Invoke it by name in chat (e.g., “Use llm-council to review …”). Follow its instructions to prepare `.llm-council/context.md` and run the critics (parallel by default, with an optional sequential mode).

See `orchestration/failure-handling.md` for the escalation ladder.

## Execution model (Codex CLI)

Codex CLI skills typically run **single-agent** (no separate Task subagents). Use a **stage-gated workflow**:

- Follow the **5 phases** below sequentially.
- Track progress in a **state file** so you can resume safely:
  - `{artifact_dir}/state.json` (recommended)
- If you need isolation for a tricky stage, you may run that stage in a fresh session via `codex exec` from your repo root, but you must still read/update `{artifact_dir}/state.json`.

The detailed phase/stage prompts live in:
- `orchestration/task-prompts.md`

The workflow logic and stage structure live in:
- `orchestration/workflow.md`

## Phase/Stage Kickoff Plan (Required)

At the start of **every phase and stage**, you **must invoke the `plan` skill** to create or update a **phase/stage plan**. The plan must include a **micro‑plan (3–7 concrete steps)**.

Rules:
- **No open questions by default**: convert unknowns into **action items** (measure, inspect code, run profiler).
- If blocked by **user input or missing hardware**, add a short **Inputs Required** section and pause.
- Keep micro‑plans concise; they are meant to guide execution, not replace the phase artifacts.
- After creating/updating the plan, copy the micro‑plan into the phase artifact (e.g., top of `constraints.md`, `optimization_plan.md`, `validation_results.md`) or record it in `{artifact_dir}/state.json`.

**Plan naming**: use a stable, lower‑case, hyphenated name per phase/stage (e.g., `moe-monokernel-{model}-{hardware}-{dtype}-p3-gemm`). If a plan already exists for that phase/stage, **update it** rather than creating a new file.

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
- Capture combined routing+experts CUDA‑graph baseline (if hardware available); otherwise mark as unavailable with reason

After Phase 1: consider invoking `llm-council` to review constraints (recommended if model semantics are unclear or high-risk).

### Phase 2: Optimization Planning

Produce `{artifact_dir}/optimization_plan.md` with key decisions:

- **Decision 0**: Saturation score + routing distribution (from `references/algorithmic-branching.md`)
- **Decision 0b**: Per-expert M under uniform routing (M_avg = BS * top_k / E_global) or measured histogram
- **Decision 0c**: Ownership model + fusion boundary (token-major vs expert-major; cooperative vs split)
- **Decision 0d**: Baseline reference profiling summary (optional but recommended)
- **Decision 0e**: Baseline delta requirements (target savings vs combined‑graph baseline)
- **Decision 1**: Output path (atomics vs direct write, depends on ownership)
- **Decision 2**: Shared expert strategy (from `references/architecture-pattern.md`)
- **Decision 3**: GEMM strategy (per-pair GEMV vs expert-grouped GEMM)
- **Decision 4**: SRAM Tetris (from `references/tiling-config.md`) — dtype affects buffer sizes
- **Decision 5**: Warp configuration
- **Decision 6**: MMA instruction selection (based on dtype + SM)

After Phase 2: invoke `llm-council` to review the plan.

### Phase 3: Implementation (4 stages)

Phase 3 is staged to keep GEMM work together. Implement stages **sequentially**:

| Stage | Components | Notes |
|-------|------------|-------|
| routing_and_prepare | router + prepare | non-GEMM, tightly coupled |
| activation_quantization | scale_inputs | conditional (FP8 only) |
| **gemm_implementation** | **up_proj + down_proj** | implement together only for single‑kernel expert‑major; split/hybrid may diverge |
| kernel_assembly | output + main kernel | wire everything + cooperative launch |

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

See `validation/README.md` and `orchestration/investigation-prompts.md`.

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
3. Continue with the next phase/stage prompt from `orchestration/task-prompts.md`
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
