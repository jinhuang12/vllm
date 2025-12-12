---
description: Design + implement fused MoE monokernel optimizations for vLLM inference (router + quant + GEMMs fused).
argument-hint: MODEL_ID="<hf_id_or_local_path>" [HARDWARE="<gpu>"] [DTYPE="<fp8|bf16|fp16>"] [TP=<int>] [TOPK=<int>] [MODE="<plan|implement|validate|integrate|full>"] [PHASE=<1-5>] [ARTIFACT_DIR="<path>"]
---

# MoE Monokernel Optimizer (vLLM)

You are operating **inside the vLLM repo root**.

Goal: design, implement, validate, and (optionally) integrate a **single cooperative CUDA “monokernel”** that fuses:
router → token grouping/sorting → (optional) activation quant → up-proj GEMM → activation (e.g., SiLU/SwiGLU) → down-proj GEMM → accumulation.

## Inputs

- MODEL_ID: `$MODEL_ID`
- HARDWARE: `$HARDWARE`
- DTYPE: `$DTYPE`
- TP: `$TP`
- TOPK: `$TOPK`
- MODE: `$MODE`  (default: `full`)
- PHASE: `$PHASE` (optional; run only that phase then stop)
- ARTIFACT_DIR: `$ARTIFACT_DIR` (optional; default: `moe_monokernel_artifacts/<model>_<hardware>_<dtype>_tp<TP>/`)

If any field above is missing or still looks like a literal `$PLACEHOLDER`, **auto-detect** where safe (e.g., GPU via `nvidia-smi`, SM count via CUDA, TP from vLLM config), otherwise **ask** (interactive) or **derive a best-effort default and document the assumption** (non-interactive).

## Canonical outputs (always write these)

Write all artifacts into `ARTIFACT_DIR`:

1. `constraints.md` — model semantics + vLLM code pointers + hardware/dtype constraints
2. `optimization_plan.md` — branching decisions + tiling + kernel structure
3. `validation_results.md` — correctness + perf + profiler links/commands
4. `integration_notes.md` — how to wire into vLLM (or a patch file if requested)
5. `state.json` — resumable state (phase/stage completion)

## Non-interactive safety rules (CI / codex exec)

When running non-interactively:
- Never invoke interactive editors (no `vim`, `nano`, pager prompts, etc.)
- Ensure commands won’t block on auth (set `GIT_TERMINAL_PROMPT=0`)
- Prefer writing patch files and logs into `ARTIFACT_DIR`
- Keep output deterministic; avoid “maybe” steps without verifying

---

# Phase gate: should we even do a monokernel?

Use a quick “saturation score” heuristic:

```
saturation = (BS_decode * TOPK) / SM_count
```

Rule of thumb: if `BS_decode ≤ 64` and `saturation < 0.5`, a monokernel is likely beneficial (decode is under-saturating the GPU, so kernel launch + intermediate memory traffic dominates).

Common SM counts (approx):
- H100/H200: 132 SMs
- L40S: 142 SMs
- A100 80GB: 108 SMs

If `BS_decode` is unknown:
- measure from a representative decode trace, or
- infer a conservative range and state the assumption.

If the gate says “no”, document why stock `fused_moe` is likely sufficient and stop unless explicitly told to proceed.

For more detail: `.codex/moe-monokernel-optimizer/references/algorithmic-branching.md`

---

# Phase 1 — Gather constraints (must be concrete)

### 1A) Locate vLLM MoE implementation for MODEL_ID
- Find the model entrypoint under `vllm/model_executor/models/**`.
- Trace: router logits → top-k → expert dispatch → accumulation.
- Record file paths + key function names + any semantic deltas vs “typical” MoE.

### 1B) Extract config + runtime constraints
Capture:
- Do a web search of the model ID's config.json from huggingface.com to get model details
- `top_k`, number of experts, shared experts (if any), expert parallelism strategy
- Hidden size `K`, intermediate size `N` (per TP shard), activation function
- DType path (fp8/bf16/fp16) and any quant scheme (FP8 dynamic scaling vs blockwise)
- Hardware: GPU name, SM count, shared-mem budget, cooperative launch constraints

**Deliverable:** write `constraints.md`.

If `PHASE == 1`, stop after writing `constraints.md`.

---

# Phase 2 — Optimization plan (branching + tiling + structure)

## 2A) Correctness-sensitive branching

**A. Output accumulation**
- If `TOPK == 1`: **direct write** (no atomics).
- If `TOPK > 1`: **atomic accumulation** (or an equivalent safe reduction).

**B. Weight application order**
- If `TOPK == 1`: weight may be folded early (before activation) **only if** you prove equivalence for this model.
- If `TOPK > 1`: apply weights **after** activation unless proven equivalent.

**C. Token grouping/sorting**
- BS tiny: bitfield/packed experts
- BS normal: histogram/prefix-sum based grouping

References:
- `.codex/moe-monokernel-optimizer/references/expert-grouping.md`
- `.codex/moe-monokernel-optimizer/references/router-design.md`

## 2B) Shared memory “SRAM Tetris” tiling (inline quick solver)

Pick defaults:
- `K_t = 64` (MMA-friendly)
- `M_t = 8` (decode) or `16` (prefill)
- `S_buf = 3` on Hopper (sm_90, TMA-friendly), else `2`

Element sizes:
- `sz_W = 1` for FP8 weights
- `sz_A = 1` for FP8 activations; `2` for BF16/FP16 activations
- accumulator always FP32 (`sz_acc = 4`)

Budget (per SM, typical):
- H100/H200: ~220KB usable (out of 228KB)
- L40S: ~96KB usable (out of 100KB)
- A100: ~156KB usable (out of 164KB)

Solve for `N_t`:

```
SMEM_A   = S_buf * M_t * K_t * sz_A
SMEM_meta≈ 4096

available = SMEM_budget - SMEM_A - SMEM_meta

denom = (S_buf * K_t * sz_W) + (M_t * sz_acc)
N_t = floor(available / denom)
N_t = floor(N_t / 16) * 16   # round down to 16
```

If `N_t < 32`: reduce buffering, reduce `M_t`, or K-chunk (see full solver):
`.codex/moe-monokernel-optimizer/references/tiling-config.md`

Also decide:
- warp role split (good default): 12 warps (384 threads) with 8 compute / 4 prefetch
- sync strategy: named barrier for compute warps only (avoid full `__syncthreads`)

## 2C) Implementation structure (4 stages)

Implement in 4 stages to avoid incomplete MMA loops:

1. `routing_and_prepare`
2. `activation_quantization` (only if dtype path needs it)
3. `gemm_implementation` (**up + down together**)
4. `kernel_assembly` (wiring + cooperative sync + outputs)

**Deliverable:** write `optimization_plan.md`.

If `PHASE == 2`, stop after writing `optimization_plan.md`.

---

# Phase 3 — Implementation

Create a new kernel directory:
- `csrc/moe/moe_monokernel_<model>_<hardware>_<dtype>_tp<TP>/`

Rules:
- No TODOs inside the MMA hot loops when you claim a stage is “done”.
- Prefer templates and scaffolds from `.codex/moe-monokernel-optimizer/references/code-templates.md`.
- Follow controller/worker and cooperative-sync patterns in `.codex/moe-monokernel-optimizer/references/architecture-pattern.md`.
- Use optimization techniques from `.codex/moe-monokernel-optimizer/references/optimization-techniques.md`.

Minimum build/compile verification:
- Build the relevant CUDA extension / vLLM C++ targets for your changes.
- Run at least one smoke test that exercises the MoE path (even if tiny).

If `PHASE == 3`, stop after compiling + writing `implementation_notes.md` into `ARTIFACT_DIR` (what changed, what still missing).

---

# Phase 4 — Validation (correctness + performance)

**Validation README**: `.codex/moe-monokernel-optimizer/validation/README.md`

Correctness:
- Compare monokernel output vs stock vLLM MoE (reference) on representative inputs.
- Report max/mean error; target `max_abs_diff < 1e-2` unless model/dtype demands stricter.

Performance:
- Benchmark across decode batch sizes and top-k settings.
- Benchmark end to end using `vllm bench latency`

Profiling:
- `nsys` for timeline (kernel launch gaps, overlap)
- `ncu` for kernel-level metrics (occupancy, memory, tensor cores)
- `torch.profiler` for end-to-end attribution

**Deliverable:** write `validation_results.md` (include exact commands you ran).

If `PHASE == 4`, stop after writing `validation_results.md`.

---

# Phase 5 — Integration (optional)

Integrate as a fast-path in vLLM:
- Wire build system (CMake) + bindings.
- Add dispatch conditions: match `(model, tp, dtype, hardware, top_k, dims)`.

Deliverable:
- `integration_notes.md`
- patch file `ARTIFACT_DIR/monokernel.patch` (preferred over auto-commits unless asked)

If `PHASE == 5`, stop after writing integration outputs.

---

# State file (always keep current)

Maintain `state.json` in `ARTIFACT_DIR` with:
- `current_phase`
- per-phase status
- per-stage status (Phase 3)
- resolved parameters (`model_id`, `hardware`, `dtype`, `tp`, `top_k`)
- links to output files

Update it after every phase/stage.

---

# Failure handling (inline summary + full doc)

Escalation ladder:
1. **Self-fix** (rerun with better logging, simplify)
2. **Document blocker** (`blockers.md` with command/log + hypotheses)
3. **Retry heuristics** (smaller scope, isolate kernel, reduce variables)
4. **Council protocol** (structured multi-hypothesis analysis)
5. **Post-council retries** (bounded experiments)
6. **Fallback implementation** (ship a safe non-monokernel improvement)

Full detail: `.codex/moe-monokernel-optimizer/orchestration/failure-handling.md`

Minimum blocker format in `ARTIFACT_DIR/blockers.md`:
- what failed
- exact command/log excerpt
- hypotheses
- next experiment
- rollback/safe fallback

---

## First action now

1. Create `ARTIFACT_DIR` (or infer it) and initialize `state.json`.
2. Run Phase 1 and write `constraints.md`.
