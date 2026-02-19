---
name: moe-monokernel-optimizer
description: Optimize vLLM Mixture-of-Experts (MoE) inference (fused_moe/FusedMoE) by designing/fusing custom CUDA kernels (cooperative monokernel, hybrid large-grid fusion, or split-kernel) and validating speedups under production settings (CUDA graphs, torch.compile). Use when asked to beat vLLM’s MoE baseline for a specific model+GPU+dtype+TP/EP, or to decide routing/semantics, fusion boundaries, tiling/ownership, and fast-path dispatch.
---

# MoE Monokernel Optimizer

Design MoE kernel-fusion optimizations for **vLLM inference** that beat the **production-parity baseline** (CUDA graphs / torch.compile), without regressing correctness.

## Search anchors

production parity, CUDA graphs, torch.compile, fused_moe, FusedMoE, router logits, top_k, renorm, TP, EP, token-major, expert-major, shared experts, cooperative_groups, SM80, SM90, FP8, BF16, nsys, ncu

## Workflow **CRITICAL**

- You **MUST** follow the stage-gated checklist in `orchestration/task-guide.md` (canonical).
- Use `rg`/grep with the Search anchors above to locate details in references without reading everything.
- Track resumable progress in `{artifact_dir}/state.json` (recommended).
- If Phase 4 fails, use `orchestration/investigation-prompts.md` (do not thrash).
- Use `orchestration/llm-council.md` to decide when to invoke the separate `llm-council` skill.

## Non‑negotiables

1. **Measure the right baseline (production parity)**
   - Benchmark **combined routing + experts** under the same settings as production (CUDA graphs, torch.compile, TP/EP, batch bucketing).
   - Under CUDA graphs, “fewer kernels” is rarely enough; require **GPU kernel-time** savings (or eliminated DRAM round‑trips).
   - Use: `references/profiling-launch-vs-kernel.md`, `references/e2e-delta-math.md`.

2. **Do not assume model semantics or baseline fusion facts**
   - Verify routing math and accumulation behavior by reading the model’s vLLM implementation (`vllm/model_executor/models/...`).
   - Record **(a) semantics + topology** and **(b) baseline “truth snapshot”** in `{artifact_dir}/constraints.md` using `references/moe-parameters-template.md`.
   - Implement *exactly*: scoring (softmax vs sigmoid), renorm, scaling factors, tie-breaks, shared experts, accumulation/reduction rules.

3. **Pick the fusion boundary deliberately**
   - Use `references/route-selection-decision-tree.md` to choose:
     - **A)** Cooperative monokernel
     - **B)** Hybrid large-grid fusion (fuse around baseline GEMMs)
     - **C)** Split-kernel graph-captured route
   - Do not default to “single mega-kernel” if Phase 1 profiling shows the win can’t amortize barriers/SMEM/regs.

4. **CUDA graphs safety is required**
   - Custom ops must be graph-safe: correct stream, no hidden allocations, stable shapes within buckets.
   - Use: `references/cudagraph-safety.md`.

5. **No ungrounded performance claims**
   - Never claim a win without measurement on the target bucket(s) with CUDA graphs enabled.
   - If a change is “plausibly faster” but not measured, label it explicitly as a hypothesis and schedule measurement.

6. **Fast-path enablement must be bounded**
   - Guard the monokernel by the exact envelope you validated (model id, dtype, TP/EP, shapes, batch buckets).
   - Preserve a correct fallback outside the envelope.

7. **Do not skip full-model E2E because “weights aren’t available”**
   - If you need E2E to compute MoE share `f` (Phase 1) or to validate speedup (Phase 4.3), and the model isn’t cached locally: **download the weights**.
   - Only skip E2E if the **user explicitly waives** the E2E requirement (or the model is gated and the user cannot/does not want to provide access).

## Deterministic helper scripts (optional)

These are **measurement + reporting plumbing**. They are designed to be safe defaults and to fail-fast when the target is underspecified.

- `scripts/new_target.py`: scaffold `{artifact_dir}` + `state.json` + `target.json`.
- `scripts/collect_env.py`: capture `env.json`/`env.md` for reproducible reporting.
- `scripts/run_vllm_bench_latency_sweep.py`: run baseline vs optimized **batch-size sweep** via `vllm bench latency` (loads the model once per label by default; archives existing output dir automatically; emits `logs/` + `status/` heartbeats; supports per-label flags via `bench.{baseline,opt}_extra_args`; use `--out-name ...` to avoid collisions, and `--execution-mode cli_per_bs` for legacy per-batch-size CLI runs).
- `scripts/generate_validation_report.py`: generate `{artifact_dir}/validation_results.md` from recorded evidence (no guessing).

## Canonical references (read as needed)

- **Route + ownership decisions**: `references/route-selection-decision-tree.md`
- **Triage math + stop conditions**: `references/e2e-delta-math.md`, `references/fusion-feasibility-heuristics.md`
- **Optimization plan template (Phase 2)**: `references/optimization-plan-template.md`
- **Routing implementation**: `references/router-design.md`
- **Tiling + SRAM constraints**: `references/tiling-config.md`, `references/gpu-configs.md`
- **Architecture patterns (post-route)**: `references/architecture-pattern.md`, `references/algorithmic-branching.md`
- **Code patterns**: `references/code-templates.md`
- **Graph safety**: `references/cudagraph-safety.md`
- **Validation defaults**: `references/validation-defaults.md`, `validation/E2E_LATENCY_GUIDE.md`

## Additional references and examples (optional)

Open these only when you need deeper detail or concrete examples:

- **Technique catalog**: `references/optimization-techniques.md`
- **Expert grouping/scheduling**: `references/expert-grouping.md`
- **Scope + validated examples**: `references/scope-and-support.md`
- **Model-specific baselines (examples)**: `validation/QWEN3_BASELINE.md`
- **Llama4 monokernel deep dive**: `assets/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md`
- **Reference implementation patch (top_k=1)**: `assets/LLAMA4_MONOKERNEL_PATCH.md`
- **Worked examples**:
  - `examples/MODELS_COMPARISON.md`
  - `examples/W1_EPILOGUE_FUSION.md`

## Required outputs (per target)

Write these artifacts (minimum):

- `{artifact_dir}/constraints.md` (Phase 1)
- `{artifact_dir}/optimization_plan.md` (Phase 2)
- `{artifact_dir}/implementation_notes.md` (Phase 3)
- `{artifact_dir}/validation_results.md` (Phase 4)
- `{artifact_dir}/integration.md` (Phase 5)
- `{artifact_dir}/state.json` (recommended)

## Example prompts that should trigger this skill

- “Beat vLLM MoE latency on H100 for {model} FP8 decode BS 1–64 under CUDA graphs.”
- “Decide cooperative monokernel vs hybrid large-grid fusion for {model} top_k={k} on {gpu}.”
- “My custom MoE kernel regressed under torch.compile/CUDA graphs — help investigate and fix.”
