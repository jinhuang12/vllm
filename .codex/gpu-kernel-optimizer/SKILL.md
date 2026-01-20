---
name: gpu-kernel-optimizer
description: Optimize GPU kernels for LLM inference using bundled kernel corpora (e.g., FlashInfer) + profiling-driven workflow. Use when asked to speed up CUDA/Triton/CUTLASS kernels (attention, KV cache, sampling, quantization, MoE/FFN), reduce launch/memory overhead, or integrate faster kernels into inference runtimes (vLLM/TRT-LLM/PyTorch).
---

# GPU Kernel Optimizer

Optimize GPU kernels used in LLM inference by combining:
- **Profiling-driven iteration** (baseline → bottleneck → change → re-measure)
- **Retrieval from bundled kernel corpora** under `assets/corpora/` (FlashInfer included)
- **Stage-gated execution** to avoid correctness regressions

This skill is designed so Codex can stay efficient: **start from indexes**, then open only the specific files/functions you need.

## Operating rules

1. **Never claim a speedup without numbers.** Always provide: workload shape(s), GPU, dtype, baseline vs optimized, and the exact benchmark command.
2. **Match production settings.** If the target runtime uses CUDA graphs / torch.compile / persistent buffers, measure under the same conditions.
3. **Prefer “small, reversible” diffs.** Keep changes isolated and guard with feature flags when integrating into a larger runtime.
4. **Use corpora as patterns, not gospel.** Borrow proven layouts/scheduling ideas, but re-validate assumptions for the target codebase.

## Kernel “problem signature” (write this down first)

Before touching code, write a 5–10 line signature:

- **Operation:** (attention decode/prefill/paged, GEMM/FFN, MoE routing, sampling/top-k/top-p, RoPE, norm, quant/dequant, KV-cache ops, etc.)
- **Shapes:** batch, seq_len, heads, head_dim, hidden, top_k, etc.
- **Dtype / quant:** fp16/bf16/fp8/int8/fp4, scaling scheme, accumulation type
- **Layout:** QKV layout, KV-cache layout (paged vs ragged), strides/alignment
- **Runtime constraints:** CUDA graphs? torch.compile? multi-GPU comm? max SMEM / registers / launch bounds
- **Correctness tolerance:** exact vs approximate, acceptable max error / ULP

## Efficient corpus querying (how to use the RAG corpora)

1. Open **`references/corpus-registry.md`** and pick 1–2 corpora that match the problem signature.
2. Open the chosen corpus index file(s) (for example: **`references/corpus.flashinfer.md`**).
3. Use the index’s **“Where to look first”** + **“Grep recipes”** to jump to the exact implementation.
4. Only then open code under `assets/corpora/<corpus>/...` (avoid “read the whole repo”).

See also: `references/query-playbook.md`.

## Optimization loop

Follow the detailed checklist in `references/optimization-workflow.md`. The high-level stages:

1. **Baseline & isolate**
   - Reproduce the slowdown with a minimal benchmark.
   - Use a profiler to confirm the kernel(s) and whether you are launch-bound or memory/compute-bound.

2. **Find reference kernels**
   - Use the corpus registry + indexes to find similar kernels (same op + similar layout/dtype).
   - Extract 2–3 concrete ideas (tiling, fusion boundary, epilogue strategy, memory layout, scheduling).

3. **Plan the change**
   - Write a micro-plan (3–7 steps) and explicit kill criteria (what would make you revert).

4. **Implement + benchmark**
   - Keep diffs small; test frequently.
   - Measure both microbenchmarks and end-to-end inference impact if possible.

5. **Validate correctness**
   - Add/extend unit tests or a correctness harness.
   - Verify numerics across representative shapes and edge cases.

6. **Integrate safely**
   - Add dispatch guards (arch/dtype/shape).
   - Document how to reproduce the benchmark and how to roll back.

## Adding new corpora

Use the separate skill **`kernel-optimization-indexer`** to import a repo/zip and generate:
- `assets/corpora/<new-corpus>/...`
- `references/corpus.<new-corpus>.md`
- an updated `references/corpus-registry.md`
