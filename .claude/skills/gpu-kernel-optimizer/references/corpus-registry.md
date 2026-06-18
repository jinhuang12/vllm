# Kernel Corpus Registry

Start here when you need a reference implementation or proven kernel patterns.

**Path convention:** paths in this registry are written **relative to the skill folder**.
- If you are operating from your repo root, prefix with: `.codex/skills/gpu-kernel-optimizer/`

**How to use this registry**
1. Write the kernel “problem signature” (op + shapes + dtype + layout + runtime constraints).
2. Pick 1–2 corpora from the table.
3. Open the corpus index file and follow its “Where to look first” section.

## Corpora (auto-maintained)

<!-- CORPUS_REGISTRY:START -->
| Corpus | Location | Index | Best for | Notes |
|---|---|---|---|---|
| flashinfer | `assets/corpora/flashinfer/` | `references/corpus.flashinfer.md` | Attention (dense/paged), sampling, quantization, fused MoE, RoPE, norm | plan/run staging; JIT codegen + optional prebuilt cubins |
| sglang | `assets/corpora/sglang/` | `references/corpus.sglang.md` | Serving runtime integration, attention backends, custom all-reduce, JIT kernels, Triton fallbacks | wraps/reuses FlashInfer; strong CUDA-graph workspace + fallback patterns |
<!-- CORPUS_REGISTRY:END -->

## Routing hints (quick heuristics)

- If the task mentions **paged KV cache**, **batch decode**, **cascade attention**, or **cudagraph-friendly attention** → start with **flashinfer**.
- If you need examples of **CUDA-graph capture/replay**, **workspace planning**, or **custom all-reduce integration** → start with **sglang**.
- If you need a **minimal microbenchmark harness** → check the corpus’s `benchmarks/` and `tests/` (the index file lists the best entry points).
