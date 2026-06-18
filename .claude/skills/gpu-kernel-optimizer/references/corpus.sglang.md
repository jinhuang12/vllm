# Corpus: sglang

SGLang is a high-performance serving stack that also ships a **kernel library** (`sgl-kernel`) plus a set of **Triton kernels** and **runtime integration patterns** (backend selection, CUDA-graph support, workspaces, fallbacks).

This corpus is curated from `sglang-main.zip`.

## What this corpus is useful for

- **End-to-end integration patterns**: “serving runtime → Python wrapper → torch extension → CUDA kernel”
- **Kernel API design**: a single op namespace (`torch.ops.sgl_kernel.*`) routing many inference operators
- **Attention backends**:
  - FlashAttention wrapper + feature gating
  - Cutlass MLA decode backend with workspace planning + CUDA-graph replay
  - Triton fallbacks for unsupported shapes
- **Communication primitives**: custom all-reduce + graph-friendly buffer registration
- **JIT kernel loading** patterns (template-parameterized CUDA headers compiled on demand)
- **Triton feature-detection** patterns for version skew (e.g., gather / TMA descriptors)

## Design DNA (portable motifs)

Use these motifs when you’re trying to port ideas into your own kernels / runtime.

| Motif | What to steal | Where to study |
|---|---|---|
| Unified op surface via `TORCH_LIBRARY_FRAGMENT` | Stable operator names + per-backend impl registration (`torch::kCUDA`, etc.) | `assets/corpora/sglang/sgl-kernel/csrc/common_extension.cc` |
| “Wrap, don’t rewrite” (reuse FlashInfer kernels) | Thin adapters around proven kernels (with SGLang’s API + guards) | `assets/corpora/sglang/sgl-kernel/csrc/attention/cascade.cu` (MergeState), `assets/corpora/sglang/sgl-kernel/csrc/elementwise/activation.cu`, `assets/corpora/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu` |
| Backend fallback rules (CUDA → Triton) | Guardrails for dtype/head-dim/alignment + safe fallback path | `assets/corpora/sglang/python/sglang/srt/layers/attention/merge_state.py` |
| Workspace planning + CUDA-graph friendliness | Pre-allocate workspaces and “replay-safe” buffers; separate init/capture/replay | `assets/corpora/sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py` |
| CUTLASS-based SM100 MLA decode wrapper | Template-driven CUTLASS FMHA invocation + scheduler selection + CUDA version gating | `assets/corpora/sglang/sgl-kernel/csrc/attention/cutlass_mla_kernel.cu` |
| Custom all-reduce with IPC + graph buffers | IPC handle exchange, registered buffers, graph capture support | `assets/corpora/sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cu`, `assets/corpora/sglang/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py` |
| “Small JIT kernel factory” | Build tiny CUDA kernels from templated headers; cache modules in a torch.compile-safe way | `assets/corpora/sglang/python/sglang/jit_kernel/utils.py`, `assets/corpora/sglang/python/sglang/jit_kernel/norm.py` |
| Triton version compatibility via feature detection | `hasattr(...)` gates + fallback stubs to keep compiler happy | `assets/corpora/sglang/python/sglang/srt/layers/attention/fla/op.py` |

## Where to look first (by task)

### Attention: decode/prefill backends + metadata

- Runtime backend code (how the serving stack calls kernels):
  - `assets/corpora/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`
  - `assets/corpora/sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py`
  - `assets/corpora/sglang/python/sglang/srt/layers/attention/merge_state.py` (CUDA→Triton fallback)
- Kernel library wrappers:
  - `assets/corpora/sglang/sgl-kernel/python/sgl_kernel/flash_attn.py`
  - `assets/corpora/sglang/sgl-kernel/python/sgl_kernel/flash_mla.py`
- CUDA kernels:
  - `assets/corpora/sglang/sgl-kernel/csrc/attention/`

### Communication / all-reduce

- CUDA custom all-reduce implementation:
  - `assets/corpora/sglang/sgl-kernel/csrc/allreduce/custom_all_reduce.cu`
  - `assets/corpora/sglang/sgl-kernel/csrc/allreduce/mscclpp_allreduce.cu`
- Python integration (device communicators):
  - `assets/corpora/sglang/python/sglang/srt/distributed/device_communicators/`
- Microbench harness:
  - `assets/corpora/sglang/benchmark/kernels/all_reduce/`

### Norm / activation / small fusions

- CUDA-side wrappers (often call into FlashInfer headers):
  - `assets/corpora/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu`
  - `assets/corpora/sglang/sgl-kernel/csrc/elementwise/activation.cu`
- JIT norm kernel example (template header compiled on-demand):
  - `assets/corpora/sglang/python/sglang/jit_kernel/norm.py`
  - `assets/corpora/sglang/python/sglang/jit_kernel/csrc/elementwise/qknorm.cuh`

### Quantization / low-precision GEMM

- `assets/corpora/sglang/sgl-kernel/csrc/gemm/`
- `assets/corpora/sglang/sgl-kernel/csrc/quantization/`
- `assets/corpora/sglang/python/sglang/srt/layers/quantization/` (how kernels are invoked from the runtime)
- `assets/corpora/sglang/benchmark/kernels/quantization/`

### Triton kernels (esp. attention variants)

- Flash Linear Attention (FLA) ops:
  - `assets/corpora/sglang/python/sglang/srt/layers/attention/fla/`
- Triton fallbacks and utilities:
  - `assets/corpora/sglang/python/sglang/srt/layers/attention/triton_ops/`

## Trace recipes (API → binding → kernel)

These are the fastest “follow the wires” traces when you want to port/optimize.

### Trace 1 — MergeState with CUDA→Triton fallback

1. Runtime decision:
   - `assets/corpora/sglang/python/sglang/srt/layers/attention/merge_state.py` → `merge_state(...)`
2. CUDA fast path:
   - calls `sgl_kernel.merge_state_v2(...)`
3. Torch binding:
   - `assets/corpora/sglang/sgl-kernel/csrc/common_extension.cc` (`m.def("merge_state_v2(...)")`)
4. CUDA wrapper + underlying algorithm:
   - `assets/corpora/sglang/sgl-kernel/csrc/attention/cascade.cu` → `flashinfer::MergeState(...)`

### Trace 2 — CUTLASS MLA decode (SM100)

1. Runtime backend:
   - `assets/corpora/sglang/python/sglang/srt/layers/attention/cutlass_mla_backend.py` → `CutlassMLABackend.forward_decode(...)`
2. Workspace planning / CUDA-graph safe buffers:
   - `cutlass_mla_get_workspace_size(...)`, `init_cuda_graph_state(...)`
3. Torch binding:
   - `assets/corpora/sglang/sgl-kernel/csrc/common_extension.cc` (`m.def("cutlass_mla_decode(...)")`)
4. CUTLASS kernel wrapper:
   - `assets/corpora/sglang/sgl-kernel/csrc/attention/cutlass_mla_kernel.cu`

### Trace 3 — Fused Add + RMSNorm (FlashInfer-backed)

1. Torch op binding:
   - `assets/corpora/sglang/sgl-kernel/csrc/common_extension.cc` (`m.def("fused_add_rmsnorm(...)")`)
2. CUDA wrapper:
   - `assets/corpora/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu`
3. Underlying kernel implementation:
   - uses `flashinfer::norm::FusedAddRMSNorm(...)` (see FlashInfer corpus for internals)

### Trace 4 — JIT QK-Norm (templated header compiled on demand)

1. Python entry:
   - `assets/corpora/sglang/python/sglang/jit_kernel/norm.py` → `fused_inplace_qknorm(...)`
2. JIT loader:
   - `assets/corpora/sglang/python/sglang/jit_kernel/utils.py` → `load_jit(...)` (+ caching)
3. CUDA header:
   - `assets/corpora/sglang/python/sglang/jit_kernel/csrc/elementwise/qknorm.cuh` (`QKNormKernel<...>::run`)

## Benchmarks and tests

- Kernel microbench harnesses:
  - `assets/corpora/sglang/benchmark/kernels/`
    - All-reduce: `all_reduce/`
    - Attention (Triton): `decoding_attention_triton/`, `sliding_window_attention_triton/`
    - Quantization: `quantization/`
- Kernel library tests:
  - `assets/corpora/sglang/sgl-kernel/tests/`
  - JIT-kernel tests:
    - `assets/corpora/sglang/python/sglang/jit_kernel/tests/`

## Grep recipes (fast navigation)

- Find the torch op registration for an operator:
  - `grep -n "m.def(\"cutlass_mla_decode" -n assets/corpora/sglang/sgl-kernel/csrc/common_extension.cc`
- Find where FlashInfer headers are reused:
  - `grep -R -n "#include <flashinfer/" assets/corpora/sglang/sgl-kernel/csrc | head`
- Find where the runtime selects the attention backend:
  - `grep -R -n "class .*Backend" assets/corpora/sglang/python/sglang/srt/layers/attention | head`
- Find JIT entrypoints:
  - `grep -R -n "load_jit(" assets/corpora/sglang/python/sglang/jit_kernel | head`
- Find where `enable_pdl` is threaded through:
  - `grep -R -n "enable_pdl" assets/corpora/sglang | head`

## Overlap note vs existing corpora

- **Most similar to:** `flashinfer` (SGLang frequently *wraps* FlashInfer kernels or ports their structure).
- **New value add:** end-to-end serving integration patterns (backend selection, CUDA-graph workflows, workspaces), plus additional kernel families (custom all-reduce, CUTLASS MLA decode, JIT kernel factory).
- **What we intentionally did not import:** the full SGLang serving stack (HTTP/engine/model code) and non-kernel benchmarks; this corpus is focused on kernel + integration “surface area”.
