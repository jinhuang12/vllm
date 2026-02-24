# Production parity requirements (vLLM fusion mining)

## Table of contents
1. Why parity matters
2. CUDA graphs parity checklist
3. `torch.compile` parity checklist
4. Warmup / steady-state discipline
5. Apples-to-apples rules

## 1) Why parity matters

Fusion opportunities found in eager mode can disappear (or invert) under:
- CUDA graphs replay
- `torch.compile` codegen changes
- allocator + workspace reuse patterns

Mining should reflect the real deployment mode.

## 2) CUDA graphs parity checklist

Record (do not guess):
- whether CUDA graphs are enabled
- what shapes/buckets are captured (confirm from vLLM config/code)
- whether there are graph breaks/fallbacks

Rules:
- Warm up long enough to populate any graph caches.
- Do not compare “captured” vs “uncaptured” runs as if they’re equivalent.

## 3) `torch.compile` parity checklist

Record (do not guess):
- whether `torch.compile` is enabled by vLLM
- compile mode/flags/caches (confirm from vLLM config/code)
- whether compilation happens inside the profiled window (avoid if possible)

Rules:
- Compile first, then profile steady-state.
- Treat compile-time profiling as a separate run/tag.

## 4) Warmup / steady-state discipline

For each run:
- warm up before measurement/profiling
- keep measurement iterations consistent across baseline and variants

## 5) Apples-to-apples rules

- Same model + quantization + parallelism
- Same input shapes and batching policy
- Same graphs/compile state
- Same env vars affecting kernels/allocators
- Prefer p50/p95 over single-shot timing

