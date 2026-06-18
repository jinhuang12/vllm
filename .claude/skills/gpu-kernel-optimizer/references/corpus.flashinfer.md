# Corpus: flashinfer

- **Location:** `assets/corpora/flashinfer/`
- **Upstream:** `flashinfer-ai/flashinfer`
- **Version in this skill:** `0.6.0` (see `assets/corpora/flashinfer/version.txt`)
- **License:** Apache-2.0 (see `assets/corpora/flashinfer/LICENSE`)

FlashInfer is a kernel library + kernel generator focused on **LLM serving / inference** (attention variants, paged KV cache, sampling, quantization, fused operators).

## What this corpus is useful for

- **Attention kernels**: single/batch decode + prefill, paged/ragged KV-cache layouts, sparse attention, GQA optimizations
- **Plan/run staging** patterns for variable-length batching and CUDA-graph friendliness
- **LLM operators**: sampling (top-k/top-p/min-p), topk, RoPE, norm, activation, quant/dequant
- **MoE / routed GEMM** building blocks (DeepSeek-style routing, fused MoE, comm helpers)
- **Benchmark scripts** for many operator families (good repro harnesses)

---

## High-level insights (Design DNA / portable motifs)

Use these when you’re porting *ideas* out of FlashInfer (not just locating a file).

### Insight 1 — Traits → SharedStorage → Mainloop/Epilogue (kernel family structure)
- Why it matters: scales complexity while keeping tiling/staging/layout decisions explicit and reusable.
- Where it lives (evidence):
  - `assets/corpora/flashinfer/include/flashinfer/attention/hopper/kernel_traits.cuh`
  - `assets/corpora/flashinfer/include/flashinfer/attention/hopper/mainloop.cuh`
  - `assets/corpora/flashinfer/include/flashinfer/attention/hopper/epilogue.cuh`
- Core mechanism: compile-time “knobs” + fixed SMEM layout + pipelined loop.
- Porting recipe: define Traits; define SharedStorage; write mainloop/epilogue split; dispatch over a tiny config set.
- Constraints: best payoff for complex kernels (attention, fused epilogues); requires disciplined config management.
- How to validate: compare perf across a small tile/stage grid; confirm register/SMEM tradeoffs don’t regress.

### Insight 2 — Controlled specialization + dispatch (variant management)
- Why it matters: SOTA usually comes from picking the right (arch, head_dim, tile, stages), not one mega-kernel.
- Where it lives (evidence):
  - `assets/corpora/flashinfer/include/flashinfer/attention/variants.cuh`
  - `assets/corpora/flashinfer/include/flashinfer/attention/variant_helper.cuh`
  - JIT/codegen: `assets/corpora/flashinfer/flashinfer/jit/` and `assets/corpora/flashinfer/csrc/*.jinja`
- Core mechanism: curated variant set + dispatch tables + optional codegen/JIT instantiation.
- Porting recipe: keep 5–20 configs; bucket by key params; add guards + fallback.
- Constraints: avoid “autotune everything”; keep the config surface small and motivated.
- How to validate: per-bucket regression plots (seq buckets, head_dim buckets).

### Insight 3 — Memory movement as an API (vector loads + swizzled/permuted SMEM)
- Why it matters: coalescing + fewer transactions + fewer bank conflicts often dominate.
- Where it lives (evidence):
  - `assets/corpora/flashinfer/include/flashinfer/cp_async.cuh`
  - `assets/corpora/flashinfer/include/flashinfer/permuted_smem.cuh`
- Core mechanism: explicit vector width/alignment + SMEM index swizzle/permutation for conflict-free matrix ops.
- Porting recipe: decide vector width early; enforce alignment; choose SMEM layout; verify bank conflicts.
- Constraints: alignment and layout assumptions must be guarded and tested.
- How to validate: memory transactions, shared bank conflict metrics, and end-to-end kernel time.

### Insight 4 — Async pipelines (cp.async → TMA) to overlap memory with compute
- Why it matters: hides HBM latency so tensor cores stay fed.
- Where it lives (evidence):
  - SM90 mainloop: `assets/corpora/flashinfer/include/flashinfer/attention/hopper/mainloop.cuh`
  - SM100 TMA warp-specialized: `assets/corpora/flashinfer/include/flashinfer/attention/blackwell/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`
- Core mechanism: staged producer/consumer pipeline with barriers.
- Porting recipe: implement a multi-stage copy/compute pipeline; keep stages minimal; validate overlap.
- Constraints: arch-specific; requires careful barrier correctness.
- How to validate: reduced “memory dependency” stalls; improved pipe utilization.

### Insight 5 — Warp-group specialization + register rebalancing (Hopper/Blackwell SOTA style)
- Why it matters: splits “load” and “compute” roles and frees register budget where it matters.
- Where it lives (evidence):
  - `assets/corpora/flashinfer/include/flashinfer/attention/hopper/prefill_sm90.cuh`
- Core mechanism: dedicated warp-groups for memory movement vs MMA; explicit reg pressure management.
- Porting recipe: isolate loader roles; reduce loader regs; maximize compute regs; keep synchronization minimal.
- Constraints: mainly SM90+ style kernels; sensitive to occupancy/reg spills.
- How to validate: spills/occupancy metrics + kernel runtime; verify no correctness drift from reordering.

### Insight 6 — Plan/Run staging + tile scheduler for ragged/variable lengths
- Why it matters: variable-length batching is a scheduling problem; avoids load imbalance and improves graphs-friendliness.
- Where it lives (evidence):
  - Scheduler: `assets/corpora/flashinfer/include/flashinfer/attention/scheduler.cuh`
  - Tile sched (SM90): `assets/corpora/flashinfer/include/flashinfer/attention/hopper/tile_scheduler.cuh`
  - Launch sites: `assets/corpora/flashinfer/csrc/batch_decode.cu`, `assets/corpora/flashinfer/csrc/batch_prefill.cu`
- Core mechanism: CPU plan computes work partitioning; GPU run executes balanced tiles.
- Porting recipe: separate plan/run; represent work as tile list; split KV when needed.
- Constraints: needs stable plan caching if you want CUDA graphs.
- How to validate: tail latency (p95/p99) and per-CTA utilization uniformity.

### Insight 7 — Online reductions to avoid global traffic (softmax/sampling/topk)
- Why it matters: avoids multi-pass global reads/writes that dominate latency.
- Where it lives (evidence):
  - `assets/corpora/flashinfer/include/flashinfer/sampling.cuh`
  - `assets/corpora/flashinfer/include/flashinfer/topk.cuh`
- Core mechanism: streaming max/sum; selection without full sort.
- Porting recipe: design reductions to stay on-chip; only write final outputs.
- Constraints: numeric stability must be tested (extreme logits).
- How to validate: correctness under extreme values + perf at large vocab sizes.

### Insight 8 — Built-in micro-profiling hooks for faster iteration
- Why it matters: phase attribution inside a kernel (mainloop vs epilogue vs loads) speeds tuning.
- Where it lives (evidence):
  - `assets/corpora/flashinfer/include/flashinfer/profiler.cuh`
  - `assets/corpora/flashinfer/profiler/`
- Core mechanism: lightweight timestamping / tagging to complement Nsight.
- Porting recipe: add optional compile-time instrumentation; sample, don’t spam.
- Constraints: ensure it’s off by default; don’t perturb perf measurements.
- How to validate: correlate with Nsight; ensure negligible overhead when disabled.

---

## Delta vs existing corpora in this skill (if any)

- flashinfer is currently the primary “pattern library” corpus in this skill:
  - strong coverage for attention + scheduling + sampling + quantization
  - also provides codegen/JIT examples and reproducible microbenches

---

## Where to look first (by task)

### Attention (decode / prefill / paged / ragged / sparse)

- Python API entry points:
  - `assets/corpora/flashinfer/flashinfer/attention.py` (BatchAttention plan/run + backend selection)
  - `assets/corpora/flashinfer/flashinfer/decode.py`, `assets/corpora/flashinfer/flashinfer/prefill.py`
  - `assets/corpora/flashinfer/flashinfer/page.py`, `assets/corpora/flashinfer/flashinfer/cascade.py`, `assets/corpora/flashinfer/flashinfer/sparse.py`
- JIT/codegen:
  - `assets/corpora/flashinfer/flashinfer/jit/` (module generation + build_and_load)
  - `assets/corpora/flashinfer/csrc/*.jinja` (kernel instantiation templates)
- CUDA/C++:
  - `assets/corpora/flashinfer/csrc/batch_decode.cu`, `assets/corpora/flashinfer/csrc/batch_prefill.cu`, `assets/corpora/flashinfer/csrc/batch_attention.cu`
  - `assets/corpora/flashinfer/csrc/fmha_v2/` (fused multi-head attention v2 kernel family)
  - `assets/corpora/flashinfer/csrc/flat/` (newer “flat” kernels; see `flat/prefill/`)

### KV-cache layout / paging

- Docs/tutorials:
  - `assets/corpora/flashinfer/docs/tutorials/kv_layout.rst`
- Implementation:
  - `assets/corpora/flashinfer/flashinfer/page.py`
  - `assets/corpora/flashinfer/csrc/flashinfer_page_binding.cu`

### Sampling / logits processing (top-k / top-p / min-p)

- Python:
  - `assets/corpora/flashinfer/flashinfer/sampling.py`
  - `assets/corpora/flashinfer/flashinfer/topk.py`
  - `assets/corpora/flashinfer/flashinfer/logits_processor/`
- CUDA binding:
  - `assets/corpora/flashinfer/csrc/flashinfer_sampling_binding.cu`
  - `assets/corpora/flashinfer/csrc/flashinfer_topk_binding.cu`
- Bench:
  - `assets/corpora/flashinfer/benchmarks/bench_sampling.py`
  - `assets/corpora/flashinfer/benchmarks/bench_topk.py`

### Quantization / low precision (fp8/int8/fp4)

- Python:
  - `assets/corpora/flashinfer/flashinfer/quantization.py`
  - `assets/corpora/flashinfer/flashinfer/fp8_quantization.py`
  - `assets/corpora/flashinfer/flashinfer/fp4_quantization.py`
- CUDA binding:
  - `assets/corpora/flashinfer/csrc/flashinfer_quantization_binding.cu`
- Bench:
  - `assets/corpora/flashinfer/benchmarks/bench_fp8_prefill.py`
  - `assets/corpora/flashinfer/benchmarks/bench_rope_quantize_fp8.py`

### RoPE / positional encoding

- Python:
  - `assets/corpora/flashinfer/flashinfer/rope.py`
- CUDA binding:
  - `assets/corpora/flashinfer/csrc/flashinfer_rope_binding.cu`
- Bench:
  - `assets/corpora/flashinfer/benchmarks/bench_rope.py`

### Norm / activation / fused epilogues

- Python:
  - `assets/corpora/flashinfer/flashinfer/norm.py`
  - `assets/corpora/flashinfer/flashinfer/activation.py`
  - `assets/corpora/flashinfer/flashinfer/cute_dsl/`
- CUDA binding:
  - `assets/corpora/flashinfer/csrc/flashinfer_norm_binding.cu`
- Bench:
  - `assets/corpora/flashinfer/benchmarks/bench_rmsnorm.py`

### MoE / routed GEMM / all-to-all

- Python:
  - `assets/corpora/flashinfer/flashinfer/fused_moe/`
  - `assets/corpora/flashinfer/flashinfer/gemm/`
  - `assets/corpora/flashinfer/flashinfer/comm/`
- CUDA/C++:
  - `assets/corpora/flashinfer/csrc/dsv3_router_gemm.cu`
  - `assets/corpora/flashinfer/csrc/flashinfer_gemm_binding.cu`
- Bench:
  - `assets/corpora/flashinfer/benchmarks/bench_cutlass_fused_moe.py`

### Benchmarks / repro harnesses

Start at:
- `assets/corpora/flashinfer/benchmarks/README.md`
- `assets/corpora/flashinfer/benchmarks/flashinfer_benchmark.py`
- `assets/corpora/flashinfer/benchmarks/test_flashinfer_benchmark.py`

---

## Trace recipes (API → binding → kernel)

Run these from the skill folder (or prefix `.codex/skills/gpu-kernel-optimizer/` from repo root).

### Trace 1 — Batch attention: python → csrc launcher → include mainloop
- Start: `assets/corpora/flashinfer/flashinfer/attention.py`
- Find the C++ binding call-sites:
  - `grep -R -n "batch_(decode|prefill|attention)" assets/corpora/flashinfer/flashinfer/attention.py`
- Jump to launch sites:
  - `grep -R -n "batch_(decode|prefill|attention)" assets/corpora/flashinfer/csrc | head`
- Study the “SOTA” path:
  - `assets/corpora/flashinfer/include/flashinfer/attention/hopper/*` (SM90)
  - `assets/corpora/flashinfer/include/flashinfer/attention/blackwell/*` (SM100)

### Trace 2 — Sampling: python → binding → include/sampling.cuh
- Start: `assets/corpora/flashinfer/flashinfer/sampling.py`
- Find binding:
  - `grep -R -n "sampling" assets/corpora/flashinfer/csrc/flashinfer_sampling_binding.cu | head`
- Study kernel primitives:
  - `assets/corpora/flashinfer/include/flashinfer/sampling.cuh`

### Trace 3 — TopK: python → binding → include/topk.cuh
- Start: `assets/corpora/flashinfer/flashinfer/topk.py`
- Binding:
  - `assets/corpora/flashinfer/csrc/flashinfer_topk_binding.cu`
- Kernel:
  - `assets/corpora/flashinfer/include/flashinfer/topk.cuh`

---

## Directory map (top-level)

| Path | What’s inside | Why you’d read it |
|---|---|---|
| `flashinfer/` | Python API + dispatch + JIT module builders | quickest trace: API → binding → kernel |
| `csrc/` | CUDA/C++ kernels, bindings, template instantiations (`.jinja`) | where perf work often lands |
| `include/flashinfer/` | header-only/C++ primitives | kernel motifs, tiling, pipelines |
| `docs/` | Sphinx docs (`.rst`) | conceptual docs + API surface |
| `benchmarks/` | microbench scripts | fastest repro harnesses |
| `tests/` | correctness tests | validate changes and edge cases |
| `profiler/` | profiling utilities | kernel-phase attribution |

## Grep recipes

### Trace python API → binding → kernel

- Find the python wrapper:
  - `grep -R -n "def single_decode_with_kv_cache" assets/corpora/flashinfer/flashinfer | head`
- Find the C++ binding symbol:
  - `grep -R -n "single_decode_with_kv_cache" assets/corpora/flashinfer/csrc | head`
- Find batch decode kernel entry points:
  - `grep -R -n "batch_decode" assets/corpora/flashinfer/csrc | head`

### Find paged attention paths

- `grep -R -n "paged" assets/corpora/flashinfer/flashinfer | head`
- `grep -R -n "PagedKV" assets/corpora/flashinfer | head`

### Find JIT codegen sources

- `grep -R -n "build_and_load" assets/corpora/flashinfer/flashinfer/jit | head`
- `find assets/corpora/flashinfer/csrc -name "*.jinja" | head`

## Notes & gotchas

- FlashInfer frequently uses **plan/run staging** (plan on CPU, run on GPU) to handle variable-length batching and to stay CUDA-graph friendly.
- Many kernels are **generated/instantiated** via `.jinja` templates and JIT build logic under `flashinfer/jit/`.
- When porting ideas, prioritize: layout (paged KV), scheduling (tile scheduler), memory movement (vector loads + swizzled SMEM), and fusion boundaries (epilogues).
