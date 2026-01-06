# Nsight Systems playbook for vLLM fusion mining

## Table of contents
1. Goals and non-goals
2. Capture modes (full-run vs delimited)
3. Recommended flags and trace hygiene
4. Exporting CSV reports
5. What each report answers
6. Multi-process / multi-GPU tips
7. Attribution workflow (kernel → code)
8. Follow-up with Nsight Compute (when and why)

## 1) Goals and non-goals

Goal:
- Identify where GPU time goes and which kernel sequences repeat in steady-state vLLM inference.

Non-goals:
- Treat Nsight Systems timing as a microbenchmark oracle. Use it to find candidates and validate “did time disappear?”
- Use Nsight Compute (NCU) for “why is this kernel slow?” investigations.

## 2) Capture modes

### A) Full-run capture (robust default)
- Works without modifying the workload.
- Keep the workload short (few iterations) so the trace is small and analyzable.

### B) Delimited capture (best signal; needs delimiters)
- Use `--capture-range=cudaProfilerApi` and have the workload call `cudaProfilerStart/Stop`, or
- Use `--capture-range=nvtx` and annotate the region.

Use delimited capture when you can do it cleanly without changing semantics.

## 3) Recommended flags and trace hygiene

General guidance:
- Trace CUDA + NVTX.
- Disable CPU sampling unless you have a CPU-side question.
- Keep traces short and comparable across baseline vs variants.

CUDA graphs + `torch.compile`:
- Prefer profiling steady-state replay, not compilation or initial graph capture.
- If you must profile compile/capture, tag it as a separate run.

## 4) Export CSV reports (minimum useful set)

From a `.nsys-rep`, export:
- `cuda_gpu_kern_sum.csv` — per-kernel total GPU time summary
- `cuda_gpu_trace.csv` — chronological kernel list (for chain mining)
- `nvtx_sum.csv` (if NVTX exists) — attribution by NVTX ranges

## 5) What each report answers

- `cuda_gpu_kern_sum.csv`:
  - “What kernels dominate total GPU time?”
  - “Is the hot path heavy-kernel dominated (GEMM/attention) or micro-kernel dominated?”

- `cuda_gpu_trace.csv`:
  - “What kernel sequences repeat?”
  - “Where is micro-kernel soup between heavy kernels?”

- `nvtx_sum.csv`:
  - “Which model component / stage owns the time?”

## 6) Multi-process / multi-GPU tips

- Start on single-GPU if possible.
- If running TP/PP, trace one rank first (smaller/faster iteration), then confirm representativeness across ranks.

## 7) Attribution workflow (kernel → code)

Attribution is often the hardest part.

Recommended approach:
1) Start with NVTX:
   - If vLLM already emits NVTX, use it.
   - If not, add temporary NVTX around coarse modules: attention / MLP / KV-cache update / sampling / MoE.
2) Use kernel names as search hints:
   - Triton kernels often retain readable names.
   - cuBLAS/CUTLASS kernels are long; rely more on adjacency + NVTX.
3) Re-trace after adding NVTX to confirm the mapping.

## 8) Nsight Compute follow-up (when and why)

Use NCU when:
- A fused kernel regresses.
- You need to inspect spills/occupancy/memory throughput/tensor core utilization.

