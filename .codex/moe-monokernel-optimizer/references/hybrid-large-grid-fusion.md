# Hybrid Large‑Grid Fusion (Playbook)

Use this when the baseline expert GEMM(s) are already strong (high occupancy / good concurrency), and cooperative monokernel loses due to `grid.sync`, large dynamic shared memory, or 1‑CTA/SM bottlenecks.

## Core idea

Keep the baseline **large‑grid expert GEMM kernel(s)** (often Triton `fused_moe_kernel`) and fuse *around* them:
- Replace/optimize **routing + prepare** with a small CUDA kernel (or fuse two small kernels into one).
- Fuse **W1 epilogue** stages into the W1 GEMM kernel (activation + quantization for W2 input).
- Keep W2 GEMM as baseline unless you can fuse a safe epilogue without atomics.

This avoids the “single cooperative grid” trap: low effective occupancy + global barriers can dominate even when you reduce memory traffic.

## Phase 1: Identify what is already fused (don’t double‑count)

Before implementing any “fusion”, confirm (by code + profiling) which stages are already fused in your baseline for your config:
- Routing: is `grouped_topk` already a fused CUDA op for this model/config, or is it Python fallback?
- Router‑weight multiply: does the expert GEMM support routed‑weight folding (flagged epilogue)?
- Activation: is the activation (e.g., SiLU/SwiGLU) separate from W1?
- Quantize: is W2 input quantization separate from W1?
- Reduce: is `moe_sum` separate from W2?

If a stage is already fused, it is not an opportunity.

## High‑impact opportunities (typical)

### 1) Fuse W1 → activation → W2‑input quantization (single write)

Baseline often does:
1. W1 GEMM writes `[M*topk, 2*N]` to global
2. Activation (SiLU/SwiGLU) kernel reads + writes `[M*topk, N]`
3. Quant kernel reads `[M*topk, N]` and writes FP8 `[M*topk, N]` (+ scales)

In a hybrid design, W1 GEMM kernel:
- Computes the W1 tile
- Applies activation in registers
- Quantizes directly to FP8 W2‑input buffer
- Writes only the final FP8 activations (+ scale metadata)

This removes at least one full global‑memory round‑trip for large intermediates and deletes 1–2 kernels.

### 2) Fuse routing + prepare when E is small and total_pairs is small

When `E_local` is moderate (e.g., 128) and `total_pairs = BS * top_k` is small (e.g., ≤512):
- A single CTA can do deterministic **count → scan → scatter** in shared memory.
- Avoid atomics; prefer stable ordering (pair index order).

Routing top‑k itself can be improved, but only pursue it if routing is a measurable share under CUDA graphs.

### 3) Avoid W2+reduce fusion unless you accept atomics (usually not)

`moe_sum` often requires reducing multiple expert contributions into a single output token vector.
- If ownership implies overlap, fusing into W2 typically needs atomics or a major remap.
- Treat this as “later”; measure first.

## Profiling gates (don’t guess)

### CUDA graphs baseline (required)

Measure the *combined* MoE graph (routing + experts) under CUDA graphs. Then compute:
- How much time is in GEMMs vs non‑GEMM stages
- Whether your target fusion affects kernel execution time or only launch/API time

### Nsight Systems: API vs kernel time

Use Nsight Systems to distinguish:
- CUDA API time (e.g., `cudaLaunchKernel`, `cudaGraphLaunch`)
- GPU kernel time (actual durations)

This prevents over‑estimating “launch overhead savings” when graphs are enabled.

### Nsight Compute: verify you didn’t break the GEMM

After epilogue fusion, use NCU to check:
- Achieved occupancy and warp stalls
- Tensor Core utilization (if applicable)
- Register pressure / spills

## Integration pattern (recommended)

- Guard each optimization with a model+hw+dtype+tp specific env var (fast rollback).
- Keep a correctness test for the model MoE op (compare vs baseline).
- Add a microbenchmark script that prints per‑stage time for: routing, prepare, W1 GEMM, activation/quant, W2 GEMM, reduce.

## “Tuning-only” is not a hybrid outcome (unless proven)

Kernel tuning / config generation (e.g., Triton MoE config files) is often a **baseline normalization** step that should happen early if the baseline is falling back to defaults.

If you choose the **hybrid large-grid** route, you must do one of:
- Implement at least one fusion that removes real GPU kernel work around the GEMM(s) (e.g., W1 epilogue fusion or routing+prepare fusion), **or**
- Document a “no fusion opportunities” proof with evidence:
  - Nsight Systems per-kernel breakdown showing the suspected stages are already fused or negligible under CUDA graphs, and
  - a clear kill criterion that justifies stopping at tuning/config improvements.

## When to stop pursuing full monokernel

If all are true, hybrid is usually the right finish line:
- Baseline expert GEMM(s) dominate and already scale with large grids
- Cooperative design needs >1 `grid.sync` and/or >~96KB dynamic SMEM per CTA
- NCU shows reduced occupancy or spills in the monokernel path
