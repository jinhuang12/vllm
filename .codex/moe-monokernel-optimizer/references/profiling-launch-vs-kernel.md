# Profiling: Launch/API Time vs Kernel Execution Time

> For practical vLLM nsys commands and MoE-specific workflow, see `nsys-profiling-guide.md`.

MoE optimization discussions often conflate:
- **CPU launch/API overhead** (Python → CUDA API calls)
- **GPU kernel execution time** (actual GPU work)

Under CUDA graphs, the CPU launch path changes significantly; kernel execution time does not.

## What you can (and can’t) measure

- **Nsight Systems (nsys)**: shows both CUDA API time and GPU kernel time on a timeline.
- **Nsight Compute (ncu)**: profiles *inside a kernel* (no CPU launch overhead).
- **CUDA events**: measure elapsed time on a stream (includes kernels and device-side waits on that stream, excludes CPU overhead).

## CUDA graphs implications

In eager mode:
- You typically see many `cudaLaunchKernel` (or driver equivalents) plus Python overhead.

In CUDA graphs:
- You typically see `cudaGraphLaunch` plus the kernel nodes.
- Per‑kernel launch overhead is reduced, but not “free”; the dominant remaining cost is GPU kernel time.

## Recommended baseline: nsys first, then ncu

### 1) Nsight Systems: timeline and attribution

Goal: for the MoE subgraph, quantify:
- total GPU time (sum kernel durations)
- whether API time is material
- which kernels dominate

Practical workflow:
- Run a small batch bucket that you care about (e.g., BS=4/8) under the *same* CUDA graph settings as production.
- Capture a short trace focused on MoE.
- Identify kernel durations and any unexpected gaps.

When graphs are enabled, pay attention to:
- Graph breaks (kernels missing from capture)
- Host sync points
- Unexpected memcpy or memset nodes

### 2) Nsight Compute: “did fusion hurt the GEMM?”

After you fuse an epilogue into a GEMM kernel, use NCU to validate that you did not:
- explode register count (spills)
- reduce occupancy below what the baseline needs for latency hiding
- introduce new stalls (e.g., memory throttling, barrier waits)

NCU is the right tool for “why is my fused kernel slower even though it does less memory traffic”.

## Interpretation shortcuts

- If your baseline already runs under CUDA graphs and kernel time dominates, “launch overhead” savings are often single‑digit µs.
- If your fused kernel increases dynamic shared memory enough to go from 2 CTAs/SM → 1 CTA/SM, it can lose even if it saves memory traffic.
- If your approach adds `grid.sync`, treat it like a tax that must be amortized by substantial work per barrier.
