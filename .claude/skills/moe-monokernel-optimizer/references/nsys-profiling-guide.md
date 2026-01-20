# Profiling vLLM with Nsight Systems (nsys) and Nsight Compute (ncu)

This is a practical guide for profiling vLLM to:
- Use **Nsight Systems (nsys)** for **end-to-end** timelines: identify fusion opportunities, launch overhead, sync/memcpy gaps, and system-level bottlenecks.
- Use **Nsight Compute (ncu)** for **kernel-level** analysis: identify device bottlenecks (memory vs compute, occupancy limits, stall reasons).

Rule of thumb:
1) Start with **nsys** to determine what to optimize and which kernels matter in steady state.
2) Use **ncu** on 1–3 selected kernels to determine why they are slow and what knob to turn.

Always record the exact commands and key findings in `validation_results.md`.

## Search Anchors

nsys profile, nsys stats, CUDA graphs, torch.compile, fused_moe, fused_experts, topk_softmax, kernel timing, VLLM_WORKER_MULTIPROC_METHOD, ncu, Nsight Compute

## Table of contents
1. Goals and non-goals
2. Make runs comparable (vLLM setup)
3. Nsight Systems (nsys): capture and analysis (E2E)
4. Nsight Compute (ncu): targeted kernel profiling (device bottlenecks)
5. Multi-process / multi-GPU gotchas
6. Interpreting results: fusion opportunities vs device bottlenecks
7. Worked example: MoE extraction (nsys → CSV → ncu) + example table
8. References

## 1) Goals and non-goals

Goals:
- Identify where GPU time goes and which kernel sequences repeat in steady-state vLLM inference (nsys).
- Identify device bottlenecks for specific hot kernels (ncu): memory vs compute vs occupancy vs latency.

Non-goals:
- Treat nsys timing as a microbenchmark oracle. Use it to find candidates and validate "did time disappear?"
- Use ncu to estimate end-to-end latency. Kernel replay can distort timing; use ncu for attribution.

## 2) Make runs comparable (vLLM setup)

### 2.1 Prefer steady state
- Always include warmup iterations.
- If you use CUDA graphs and/or compilation, profile steady-state replay separately from compile/capture.

### 2.2 vLLM multiprocessing (important for profilers)

vLLM can spawn worker processes. For cleaner profiler behavior:

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
```

### 2.3 NVTX ranges (recommended)

NVTX improves attribution (stage → kernels) for both nsys and ncu (via NVTX filtering).

```bash
export VLLM_NVTX_SCOPES_FOR_PROFILING=1
```

If NVTX is enabled but you see import errors, install the Python package:

```bash
python -c "import nvtx" || pip install nvtx
```

Optional (more granular, validate in your configuration):
- `--enable-layerwise-nvtx-tracing` (may be incompatible with CUDA graph in some modes; confirm in your run).

### 2.4 Keep traces small and comparable
- Keep `--num-iters` low (often `1`) after warmup.
- Keep input/output lengths constant across baseline vs variants.
- Start with single GPU (`CUDA_VISIBLE_DEVICES=0`) unless the bottleneck is distributed.

## 3) Nsight Systems (nsys): capture and analysis (E2E)

### 3.1 Capture modes

**A) Full-run capture (robust default)**
- Works without modifying the workload.
- Keep the workload short so the trace is small and analyzable.

**B) Delimited capture (best signal; needs delimiters)**
- Use `--capture-range=cudaProfilerApi` and a workload that toggles capture, or
- Use `--capture-range=nvtx` and target specific NVTX ranges.

Use delimited capture when you can do it cleanly without changing semantics.

### 3.2 Recommended nsys flags (vLLM baseline)

Recommended flags for vLLM (especially with worker processes and CUDA graphs):
- `--trace-fork-before-exec=true` (trace worker processes)
- `--cuda-graph-trace=node` (expand CUDA graph nodes into kernels)

For CSV attribution (recommended):
- `--trace=cuda,nvtx` (include NVTX if enabled)
- `--sample=none` (disable CPU sampling unless you have a CPU-side question)

### 3.3 Quick reference: offline inference (bench latency)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o {artifact_dir}/nsys/baseline_bs8 \
  vllm bench latency \
    --model {model_id} \
    --batch-size 8 \
    --input-len 1024 \
    --output-len 32 \
    --num-iters-warmup 5 \
    --num-iters 1
```

Tip: If `vllm` is not on your PATH, you can run the CLI via Python:

```bash
python -m vllm.entrypoints.cli.main bench latency --help
```

### 3.4 Quick reference: server profiling (dynamic capture)

Use this pattern when profiling `vllm serve` (capture only around request handling):

```bash
nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end repeat \
  vllm serve {model_id} --profiler-config.profiler cuda
```

### 3.5 Export the minimum useful CSV reports

From a `.nsys-rep`, export:
- `cuda_gpu_kern_sum` — per-kernel GPU time totals (what dominates)
- `cuda_gpu_trace` — chronological kernel list (repeat patterns, gaps)
- `nvtx_sum` and `nvtx_kern_sum` (if NVTX exists) — stage attribution
- `cuda_api_sum` (optional) — CPU/CUDA API overhead and syncs

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output {artifact_dir}/nsys/baseline_bs8_cuda_gpu_kern_sum \
  {artifact_dir}/nsys/baseline_bs8.nsys-rep

nsys stats --report cuda_gpu_trace --format csv \
  --output {artifact_dir}/nsys/baseline_bs8_cuda_gpu_trace \
  {artifact_dir}/nsys/baseline_bs8.nsys-rep

# Optional (requires NVTX)
nsys stats --report nvtx_sum --format csv \
  --output {artifact_dir}/nsys/baseline_bs8_nvtx_sum \
  {artifact_dir}/nsys/baseline_bs8.nsys-rep

nsys stats --report nvtx_kern_sum --format csv \
  --output {artifact_dir}/nsys/baseline_bs8_nvtx_kern_sum \
  {artifact_dir}/nsys/baseline_bs8.nsys-rep

# Optional: look for sync/API overhead
nsys stats --report cuda_api_sum --format csv \
  --output {artifact_dir}/nsys/baseline_bs8_cuda_api_sum \
  {artifact_dir}/nsys/baseline_bs8.nsys-rep
```

### 3.6 What each nsys report answers

- `cuda_gpu_kern_sum`:
  - "What kernels dominate total GPU time?"
  - "Is the hot path heavy-kernel dominated (GEMM/attention) or micro-kernel dominated (fusion/launch overhead)?"

- `cuda_gpu_trace`:
  - "What kernel sequences repeat in steady state?"
  - "Where is micro-kernel soup between heavy kernels?"
  - "Are there gaps/bubbles between kernels?"

- `nvtx_sum` / `nvtx_kern_sum`:
  - "Which vLLM stage owns time?"
  - "Which kernels are inside that stage?"

- `cuda_api_sum`:
  - "Are we spending time in `cudaStreamSynchronize` / `cudaMemcpy*` / graph breaks / CPU launch overhead?"

## 4) Nsight Compute (ncu): targeted kernel profiling (device bottlenecks)

### 4.1 How to pick kernels for ncu

Use nsys `cuda_gpu_kern_sum` to select:
- 1–2 kernels with high total time and meaningful instances.
- Optionally 1 short kernel that runs extremely often (fusion candidate).

Treat kernel name matching as approximate (Triton and CUTLASS names can vary). Use NVTX ranges and adjacency in the trace when names are long.

### 4.2 ncu recommended starting point

Start small:
- `--set basic` first (lower overhead)
- keep `--launch-count` small
- use `--launch-skip` to avoid warmup noise
- enable NVTX filtering if available (`--nvtx`, `--nvtx-include`)

Example (MoE kernel pattern, forward stage):

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_NVTX_SCOPES_FOR_PROFILING=1

ncu \
  --set basic \
  --target-processes all \
  --nvtx \
  --nvtx-include "gpu_model_runner: forward" \
  --kernel-name "fused_moe" \
  --launch-skip 50 \
  --launch-count 5 \
  -o {artifact_dir}/ncu/fused_moe_basic \
  vllm bench latency \
    --model {model_id} \
    --batch-size 8 \
    --input-len 1024 \
    --output-len 32 \
    --num-iters-warmup 5 \
    --num-iters 1
```

If you need more detail, move to:

```bash
ncu --set detailed ...
```

### 4.3 Exporting ncu output for review

ncu can print CSV to stdout (useful for quick comparisons):

```bash
ncu --set basic --csv ...
```

For deeper inspection, prefer saving a report with `-o`/`--export` and opening it in the Nsight Compute UI or importing via CLI workflows.

## 5) Multi-process / multi-GPU gotchas

### 5.1 nsys
- Use `--trace-fork-before-exec=true` to follow vLLM worker processes.
- Prefer tracing one GPU first: `CUDA_VISIBLE_DEVICES=0`.
- For CUDA graphs, include `--cuda-graph-trace=node` to see per-kernel detail.

### 5.2 ncu
- Use `--target-processes all` to include worker processes.
- Keep `--launch-count` small; replay overhead grows quickly.
- If you only want one rank/process, use `--target-processes-filter` and/or run single GPU.

## 6) Interpreting results: fusion opportunities vs device bottlenecks

### 6.1 Fusion opportunities (best found with nsys)

Signals:
- Many short pointwise kernels between heavy kernels (launch overhead + HBM round-trips).
- Repeated chains like: cast → add → mul → activation → cast.
- Quant/dequant kernels interleaved with compute kernels.
- Large gaps/bubbles between kernels (CPU overhead, sync points, graph breaks).

Actions:
- Fuse pointwise chains (Triton/CUDA fusion).
- Reduce intermediate reads/writes to HBM (fuse producer/consumer).
- Pull small ops into an existing heavy kernel (when safe).
- Fix graph breaks / unintended syncs to reduce bubbles.

### 6.2 Device bottlenecks (best found with ncu)

Common diagnoses:
- **Memory bound**: bandwidth saturated, low compute utilization → improve data layout/reuse/coalescing, reduce traffic.
- **Compute bound**: high SM/Tensor utilization → improve tensor core path, tiling, pipelining.
- **Occupancy limited**: capped by registers/shared memory → reduce regs/smem, adjust tiling.
- **Latency/launch limited**: extremely short kernels → fix with fusion (nsys typically shows this best).

## 7) Worked example: MoE extraction (nsys → CSV → ncu) + example table

This section is MoE-specific by design. Keep it as a worked example so the rest of the guide remains general.

### 7.1 Capture a short nsys trace (MoE workload)

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn

nsys profile \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  -o {artifact_dir}/nsys/moe_bs8 \
  vllm bench latency \
    --model {model_id} \
    --batch-size 8 \
    --input-len 1024 \
    --output-len 32 \
    --num-iters-warmup 5 \
    --num-iters 1
```

### 7.2 Extract MoE kernel times (nsys stats + grep)

```bash
# Full kernel summary (sorted by total time)
nsys stats --report cuda_gpu_kern_sum {artifact_dir}/nsys/moe_bs8.nsys-rep

# Filter common MoE-related kernels
nsys stats --report cuda_gpu_kern_sum {artifact_dir}/nsys/moe_bs8.nsys-rep \
  | grep -E "fused_moe|fused_experts|topk|topk_softmax|moe_align"
```

MoE kernel name patterns you may see:

| Pattern | What it is |
|---------|------------|
| `fused_moe*` | Main fused MoE kernel |
| `fused_experts*` | Expert execution kernel |
| `topk*` / `topk_softmax*` | Router/gating kernel |
| `moe_align*` | Token alignment/preparation |
| `xmma_gemm*` | GEMM kernels (may be inside MoE) |
| `act_and_mul*` / `silu*` | Activation kernels |

### 7.3 Follow-up with ncu (pick one MoE kernel)

Take the top MoE kernel from `cuda_gpu_kern_sum` and run a small `ncu` capture to diagnose whether it is memory bound, compute bound, or occupancy limited.

```bash
ncu \
  --set basic \
  --target-processes all \
  --kernel-name "fused_moe" \
  --launch-skip 50 \
  --launch-count 5 \
  -o {artifact_dir}/ncu/moe_fused_moe_basic \
  vllm bench latency \
    --model {model_id} \
    --batch-size 8 \
    --input-len 1024 \
    --output-len 32 \
    --num-iters-warmup 5 \
    --num-iters 1
```

### 7.4 Example: constraints.md entry (keep as a template)

```markdown
## Baseline Truth Snapshot

nsys profile: {artifact_dir}/nsys/moe_bs8.nsys-rep

### MoE Kernel Timings (BS=8, decode step)

| Kernel | Avg (us) | % of MoE |
|--------|----------|----------|
| fused_moe_kernel | 523.4 | 78% |
| topk_softmax | 45.2 | 7% |
| moe_align_block | 32.1 | 5% |
| scaled_fp8_quant | 28.9 | 4% |
| (accumulate) | 41.5 | 6% |
| **Total MoE** | **671.1** | **100%** |

### Derived Values
- P = 8 * 8 = 64 (token-expert pairs, top_k=8)
- M_avg = 64 / 128 = 0.5 tokens/expert
- Saturation = 64 / 108 = 0.59 (108 SMs example)
```

### 7.5 Red flags in traces (use as a checklist)

| Symptom | Possible cause | Action |
|---------|----------------|--------|
| Unexpected `cudaMemcpy*` | Host-device transfer on hot path | Find the caller; keep transfers off the step loop |
| `cudaStreamSynchronize` | Graph breaks / explicit sync | Identify the sync site; restore async/graph capture |
| Large gaps between kernels | CPU overhead or sync | Confirm CUDA graphs are active; reduce Python overhead |
| Missing kernels in trace | Graph not expanded | Ensure `--cuda-graph-trace=node` is set |
| High variance in kernel times | Contention / throttling | Profile on an isolated GPU / stable clocks |

## Installation (if needed)

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
  | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

## 8) References

- vLLM profiling docs: https://docs.vllm.ai/en/latest/contributing/profiling/
- Nsight Systems User Guide: https://docs.nvidia.com/nsight-systems/UserGuide/index.html
- Nsight Compute User Guide: https://docs.nvidia.com/nsight-compute/
- `profiling-launch-vs-kernel.md` - Conceptual background on launch vs kernel time
