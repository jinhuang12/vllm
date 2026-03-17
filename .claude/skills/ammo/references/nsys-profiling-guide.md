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

## Scope

This guide covers **profiling (trace capture)** for kernel analysis in Stage 1.

For **validation latency measurements** (Stages 5-6), use `scripts/run_vllm_bench_latency_sweep.py`
instead — see `references/validation-defaults.md`.

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

**A) Full-run capture (small models, TP=1 only)**
- Works without modifying the workload.
- Keep the workload short so the trace is small and analyzable.
- Acceptable for single-GPU models where torch.compile + CUDA graph capture takes <60s.

**B) Two-step delimited capture (REQUIRED for TP>1 or large models)**

Full-run capture with `--cuda-graph-trace=node` can hang indefinitely. The hang can occur during CUDA graph **replay** under `--cuda-graph-trace=node`, not only during graph creation or torch.compile. Evidence from Qwen3.5-35B-A3B-FP8 on B200: graph capture completed successfully (15s, 102 graphs), BS=1 profiling succeeded (5 iterations), but BS=8 hung after `cudaProfilerStart()` during the first graph replay. Likely cause: per-node replay instrumentation overwhelmed by graph complexity or GPU memory pressure from ~2,142 CUDAGraph objects (50 default capture sizes x ~41 piecewise subgraphs + 50 FULL graphs).

The two-step approach reduces compile/capture overhead in the profiled region (even though it does not fully prevent replay hangs -- see also section 3.1C for reducing the CUDA graph capture surface):

1. **Pre-warm** (no nsys): Run the workload once to populate torch.compile and Triton autotuning caches on disk. Note: CUDA graphs are in-memory only and will be recaptured in Step 2 (see note on caching below).
2. **Profile with delayed capture**: Use `--capture-range=cudaProfilerApi` so nsys idles through model load, compile, and graph capture, then traces only the profiled iteration.

vLLM's `--profile --profiler-config '{"profiler": "cuda"}'` flag calls `cudaProfilerStart/Stop` around exactly one benchmark iteration, which nsys hooks into.

**When to use two-step capture**: Use it whenever `--cuda-graph-trace=node` is needed (which is always for accurate decode-step breakdowns) AND any of these apply:
- TP > 1 (multiple worker processes)
- Model has >10B parameters
- torch.compile time >60 seconds
- Previous full-run capture attempt timed out or produced a trace >500 MB

If in doubt, use two-step — it is strictly better than full-run for steady-state decode profiling.

**Note on caching**: Only `torch.compile` and Triton autotuning artifacts cache to disk. CUDA graphs are stored in-memory only (`CUDAGraphWrapper.concrete_cudagraph_entries` dict in `vllm/compilation/cuda_graph.py` — a plain Python dict that evaporates on process exit). Pre-warming still helps significantly: it eliminates torch.compile latency (~840s down to ~18s on subsequent runs), but CUDA graph capture still runs every process launch (~12-15s for large MoE models).

**C) Reducing CUDA graph capture surface for nsys profiling**

vLLM's default compilation config captures CUDA graphs for ~50 batch sizes (`[1, 2, 4, 8, 16, 24, 32, ..., 512]`), producing ~2,142 CUDAGraph objects total (50 capture sizes x ~41 piecewise subgraphs + 50 FULL graphs for a large MoE model). This memory pressure can cause `--cuda-graph-trace=node` replay instrumentation to hang.

**Mitigation**: Restrict capture to only the batch sizes being profiled using the `--cudagraph-capture-sizes` CLI argument:

```bash
vllm bench latency \
  --model {model_id} \
  --cudagraph-capture-sizes 1 8 32 \
  --batch-size 8 \
  --input-len 64 --output-len 32 \
  --num-iters-warmup 2 --num-iters 1
```

This reduces graph objects from ~2,142 to ~126 (a 17x reduction), substantially lowering the instrumentation burden on nsys.

**Important**: `VLLM_CUDAGRAPH_CAPTURE_SIZES` does not exist as an environment variable. Use the CLI argument `--cudagraph-capture-sizes` or pass it via `--compilation-config '{"cudagraph_capture_sizes": [1, 8, 32]}'`.

**Production parity**: The profiled batch sizes (e.g., `[1, 8, 32]`) are exact matches in vLLM's default capture list, so the restricted graphs are identical to what production would use for those sizes. Restricting capture sizes is NOT a parity violation — it simply skips graphs for batch sizes that are not being profiled.

**Automated in sweep script**: The sweep script (`run_vllm_bench_latency_sweep.py`) does this automatically when `--nsys-profile` is active — it sets `--cudagraph-capture-sizes` to match `workload.batch_sizes` from `target.json`.

### 3.2 Recommended nsys flags (vLLM baseline)

Recommended flags for vLLM (especially with worker processes and CUDA graphs):
- `--trace-fork-before-exec=true` (trace worker processes)
- `--cuda-graph-trace=node` (expand CUDA graph nodes into kernels)

For CSV attribution (recommended):
- `--trace=cuda,nvtx` (include NVTX if enabled)
- `--sample=none` (disable CPU sampling unless you have a CPU-side question)

### 3.3 Quick reference: offline inference (bench latency)

**Preferred: Two-step delimited capture (works for any model size / TP)** 

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn                                                                                                                                                                                                                                                                           
export HF_HOME=<path_to_hf_cache>  # if model weights are cached elsewhere                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                   
# Step 1: Pre-warm (populates torch.compile + Triton autotuning disk caches, no nsys)                                                                                                                                                                                                                                          
vllm bench latency \                                                                                                                                                                                                                                                                                               
  --model {model_id} \                                                                                                                                                                                                                                                                                             
  --tensor-parallel-size {tp} \                                                                                                                                                                                                                                                                                    
  --batch-size 8 \                                                                                                                                                                                                                                                                                                 
  --input-len 64 \                                                                                                                                                                                                                                                                                                 
  --output-len 32 \                                                                                                                                                                                                                                                                                                
  --num-iters-warmup 1 \                                                                                                                                                                                                                                                                                           
  --num-iters 1                                                                                                                                                                                                                                                                                                    
                                                                                                                                                                                                                                                                                                                   
# Step 2: Profile with delayed capture (only traces the --profile iteration)
nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \ 
  --capture-range-end=stop \
  -o {artifact_dir}/nsys/baseline_bs8 \
  vllm bench latency \
    --model {model_id} \
    --tensor-parallel-size {tp}
    --batch-size 8 \
    --input-len 64 \                                                                                                                                                                                                                                                                                        
    --output-len 32 \                                                                                                                                                                                                                                                                                       
    --num-iters-warmup 2 \                                                                                                                                                                                                                                                                                  
    --profile \                                                                                                                                                                                                                                                                                             
    --profiler-config '{"profiler": "cuda"}'                                                                                                                                                                                                                                                                
```                                                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                            
Key flags explained:                                                                                                                                                                                                                                                                                        
- `--capture-range=cudaProfilerApi`: nsys idles until vLLM calls `cudaProfilerStart()`, skipping model load + compile + graph capture entirely                                                                                                                                                              
- `--capture-range-end=stop`: stops capture at `cudaProfilerStop()`                                                                                                                                                                                                                                         
- `--profile --profiler-config '{"profiler": "cuda"}'`: vLLM brackets exactly one iteration with cudaProfiler start/stop                                                                                                                                                                                    
- `--num-iters-warmup 2` in Step 2: ensures CUDA graphs are replayed (warmed) before the profiled iteration                                                                                                                                                                                                 
- `--output-len 32`: short generation keeps trace small (~20 decode steps) while capturing full steady-state behavior                                                                                                                                                                                       
- Step 1 pre-warm: torch.compile and Triton autotuning caches persist on disk so Step 2 skips recompilation; CUDA graphs are in-memory only and are recaptured in Step 2                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                                                                            
**Fallback: Full-run capture (small TP=1 models only)**                                                                                                                                                                                                                                                     
                                                                                                                                                                                                                                                                                                            
```bash                                                                                                                                                                                                                                                                                                     
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

Only use full-run capture for small single-GPU models where torch.compile takes <60s.

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

### 3.5 Automated per-bucket profiling via sweep script

Instead of manually running nsys per batch size (which reloads the model each time), use the sweep script's `--nsys-profile` flag to profile all buckets in a single model load:

```bash
python scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --nsys-profile
```

This produces **one `.nsys-rep` per bucket** in `{artifact_dir}/e2e_latency/nsys/` (e.g., `baseline_bs1.nsys-rep`, `baseline_bs8.nsys-rep`).

**How it works**: The sweep script wraps the child process with `nsys profile --capture-range=cudaProfilerApi --capture-range-end=repeat:N` and calls `cudaProfilerStart()/Stop()` with `torch.cuda.synchronize()` around each bucket's measured iterations (after warmup). nsys's repeat mode automatically splits each capture range into a separate `.nsys-rep` file. The script renames the numbered files to match bucket tags.

**TP > 1 compatibility**: The script sets `VLLM_WORKER_MULTIPROC_METHOD=spawn` and uses `--trace-fork-before-exec=true`. nsys captures all traced processes (including TP workers) when any process triggers the capture range — this is an nsys-level mechanism, not CUDA profiler propagation. Each `.nsys-rep` contains traces from all GPUs.

**Workload matrix support**: The sweep script also supports sweeping `(input_len, output_len, batch_size)` tuples via the `workload_matrix` field in `target.json` — see `references/e2e-latency-guide.md`.

Each output file is independently analyzable:

```bash
nsys stats --report cuda_gpu_kern_sum {artifact_dir}/e2e_latency/nsys/baseline_bs8.nsys-rep
```

### 3.6 Traces without `--cuda-graph-trace=node` are misleading (CRITICAL)                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                   
If `--cuda-graph-trace=node` is omitted (or the trace hangs and you fall back to a non-expanded trace), the resulting data has serious distortions:                                                                                                                                                                
                                                                                                                                                                                                                                                                                                                   
1. **FULL CUDA graph replays appear as single opaque `cudaGraphLaunch` events.** Individual kernels inside the graph are NOT visible. Since steady-state decode runs entirely inside FULL CUDA graphs, the per-kernel breakdown is missing for the most important region.                                          
                                                                                                                                                                                                                                                                                                                   
2. **All-reduce spin-wait inflates communication time.** Custom all-reduce kernels (e.g., `multimem_all_reduce_kernel`) report wall time including barrier spin-wait. In a trace, this looks like 50-80% of GPU time is communication — but the actual added latency is only ~5-15 us per call. The spin-wait overlaps with compute on peer GPUs and is NOT additive to the critical path.                                                                                                                                                                                                                             
                                                                                                                                                                                                                                                                                                                   
3. **Piecewise graph kernel times overestimate decode costs.** Kernels visible in piecewise graph regions have different scheduling behavior than those in FULL graph replay. In one measured case, MoE routing overhead was 75.9 us/call in piecewise regions but only 14.8 us/call under FULL CUDA graph replay — a 5.1x overestimate.                                                                                                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                                                                   
**If you must work with a non-expanded trace**, document these caveats prominently in `bottleneck_analysis.md` and flag all `f_decode` estimates as approximate with explicit uncertainty bounds. Prefer CUDA-graph micro-experiments (like those in Stage 3 debate) to validate nsys-extrapolated timings before committing to optimization targets.                                                                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                                                                                   
### 3.7 Export the minimum useful CSV reports                                                                                                                                                                                                                                                                      
    

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

### 3.8 What each nsys report answers

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
**For TP > 1**: Use the two-step delimited capture (section 3.1B, section 3.3) and consider reducing the CUDA graph capture surface (section 3.1C). Full-run capture with `--cuda-graph-trace=node` can hang during graph replay when per-node instrumentation is overwhelmed by the number of CUDAGraph objects. The two-step approach (pre-warm without nsys, then `--capture-range=cudaProfilerApi`) reduces overhead, and restricting `--cudagraph-capture-sizes` further mitigates replay hangs.                                                                                                                                                                                                                                                    
- Single-GPU tracing (`CUDA_VISIBLE_DEVICES=0`) is useful for kernel-level analysis but cannot capture multi-GPU communication patterns. For TP models, trace all GPUs but use delimited capture to keep trace size manageable.
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
| Kernel appears in piecewise graph but not FULL decode graph | One-time init, prefill-only, or framework overhead | Compute f_decode separately; if f_decode ≈ 0, this kernel is not worth optimizing for decode-heavy workloads |
| Kernel instance count >> (num_layers × num_decode_steps) | Autotuning, JIT, or graph capture artifact | Cross-check with multi-iteration run or FULL CUDA graph extraction |

### 7.6 Transient vs steady-state overhead

Traces with `--num-iters 1` capture one-time costs (Triton autotuning, JIT, graph capture) as a large fraction of GPU time. These are amortized in production. The researcher should extract the FULL CUDA graph (decode) region and compute `f_decode` as the primary optimization target. Use `--num-iters-warmup 5` to ensure warmup overhead completes before the bench iteration.

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
