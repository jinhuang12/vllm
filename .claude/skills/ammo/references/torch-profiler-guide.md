# Profiling vLLM with torch.profiler Chrome Traces

This is a practical guide for extracting kernel-level insights from
torch.profiler Chrome trace files (`.pt.trace.json.gz`). It is the companion
to `nsys-profiling-guide.md` and covers the Tier 1 profiling methodology in the
AMMO tiered profiling strategy.

## Search Anchors

torch.profiler, Chrome trace, CUPTI, CUDA graphs, multi-rank, kernel chain,
straggler analysis, AllReduce, bandwidth utilization, occupancy caveat,
Blackwell, SM100, B200, traceEvents, pt.trace.json.gz

## Scope

This guide covers **analysis of torch.profiler Chrome trace files** for kernel
attribution in Stages 1-2. For capture setup (nsys probe, vLLM bench flags),
see `nsys-profiling-guide.md`. For validation latency measurements (Stages
5-6), see `validation-defaults.md`.

## Table of contents

1. Overview
2. Capture methods
3. Chrome trace JSON format
4. Multi-rank analysis
5. Kernel chain analysis
6. Launch configuration analysis
7. Bandwidth utilization estimation
8. Occupancy caveat on Blackwell + CUDA graphs
9. Limitations vs nsys
10. Practical recipes

---

## 1) Overview

### What torch.profiler provides

torch.profiler captures per-kernel CUDA activity via CUPTI activity tracing.
When used with CUDA graph replay, it records production-representative timing
for every kernel in the replayed graph -- unlike nsys `--cuda-graph-trace=graph`
which only captures kernel timing during the graph capture phase (not replay).

### The tiered profiling strategy

| Tier | Tool | When to use | Key advantage |
|------|------|-------------|---------------|
| **0** | nsys `--cuda-graph-trace=node` | Models <~50B where nsys probe passes | Strictly superior: all fields, per-kernel replay timing, single file for all ranks |
| **1** | **torch.profiler Chrome trace** | **PRIMARY default for large models. Always works.** | **Production-representative CUPTI timing. Sees through CUDA graph replays. Multi-rank per-file. Chronological kernel ordering.** |
| **2** | nsys `--cuda-graph-trace=graph` | ENRICHMENT only | Adds SM100 cluster dims (HIGH value), NVLink traffic (MEDIUM), static/dynamic smem split (LOW-MED). **Kernel timings are from capture phase, NOT production -- do not use for rankings.** |

### When to use this guide

Use torch.profiler (Tier 1) as the primary profiling tool when:
- The nsys probe fails (model >~50B params, or `--cuda-graph-trace=node` times out)
- You need production-accurate kernel timing under CUDA graph replay
- You need multi-rank analysis from separate per-rank trace files
- You need chronological kernel ordering within a single decode step

Chrome trace provides ~85% of analytical value compared to nsys node mode. The
remaining ~15% (cluster dimensions, NVLink traffic, static/dynamic smem split)
comes from nsys Tier 2 enrichment.

---

## 2) Capture method

Use the sweep script with `--torch-profile` to capture traces for all batch sizes
in a single model load:

```bash
python scripts/run_vllm_bench_latency_sweep.py \
  --artifact-dir {artifact_dir} \
  --torch-profile
```

The sweep script configures `ProfilerConfig` on the LLM engine at load time:
- `profiler: "torch"` -- selects torch.profiler (not the `"cuda"` profiler)
- `torch_profiler_dir` -- per-bucket output directory, updated before each bucket's capture

Other `ProfilerConfig` fields use vLLM defaults:
- `torch_profiler_use_gzip: true` -- compresses output as `.pt.trace.json.gz`
- `torch_profiler_with_stack: true` -- includes Python stack traces

Activities are hardcoded internally (`["CPU", "CUDA"]` in `gpu_worker.py`) and cannot be
configured via `ProfilerConfig`.

Per-bucket capture is managed by `llm.start_profile()` / `llm.stop_profile()` calls
bracketing each bucket's benchmark iterations. This produces per-BS, per-rank trace files
(~50 MB per rank for large models, viewable in [Perfetto](https://ui.perfetto.dev/)):

```
{artifact_dir}/torch_profile/baseline_bs1/dp0_pp0_tp0_*.pt.trace.json.gz
{artifact_dir}/torch_profile/baseline_bs1/dp0_pp0_tp1_*.pt.trace.json.gz
{artifact_dir}/torch_profile/baseline_bs1/dp0_pp0_tp2_*.pt.trace.json.gz
{artifact_dir}/torch_profile/baseline_bs1/dp0_pp0_tp3_*.pt.trace.json.gz
{artifact_dir}/torch_profile/baseline_bs8/dp0_pp0_tp0_*.pt.trace.json.gz
...
```

### 2.1 Output file naming and location

Trace files follow the pattern:

```
dp{dp}_pp{pp}_tp{tp}_dcp{dcp}_ep{ep}_rank{global}.{nanosecond_ts}.pt.trace.json.gz
```

Use `dp0_pp0_tp{N}_*.pt.trace.json.gz` as the glob pattern — the wildcard safely
captures the additional suffixes. For a TP=4 setup profiling BS=1, you get 4 files:
- `dp0_pp0_tp0_*.pt.trace.json.gz` (Rank 0)
- `dp0_pp0_tp1_*.pt.trace.json.gz` (Rank 1)
- `dp0_pp0_tp2_*.pt.trace.json.gz` (Rank 2)
- `dp0_pp0_tp3_*.pt.trace.json.gz` (Rank 3)

Each file contains the full trace for that rank's single profiled decode step.

---

## 3) Chrome trace JSON format

### 3.1 Top-level structure

Chrome trace files are gzip-compressed JSON. The top-level structure:

```json
{
  "traceEvents": [
    { "cat": "kernel", "name": "...", "dur": 9.471, "ts": 12345678.0, "args": {...} },
    { "cat": "cpu_op", "name": "...", "args": {...} },
    { "cat": "runtime", "name": "cudaGraphLaunch", "args": {...} },
    ...
  ]
}
```

CUDA kernel events have `"cat": "kernel"`. Filter on this to extract GPU kernel
data.

### 3.2 Per-kernel fields

Each kernel event has fields at two levels — some are top-level event keys,
others are nested in the `args` dictionary:

**Top-level event keys** (access via `event['field']`):

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Full demangled kernel name (e.g., `void nvjet_32x64_TNN_Bfloat16_128_1<...>`) |
| `cat` | string | Category — `"kernel"` for CUDA kernels |
| `dur` | float | Duration in microseconds |
| `ts` | float | Start timestamp in microseconds (relative to trace start) |

**Nested in `args` dictionary** (access via `event['args']['field']`):

| Field | Type | Description |
|-------|------|-------------|
| `grid` | `[X,Y,Z]` | Grid dimensions (number of CTAs per dimension) |
| `block` | `[X,Y,Z]` | Block dimensions (threads per CTA per dimension) |
| `registers per thread` | int | Registers consumed per thread |
| `shared memory` | int | Shared memory in bytes (combined static + dynamic) |
| `blocks per SM` | float | Average blocks per SM (grid_total / num_SMs) |
| `warps per SM` | float | Average warps per SM |
| `est. achieved occupancy %` | float | CUPTI occupancy estimate (see section 8 for caveats) |
| `device` | int | GPU device ID |
| `stream` | int | CUDA stream ID |
| `correlation` | int | Correlation ID linking to CPU-side launch event |

### 3.3 Python parsing pattern

```python
import gzip
import json

def load_chrome_trace(path):
    """Load a Chrome trace file and extract CUDA kernel events."""
    with gzip.open(path, 'rt') as f:
        trace = json.load(f)
    kernels = [
        e for e in trace['traceEvents']
        if e.get('cat') == 'kernel'
    ]
    return kernels
```

Each element in `kernels` is a dict with `name`, `dur`, `ts`, and `args`
containing the launch configuration fields. Some fields (`grid`, `block`,
`registers per thread`, etc.) may appear directly in the event dict or nested
in `args` depending on the PyTorch version. Always check both:

```python
def get_field(event, field):
    """Get a field from event or event['args']."""
    if field in event:
        return event[field]
    return event.get('args', {}).get(field)
```

---

## 4) Multi-rank analysis

Multi-rank analysis is the highest-value methodology unique to Chrome trace
Tier 1 profiling. It reveals straggler ranks, AllReduce barrier skew, and
compute imbalance -- none of which are visible from a single-rank trace.

### 4.1 Load ALL rank files

For a TP=4 setup, always load all 4 rank traces:

```python
import glob

def load_all_ranks(trace_dir, bs):
    """Load Chrome traces for all ranks at a given batch size."""
    pattern = f"{trace_dir}/torch_profile/baseline_bs{bs}/dp0_pp0_tp*_*.pt.trace.json.gz"
    files = sorted(glob.glob(pattern))
    rank_kernels = {}
    for f in files:
        # Extract rank from filename: dp0_pp0_tp{rank}_...
        rank = int(f.split('_tp')[1].split('_')[0])
        rank_kernels[rank] = load_chrome_trace(f)
    return rank_kernels
```

### 4.2 Per-rank total CUDA time

The first diagnostic: does total GPU time vary across ranks?

```python
for rank, kernels in rank_kernels.items():
    total_us = sum(k['dur'] for k in kernels)
    print(f"Rank {rank}: {total_us:.0f} us total GPU time")
```

Example output (from Qwen3.5-397B-A17B, BS=1, TP=4):
```
Rank 0:  7338 us
Rank 1:  8735 us
Rank 2:  7676 us
Rank 3:  9922 us   <-- straggler (+35% vs Rank 0)
```

### 4.3 Straggler identification

A straggler rank has the longest total GPU time. Determine whether it is
compute-bound or communication-bound:

```python
def compute_ar_split(kernels):
    """Split kernel time into AllReduce and non-AllReduce components."""
    ar_time = sum(k['dur'] for k in kernels
                  if 'cross_device_reduce' in k['name']
                  or 'allreduce' in k['name'].lower()
                  or 'multimem_all_reduce' in k['name'])
    total = sum(k['dur'] for k in kernels)
    return {'total': total, 'allreduce': ar_time, 'compute': total - ar_time}

for rank, kernels in rank_kernels.items():
    split = compute_ar_split(kernels)
    print(f"Rank {rank}: total={split['total']:.0f} us, "
          f"AR={split['allreduce']:.0f} us, "
          f"compute={split['compute']:.0f} us")
```

### 4.4 AllReduce barrier skew

In AllReduce, all ranks must synchronize. The rank that arrives last (most
compute before the barrier) forces other ranks to spin-wait. This spin-wait
inflates the AllReduce duration for the waiting ranks:

- **Rank with LEAST AllReduce time** = arrives last = is the bottleneck
  (other ranks spin waiting for it)
- **Rank with MOST AllReduce time** = arrives first = spends time spinning

Example:
```
Rank 0: AR=  947 us  (arrives LAST -- least spin, most compute before barrier)
Rank 1: AR= 2356 us
Rank 2: AR= 1220 us
Rank 3: AR= 3404 us  (arrives FIRST -- most spin, least compute before barrier)
```

### 4.5 Non-AllReduce compute balance

Subtract AllReduce time from total to get pure compute time. For a balanced
workload, compute should be within ~2% across ranks:

```python
compute_times = [compute_ar_split(k)['compute'] for k in rank_kernels.values()]
mean_compute = sum(compute_times) / len(compute_times)
max_dev = max(abs(c - mean_compute) / mean_compute for c in compute_times)
print(f"Compute CV: {max_dev*100:.1f}%")
# Expected: <2% for balanced EP/TP; >5% indicates load imbalance
```

### 4.6 Per-kernel cross-rank variance

For high-variance kernels, check whether the variance is real (different work
per rank) or an artifact (measurement noise):

```python
from collections import defaultdict
import statistics

def cross_rank_variance(rank_kernels):
    """Compute per-kernel-name mean/stddev across ranks."""
    # Aggregate total time per kernel name per rank
    rank_totals = {}
    for rank, kernels in rank_kernels.items():
        totals = defaultdict(float)
        for k in kernels:
            totals[k['name']] += k['dur']
        rank_totals[rank] = totals

    # Find all kernel names
    all_names = set()
    for t in rank_totals.values():
        all_names.update(t.keys())

    results = []
    for name in all_names:
        vals = [rank_totals[r].get(name, 0) for r in sorted(rank_totals)]
        if max(vals) > 0:
            results.append({
                'name': name,
                'mean': statistics.mean(vals),
                'stddev': statistics.stdev(vals) if len(vals) > 1 else 0,
                'per_rank': vals,
            })
    return sorted(results, key=lambda x: -x['mean'])
```

---

## 5) Kernel chain analysis

Kernel chain analysis identifies the repeating per-layer kernel sequence in a
decode step. This MUST be done from trace event ordering, NOT inferred from
model architecture. Fused kernels do not appear as separate events, and
architecture diagrams may not reflect the actual execution.

### 5.1 Sort events by timestamp

```python
def get_kernel_chain(kernels):
    """Return kernels sorted by start timestamp."""
    return sorted(kernels, key=lambda k: k['ts'])
```

### 5.2 Identify per-layer sequences

Count kernels between AllReduce calls. Each AllReduce boundary typically marks
the end of one layer's compute:

```python
def segment_by_allreduce(kernels):
    """Split kernel chain into segments delimited by AllReduce."""
    chain = get_kernel_chain(kernels)
    segments = []
    current = []
    for k in chain:
        if 'cross_device_reduce' in k['name'] or 'allreduce' in k['name'].lower():
            if current:
                segments.append(current)
            segments.append([k])  # AllReduce as its own segment
            current = []
        else:
            current.append(k)
    if current:
        segments.append(current)
    return segments
```

For a model with 60 layers and 2 AllReduce calls per layer (attention +
MoE/FFN) plus 1 final AllReduce, expect ~121 AllReduce segments and ~121
compute segments.

### 5.3 Verify fusion state from the trace

A critical principle: if a kernel X is claimed to exist between kernels Y and
Z, CHECK the trace. Fused kernels will not appear as separate events.

Example: the MoE dispatch chain was initially assumed to have 6 kernels based
on architecture diagrams. Trace analysis revealed only 4 -- the routing and
normalization steps were fused into a single TRT-LLM monolithic kernel:

```
Claimed (architecture): FP4_QUANT -> ROUTING -> RENORM -> W1 -> ACT -> W2 -> FINALIZE
Actual (trace):         FP4_QUANT -> ROUTING_RENORM_W1_W2_FINALIZE
                        (monolithic TRT-LLM kernel)
```

Always verify by printing the actual kernel names in sequence:

```python
segments = segment_by_allreduce(kernels)
# Print the first MoE compute segment
for i, seg in enumerate(segments):
    if len(seg) > 3:  # Skip AllReduce-only segments
        print(f"--- Segment {i} ({len(seg)} kernels) ---")
        for k in seg:
            print(f"  {k['dur']:8.1f} us  {k['name'][:80]}")
        break
```

---

## 6) Launch configuration analysis

### 6.1 Extract launch configuration

```python
def get_launch_config(kernel):
    """Extract launch configuration from a kernel event."""
    return {
        'name': kernel['name'],
        'dur': kernel['dur'],
        'grid': get_field(kernel, 'grid'),
        'block': get_field(kernel, 'block'),
        'registers': get_field(kernel, 'registers per thread'),
        'smem': get_field(kernel, 'shared memory'),
        'blocks_per_sm': get_field(kernel, 'blocks per SM'),
        'warps_per_sm': get_field(kernel, 'warps per SM'),
    }
```

### 6.2 Flag low-utilization kernels

Kernels with `grid < num_SMs` leave SMs idle. On B200 (132 SMs), a kernel with
`grid=[1,1,1]` uses only 1 of 132 SMs -- 0.76% hardware utilization:

```python
NUM_SMS = 132  # B200

def flag_low_utilization(kernels, threshold=0.5):
    """Flag kernels using fewer than threshold fraction of SMs."""
    flagged = []
    for k in kernels:
        grid = get_field(k, 'grid')
        if grid:
            total_blocks = grid[0] * grid[1] * grid[2]
            utilization = min(total_blocks / NUM_SMS, 1.0)
            if utilization < threshold:
                flagged.append({
                    'name': k['name'][:60],
                    'grid': grid,
                    'blocks': total_blocks,
                    'sm_util': f"{utilization*100:.1f}%",
                    'dur': k['dur'],
                })
    return flagged
```

Example finding: `gemv2N` (router gate GEMV) runs with `grid=[1,1,1]` on B200,
using a single SM. This is 5.3% of peak HBM bandwidth and is a clear
optimization target.

### 6.3 Flag high register pressure

Kernels with >128 registers per thread have limited occupancy. On SM100,
the maximum registers per thread before occupancy drops below 50% is
architecture-dependent, but >128 is a useful heuristic:

```python
def flag_high_registers(kernels, threshold=128):
    """Flag kernels with high register pressure."""
    return [
        {'name': k['name'][:60],
         'registers': get_field(k, 'registers per thread'),
         'dur': k['dur']}
        for k in kernels
        if (get_field(k, 'registers per thread') or 0) > threshold
    ]
```

### 6.4 Shared memory note

The `shared memory` field in Chrome trace is the combined total of static and
dynamic shared memory. It does NOT distinguish between the two. To determine
whether shared memory is compile-time fixed (static) or runtime-configurable
(dynamic), you need nsys Tier 2 data which provides separate
`staticSharedMemory` and `dynamicSharedMemory` fields.

For most optimization decisions, the combined total is sufficient. The
static/dynamic distinction matters only when you need to know whether smem
usage can be adjusted via the launch API (dynamic only) or requires
recompilation (static).

---

## 7) Bandwidth utilization estimation

### 7.1 Methodology for BW-bound kernels

For bandwidth-bound kernels (GEMV, elementwise, reductions), estimate achieved
bandwidth:

```
achieved_BW = data_size_bytes / kernel_time_seconds
```

Where `data_size_bytes` comes from the model config (weight dimensions times
dtype bytes) and `kernel_time_seconds` = `dur * 1e-6`.

### 7.2 Data size from model config

Example for a router gate GEMV (`gemv2N`):
- Weight shape: `[hidden_size, num_experts]` = `[4096, 512]`
- Dtype: BF16 (2 bytes)
- Data size: `4096 * 512 * 2` = 4,194,304 bytes = 4.0 MB
- Input vector: `4096 * 2` = 8,192 bytes (negligible)
- Total: ~4.0 MB

### 7.3 Compare to GPU peak bandwidth

```python
def compute_bw_utilization(data_bytes, dur_us, peak_bw_tbps=8.0):
    """Compute BW utilization for a kernel.

    Args:
        data_bytes: Total bytes read+written by the kernel.
        dur_us: Kernel duration in microseconds.
        peak_bw_tbps: Peak HBM bandwidth in TB/s (B200 = ~8 TB/s).

    Returns:
        Dict with achieved_bw (TB/s) and utilization (fraction).
    """
    dur_s = dur_us * 1e-6
    achieved_bw = data_bytes / dur_s / 1e12  # TB/s
    utilization = achieved_bw / peak_bw_tbps
    return {
        'achieved_bw_tbps': achieved_bw,
        'utilization': utilization,
        'utilization_pct': f"{utilization*100:.1f}%",
    }

# Example: gemv2N on B200
# 4.0 MB weight, 9.8 us mean duration
result = compute_bw_utilization(4_194_304, 9.8, peak_bw_tbps=8.0)
# achieved_bw_tbps ~ 0.428, utilization ~ 5.3%
```

### 7.4 L2 cache consideration

If the working set fits in the GPU's L2 cache, the effective bandwidth ceiling
is L2 bandwidth, not HBM bandwidth. L2 is typically 3-5x faster than HBM.

| GPU | L2 cache size | HBM peak BW | Approximate L2 BW |
|-----|--------------|-------------|-------------------|
| B200 | 96 MB | ~8 TB/s | ~30 TB/s |
| H100 | 50 MB | ~3.35 TB/s | ~12 TB/s |

Example: `gemv2N` accesses 4.0 MB of weight data, which fits entirely in the
96 MB L2 cache on B200. After the first invocation warms the cache, subsequent
calls may be served from L2 at up to ~30 TB/s. The measured 0.43 TB/s (5.3% of
HBM peak) is therefore even more concerning -- it is ~1.4% of L2 bandwidth,
indicating the bottleneck is not memory bandwidth but the single-SM launch
configuration (`grid=[1,1,1]`).

---

## 8) Occupancy caveat on Blackwell + CUDA graphs

### 8.1 The problem

The `est. achieved occupancy %` field reports **0% for approximately 81% of
kernel events** on SM100 (Blackwell) when kernels execute under CUDA graph
replay.

### 8.2 Which kernels are affected

Essentially ALL hot kernels:

| Kernel category | Reports 0%? |
|----------------|-------------|
| nvjet cuBLAS GEMMs | YES |
| bmm MoE GEMMs (TRT-LLM) | YES |
| AllReduce (cross_device_reduce) | YES |
| FlashAttention (fmhaSm100f) | YES |
| Triton fused kernels | YES |
| splitKreduce (minor) | NO -- reports ~3% |
| routingIndicesCluster (minor) | NO -- reports ~1% |

### 8.3 Root cause

The CUPTI heuristic for occupancy estimation fails for:
1. **JIT-compiled kernels** (nvjet cuBLAS, Triton)
2. **CUDA graph-replayed kernels** (the entire decode step)
3. **SM100-specific kernel variants** (new instruction sets)

### 8.4 Recommendation

Treat the `est. achieved occupancy %` field as **"unknown"** for all kernels
that report 0%. Do NOT conclude that these kernels have zero occupancy -- the
field is simply broken for this hardware + execution mode combination.

For actual occupancy data, use `ncu` (Nsight Compute) with `--set basic` on
targeted kernels. ncu uses hardware performance counters and provides accurate
occupancy regardless of CUDA graph mode or kernel origin.

For rough manual estimation, use the raw fields from nsys Tier 2:
`registersPerThread`, grid/block dims, `sharedMemoryExecuted`, and SM
specifications to compute theoretical occupancy via the CUDA Occupancy
Calculator.

---

## 9) Limitations vs nsys

Chrome trace provides the backbone of kernel analysis but has specific gaps
that nsys fills. Understanding these gaps prevents over-reliance on a single
tool.

### 9.1 Fields missing from Chrome trace

| Data | Chrome trace | nsys | Impact |
|------|-------------|------|--------|
| Static/dynamic smem split | Combined total only | Separate `staticSharedMemory` + `dynamicSharedMemory` | **LOW-MED** -- combined total usually sufficient; split needed only for smem optimization strategy |
| SM100 cluster dimensions (CGA) | Not available | `clusterX`, `clusterY`, `clusterZ` | **HIGH** on Blackwell -- critical for understanding cooperative kernel launches (e.g., fmhaSm100f uses 8-SM clusters) |
| NVLink peer-to-peer traffic | ~21 HtoD events, 0.01 MB total | 780+ PtoP copies, 301 MB with `srcDeviceId` -> `dstDeviceId` | **MEDIUM** -- Chrome trace sees AllReduce kernel duration but not bytes transferred or NVLink topology |
| Full CUDA runtime API | ~77 events from profiled window | 19K+ events across full session | **LOW** -- Chrome captures the profiled step adequately; nsys adds warmup/capture overhead visibility |
| Kernel-to-API correlation | Limited `correlation` field (~1.4% of kernels, eager-dispatch only) | `correlationId` links all kernels to CPU launch | **LOW** -- rarely needed for bottleneck analysis |
| `sharedMemoryExecuted` (L1 partition) | Not available | Actual L1/smem partition at runtime | **LOW** -- niche value for understanding cache partitioning |
| `localMemoryPerThread` | Not available | Register spill to local memory | **LOW** -- relevant only for register spilling investigation |

### 9.2 Timing differences

| Metric | Chrome trace | nsys (`--cuda-graph-trace=graph`) | Which is authoritative |
|--------|-------------|-----------------------------------|----------------------|
| Kernel duration (graph replay) | Production timing via CUPTI activity | Not available (only capture-phase timing) | **Chrome trace** |
| Kernel duration (warmup) | Not captured (filtered by delay_iterations) | Full session timing | **nsys** (more data) |
| `cudaGraphLaunch` duration | 1 event, may be inflated (e.g., 2077 us) | 12+ events, steady-state average (e.g., 370 us) | **nsys** (more events, representative) |

### 9.3 Coverage comparison

| Dimension | Chrome trace | nsys |
|-----------|-------------|------|
| Kernel events per decode step | ~1,834 (exact, clean boundary) | ~32,868 (full session, must isolate step manually) |
| Ranks per file | 1 | All ranks in one file |
| Memory copy events | ~21 (metadata only) | ~1,028 (HtoD + DtoH + PtoP NVLink) |

### 9.4 Summary

Chrome trace is the primary analytical tool for kernel ranking, chain analysis,
and multi-rank comparison. nsys adds enrichment data -- primarily SM100 cluster
dimensions (HIGH value for Blackwell) and NVLink traffic quantification
(MEDIUM). No nsys-exclusive finding changes the kernel ranking or candidate
selection.

---

## 10) Practical recipes

### Recipe 1: Top-10 kernels by total time

```python
import gzip, json
from collections import defaultdict

def top_kernels(trace_path, n=10):
    """Print top-N kernels by total time from a Chrome trace."""
    with gzip.open(trace_path, 'rt') as f:
        trace = json.load(f)

    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
    totals = defaultdict(lambda: {'total': 0.0, 'count': 0, 'durs': []})
    for k in kernels:
        name = k['name']
        totals[name]['total'] += k['dur']
        totals[name]['count'] += 1
        totals[name]['durs'].append(k['dur'])

    total_gpu = sum(k['dur'] for k in kernels)
    ranked = sorted(totals.items(), key=lambda x: -x[1]['total'])

    print(f"{'Kernel':<60} {'Total(us)':>10} {'Count':>6} {'Mean(us)':>10} {'f_decode':>8}")
    print("-" * 100)
    for name, data in ranked[:n]:
        mean = data['total'] / data['count']
        f_decode = data['total'] / total_gpu
        short_name = name[:58] + '..' if len(name) > 60 else name
        print(f"{short_name:<60} {data['total']:>10.1f} {data['count']:>6} "
              f"{mean:>10.1f} {f_decode:>7.1%}")
    print(f"\nTotal GPU time: {total_gpu:.0f} us ({len(kernels)} kernel events)")
```

### Recipe 2: Cross-rank AllReduce variance

```python
import glob, gzip, json, statistics

def allreduce_variance(trace_dir, bs):
    """Compute AllReduce time variance across ranks."""
    pattern = f"{trace_dir}/torch_profile/baseline_bs{bs}/dp0_pp0_tp*_*.pt.trace.json.gz"
    files = sorted(glob.glob(pattern))

    rank_ar = {}
    rank_total = {}
    for f in files:
        rank = int(f.split('_tp')[1].split('_')[0])
        with gzip.open(f, 'rt') as fh:
            trace = json.load(fh)
        kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']
        ar_names = ['cross_device_reduce', 'allreduce', 'multimem_all_reduce']
        ar_time = sum(k['dur'] for k in kernels
                      if any(n in k['name'].lower() for n in ar_names))
        total = sum(k['dur'] for k in kernels)
        rank_ar[rank] = ar_time
        rank_total[rank] = total

    print(f"{'Rank':<6} {'Total(us)':>10} {'AR(us)':>10} {'Compute(us)':>12}")
    print("-" * 42)
    for rank in sorted(rank_ar):
        compute = rank_total[rank] - rank_ar[rank]
        print(f"{rank:<6} {rank_total[rank]:>10.0f} {rank_ar[rank]:>10.0f} {compute:>12.0f}")

    ar_values = list(rank_ar.values())
    if len(ar_values) > 1:
        cv = statistics.stdev(ar_values) / statistics.mean(ar_values) * 100
        print(f"\nAllReduce CV: {cv:.1f}%")
        straggler = min(rank_ar, key=rank_ar.get)
        print(f"Straggler: Rank {straggler} (least AR time = arrives last)")
```

### Recipe 3: Extract MoE kernel chain per layer

```python
import gzip, json

def moe_kernel_chains(trace_path, max_layers=3):
    """Extract and print kernel chains between AllReduce boundaries."""
    with gzip.open(trace_path, 'rt') as f:
        trace = json.load(f)

    kernels = sorted(
        [e for e in trace['traceEvents'] if e.get('cat') == 'kernel'],
        key=lambda k: k['ts']
    )

    # Segment by AllReduce
    segments = []
    current = []
    for k in kernels:
        is_ar = ('cross_device_reduce' in k['name']
                 or 'allreduce' in k['name'].lower())
        if is_ar:
            if current:
                segments.append(current)
            current = []
        else:
            current.append(k)
    if current:
        segments.append(current)

    # Print first N non-trivial segments (likely MoE or attention layers)
    printed = 0
    for i, seg in enumerate(segments):
        if len(seg) < 3:
            continue
        print(f"\n=== Layer segment {i} ({len(seg)} kernels, "
              f"{sum(k['dur'] for k in seg):.1f} us total) ===")
        for k in seg:
            grid = k.get('args', {}).get('grid', k.get('grid', '?'))
            print(f"  {k['dur']:8.1f} us  grid={grid}  {k['name'][:70]}")
        printed += 1
        if printed >= max_layers:
            break
```

### Recipe 4: Compute BW utilization for GEMV kernels

```python
import gzip, json

def gemv_bandwidth(trace_path, weight_configs, peak_bw_tbps=8.0):
    """Estimate BW utilization for known GEMV kernels.

    Args:
        trace_path: Path to Chrome trace .pt.trace.json.gz file.
        weight_configs: Dict mapping kernel name substring to
            (rows, cols, dtype_bytes) describing the weight matrix.
        peak_bw_tbps: Peak HBM bandwidth in TB/s.
    """
    with gzip.open(trace_path, 'rt') as f:
        trace = json.load(f)
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']

    for pattern, (rows, cols, dtype_bytes) in weight_configs.items():
        matching = [k for k in kernels if pattern in k['name']]
        if not matching:
            print(f"No kernels matching '{pattern}'")
            continue

        data_bytes = rows * cols * dtype_bytes
        mean_dur = sum(k['dur'] for k in matching) / len(matching)
        achieved_bw = data_bytes / (mean_dur * 1e-6) / 1e12  # TB/s
        util = achieved_bw / peak_bw_tbps * 100

        grid = matching[0].get('args', {}).get(
            'grid', matching[0].get('grid', '?'))
        print(f"{pattern}:")
        print(f"  Weight: [{rows}, {cols}] x {dtype_bytes}B = "
              f"{data_bytes/1e6:.1f} MB")
        print(f"  Mean duration: {mean_dur:.1f} us "
              f"({len(matching)} instances)")
        print(f"  Achieved BW: {achieved_bw:.2f} TB/s = "
              f"{util:.1f}% of {peak_bw_tbps} TB/s peak")
        print(f"  Grid: {grid}")

# Example usage:
# gemv_bandwidth(
#     "torch_profile/baseline_bs1/dp0_pp0_tp0_*.pt.trace.json.gz",
#     {"gemv2N": (4096, 512, 2)},  # Router gate: [hidden, num_experts] BF16
#     peak_bw_tbps=8.0,
# )
```

### Recipe 5: Identify single-SM kernels (grid=[1,1,1])

```python
import gzip, json
from collections import defaultdict

def find_single_sm_kernels(trace_path):
    """Find kernels that launch on a single SM (grid=[1,1,1])."""
    with gzip.open(trace_path, 'rt') as f:
        trace = json.load(f)
    kernels = [e for e in trace['traceEvents'] if e.get('cat') == 'kernel']

    single_sm = defaultdict(lambda: {'total': 0.0, 'count': 0})
    for k in kernels:
        grid = k.get('args', {}).get('grid', k.get('grid'))
        if grid and grid == [1, 1, 1]:
            name = k['name']
            single_sm[name]['total'] += k['dur']
            single_sm[name]['count'] += 1

    if not single_sm:
        print("No single-SM kernels found.")
        return

    total_gpu = sum(k['dur'] for k in kernels)
    ranked = sorted(single_sm.items(), key=lambda x: -x[1]['total'])

    print(f"{'Kernel':<60} {'Total(us)':>10} {'Count':>6} {'f_decode':>8}")
    print("-" * 88)
    for name, data in ranked:
        f_decode = data['total'] / total_gpu
        short = name[:58] + '..' if len(name) > 60 else name
        print(f"{short:<60} {data['total']:>10.1f} {data['count']:>6} "
              f"{f_decode:>7.1%}")

    total_single = sum(d['total'] for d in single_sm.values())
    print(f"\nTotal single-SM time: {total_single:.0f} us "
          f"({total_single/total_gpu*100:.1f}% of decode step)")
```

---

## References

- PyTorch Profiler docs: https://pytorch.org/docs/stable/profiler.html
- Chrome Trace Format: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
- CUPTI Activity API: https://docs.nvidia.com/cupti/modules.html
- Companion guide (nsys): `nsys-profiling-guide.md`
- Tiered profiling decision tree: `nsys-profiling-guide.md` §3.10
