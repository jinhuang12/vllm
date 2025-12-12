---
name: pytorch-profiler
description: Parse, analyze, and extract insights from PyTorch Profiler JSON traces (Chrome trace format). Use when working with .json or .json.gz profiler outputs from torch.profiler, analyzing GPU kernel performance, tracing Inductor-generated code to kernels, building call hierarchies, or summarizing vLLM/inference workloads. Handles compressed traces efficiently via streaming.
---

# PyTorch Profiler Trace Analysis

## Quick Start

Load a trace (supports .json and .json.gz):

```python
from scripts.pt_trace import PTTrace
trace = PTTrace("/path/to/trace.json.gz")

# Executive summary
trace.summary()

# Top GPU kernels by time
trace.top_kernels(n=20)

# Time breakdown by category
trace.breakdown()
```

## Core Concepts

### Event Categories

PyTorch profiler traces contain events in these categories:

| Category | Description | Key Fields |
|----------|-------------|------------|
| `kernel` | GPU kernel executions | `name`, `dur`, `args.correlation`, `args.grid`, `args.block` |
| `cpu_op` | PyTorch operators (aten::, vllm::, etc.) | `name`, `dur`, `args.External id` |
| `python_function` | Python call stack | `name`, `args.Python id`, `args.Python parent id` |
| `cuda_runtime` | CUDA API calls | `name`, `args.correlation`, `args.External id` |
| `user_annotation` | torch.profiler.record_function regions | `name`, `dur` |
| `gpu_user_annotation` | GPU-side annotations (NCCL, etc.) | `name`, `dur` |

### Correlation Chain

To trace a GPU kernel back to its source:

```
GPU kernel (correlation=X)
  ↓ args.correlation
cuda_runtime event (correlation=X, External id=Y)
  ↓ args.External id
cpu_op event (External id=Y)
  ↓ timestamp overlap
python_function event (innermost by duration)
```

### Inductor File Mapping

Torch Inductor generates files in `/tmp/torchinductor_*/`. Map them to kernels:

```python
# Get Inductor file → kernel mapping
trace.inductor_mapping()

# Trace a specific kernel back to source
trace.trace_kernel("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0")
```

## Common Workflows

### 1. Identify Bottlenecks

```python
# Categorical breakdown
breakdown = trace.breakdown()
# Returns: {'NCCL AllReduce': 363.4, 'MoE Kernels': 298.0, ...}

# Find optimization targets
trace.top_kernels(n=10, sort_by='total')  # Most total time
trace.top_kernels(n=10, sort_by='count')  # Most invocations (fusion candidates)
trace.small_kernels(threshold_us=10)       # Tiny kernels to fuse
```

### 2. Understand Model Structure

```python
# Infer layers from kernel counts
trace.infer_structure()
# Returns: {'attention_layers': 64, 'moe_layers': 64, 'tp_degree': 4}

# Get execution phases
trace.phases()  # Returns user_annotation regions with timing
```

### 3. Trace Inductor Fusions

```python
# What ops did Inductor fuse into each Triton kernel?
trace.decode_fusion("triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0")
# Returns: ['_to_copy', 'add', 'mean', 'mul', 'pow', 'rsqrt'] - RMSNorm!

# Full Inductor file analysis
for file, kernels in trace.inductor_mapping().items():
    print(f"{file}: {kernels}")
```

### 4. CUDA Graph Analysis

When CUDA Graphs are used, kernels launch via `cudaGraphLaunch`:

```python
# Check if trace uses CUDA Graphs
trace.uses_cuda_graphs()  # True/False

# Get graph capture vs replay phases
trace.graph_phases()
```

## Script Reference

### `scripts/pt_trace.py`

Main analysis class. Run directly for CLI usage:

```bash
python scripts/pt_trace.py /path/to/trace.json.gz --summary
python scripts/pt_trace.py /path/to/trace.json.gz --top-kernels 20
python scripts/pt_trace.py /path/to/trace.json.gz --breakdown
python scripts/pt_trace.py /path/to/trace.json.gz --inductor
```

### `scripts/trace_to_sqlite.py`

Convert trace to SQLite for complex queries:

```bash
python scripts/trace_to_sqlite.py /path/to/trace.json.gz -o trace.db
```

Then query directly:

```sql
-- Top kernels
SELECT name, COUNT(*) as cnt, SUM(dur)/1e3 as total_ms
FROM events WHERE cat='kernel'
GROUP BY name ORDER BY total_ms DESC LIMIT 20;

-- Inductor files active during kernel launch
SELECT DISTINCT pf.name FROM events k
JOIN events cr ON k.correlation = cr.correlation AND cr.cat='cuda_runtime'
JOIN events pf ON pf.cat='python_function' 
  AND pf.ts <= cr.ts AND cr.ts <= pf.ts + pf.dur
  AND pf.name LIKE '%torchinductor%'
WHERE k.cat='kernel' AND k.name LIKE 'triton%';
```

## Kernel Name Patterns

| Pattern | Meaning |
|---------|---------|
| `triton_poi_fused_*` | Pointwise fused kernel (elementwise) |
| `triton_red_fused_*` | Reduction fused kernel (sum, mean, norm) |
| `triton_per_fused_*` | Persistent reduction kernel |
| `triton_tem_fused_*` | Template-based kernel |
| `_w8a8_*` | W8A8 quantized matmul |
| `fused_moe_kernel` | vLLM MoE dispatch + compute |
| `flash_fwd_*` | Flash Attention forward |
| `flash_fwd_splitkv_*` | Flash Attention decode (split KV) |
| `ncclDevKernel_*` | NCCL collective operations |

## Device Info Access

```python
trace.devices()  # List of GPU info dicts
trace.distributed_info()  # {'backend': 'nccl', 'rank': 0, 'world_size': 4, ...}
```

## Performance Tips

1. **Large traces**: Use `trace_to_sqlite.py` for traces >500MB - SQL queries are faster than repeated Python iteration
2. **Streaming**: The loader streams gzipped files, but still needs to parse full JSON
3. **Filtering**: Use category filters early: `trace.events(cat='kernel')` not `[e for e in trace.all_events() if e['cat']=='kernel']`
