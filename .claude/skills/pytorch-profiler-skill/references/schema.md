# PyTorch Profiler JSON Schema Reference

## Top-Level Structure

```json
{
  "schemaVersion": 1,
  "deviceProperties": [...],
  "distributedInfo": {...},
  "cupti_version": 26,
  "cuda_runtime_version": 12080,
  "cuda_driver_version": 13000,
  "with_stack": 1,
  "trace_id": "...",
  "displayTimeUnit": "ms",
  "baseTimeNanoseconds": 1759300074000000000,
  "traceEvents": [...]
}
```

## deviceProperties

```json
{
  "id": 0,
  "name": "NVIDIA H100 80GB HBM3",
  "totalGlobalMem": 85899345920,
  "computeMajor": 9,
  "computeMinor": 0,
  "maxThreadsPerBlock": 1024,
  "maxThreadsPerMultiprocessor": 2048,
  "regsPerBlock": 65536,
  "regsPerMultiprocessor": 65536,
  "warpSize": 32,
  "sharedMemPerBlock": 49152,
  "sharedMemPerBlockOptin": 232448,
  "sharedMemPerMultiprocessor": 233472,
  "numSms": 132
}
```

## distributedInfo

```json
{
  "backend": "nccl",
  "rank": 0,
  "world_size": 8,
  "pg_count": 31,
  "pg_config": [
    {
      "pg_name": "0",
      "pg_desc": "default_pg",
      "backend_config": "cuda:nccl",
      "pg_size": 8,
      "ranks": [0, 1, 2, 3, 4, 5, 6, 7]
    }
  ],
  "nccl_version": "2.27.5"
}
```

## traceEvents

All events follow Chrome Trace Format with `ph` (phase) field:
- `"X"` = Complete event (has duration)
- `"B"` = Begin event
- `"E"` = End event
- `"i"` = Instant event

### Common Fields

| Field | Type | Description |
|-------|------|-------------|
| `ph` | string | Event phase (X, B, E, i) |
| `cat` | string | Category (kernel, cpu_op, etc.) |
| `name` | string | Event name |
| `pid` | int | Process ID |
| `tid` | int | Thread ID |
| `ts` | float | Timestamp in microseconds |
| `dur` | float | Duration in microseconds (for ph=X) |
| `args` | object | Category-specific arguments |

---

## Event Categories

### kernel

GPU kernel execution events.

```json
{
  "ph": "X",
  "cat": "kernel",
  "name": "void flash::flash_fwd_kernel<...>",
  "pid": 0,
  "tid": 7,
  "ts": 5387313688269.031,
  "dur": 125.6,
  "args": {
    "queued": 0,
    "device": 0,
    "context": 1,
    "stream": 7,
    "correlation": 125,
    "registers per thread": 128,
    "shared memory": 65536,
    "blocks per SM": 2.0,
    "warps per SM": 8.0,
    "grid": [128, 1, 1],
    "block": [128, 1, 1],
    "est. achieved occupancy %": 50
  }
}
```

**Key args:**
- `correlation`: Links to cuda_runtime event that launched this kernel
- `grid`, `block`: CUDA grid/block dimensions
- `registers per thread`: Register usage
- `shared memory`: Shared memory in bytes
- `est. achieved occupancy %`: SM occupancy percentage

### cpu_op

PyTorch operator events (aten::, vllm::, custom ops).

```json
{
  "ph": "X",
  "cat": "cpu_op",
  "name": "aten::mm",
  "pid": 2501128,
  "tid": 2501128,
  "ts": 5387313686629.708,
  "dur": 45.2,
  "args": {
    "External id": 13,
    "Record function id": 0,
    "Ev Idx": 12
  }
}
```

**Key args:**
- `External id`: Links to cuda_runtime events launched by this op

### python_function

Python call stack events (when `with_stack=True`).

```json
{
  "ph": "X",
  "cat": "python_function",
  "name": "vllm/model_executor/models/llama.py(245): forward",
  "pid": 2501128,
  "tid": 2501128,
  "ts": 5387313589433.432,
  "dur": 2465844.731,
  "args": {
    "Python parent id": 42,
    "Python id": 43,
    "Ev Idx": 21509
  }
}
```

**Key args:**
- `Python id`: Unique ID for this function call
- `Python parent id`: ID of calling function (null for root)

**Building call stack:**
```python
def get_call_stack(events, python_id):
    id_to_event = {e['args']['Python id']: e for e in events if e['cat'] == 'python_function'}
    stack = []
    current = id_to_event.get(python_id)
    while current:
        stack.append(current['name'])
        parent_id = current['args'].get('Python parent id')
        current = id_to_event.get(parent_id)
    return stack[::-1]  # Root first
```

### cuda_runtime

CUDA API calls.

```json
{
  "ph": "X",
  "cat": "cuda_runtime",
  "name": "cudaLaunchKernel",
  "pid": 2501128,
  "tid": 2501128,
  "ts": 5387313688237.449,
  "dur": 12.5,
  "args": {
    "External id": 85,
    "correlation": 125
  }
}
```

**Key args:**
- `correlation`: Links to kernel event this call launched
- `External id`: Links to cpu_op that triggered this call

**Common cuda_runtime names:**
- `cudaLaunchKernel`: Direct kernel launch
- `cudaGraphLaunch`: CUDA Graph replay
- `cudaMemcpyAsync`: Async memory copy
- `cudaStreamSynchronize`: Stream sync
- `cudaStreamBeginCapture`/`cudaStreamEndCapture`: Graph capture

### user_annotation

`torch.profiler.record_function()` regions.

```json
{
  "ph": "X",
  "cat": "user_annotation",
  "name": "## Call CompiledFxGraph abc123 ##",
  "pid": 2501128,
  "tid": 2501128,
  "ts": 5387313707025.251,
  "dur": 948595.408,
  "args": {
    "External id": 1,
    "Record function id": 0,
    "Ev Idx": 0
  }
}
```

**Common patterns:**
- `execute_new_*_cached_0`: First execution (graph capture)
- `execute_new_*_cached_1`: Subsequent executions (graph replay)
- `## Call CompiledFxGraph <hash> ##`: Inductor compiled graph

### gpu_user_annotation

GPU-side annotations (NCCL collectives).

```json
{
  "ph": "X",
  "cat": "gpu_user_annotation",
  "name": "nccl:_all_gather_base",
  "pid": 0,
  "tid": 7,
  "ts": 5387316049276.842,
  "dur": 72.354,
  "args": {
    "External id": 21462
  }
}
```

### ac2g

Async CPU-to-GPU launch correlation.

```json
{
  "ph": "s",
  "cat": "ac2g",
  "name": "ac2g",
  "pid": 2501128,
  "tid": 2501128,
  "ts": 5387313688250.0,
  "id": 125,
  "args": {}
}
```

---

## Correlation Chains

### Kernel → Source Code

```
kernel.args.correlation = X
    ↓
cuda_runtime (correlation=X).args.External id = Y
    ↓
cpu_op (External id=Y)
    ↓ (by timestamp overlap)
python_function (ts ≤ cpu_op.ts ≤ ts+dur)
```

### Inductor File → Kernels

```
python_function.name contains "/tmp/torchinductor_"
    ↓ (by timestamp overlap)
cpu_op (ts within python_function time range)
    ↓ (filter by name)
triton_* or _w8a8_* kernels
```

---

## Kernel Name Patterns

### Inductor-Generated Triton Kernels

Format: `triton_{type}_fused_{ops}_{variant}`

**Types:**
- `poi` = Pointwise (elementwise ops)
- `red` = Reduction (sum, mean, norm)
- `per` = Persistent (keeps data in registers across iterations)
- `tem` = Template-based

**Fused ops decode:**
```
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0
                 └─────────────────────────────┘
                 _to_copy + add + mean + mul + pow + rsqrt = RMSNorm
```

### vLLM Custom Kernels

| Pattern | Description |
|---------|-------------|
| `fused_moe_kernel` | MoE expert computation |
| `_w8a8_triton_block_scaled_mm` | W8A8 quantized matmul |
| `reshape_and_cache_flash_kernel` | KV cache update |
| `topkGatingSoftmax` | MoE top-k routing |
| `moe_align_block_size_kernel` | MoE token padding |

### NCCL Collectives

| Pattern | Description |
|---------|-------------|
| `ncclDevKernel_AllReduce_*` | Tensor parallel sync |
| `ncclDevKernel_AllGather_*` | Gather distributed tensors |
| `ncclDevKernel_ReduceScatter_*` | Reduce + scatter |

### Flash Attention

| Pattern | Description |
|---------|-------------|
| `flash_fwd_kernel` | Prefill attention |
| `flash_fwd_splitkv_kernel` | Decode attention (split KV) |
| `flash_fwd_splitkv_combine_kernel` | Combine split attention outputs |

---

## Common SQL Queries

### Top kernels by time

```sql
SELECT name, COUNT(*) as cnt, SUM(dur)/1e3 as total_ms, AVG(dur) as avg_us
FROM events WHERE cat='kernel'
GROUP BY name ORDER BY total_ms DESC LIMIT 20;
```

### Time breakdown by category

```sql
SELECT 
  CASE 
    WHEN name LIKE '%nccl%AllReduce%' THEN 'NCCL AllReduce'
    WHEN name LIKE '%fused_moe%' THEN 'MoE'
    WHEN name LIKE '%flash%' THEN 'Attention'
    WHEN name LIKE '%triton%' THEN 'Triton Other'
    ELSE 'Other'
  END as category,
  SUM(dur)/1e6 as total_s
FROM events WHERE cat='kernel'
GROUP BY category ORDER BY total_s DESC;
```

### Find small kernels (fusion candidates)

```sql
SELECT name, COUNT(*) as cnt, AVG(dur) as avg_us
FROM events WHERE cat='kernel' AND dur < 10
GROUP BY name ORDER BY cnt DESC LIMIT 20;
```

### Trace kernel to Python source

```sql
-- Find cuda_runtime for a kernel
SELECT cr.* FROM events k
JOIN events cr ON k.correlation = cr.correlation AND cr.cat='cuda_runtime'
WHERE k.cat='kernel' AND k.name LIKE '%flash_fwd%'
LIMIT 1;

-- Find python_function active during launch
SELECT pf.name FROM events pf
WHERE pf.cat='python_function'
  AND pf.ts <= 5387313688237.449  -- cr.ts from above
  AND pf.ts + pf.dur >= 5387313688237.449
  AND pf.name LIKE '%vllm%'
ORDER BY pf.dur ASC  -- Innermost first
LIMIT 5;
```

### Find Inductor files

```sql
SELECT DISTINCT 
  SUBSTR(name, INSTR(name, 'torchinductor_'), 60) as inductor_file
FROM events 
WHERE cat='python_function' AND name LIKE '%torchinductor_%';
```
