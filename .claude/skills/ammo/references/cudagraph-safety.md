# CUDA Graphs Safety Checklist (Custom CUDA Ops)

A common failure mode in MoE optimization work is a custom CUDA op that:
- works in eager mode, but
- fails during CUDA graph capture, causes graph breaks, or silently regresses latency under graphs.

Use this checklist for *every* new custom op on the hot path.

## 1) Stream correctness (must)

**Rule:** launch on the **current PyTorch CUDA stream** for the current device.

In C++/CUDA extensions, prefer:

```cpp
#include <ATen/cuda/CUDAContext.h>
cudaStream_t stream = at::cuda::getDefaultCUDAStream();   // usually NOT what you want
cudaStream_t stream = at::cuda::getCurrentCUDAStream();   // what you want for graph capture
```

Rationale:
- vLLM / PyTorch may run work on non-default streams (especially with graphs).
- Launching on the default stream can introduce hidden sync and/or break capture assumptions.

## 2) Device + stream guards (must)

- Use `at::cuda::CUDAGuard` (or equivalent) to set the correct device before launching.
- Do not call `cudaDeviceSynchronize()` or introduce any host-side synchronization in code that can run during capture.
- If you must synchronize for debugging, gate it behind a debug env var and keep it off by default.

## 3) Allocations + shape stability (must)

- **No allocations during capture**:
  - Avoid `at::empty`, `new`, `malloc`, or creating temporary tensors inside the captured region.
  - Prefer: explicit workspaces that are preallocated and reused.
- **Stable shapes per bucket**:
  - CUDA graphs require stable shapes for captured paths.
  - Ensure the op sees consistent shapes within a bucket (same hidden dims, same top_k, same quant format, same workspace sizes).
- If your kernel uses dynamic shared memory, keep the dynamic SMEM size constant within a bucket.

## 4) Error visibility (debug toolkit)

When capture fails or output is wrong:
- `CUDA_LAUNCH_BLOCKING=1` (slow, but surfaces the true failing op)
- `TORCH_SHOW_CPP_STACKTRACES=1` (better call stacks)

## 5) Minimum verification procedure

1. Eager correctness test (unit test or small harness).
2. The *same* correctness test under CUDA graphs (or within `vllm bench latency` with graphs enabled).
3. Nsight Systems trace:
   - Confirm your op is captured (no graph breaks).
   - Confirm there are no unexpected gaps, memcpys, or sync nodes.

## 6) Common gotchas (quick scan)

- Default-stream launches in a graph-captured workload.
- Hidden device mismatch in multi-GPU (TP/EP) setups.
- Implicit temporary allocations during capture.
- Divergent control flow by bucket that changes workspace or dynamic SMEM requirements.
