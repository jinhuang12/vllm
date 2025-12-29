# CUDA Graphs Safety Checklist (Custom CUDA Ops)

This is a high‑frequency failure mode in MoE optimization work: a kernel “works” in eager but fails (or silently slows down) under CUDA graphs.

## Required: stream correctness

- Launch kernels on the **current PyTorch stream**, not the CUDA default stream.
  - In C++/CUDA extensions: use `at::cuda::getDefaultCUDAStream()` only if you have explicitly switched; otherwise use `at::cuda::getDefaultCUDAStream()` is wrong for graph capture in typical vLLM flows.
  - Prefer: `cudaStream_t stream = at::cuda::getDefaultCUDAStream()` only when PyTorch is actually running on that stream; otherwise use `at::cuda::getDefaultCUDAStream()` is misleading.
  - Practical rule: use `at::cuda::getDefaultCUDAStream()` **never** for vLLM; use `at::cuda::getCurrentCUDAStream()` instead.

## Required: device + stream guards

- Use `at::cuda::CUDAGuard` (or equivalent) to ensure the right device is set before launching.
- Do not call `cudaDeviceSynchronize()` or introduce host syncs inside code that may run during capture.

## Required: allocations and shape stability

- Avoid allocating temporary tensors inside the capture region.
  - Prefer: preallocate scratch buffers (workspace) and reuse.
- Ensure capture uses stable shapes:
  - Same batch buckets, same hidden dims, same top_k, same quant format.
  - If your kernel has dynamic shared memory size, keep it constant within a bucket.

## Required: error visibility

For debugging capture failures:
- Set `CUDA_LAUNCH_BLOCKING=1` to surface the true failing op (slow, but clarifies).
- Use `TORCH_SHOW_CPP_STACKTRACES=1` for better call stacks.

## Verification procedure (minimum)

1. Run eager correctness test.
2. Run the same correctness test with CUDA graphs enabled (or within `vllm bench latency`).
3. Nsight Systems trace: confirm your op is captured and not causing graph breaks.

## Common gotchas

- Accidentally using the default stream in a custom op: passes eager, crashes or misbehaves in capture.
- Hidden device mismatch (multi‑GPU/TP): kernel launches on a different device than inputs.
- Implicit allocations during capture (e.g., creating small tensors, calling `at::empty`).
- Divergent control flow by batch size bucket that changes dynamic SMEM requirements.

