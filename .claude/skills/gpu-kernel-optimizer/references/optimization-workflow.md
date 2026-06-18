# GPU Kernel Optimization Workflow

Use this checklist when improving a GPU kernel used for LLM inference.

## Stage 1: Baseline & isolate

- Reproduce the issue with a minimal script:
  - isolate one operator if possible (attention / GEMM / sampling / etc.)
  - pin shapes (batch, seq_len, heads, head_dim, hidden)
- Measure **latency and throughput** with enough warmup + iterations.
- Profile once to classify the bottleneck:
  - **launch-bound** (many tiny kernels / high CPU CUDA API time)
  - **memory-bound** (high DRAM / low math utilization)
  - **compute-bound** (high math utilization / low memory pressure)
- Record the baseline:
  - GPU model + driver/CUDA version
  - dtype/quant details
  - command line + env vars
  - baseline numbers

## Stage 2: Retrieval (find reference implementations)

- Use `references/corpus-registry.md` to select corpora.
- Use the corpus index to find:
  - similar kernel family (decode vs prefill differs!)
  - same KV layout / paging strategy
  - same dtype/quant scheme
  - same arch specialization (sm80/sm90/sm100/etc.)

Extract 2–3 concrete transferable ideas:
- tiling shape / warp layout
- fusion boundary / epilogue structure
- shared-memory staging strategy
- load-balancing or plan/run split

## Stage 3: Design (write a micro-plan)

Write:
- the hypothesis (“this reduces X bytes moved” / “this removes Y launches”)
- the specific code change(s)
- how you will test correctness
- how you will measure performance
- kill criteria (when to revert)

## Stage 4: Implementation

- Keep diffs small and readable.
- Add guards for:
  - SM architecture
  - dtype / alignment
  - shape constraints
- Prefer compile-time specialization only when it wins measurably.

## Stage 5: Correctness validation

At minimum:
- compare against a reference implementation (PyTorch / naive kernel)
- test multiple shapes including edge cases
- validate numerics (max/mean error; attention lse if applicable)

For inference-critical kernels, also:
- test under CUDA graphs if used in production
- test with realistic strides/layouts (not just contiguous tensors)

## Stage 6: Performance validation

- Re-run the baseline benchmark and compare:
  - end-to-end latency (if possible)
  - kernel-level time
  - launch count (if launch-bound)
- If using Nsight Compute, sanity check:
  - memory throughput changes match your hypothesis
  - occupancy / register pressure didn’t regress unexpectedly

## Stage 7: Integration

- Add a dispatch rule that preserves correctness and fallback paths.
- Add regression tests / benchmarks if the repo supports them.
- Document:
  - how to reproduce the benchmark
  - expected speedup range
  - supported GPUs + dtypes
