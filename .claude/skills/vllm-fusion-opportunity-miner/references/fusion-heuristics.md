# Fusion heuristics (first principles) for vLLM

## Table of contents
1. What “fusion opportunity” means
2. First-principles scoring
3. High-ROI fusion patterns in LLM inference
4. Detecting candidates from profiler outputs
5. Feasibility checklist (graphs/compile/multi-GPU)
6. Stop-ship list

## 1) What “fusion opportunity” means

A fusion opportunity exists when:
- adjacent kernels operate on the same intermediate(s), and
- intermediates are written to and read from global memory, and
- combining kernels plausibly reduces memory traffic and/or improves locality.

Launch overhead matters less under CUDA graphs, so focus on:
- removing memory round-trips
- enabling better epilogues / better layout feeding heavy kernels

## 2) First-principles scoring

Use:
- **Time share**: `candidate_gpu_time / total_gpu_time` in the target regime
- **Plausible savings** (conservative):
  - removing intermediate read+write
  - eliminating pure pointwise/conversion micro-kernels
  - avoiding redundant quant/dequant passes
- **Feasibility gates**:
  - shapes stable under capture buckets
  - determinism requirements satisfied
  - no fusion across collectives or hard sync boundaries
  - numerical behavior matchable (tolerance declared)

If share is tiny, deprioritize: a 1% share kernel cannot buy a 5% e2e win without unrealistic speedups.

## 3) High-ROI fusion patterns in LLM inference

Common winners:
1) GEMM epilogues: bias + activation + (optional) quant/dequant + residual add
2) Attention plumbing: rotary/scale/layout transforms adjacent to attention kernels
3) KV-cache update paths: reshape/transpose/store chains
4) Norm + residual patterns: RMSNorm/LayerNorm fused with residual + scale/bias
5) Quantization glue: packing/unpacking kernels adjacent to GEMMs

Common losers:
- fusing across a large library kernel when it forces re-implementing the library kernel without a clear replacement plan

## 4) Detecting candidates from profiler outputs

From kernel summaries:
- Identify top kernels (≤20) by total GPU time.
- Group by class: heavy kernels (GEMM/attention), micro-kernels, memcopy/memset, NCCL/collectives.

From repeated chains:
- Look for repeated adjacent chains where:
  - each kernel is small/memory-bound
  - the chain repeats per layer/token
  - total chain time is material

## 5) Feasibility checklist

Before proposing fusion, answer (with evidence):
- Where in code is this kernel/chain emitted?
- Is it Python torch ops, Triton, or a CUDA extension?
- Can you fuse without changing semantics?
- Is it capture-safe under CUDA graphs?
- Will `torch.compile` change the op sequence (and does the plan still apply)?
- For multi-GPU: is it near collectives or sync boundaries?

## 6) Stop-ship list

Do not claim a “fusion win” if:
- baseline vs variant differ in graphs/compile settings
- traces show graph breaks or different capture coverage
- numerical differences exceed declared tolerances or determinism changes user-visible outputs
- “win” is only CUDA API time with graphs enabled (kernel time unchanged)

