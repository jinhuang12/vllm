# Nsight Systems / Nsight Compute Profiling Guide (Generic)

Use this guide to capture comparable profiling evidence for any NVIDIA GPU-heavy project.

## Goal

- use Nsight Systems (`nsys`) for system timeline and bottleneck ranking
- use Nsight Compute (`ncu`) for targeted kernel root-cause analysis

## Workflow

1. Capture baseline with production parity settings.
2. Export kernel summary and timeline artifacts.
3. Rank hotspots by total time and frequency.
4. Use focused ncu passes on top hotspots only.

## Suggested nsys baseline command pattern

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  -o {artifact_dir}/profiles/baseline \
  <your_baseline_command>
```

## Useful nsys exports

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  --output {artifact_dir}/profiles/baseline_cuda_gpu_kern_sum \
  {artifact_dir}/profiles/baseline.nsys-rep

nsys stats --report cuda_gpu_trace --format csv \
  --output {artifact_dir}/profiles/baseline_cuda_gpu_trace \
  {artifact_dir}/profiles/baseline.nsys-rep
```

## Suggested ncu focused command pattern

```bash
ncu \
  --set basic \
  --target-processes all \
  --launch-count 5 \
  -o {artifact_dir}/profiles/hot_kernel_basic \
  <your_candidate_command>
```

## Reporting requirements

For each candidate include:
- trace paths
- top-kernel summary
- hotspot attribution
- interpretation: launch-bound, memory-bound, compute-bound, or sync-bound

## Anti-patterns

- profiling non-parity workloads and using results for ship decisions
- optimizing kernels not present in top-ranked hotspots
- using ncu as E2E proof (it is attribution, not end-to-end validation)
