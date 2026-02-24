# Bottleneck Mining Guide (Stage 2)

Systematic identification of optimization opportunities from nsys profiling data.

## Search Anchors

bottleneck mining, nsys sqlite, kernel ranking, fusion candidates, opportunity scoring, kernel chains

## Overview

Stage 2 takes the nsys profile from Stage 1 and systematically identifies:
1. Which kernels consume the most GPU time
2. Which kernel chains repeat and could be fused
3. Which optimizations are feasible given hardware constraints
4. A ranked list of opportunities for Stage 3 planning

## Step 1: Export nsys traces to SQLite

```bash
nsys export --type sqlite -o {artifact_dir}/runs/baseline_sqlite {artifact_dir}/runs/baseline.nsys-rep
```

This creates a SQLite database with tables for GPU kernels, API calls, memory operations, etc.

## Step 2: Extract top-K kernels by GPU time

```sql
-- Query CUDA_GPU_KERN_SUM for top kernels
SELECT
    demangledName as kernel_name,
    count(*) as invocations,
    sum(end - start) / 1e3 as total_us,
    avg(end - start) / 1e3 as avg_us,
    max(end - start) / 1e3 as max_us
FROM CUPTI_ACTIVITY_KIND_KERNEL
GROUP BY demangledName
ORDER BY total_us DESC
LIMIT 20;
```

Alternatively, use the nsys CLI:
```bash
nsys stats --report cuda_gpu_kern_sum {artifact_dir}/runs/baseline.nsys-rep
```

Record the top-10 kernels by total GPU time in `{artifact_dir}/bottleneck_analysis.md`.

## Step 3: Identify repeated kernel chains (fusion candidates)

Look for sequences of kernels that always appear together:

```sql
-- Find kernel sequences (simplified; adjust window)
SELECT
    k1.demangledName as kernel_1,
    k2.demangledName as kernel_2,
    count(*) as chain_count,
    avg(k2.start - k1.end) / 1e3 as avg_gap_us
FROM CUPTI_ACTIVITY_KIND_KERNEL k1
JOIN CUPTI_ACTIVITY_KIND_KERNEL k2
    ON k2.start > k1.end
    AND k2.start - k1.end < 50000  -- 50us gap threshold
    AND k2.correlationId = k1.correlationId + 1
GROUP BY k1.demangledName, k2.demangledName
HAVING chain_count > 5
ORDER BY chain_count DESC;
```

Fusion candidates are chains where:
- The intermediate data between kernels is large (potential memory savings)
- The gap between kernels is small (launch overhead opportunity)
- The chain repeats many times per forward pass

## Step 4: Map kernel names back to vLLM code paths

For each top kernel, identify its source in vLLM:

1. **Search by kernel name**: `rg "kernel_name_fragment" vllm/`
2. **Search by op registration**: `rg "torch.ops.*kernel_name" vllm/`
3. **Check Triton kernels**: `rg "@triton.jit" vllm/ | grep -i "kernel_name"`
4. **Check CUDA extensions**: `rg "kernel_name" csrc/`

Document the mapping: kernel name -> vLLM file -> function -> call site.

## Step 5: Compute feasibility bounds

For each fusion candidate, compute the theoretical maximum savings:

```
bytes_saved = 2 * intermediate_tensor_size  (store + load removed)
time_saved_max_us = bytes_saved / BW_eff * 1e6

where BW_eff ~ 1 TB/s (conservative Hopper-class back-of-envelope)
```

See `references/fusion-feasibility-heuristics.md` for detailed heuristics.

## Step 6: Rank opportunities by (Impact, Feasibility) score

Use a consistent scoring rubric:

| Score | Impact (0-5) | Feasibility (0-5) |
|-------|-------------|-------------------|
| 5 | >20% of target kernel time | Drop-in replacement, no API changes |
| 4 | 10-20% of target kernel time | Minor integration (new dispatch, guards) |
| 3 | 5-10% of target kernel time | Moderate work (new kernel, testing) |
| 2 | 2-5% of target kernel time | Significant effort (architecture changes) |
| 1 | <2% of target kernel time | Major research required |
| 0 | No measurable impact | Infeasible given constraints |

**Priority = Impact + Feasibility** (higher is better)

## Step 7: Use vllm-fusion-opportunity-miner (recommended)

If available, use the `vllm-fusion-opportunity-miner` skill for automated mining:

```bash
FUSION_MINER_SKILL_DIR="<path-to-vllm-fusion-opportunity-miner>"
python3 "${FUSION_MINER_SKILL_DIR}/scripts/nsys_mine.py" \
  --sqlite {artifact_dir}/runs/baseline_sqlite.sqlite \
  --out-dir {artifact_dir}
```

This produces structured mining outputs that can directly populate the Stage 3 opportunity table.

## Output Format

Write results to `{artifact_dir}/bottleneck_analysis.md` with:

1. **Top-K kernels table** (by total GPU time)
2. **Kernel chain analysis** (repeated sequences, fusion candidates)
3. **Kernel-to-code mapping** (kernel name -> vLLM source location)
4. **Feasibility bounds** (bytes saved, time saved upper bound)
5. **Ranked opportunity list** (sorted by Priority score)

This output feeds directly into Stage 3 (optimization planning).
