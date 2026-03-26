# Debate Rules

Agent-facing rules for the Stage 3 adversarial debate. Champions must follow these rules during proposal generation, micro-experiments, and debate rounds.

## Evidence Tiers

Every claim in a proposal or argument requires evidence. The type of evidence required depends on the claim being made:

| Tier | Claim Type | Examples | Required Artifact | Feasibility Cap |
|------|-----------|----------|-------------------|-----------------|
| **Tier 1 — Analysis** | Theoretical bounds | Roofline calc, Amdahl projection, working-set analysis, ISA inspection | `.py` script using only `import math`/`numpy` — no GPU calls | **3/10** |
| **Tier 2 — Kernel execution** | Kernel speedup numbers | "Measured 1.34x at BS=8", kernel timing claims | `.py` script with `torch.cuda` calls + `.log` with GPU device name on line 1 (`torch.cuda.get_device_name()`) and `torch.cuda.Event` timing output | **7/10** |
| **Tier 3 — Hardware profiling** | Hardware utilization metrics | "85% occupancy", "400 GB/s achieved BW", register count | ncu CSV or nsys stats export with GPU hardware fingerprint | No cap |

**Rules**:
- Claiming a specific kernel speedup NUMBER (e.g., "1.5x faster") requires **Tier 2 or higher**. A roofline calculation showing "up to 2x theoretical" is Tier 1 — acceptable as a bound, but feasibility capped.
- Claiming specific hardware utilization metrics (occupancy %, achieved BW, register count) requires **Tier 3**. If a metric is cited, it must come from ncu/nsys measurement, not a roofline estimate.
- The `.log` file is the proof of execution. Missing log = Tier 1 regardless of script contents.
- Tier 1 is valid for architectural insight proposals (cache regime analysis, working-set estimation). These can advance but are scored conservatively.
- Strongly prefer providing Tier 3 level evidence. Running `ncu` on your baseline kernel (~60s) preempts all 4 NCU Triggers and unlocks uncapped Tier 3 scoring.

## Micro-Experiment Rules

### Allowed

| Experiment Type | Constraint |
|----------------|------------|
| Roofline calculations | Pure arithmetic, no GPU required |
| ISA inspection | `cuobjdump`, `ncu` |
| Tiny kernel prototypes | <100 lines of code, <2 min wall-clock execution |
| nsys/ncu single-kernel traces | One kernel invocation, existing binary only |
| Memory layout analysis | Static analysis of tensor shapes and strides |
| Kernel-level benchmarks | **MUST use CUDA graph capture** for both baseline and candidate kernels. Raw CUDA event timing without graph capture is INVALID for kernel speedup claims. |

### Forbidden

| Experiment Type | Reason |
|----------------|--------|
| Full-model benchmarks | Too slow, belongs in Stage 5 |
| vLLM source modifications | Belongs in Stage 4 |
| Model weight downloads | Too slow, too large |
| Any experiment >2 min | Blocks debate progress |
| Kernel benchmarks without CUDA graph capture | Inflates/deflates results due to launch overhead asymmetry |

## Cache-Sensitivity Requirements

For kernels identified as bandwidth-bound (AI < breakeven threshold), micro-experiments must report:
1. **Loop-warmed time**: 100+ iterations on same tensors
2. **Cold-cache time**: Single iteration after L2 flush or fresh random tensors

If the warm/cold ratio exceeds 1.5x, the speedup is cache-dependent. Use the cold-cache result for E2E projections and flag this in the proposal's feasibility math.

## Fusion-Specific Cache Testing

For proposals that fuse multiple kernels into one, the above cache requirements are necessary but not sufficient:

1. **Pipeline working set check**: Estimate total per-iteration working set (num_layers x per_layer_state). If this exceeds 2x the GPU's L2 cache, isolated benchmarks on small tensors overstate the fused kernel's benefit.
2. **L2-busting methodology**: Test the fused kernel with chained distinct data totaling > 2.5x L2 cache size, forcing DRAM streaming. This simulates production L2 competition.
3. **Report both**: Report speedup under (a) isolated warm-cache and (b) L2-busted cold conditions. If (a)/(b) > 1.5x, the E2E estimate MUST use the cold-cache speedup.

## Pipeline-Level Simulation

For proposals that replace or modify kernels invoked per-layer (e.g., GEMM, normalization, activation):

1. **Multi-layer benchmark**: Run the candidate kernel N times sequentially (N = num_layers or min(N, 8)) with distinct input data per invocation, totaling > 2x L2 cache size.
2. **Launch overhead accounting**: For proposals replacing a fused/monolithic kernel with multiple separate kernels, explicitly measure and report the launch overhead (typically 2-5 us per launch under CUDA graphs). The pipeline speedup must account for this overhead.
3. **Report pipeline speedup, not just kernel speedup**: The proposal's E2E projection must use pipeline speedup (including launch overhead and multi-layer L2 effects), not raw kernel speedup from isolated benchmarks.

Proposals that report only isolated kernel speedup for per-layer optimizations have their feasibility score capped at 6/10. See `debate-scoring-rubric.md` for scoring implications.

## Micro-Experiment Artifact Requirements

Every kernel benchmark must produce:

1. A `.py` script in `debate/micro_experiments/` that is independently runnable
2. A `.log` file with: GPU device name on line 1 (`torch.cuda.get_device_name()`), kernel timing in microseconds, iteration count
3. For hardware utilization claims: an ncu CSV or nsys stats export

The `.log` file is the proof of execution. Missing log = Tier 1 (theoretical) regardless of script contents, and feasibility is capped at 3/10.

## Baseline Provenance Rule

The micro-benchmark baseline must invoke the target kernel via the **same code path, API, and tensor layouts** as production. Look up the production dispatch in `bottleneck_analysis.md` Section 6 (Kernel-to-Code Mapping) and replicate it exactly.

**Requirements:**

1. **Dispatch match**: Trace the production call chain from model forward to the GPU kernel. The micro-benchmark must call the same entry point with the same tensor shapes, dtypes, strides, and contiguity. Different APIs, transposed views, or re-contiguified tensors can cause the backend to select a different kernel or tile configuration, silently changing throughput.

2. **Positive kernel identity confirmation**: Run `ncu --metrics launch__grid_size,launch__block_size,dram__bytes.sum.per_second` on the micro-benchmark baseline and report:
   - Kernel name (from ncu output)
   - CUDA launch grid dimensions
   - Achieved DRAM bandwidth

   Cross-reference against Stage 2 nsys trace and ncu sanity check for the same shape. If the kernel name matches but grid dimensions differ, investigate and explain the discrepancy before the proposal advances to scoring. Kernel name alone is insufficient — the same kernel template can be launched with different grid/block configs that produce different performance.

3. **Statement in proposal**: State the exact baseline invocation (API call, tensor layouts, shape) in the proposal's "Micro-Experiment Result" section.

**Rationale**: Backend libraries select internal kernel configurations based on tensor layout and API entry point, not just shape. Two calls that look equivalent (same shape, same dtype) can dispatch different internal paths with 20-30% throughput difference. Without launch-grid cross-referencing, the discrepancy is easily rationalized as measurement noise or environmental overhead.

## NCU Triggers (Mandatory Gates)

ncu profiling is encouraged broadly and **required** when any of these triggers fire. Proposals cannot advance to scoring until the requirement is met.

| # | Trigger Condition | Requirement |
|---|------------------|-------------|
| 1 | Triton kernel + occupancy/register claims | `ncu --metrics l1tex__t_sector_hit_rate.pct,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active` on prototype |
| 2 | Any specific hardware metric cited (BW, occupancy %, SMEM, cache hit rate) | Metric must come from ncu/nsys measurement. Theoretical estimates must be labeled as such. |
| 3 | Roofline assumes BW diverging >2x from Stage 2 nsys-derived BW | ncu verification of actual achieved BW. Corrected value replaces assumption in Amdahl calc. |
| 4 | Micro-experiment baseline BW diverges >10% from Stage 2 ncu/nsys BW | Profile both baselines with `ncu --metrics launch__grid_size,launch__block_size,dram__bytes.sum.per_second`. Unexplained grid divergence → baseline rejected, feasibility capped at 3/10. |

**Governance**: No new mandatory trigger without a documented failure mode from 2+ campaigns.

## Component Dismissal Standard

A component with `f_decode` > 30% cannot be dismissed as "near-optimal" or "not viable" based on a single experiment. To exclude a >30% f_decode component from all proposals:

1. **Two independent negative results required**: Two different champions (or the same champion with two fundamentally different approaches) must independently demonstrate the component is within 10% of its physical ceiling.
2. **No single-experiment dismissal**: If only one champion tested the component and found a negative result, at least one other champion must verify before the component can be excluded.
3. **Framing constraint**: The bottleneck analysis and debate artifacts must not label any component with measured utilization below 85% as "near-optimal." Present the gap as headroom: e.g., "73% BW utilization = 27% headroom."

## References

- `gpu-pool.md` — GPU reservation pattern for micro-experiments
- `gpu-configs.md` — hardware specs for roofline calculations (SMEM budgets, peak BW, peak compute)
- `e2e-delta-math.md` — Amdahl's formula: E2E improvement = f x kernel_speedup
- `debate-scoring-rubric.md` — scoring implications of evidence tiers and feasibility caps
