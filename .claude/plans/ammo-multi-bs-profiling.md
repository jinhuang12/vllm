# AMMO Multi-Batch-Size Profiling: Investigation and Proposal

## Section 1: Root Cause -- Why Was Only BS=8 Profiled?

### Primary Cause: `new_target.py` Default

The single batch size originates from a hardcoded default in the scaffolding script:

**File**: `/home/jinhun/vllm/.claude/skills/ammo/scripts/new_target.py`, line 301

```python
p.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
```

This is the `--batch-sizes` CLI argument for `new_target.py`. When the orchestrator scaffolded the Qwen3.5-4B campaign without explicitly passing `--batch-sizes`, it inherited `[8]` as the sole profiling batch size.

### Propagation Path

The default flows through the following chain:

1. **`new_target.py` line 301**: `default=[8]` sets the initial value.
2. **`new_target.py` line 63-74** (`_default_target_fields`): Wraps it into a `TargetFields` dataclass with `batch_sizes=args.batch_sizes`.
3. **`new_target.py` line 236-280** (`_target_json`): Writes `batch_sizes` into `target.json` under `workload.batch_sizes`.
4. **`target.json`** (actual campaign artifact at `/home/jinhun/vllm/kernel_opt_artifacts/auto_Qwen3.5-4B_L40S_bfloat16_tp1/target.json`): Contains `"batch_sizes": [8]`.
5. **`run_vllm_bench_latency_sweep.py` line 194-199**: Reads `workload.batch_sizes` from `target.json` and expands into per-bucket benchmarks.
6. **`constraints.md` line 10**: Documents `Decode buckets (batch sizes): [8]`.
7. **`bottleneck_analysis.md` title**: "Qwen3.5-4B on L40S (BS=8, bfloat16, TP=1)".

### Secondary Cause: Hardcoded `m=8` in Roofline Chart

**File**: `/home/jinhun/vllm/.claude/skills/ammo/report/scripts/generate_charts.py`, line 728

```python
m = 8  # decode batch size
```

The roofline chart generator hardcodes `m=8` when computing arithmetic intensity (AI) for GEMM data points. This means even if profiling captured multiple batch sizes, the roofline chart would only plot points for M=8, clustering them all at AI = 2*M*K*N / (2*(M*K + K*N + M*N)) which for small M simplifies to approximately M (i.e., AI around 8).

### Tertiary Cause: Nsys Profiling Guide Examples Use BS=8

**File**: `/home/jinhun/vllm/.claude/skills/ammo/references/nsys-profiling-guide.md`, lines 130, 148, 175, 325, 400, 443

All nsys profiling examples use `--batch-size 8` as the example batch size. While these are "just examples," researchers following the guide tend to replicate the example values verbatim. The sweep script's `--nsys-profile` flag (section 3.5 of the guide) does profile all batch sizes from `target.json`, but since `target.json` only contained `[8]`, this was moot.

### Confirmation from Campaign Artifacts

The actual campaign state confirms single-BS profiling:

- `/home/jinhun/vllm/kernel_opt_artifacts/auto_Qwen3.5-4B_L40S_bfloat16_tp1/target.json`: `"batch_sizes": [8]`
- `/home/jinhun/vllm/kernel_opt_artifacts/auto_Qwen3.5-4B_L40S_bfloat16_tp1/constraints.md` line 10: `Decode buckets (batch sizes): [8]`
- `/home/jinhun/vllm/kernel_opt_artifacts/auto_Qwen3.5-4B_L40S_bfloat16_tp1/bottleneck_analysis.md` line 1: Title specifies `BS=8`

The entire analysis -- GEMM shape table, BW utilization, Amdahl ceilings, and candidate ranking -- is grounded exclusively in BS=8 behavior.


## Section 2: Recommended Batch Sizes

### Default Profiling Set: [1, 8, 32]

| Batch Size | Rationale | GEMM M dimension | Expected Regime |
|------------|-----------|-------------------|-----------------|
| **1** | Single-request cold-start latency; GEMV regime where cuBLAS often has specialized paths; reveals launch-overhead-dominated workloads | M=1 | Heavily memory-bound (AI close to 1); kernel launch overhead fraction is highest |
| **8** | Light-load decode; small GEMM regime where CUTLASS/cuBLAS may underutilize SMs; current baseline | M=8 | Memory-bound (AI around 8); weight loading dominates |
| **32** | Medium-to-high batch decode; L40S heuristic threshold (from `gpu-configs.md` line 25); approaches compute-bound crossover for some GEMMs | M=32 | Transitional: some GEMMs approach ridge point; fusion savings diminish relative to GEMM time |

### Why These Three

1. **Reveals the memory-bound to compute-bound transition**: At BS=1 and BS=8, decode GEMMs are deep in the memory-bound regime. At BS=32, the arithmetic intensity shifts enough (AI around 32) that some GEMMs approach or cross the ridge point. This is critical for optimization targeting -- a custom kernel that helps at BS=8 (memory-bound) may be irrelevant at BS=32 (compute-bound).

2. **Covers realistic production operating points**: Real vLLM deployments serve requests at varying concurrency. Profiling only BS=8 misses the most common deployment patterns (BS=1 for interactive chat, BS=32+ for throughput-oriented serving).

3. **Matches L40S hardware heuristics**: The `gpu-configs.md` reference file (line 25) lists the L40S heuristic BS threshold as 32, meaning optimizations are most impactful for BS <= 32.

4. **Keeps profiling cost manageable**: Three batch sizes triple the profiling time (~3x). The sweep script's `inproc_sweep` mode loads the model once and benchmarks all batch sizes, so the model-loading overhead is not multiplied. With nsys profiling, each additional BS adds one nsys capture (~2-5 minutes), keeping total profiling under 20 minutes.

### Extended Set (Optional): [1, 2, 4, 8, 16, 32, 64]

For thorough analysis when time is not a constraint. The power-of-2 sweep reveals the exact crossover point per GEMM shape. This takes 7x longer than single-BS profiling (~35-45 minutes with nsys).

### When to Override

The `--batch-sizes` CLI argument on `new_target.py` already supports arbitrary lists, so users can always pass `--batch-sizes 1 4 8 16 32 64` explicitly. The change here is only to the **default**.


## Section 3: Implementation Plan

### Change 1: Update `new_target.py` Default Batch Sizes

**File**: `/home/jinhun/vllm/.claude/skills/ammo/scripts/new_target.py`
**Line**: 301
**Current behavior**: `default=[8]` -- only BS=8 is profiled unless overridden.
**Proposed behavior**: `default=[1, 8, 32]` -- three common operating points profiled by default.

```python
# Line 301 - change:
p.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
# to:
p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
```

This is the single most impactful change. All downstream consumers (`target.json`, sweep script, nsys profiling) automatically pick up the new batch sizes with no further modifications.

### Change 2: Fix Hardcoded `m=8` in Roofline Chart Generator

**File**: `/home/jinhun/vllm/.claude/skills/ammo/report/scripts/generate_charts.py`
**Line**: 728
**Current behavior**: `m = 8` hardcoded -- roofline points computed only for M=8 decode.
**Proposed behavior**: Generate roofline points for each batch size from `target.json` or `constraints.md`, with distinct colors/markers per BS. Each GEMM appears as N points (one per BS) on the roofline chart, showing how the same operation moves from memory-bound to compute-bound as batch size increases.

Specific changes needed:
- Parse batch sizes from `constraints.md` (the "Decode buckets" line) or accept a `batch_sizes` parameter.
- Loop over batch sizes when computing AI and performance for each GEMM.
- Use distinct markers/colors per BS with a legend.
- The function signature at line 690 (`generate_roofline`) needs a `batch_sizes: list[int]` parameter.

```python
# Line 690 - change signature:
def generate_roofline(
    bottleneck_md: str, output_path: Path,
    hw_bw_gbps: float, hw_tflops: float
) -> None:
# to:
def generate_roofline(
    bottleneck_md: str, output_path: Path,
    hw_bw_gbps: float, hw_tflops: float,
    batch_sizes: list[int] | None = None
) -> None:

# Line 728 - change:
m = 8  # decode batch size
# to:
if batch_sizes is None:
    batch_sizes_iter = [8]
else:
    batch_sizes_iter = batch_sizes

# Then wrap lines 729-733 in a loop over batch_sizes_iter,
# using different markers per BS.
```

### Change 3: Update `constraints.md` Template for Per-BS Baselines

**File**: `/home/jinhun/vllm/.claude/skills/ammo/scripts/new_target.py`
**Lines**: 77-96 (`_constraints_md` function)
**Current behavior**: Template says `Decode buckets (batch sizes): {fields.batch_sizes}` but the "Baseline Truth Snapshot" section only has one BS heading.
**Proposed behavior**: Add guidance text indicating that the researcher should capture per-BS profiling data.

This is a template change only -- the researcher agent fills in actual data. Adding a note like:

```markdown
## Baseline Truth Snapshot

> Profile each batch size separately. Per-BS nsys traces and kernel breakdowns
> should be stored as `nsys/baseline_bs{N}.nsys-rep` and summarized below.
```

### Change 4: Update Bottleneck Analysis Instructions in SKILL.md

**File**: `/home/jinhun/vllm/.claude/skills/ammo/SKILL.md`
**Lines**: 122-128 (Stages 1-2 section)
**Current behavior**: The SKILL.md instructs the researcher to "invoke ammo-researcher as a subagent for profiling, source analysis, bottleneck mining" without specifying per-BS analysis.
**Proposed behavior**: Add explicit guidance that bottleneck analysis should include:
- Per-BS kernel breakdown tables (or at minimum, a comparison of the top bottleneck's f-value across batch sizes).
- A "BS sensitivity" row in the candidate ranking table showing whether an optimization target's share changes across batch sizes.

Example addition after line 127:

```markdown
- Lead instructs researcher to produce per-BS kernel breakdowns for all batch sizes
  in target.json. At minimum, the bottleneck_analysis.md should include a table showing
  how the top 5 kernel components' f-values change across batch sizes. This reveals
  whether a target is consistently dominant or only dominant at a specific operating point.
```

### Change 5: Update Nsys Profiling Guide Examples

**File**: `/home/jinhun/vllm/.claude/skills/ammo/references/nsys-profiling-guide.md`
**Lines**: 127-154 (two-step delimited capture example), 166-180 (full-run capture example), 392-404 (MoE worked example)
**Current behavior**: All examples use `--batch-size 8` with no mention of profiling multiple batch sizes.
**Proposed behavior**: Add a note after the examples:

```markdown
> **Multi-BS profiling**: The examples above show single-BS capture for clarity.
> In practice, profile all batch sizes from `target.json` (default: [1, 8, 32]).
> Use the sweep script with `--nsys-profile` (section 3.5) to automate this --
> it produces one `.nsys-rep` per batch size without model reload.
```

Also update section 3.5 (line 203-219) to emphasize that `--nsys-profile` captures all batch sizes from target.json automatically.

### Change 6: Store Per-BS Results in Artifact Directory

**Current behavior**: The sweep script already stores per-BS results as `json/baseline_bs{N}.json` and nsys profiles as `nsys/baseline_bs{N}.nsys-rep`. No change needed to the sweep script.

**Proposed behavior (artifact directory structure)**:

```
{artifact_dir}/
  e2e_latency/
    json/baseline_bs1.json
    json/baseline_bs8.json
    json/baseline_bs32.json
    nsys/baseline_bs1.nsys-rep
    nsys/baseline_bs8.nsys-rep
    nsys/baseline_bs32.nsys-rep
  nsys/
    baseline_bs1_cuda_gpu_kern_sum.csv
    baseline_bs8_cuda_gpu_kern_sum.csv
    baseline_bs32_cuda_gpu_kern_sum.csv
```

This structure already works with the sweep script. The bottleneck analysis just needs to consume all of them.

### Change 7: Per-BS Kernel Rankings in Bottleneck Analysis

**File**: No file change needed (this is a workflow instruction, not a script change).
**Current behavior**: `bottleneck_analysis.md` has a single "Component Share Summary" table for BS=8.
**Proposed behavior**: The researcher should produce a comparison table:

```markdown
### Component Share Across Batch Sizes

| Component | f (BS=1) | f (BS=8) | f (BS=32) | Trend |
|-----------|----------|----------|-----------|-------|
| GEMM      | 85.2%    | 89.0%    | 91.3%     | Increases with BS (more compute per step) |
| GDN       | 7.1%     | 4.9%     | 3.2%      | Decreases (fixed cost diluted) |
| FlashAttn | 2.0%     | 1.2%     | 1.8%      | Increases at high BS (more KV cache work) |
```

This is enforced by the Stage 2 gate review (T5 in the task graph). The lead should add "per-BS breakdown" to the gate checklist.


## Section 4: Trade-offs and Risks

### 1. Profiling Time Increase: 3x (Mitigated)

- **Impact**: Profiling 3 batch sizes takes approximately 3x the wall-clock time of profiling 1.
- **Mitigation**: The `inproc_sweep` mode in `run_vllm_bench_latency_sweep.py` loads the model once per label. Nsys profiling with `--nsys-profile` uses cudaProfilerApi repeat mode, so the model is also loaded once. The actual time cost is approximately 3x the **measurement** time, not 3x the total time (model loading is ~60-90s and happens once).
- **Estimated overhead for Qwen3.5-4B**: ~10-15 additional minutes in Stage 1.

### 2. Larger Artifact Directory

- **Impact**: 3x more nsys traces (each ~50-200 MB), 3x more JSON results, 3x more CSV exports.
- **Mitigation**: nsys traces are the bulk. For 3 batch sizes with 512 output steps, expect ~300-600 MB total in nsys traces. This is manageable for artifact storage.

### 3. More Complex Bottleneck Analysis

- **Impact**: The researcher agent must analyze 3 kernel breakdowns instead of 1. This could increase Stage 2 time and token usage.
- **Mitigation**: The core analysis structure is the same -- parse nsys CSV, compute f-values, rank components. The comparison table adds incremental work, not multiplicative work.

### 4. Conflicting Optimization Targets Across Batch Sizes

- **Risk**: An optimization that helps at BS=1 (e.g., reducing launch overhead) might be irrelevant at BS=32 (where GEMM dominates even more). Conversely, a GEMM optimization for BS=32 might not help at BS=1 (where cuBLAS GEMV is already efficient).
- **Mitigation**: This is actually a **feature**, not a bug. The per-BS analysis reveals which optimizations have broad vs narrow applicability, enabling better-informed debate in Stage 3. Kill criteria can specify "target batch sizes" (already supported per `validation-defaults.md` line 224-229).

### 5. Roofline Chart Complexity

- **Risk**: With 7 GEMM shapes x 3 batch sizes = 21 data points, the roofline chart may become cluttered.
- **Mitigation**: Use grouped markers (same shape per GEMM, different color per BS) and only plot the top 5 GEMMs. The additional data points are actually the key value -- they show the trajectory from memory-bound (BS=1) toward compute-bound (BS=32).

### 6. Backward Compatibility

- **Risk**: Existing campaigns with `target.json` containing `"batch_sizes": [8]` are unaffected (the default only applies to new campaigns).
- **Mitigation**: The change is to `new_target.py` defaults only. In-flight campaigns continue using their existing `target.json`.

### 7. CUDA Graph Capture Overhead at Multiple Batch Sizes

- **Risk**: vLLM captures separate CUDA graphs for each batch size bucket. With 3 batch sizes, this means 3 graph captures during the warmup phase, adding ~30-60s.
- **Mitigation**: The warmup already handles multi-BS graphs in production. The `inproc_sweep` mode handles this automatically.

---

## Summary of Changes

| Priority | File | Line(s) | Change | Effort |
|----------|------|---------|--------|--------|
| **P0** | `scripts/new_target.py` | 301 | `default=[8]` -> `default=[1, 8, 32]` | 1 line |
| **P0** | `report/scripts/generate_charts.py` | 690, 728 | Accept `batch_sizes` param; loop over them for roofline points | ~30 lines |
| **P1** | `SKILL.md` | ~125-128 | Add per-BS profiling guidance for researcher dispatch | ~5 lines |
| **P1** | `scripts/new_target.py` | 77-96 | Add per-BS note in constraints.md template | ~3 lines |
| **P2** | `references/nsys-profiling-guide.md` | After 154, 180, 404 | Add multi-BS profiling notes after examples | ~10 lines |
| **P2** | Stage 2 gate review | (workflow, not file) | Add "per-BS breakdown present" to gate checklist | N/A |

Total code changes: ~50 lines across 4 files. No breaking changes. All backward compatible.
