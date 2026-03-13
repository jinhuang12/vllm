# SQLite-First Chart Generation: Implementation Plan

**Target file**: `.claude/skills/ammo/report/scripts/generate_charts.py`  
**Net new LOC**: ~300 (replaces ~100 LOC of markdown parsing in pie/roofline/timeline)  
**Scope**: Single file modification — no new files, no dependency changes

## Summary

Refactor `generate_charts.py` to use the nsys sqlite export as the primary data source
for three charts (kernel pie, roofline, timeline) instead of parsing markdown tables.
The BW bar chart already uses sqlite via `_mine_sqlite_bw`. Markdown parsing is retained
as a fallback for artifacts that lack sqlite exports (e.g., older campaigns, partial runs).

## Fallback Chain

Each chart has a prioritized data source chain. If the primary source is unavailable or
yields no data, the next source is tried automatically.

```
┌─────────────────────┬────────────────────────┬─────────────────────┬──────────────────┐
│ Chart               │ PRIMARY                │ SECONDARY           │ TERTIARY         │
├─────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Kernel Pie          │ sqlite (KERNEL_CATEGORIES│ bottleneck_md      │ constraints_md   │
│                     │ regex classification)  │ Component Share     │ Kernel Breakdown │
├─────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ BW Bar              │ sqlite (_mine_sqlite_bw│ bottleneck_md       │ constraints_md   │
│                     │ — already implemented) │ Per-GEMM / shapes   │ BW Utilization   │
├─────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Roofline            │ target.json arch dims  │ bottleneck_md       │ hardcoded        │
│                     │ + sqlite observed time │ GEMM Shape Summary  │ defaults         │
├─────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ Timeline            │ sqlite (one decode step│ constraints_md      │ hardcoded        │
│                     │ kernel sequence)       │ kernel sequence     │ generic kernels  │
├─────────────────────┼────────────────────────┼─────────────────────┼──────────────────┤
│ E2E Bar             │ JSON files (unchanged) │ —                   │ —                │
└─────────────────────┴────────────────────────┴─────────────────────┴──────────────────┘
```

## Key Data Model: NsysProfileData

A new class that centralizes all sqlite access. Instantiated once in `main()`, passed
to chart generators that need it.

### Constructor: `__init__(self, sqlite_path: Path)`

1. Open the sqlite file **read-only** via `file:` URI with `?mode=ro` flag:
   ```python
   self._db = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
   ```
   This prevents accidental modification of the profiling artifact. No indexes are
   created on the original file.

2. **Detect CUDA graph mode** — determine whether the trace was captured with
   `--cuda-graph-trace=node`:
   ```sql
   SELECT graphId, count(*) as cnt
   FROM CUPTI_ACTIVITY_KIND_KERNEL
   WHERE graphId IS NOT NULL AND graphId != 0
   GROUP BY graphId
   ORDER BY cnt DESC
   LIMIT 1
   ```
   If the result is non-empty and `cnt > 10000`, this is a CUDA graph trace.
   Store `self.dominant_graph_id` (e.g., `5144`).

3. **Detect decode step count** using a two-tier approach:
   - **PRIMARY (CUDA graph mode)**: Count replays of any single graph node:
     ```sql
     SELECT count(*)
     FROM CUPTI_ACTIVITY_KIND_KERNEL
     WHERE graphId = :dominant_graph_id AND gridId = :any_grid_id
     ```
     Pick `any_grid_id` as the first `gridId` from:
     ```sql
     SELECT DISTINCT gridId
     FROM CUPTI_ACTIVITY_KIND_KERNEL
     WHERE graphId = :dominant_graph_id
     LIMIT 1
     ```
     Each graph node appears once per replay. The replay count = decode steps
     captured in the graph. Store as `self.num_decode_steps`.

   - **FALLBACK (eager mode / no CUDA graphs)**: Count non-graph softmax calls:
     ```sql
     SELECT count(*)
     FROM CUPTI_ACTIVITY_KIND_KERNEL k
     JOIN StringIds s ON k.demangledName = s.id
     WHERE s.value LIKE '%SoftMax%'
       AND (k.graphId IS NULL OR k.graphId = 0)
     ```
     If zero (greedy model, no softmax), fall back to parsing
     `"(\d+)\s*decode\s*steps?\s*captured"` from bottleneck_md. If still zero,
     use a conservative estimate from workload config:
     `num_iters * output_len` from target.json.

4. Store `self.is_cuda_graph_trace` boolean.

### Method: `kernel_time_breakdown(self) -> list[tuple[str, float]]`

Returns `[(category_name, total_ms), ...]` sorted by total_ms descending.

1. Query per-kernel-name aggregation:
   ```sql
   SELECT s.value,
          count(*) as calls,
          sum(k.end - k.start) / 1000000.0 as total_ms
   FROM CUPTI_ACTIVITY_KIND_KERNEL k
   JOIN StringIds s ON k.demangledName = s.id
   WHERE k.graphId = :dominant_graph_id   -- graph mode
      OR (:is_eager AND 1=1)              -- eager mode: all kernels
   GROUP BY s.value
   ORDER BY total_ms DESC
   ```
   In graph mode, only aggregate kernels from the dominant graph (excludes prefill,
   warmup, and sampling kernels that run outside the graph). In eager mode, aggregate
   all kernels (prefill exclusion requires heuristic filtering by call count).

2. Classify each kernel name into a category using `KERNEL_CATEGORIES` regex map.

3. Aggregate by category.

### Method: `decode_step_kernel_sequence(self) -> list[tuple[str, str, float]]`

Returns `[(kernel_name, category, duration_us), ...]` for one representative decode step,
ordered by execution time.

1. **CUDA graph mode**: Use a time-window approach. Get the start times of the
   first graph node (lowest `gridId`):
   ```sql
   SELECT start FROM CUPTI_ACTIVITY_KIND_KERNEL
   WHERE graphId = :dominant_graph_id AND gridId = :first_grid_id
   ORDER BY start
   LIMIT 2
   ```
   The interval `[start[0], start[1])` contains exactly one decode step's kernels.

2. Query all kernels in that window:
   ```sql
   SELECT s.value, (k.end - k.start) / 1000.0 as dur_us,
          k.gridX, k.gridY, k.gridZ
   FROM CUPTI_ACTIVITY_KIND_KERNEL k
   JOIN StringIds s ON k.demangledName = s.id
   WHERE k.graphId = :dominant_graph_id
     AND k.start >= :window_start AND k.start < :window_end
   ORDER BY k.start
   ```

3. Use the **median decode step** (step N/2) instead of step 0 to avoid warmup
   artifacts. Adjust the LIMIT/OFFSET on the boundary query accordingly.

4. Classify each kernel and return the sequence.

5. **Eager mode fallback**: Not implemented in v1. Fall back to constraints_md parsing
   or hardcoded defaults.

### Method: `gemm_timing_by_grid(self) -> list[dict]`

Returns per-grid-config GEMM timing (used by the existing `_mine_sqlite_bw` and by the
new roofline chart). This extracts the same data as the current `_mine_sqlite_bw` queries
but caches the result so both charts can reuse it without duplicate queries.

### Method: `close(self)`

Closes the database connection.

## KERNEL_CATEGORIES Definition

```python
KERNEL_CATEGORIES: list[tuple[str, list[re.Pattern]]] = [
    ("GEMM", [
        re.compile(r"cutlass.*Kernel2"),
        re.compile(r"gemm", re.IGNORECASE),
        re.compile(r"cublas", re.IGNORECASE),
        re.compile(r"s\d+gemm"),
    ]),
    ("FlashAttention", [
        re.compile(r"flash_fwd"),
        re.compile(r"flash_bwd"),
        re.compile(r"flash.*splitkv"),
        re.compile(r"flash.*combine"),
    ]),
    ("SSM/Recurrent", [
        re.compile(r"fused_recurrent"),
        re.compile(r"chunk_fwd_kernel"),
        re.compile(r"chunk_local_cumsum"),
        re.compile(r"recompute_w_u"),
        re.compile(r"chunk_gated_delta"),
    ]),
    ("GDN Ops", [
        re.compile(r"gdn_gating"),
        re.compile(r"l2norm_fwd"),
    ]),
    ("Conv1d", [
        re.compile(r"causal_conv1d"),
    ]),
    ("RMSNorm", [
        re.compile(r"rms_?norm", re.IGNORECASE),
        re.compile(r"triton_red_fused.*rsqrt"),
    ]),
    ("Activation", [
        re.compile(r"silu", re.IGNORECASE),
        re.compile(r"gelu", re.IGNORECASE),
        re.compile(r"sigmoid", re.IGNORECASE),
        re.compile(r"triton_poi_fused_mul_silu"),
        re.compile(r"triton_poi_fused_mul_sigmoid"),
    ]),
    ("Softmax/Sampling", [
        re.compile(r"SoftMax", re.IGNORECASE),
        re.compile(r"topk", re.IGNORECASE),
        re.compile(r"sort_pairs"),
    ]),
    ("KV Cache", [
        re.compile(r"reshape_and_cache"),
        re.compile(r"paged_attention"),
    ]),
    ("Pointwise", [
        re.compile(r"triton_poi_fused"),
        re.compile(r"triton_per_fused"),
        re.compile(r"elementwise_kernel"),
    ]),
    ("Reduction", [
        re.compile(r"triton_red_fused"),
        re.compile(r"reduce_kernel"),
    ]),
    # Catch-all: anything not matched above
    ("Other", []),
]
```

**Classification logic** (`_classify_kernel(name: str) -> str`):
```python
def _classify_kernel(name: str) -> str:
    for category, patterns in KERNEL_CATEGORIES:
        if not patterns:  # empty list = catch-all
            return category
        if any(p.search(name) for p in patterns):
            return category
    return "Other"  # safety fallback (should not be reached)
```
The empty-list `"Other"` at the end catches everything unmatched. The post-loop
`return "Other"` is a safety fallback in case someone accidentally removes the
catch-all entry.

**Important**: The ordering matters. More specific patterns (e.g., `RMSNorm` matching
`triton_red_fused.*rsqrt`) must come BEFORE broader patterns (`Reduction` matching
`triton_red_fused`). The current ordering achieves this: RMSNorm is checked before
Reduction, and Activation (matching `triton_poi_fused_mul_silu`) is checked before
Pointwise (matching `triton_poi_fused`).

**Triton name instability note**: Triton kernel names like `triton_poi_fused_0` are
torch.compile artifacts that change across torch versions, compiler flag changes, and
even model modifications. The classification intentionally uses broad prefix patterns
(`triton_poi_fused`, `triton_red_fused`, `triton_per_fused`) rather than exact names.
This means we can classify by operation type (pointwise, reduction, persistent reduction)
but NOT by specific op (which fused op `triton_poi_fused_0` represents). For op-specific
classification, the named kernels (CUTLASS, flash, fused_recurrent, causal_conv1d,
fused_gdn_gating) are stable and can be matched precisely.

## Step-by-Step Implementation

### Step 1: Add `NsysProfileData` class and `KERNEL_CATEGORIES`

**What changes**: New class and constant added near the top of the file, after the
existing helper functions.

**Affected charts**: None directly (infrastructure).

**Details**:
- Add `KERNEL_CATEGORIES` as a module-level constant (see definition above)
- Add `_classify_kernel(name: str) -> str` helper that walks the category list
- Add `NsysProfileData` class with `__init__`, `close`, and `__enter__`/`__exit__`
  for context manager support
- Constructor opens read-only sqlite, detects graph mode, counts decode steps
- Add `kernel_time_breakdown()` method
- Add `decode_step_kernel_sequence()` method
- Add `gemm_timing_by_grid()` method (refactored from inline code in `_mine_sqlite_bw`)

**Acceptance criteria**:
- `NsysProfileData` can be instantiated with the reference sqlite path
- `kernel_time_breakdown()` returns categories matching the bottleneck_analysis.md
  Component Share Summary within 2% for each major category (GEMM ~89%, Recurrent ~5%)
- `decode_step_kernel_sequence()` returns 562 kernels for the reference profile
- `is_cuda_graph_trace` is True for the reference profile
- `num_decode_steps` is 2552 for the reference profile

### Step 2: Add `_locate_baseline_sqlite()` helper

**What changes**: New function replacing the inline glob in `_mine_sqlite_bw`.

**Affected charts**: All sqlite-dependent charts (pie, BW bar, roofline, timeline).

**Details**:
```python
def _locate_baseline_sqlite(artifact_dir: Path) -> Path | None:
    """Find the baseline nsys sqlite in the artifact tree.

    Uses a strict glob pattern to match only timestamp-named baseline
    directories, excluding OP-specific directories and investigation sqlites.
    If multiple timestamp directories exist, takes the most recent by name sort.
    """
    candidates = list(artifact_dir.glob("e2e_latency_20*Z/nsys/baseline_profile.sqlite"))
    if not candidates:
        return None
    return sorted(candidates)[-1]  # most recent by timestamp in name
```

This function is called once in `main()` and the result is passed to all chart generators.

**Acceptance criteria**:
- Returns the correct path for the reference artifact directory
- Does NOT match `e2e_latency_op002/`, `e2e_latency_op002_v2/`, or
  `tracks/*/nsys_investigation/` paths
- Returns `None` when no timestamp-named directory exists
- When multiple timestamp dirs exist, returns the latest (alphabetical sort
  on ISO 8601 timestamps produces chronological order)

### Step 3: Refactor `generate_kernel_pie` to use sqlite primary

**What changes**: Add `NsysProfileData` as an optional parameter. When available,
use `kernel_time_breakdown()` instead of markdown parsing.

**Affected charts**: `kernel_breakdown_pie.png`

**New signature**:
```python
def generate_kernel_pie(
    bottleneck_md: str,
    output_path: Path,
    model_name: str,
    batch_size: int,
    profile_data: NsysProfileData | None = None,
) -> None:
```

**Logic**:
1. If `profile_data` is not None:
   a. Call `profile_data.kernel_time_breakdown()`
   b. Convert to labels/sizes (percentages of total)
   c. Skip to chart rendering
2. Else: fall back to existing markdown parsing (unchanged)

**Acceptance criteria**:
- With sqlite data, pie chart shows categories matching bottleneck_analysis.md
- GEMM shows ~89%, SSM/Recurrent ~5%, FlashAttention ~1.2%
- Small categories (<0.5%) are grouped into "Other" for visual clarity
- Without sqlite (profile_data=None), existing behavior is preserved exactly
- The `"Other"` bucket explicitly appears when unmatched kernels exist

### Step 4: Refactor `generate_roofline` to use target.json + sqlite

**What changes**: Read GEMM shapes from `target.json` / `constraints.md` architecture
fields (architectural constants) instead of parsing the GEMM Shape Summary table.
Pair shapes with sqlite-derived observed durations by matching operation call counts.

**Affected charts**: `roofline_plot.png`

**New signature**:
```python
def generate_roofline(
    bottleneck_md: str,
    output_path: Path,
    hw_bw_gbps: float,
    hw_tflops: float,
    batch_sizes: list[int] | None = None,
    profile_data: NsysProfileData | None = None,
    constraints_md: str = "",
    target_json: dict | None = None,
) -> None:
```

**New approach for GEMM shape derivation**:

1. **PRIMARY — architectural constants from target.json/constraints.md**:
   Parse `hidden_size`, `intermediate_size`, `num_hidden_layers`,
   `num_key_value_heads`, `head_dim`, `linear_num_key_heads`,
   `linear_num_value_heads`, `linear_key_head_dim`, `linear_value_head_dim`,
   `full_attention_interval` from constraints.md model architecture section.
   
   Derive GEMM shapes:
   ```
   GDN in_proj_qkvz: K=hidden_size, N=linear_num_key_heads*key_dim + linear_num_value_heads*value_dim + linear_num_key_heads*key_dim + linear_num_value_heads*value_dim
                     (q + k + v + z projections merged: e.g., 16*128 + 16*128 + 32*128 + 32*128 = 12288)
   GDN in_proj_ba:   K=hidden_size, N=linear_num_key_heads*1 + linear_num_value_heads*1
                     (beta + alpha scalars per head: e.g., 16 + 16 + 16 + 16 = 64)
                     NOTE: actual N may differ from bottleneck_analysis.md due to merged projections
   GDN out_proj: K=linear_num_value_heads*value_dim, N=hidden_size
   MLP gate_up_proj: K=hidden_size, N=2*intermediate_size
   MLP down_proj: K=intermediate_size, N=hidden_size
   Full attn qkv_proj: K=hidden_size, N=(num_heads + 2*num_kv_heads)*head_dim
   Full attn o_proj: K=num_heads*head_dim, N=hidden_size
   ```
   Each shape has a known layer count (from `num_hidden_layers` and
   `full_attention_interval`).

2. **Pair with observed durations from sqlite** (if `profile_data` available):
   Use `profile_data.gemm_timing_by_grid()` to get per-grid-config average
   duration. Match to architectural shapes using a two-step algorithm:

   a. **Group by expected call count**: Each GEMM shape has an expected call count
      = `num_layers * num_decode_steps`. Group shapes with identical expected counts.

   b. **Disambiguate same-count shapes using gridY**: When multiple shapes have the
      same expected call count (e.g., `gate_up_proj` and `down_proj` both at 32 layers),
      use `gridY` to pick the right one. `gridY = ceil(N / tile_N)` where tile_N is
      typically 128 for CUTLASS WMMA kernels. We are NOT deriving N from gridY — we are
      using gridY to pick between known shapes:
      - `gate_up_proj` (N=18432) → gridY=144
      - `down_proj` (N=2560) → gridY=20
      For each sqlite grid group with matching call count, compute `expected_gridY =
      ceil(N / 128)` for each candidate shape and pick the closest match.

   c. **Fallback for unresolvable**: If gridY disambiguation fails (e.g., unexpected
      tile size), assign the average duration across all same-count grid groups.

   This gives us REAL observed performance (not estimated from BW utilization)
   paired with CORRECT shapes (not derived from grid dimensions). Grid dimensions
   are used ONLY for disambiguation between known shapes, never for shape inference.

3. **SECONDARY — GEMM Shape Summary from markdown** (existing fallback):
   If neither target.json nor constraints.md architecture section is parseable,
   fall back to the GEMM Shape Summary table in bottleneck_md (current behavior).

4. **Grid-to-shape derivation is REMOVED**. The `gridY -> N_tiles` mapping
   (used in `_mine_sqlite_bw`) is NOT used for roofline shape derivation.
   Grid dimensions are unreliable for shape inference because CUTLASS tile
   layout depends on transposition flags, warp tile shapes, and padding.

**Acceptance criteria**:
- Roofline points use architectural shapes (K, N match constraints.md exactly)
- Observed performance uses real sqlite timing when available
- Operating points are in the memory-bound regime (below ridge point) for M=8
- Without sqlite, falls back to markdown parsing cleanly

### Step 5: Refactor `generate_timeline` to use sqlite primary

**What changes**: When `NsysProfileData` is available, render a real decode step
kernel sequence instead of hardcoded synthetic data.

**Affected charts**: `nsys_timeline_synthetic.png`

**New signature**:
```python
def generate_timeline(
    constraints_md: str,
    output_path: Path,
    profile_data: NsysProfileData | None = None,
) -> None:
```

**Logic**:
1. If `profile_data` is not None:
   a. Call `profile_data.decode_step_kernel_sequence()`
   b. Get the kernel sequence for one decode step (median step)
   c. Classify each kernel using `_classify_kernel()`
   d. **Merge consecutive same-category kernels** that are < 2us each
      (micro-kernels between GEMMs) to avoid visual clutter
   e. Assign colors by category using a fixed color map
   f. Render real durations (some kernels will be very thin; scale
      sub-2us kernels to a minimum visible width of 2us)
   g. Show only the first N kernels that fit one decoder layer
      (approximately: up to the pattern repeat at ~layer 2)
2. Else: fall back to existing hardcoded kernel list

**Color map** (matching existing convention):
```python
CATEGORY_COLORS = {
    "GEMM": "#e74c3c",           # red
    "SSM/Recurrent": "#3498db",  # blue
    "FlashAttention": "#2980b9", # darker blue
    "RMSNorm": "#9b59b6",       # purple
    "Activation": "#f39c12",     # orange
    "Pointwise": "#95a5a6",     # gray
    "Conv1d": "#1abc9c",        # teal
    "GDN Ops": "#e67e22",       # dark orange
    "KV Cache": "#27ae60",      # green
    "Softmax/Sampling": "#8e44ad", # dark purple
    "Reduction": "#7f8c8d",     # dark gray
    "Other": "#bdc3c7",         # light gray
}
```

**Acceptance criteria**:
- With sqlite data, timeline shows real kernel durations from the median decode step
- GEMM blocks dominate visually (matching 89% share)
- Micro-kernels between GEMMs are visible but compact
- Total timeline duration approximately matches per-step time from bottleneck_md
- Without sqlite, existing hardcoded fallback renders identically to current behavior

### Step 6: Update `_mine_sqlite_bw` to use `NsysProfileData`

**What changes**: Refactor `_mine_sqlite_bw` to accept `NsysProfileData` instead of
opening its own sqlite connection. This eliminates duplicate connection management and
allows the BW chart to reuse cached query results.

**Affected charts**: `bw_utilization_bar.png`

**New signature**:
```python
def _mine_sqlite_bw(
    profile_data: NsysProfileData,
    bottleneck_md: str,
    hw_bw_gbps: float,
) -> tuple[list[str], list[float]] | None:
```

**Changes**:
- Remove inline sqlite path discovery (now done by `_locate_baseline_sqlite`)
- Remove inline `sqlite3.connect()` (now uses `profile_data._db`)
- Use `profile_data.num_decode_steps` instead of regex parsing for step count
- Reuse `profile_data.gemm_timing_by_grid()` for cached grid-grouped data
- Core BW calculation logic (grid matching, clustering, call count matching)
  remains unchanged — it is already working well

**Acceptance criteria**:
- BW bar chart output is identical to current behavior (same labels, same values)
- No duplicate sqlite connections are opened
- Step count comes from `NsysProfileData` (graph replays or softmax fallback)

### Step 7: Update `main()` to wire everything together

**What changes**: Instantiate `NsysProfileData` once, pass to all chart generators.

**Affected charts**: All.

**Details**:
```python
# In main(), after reading source data:
sqlite_path = _locate_baseline_sqlite(artifact_dir)
profile_data = None
if sqlite_path:
    try:
        profile_data = NsysProfileData(sqlite_path)
        print(f"  Loaded nsys profile: {sqlite_path.name} "
              f"({profile_data.num_decode_steps} decode steps, "
              f"{'graph' if profile_data.is_cuda_graph_trace else 'eager'} mode)")
    except Exception as e:
        print(f"  WARNING: Could not load nsys profile: {e}")

# Update chart lambda list to pass profile_data
charts = [
    ("kernel_breakdown_pie.png",
     lambda p: generate_kernel_pie(bottleneck_md, p, model_short, 8,
                                    profile_data=profile_data)),
    ("bw_utilization_bar.png",
     lambda p: generate_bw_bar(bottleneck_md, p, args.hw_bw_gbps,
                                artifact_dir=artifact_dir,
                                profile_data=profile_data)),
    ("e2e_results_bar.png",
     lambda p: generate_e2e_bar(artifact_dir, p, shipped)),
    ("roofline_plot.png",
     lambda p: generate_roofline(bottleneck_md, p, args.hw_bw_gbps,
                                  args.hw_tflops, batch_sizes,
                                  profile_data=profile_data,
                                  constraints_md=constraints_md,
                                  target_json=target_json)),
    ("nsys_timeline_synthetic.png",
     lambda p: generate_timeline(constraints_md, p,
                                  profile_data=profile_data)),
]

# ... after chart generation ...
if profile_data:
    profile_data.close()
```

Also load `target.json`:
```python
target_json = _load_json(artifact_dir / "target.json")
```

**Acceptance criteria**:
- All charts generate successfully with sqlite data present
- All charts generate successfully without sqlite data (fallback path)
- Single sqlite connection is opened and closed cleanly
- Informative log message shows decode step count and trace mode
- No Python warnings or resource leaks

## Known Limitations

### 1. TP > 1 profiles

When `tp > 1`, each GPU runs a separate CUDA graph with different kernel counts per
step (due to tensor-parallel communication kernels like NCCL allreduce). The current
implementation assumes a single dominant graph. For TP>1:
- The dominant graph (most kernel calls) will be selected, which is typically rank 0
- NCCL kernels may not appear in `CUPTI_ACTIVITY_KIND_KERNEL` (they use separate
  CUPTI activity kinds)
- Kernel timing includes only the local GPU's work, not communication overhead
- **Mitigation**: For accurate TP>1 analysis, the report should note that displayed
  kernel times represent single-GPU computation only

### 2. FP8 dtype and quantized weight sizes

The BW utilization calculation uses `weight_bytes = K * N * 2` (bf16). For FP8 models:
- Weight bytes should be `K * N * 1` (fp8) but the computation kernel may still use
  bf16 accumulators and read dequantized bf16 from shared memory
- The "BW utilization" metric is really "weight data transfer efficiency" which depends
  on the storage format, not the compute format
- **Mitigation**: Read `dtype` from `target.json` and adjust the bytes-per-element
  multiplier. This is a future enhancement, not addressed in this refactor.

### 3. Triton kernel name instability

Triton-compiled kernel names (`triton_poi_fused_0`, `triton_red_fused_1`, etc.) are:
- Numbered based on compilation order, which changes between torch versions
- Named with fused op descriptions that change when torch.compile fusion rules change
- NOT stable across different models or even different batch sizes on the same model

The `KERNEL_CATEGORIES` classification uses broad prefix patterns that are stable across
these variations. The trade-off is coarser classification (we know "pointwise fusion"
but not "which specific fused op"). For operation-specific classification, we rely on
named kernels (CUTLASS, FlashAttention, fused_recurrent, etc.) which have stable names.

### 4. Eager mode (no CUDA graphs) has limited timeline support

The `decode_step_kernel_sequence()` method uses CUDA graph replay boundaries to isolate
one decode step. In eager mode, there are no graph replays and kernels from different
steps are interleaved on the GPU timeline. Implementing eager-mode step detection would
require either:
- NVTX range markers (not present in current nsys captures)
- Heuristic boundary detection based on kernel patterns and timing gaps
This is deferred to a future enhancement; eager mode falls back to the markdown/hardcoded
timeline.

### 5. Prefill kernels excluded from pie chart

When using sqlite as the primary source in graph mode, the pie chart only shows
decode-step kernels (from the dominant CUDA graph). Prefill kernels (which use
different CUTLASS variants and run outside the graph) are excluded. This is intentional
since the optimization target is decode latency, but the chart title/subtitle should
clarify "Decode-only kernel breakdown" to avoid confusion.

### 6. In-memory sqlite copy not used

The original Finding 2 suggested creating indexes for faster queries. Testing shows:
- Read-only mode prevents index creation on the original file (correct behavior)
- In-memory copy (203 MB, 274ms to copy) would allow indexes but adds memory pressure
- Without indexes, the critical per-kernel aggregation query takes ~3s for 1.5M rows
- With an index on `(graphId, start)`, aggregation is similar (~3s) because the
  bottleneck is the JOIN + GROUP BY, not the WHERE clause
- The 3s query time is acceptable for a report generation script that runs once

**Decision**: Do NOT create indexes or use in-memory copies. The 3s query overhead
is acceptable for a batch reporting tool. If future profiles are significantly larger
(>10M rows), revisit this decision.

## Terminal Fallback Behavior

When ALL data sources fail for a chart, the generator must not raise an exception.
Each chart has a defined terminal behavior:

| Chart | Terminal fallback |
|-------|------------------|
| Kernel Pie | Print WARNING, skip chart (no PNG generated) |
| BW Bar | Print WARNING, skip chart |
| Roofline | Render roofline curve only (no data points), with annotation "No GEMM data available" |
| Timeline | Render existing hardcoded generic kernel sequence (current behavior) |
| E2E Bar | Print WARNING, skip chart (current behavior) |

This ensures the report script never crashes, and partial chart sets are acceptable
(the report template handles missing PNGs with text fallback descriptions).

## Testing Strategy

1. **Unit**: Run `generate_charts.py` against the reference artifact directory:
   ```bash
   python .claude/skills/ammo/report/scripts/generate_charts.py \
     --artifact-dir kernel_opt_artifacts/auto_Qwen3.5-4B_L40S_bfloat16_tp1
   ```
   Verify all 5 PNGs are generated without errors.

2. **Sqlite-first validation**: Compare sqlite-derived pie chart percentages against
   bottleneck_analysis.md Component Share Summary. Differences should be < 2% absolute
   for major categories (the sqlite data includes non-graph kernels differently).

3. **Fallback**: Temporarily rename the sqlite file and verify all charts still generate
   using markdown fallback paths.

4. **Empty/corrupt sqlite**: Test with a zero-byte sqlite file to ensure graceful
   fallback to markdown parsing.

5. **Eager mode**: Test with a synthetic sqlite that has `graphId = NULL` for all
   kernels (or use a real eager-mode capture if available).
