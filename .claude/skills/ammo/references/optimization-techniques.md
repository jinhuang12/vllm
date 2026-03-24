# Optimization Techniques (Catalog)

This is a technique catalog. Use it **after** you’ve identified optimization targets from bottleneck mining:
- Fusion ROI math: `references/fusion-feasibility-heuristics.md`

## Contents
- Pattern applicability table
- T1–T12 + T14 techniques (with applicability + gotchas)

## Search anchors
token-major, expert-major, MMA, buffering, cp.async, TMA, epilogue fusion, W1, SwiGLU, topk, k-way merge, sorting network, bitonic merge, SonicMoE, FlashInfer, GPU kernel, optimization

## Pattern applicability (quick index)

| Pattern | Best‑fit Techniques |
|---------|---------------------|
| Token‑major | T1 (conditional), T8, T10, T11, T14 |
| Expert‑major | T1, T5, T6, T11 |
| Cooperative | U5, T3, T6 |
| Split‑kernel | T5, T10 (routing/prepare), T11 (W1 epilogue fusion), T14 |
| Hybrid large‑grid | T10 (routing/prepare), T11 (W1 epilogue fusion) — see `hybrid-large-grid-fusion.md` |
| System / attention backend | T12 (FlashInfer kernel survey + feasibility) |

## Wrapper/dispatch guidance (keep it boring)
- Prefer a **small number** of pre-instantiated kernels (or template specializations) with **Python-side dispatch**.
- Keep dispatch keys explicit and stable under CUDA graphs (bucketed shapes, dtype, TP/EP, arch).
- Avoid hidden allocations inside the op; preallocate scratchpad and pass it in when possible (`references/cudagraph-safety.md`).
- For scaffolding patterns, see `references/code-templates.md`.

---

## T1: Full MoE Monokernel Fusion (Conditional)

Single cooperative kernel that fuses:
1. Router logits → top-k selection
2. Activation quantization (FP8 dynamic scaling)
3. Up-projection GEMM (W8A8)
4. Gated activation (SiLU/SwiGLU)
5. Down-projection GEMM → BF16 output

**Applicability**:
- Best for decode (M ≤ 64) **and** low M_avg with token-major ownership
- Avoid when M_avg is high or ownership is expert-major (prefer split-kernel / grouped GEMM)
- Less attractive for massive prefill (M > 256)

**Decision Metric**: Fusion beneficial when `kernel_launch_overhead × num_kernels > compute_time × 0.1` **and** `M_avg < M_AVG_THRESHOLD`. For BS 8-64, ~5μs per launch is significant vs ~50-100μs compute.

## T2: Template Specialization + Python Dispatch

```cpp
using Dims_BS8_E16_TP8 = MoEDimensions<8, 1024, 5120, 16>;
using Dims_BS64_E16_TP8 = MoEDimensions<64, 1024, 5120, 16>;

MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS8_E16_TP8_impl, Dims_BS8_E16_TP8)
MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS64_E16_TP8_impl, Dims_BS64_E16_TP8)
```

**How to Choose Specializations**:
1. Sample real decode workloads → histogram of M per MoE call
2. Pick 2-3 representative caps: BS8, BS32, BS64
3. For each (BS, E_local): instantiate `Dims_...` and wrapper
4. Python dispatch via if/else on `(M, E, TP)`

## T3: In-Kernel Router Selection

Use only when routing must be fused. If routing is done outside the kernel, skip T3 and pass top-k ids/weights in as inputs.

**Strategy by (M, E_local)**:

| Case | Strategy |
|------|----------|
| M ≤ 8, E ≤ 16 | One warp/token, scalar loop |
| M ≤ 8, E ≤ 128 | One warp/token, thread subset per expert |
| M ≤ 64, E ≤ 128 | Per-thread loops with warp reduction |
| E > 128 | Hierarchical: tile into 128-expert chunks |

**Performance Target**: Router ≤ 5-10% of total kernel time.

## T4: Per-Token FP8 Activation Quantization

```cpp
// For each token i:
float max_i = max_j |x_i[j]|;
float scale_i = max(max_i, eps) / FP8_MAX;  // FP8_MAX ≈ 448
float inv_scale_i = FP8_MAX / max(max_i, eps);

// Fold scale into gate weight
topk_weight_scaled_i = topk_weight_i * scale_i;
```

**Hardware Check**: Requires FP8 Tensor Core + conversion (Hopper, Ada). For A100/V100: use BF16 GEMM path.

## T5: Expert-Local Token Sorting

**BS ≤ 8 (Bitfield)**:
```cpp
uint64_t expert_mask;  // 8 bytes × 8 tokens
uint64_t expert_ids;   // unique experts packed
uint32_t expert_count = __popc(bitset);
```

**BS ≤ 64 (Histogram)**:
```cpp
__shared__ uint32_t counters[E_local][warp_size];
// Per-warp counting → prefix sum → token_indexes permutation
```

## T6: Tensor Core Tiling + Multi-Buffering

**SMEM Budget Formula**:
```
SMEM_block_max ≈ SMEM_SM / B_per_SM
// H100: 228KB, target 1 CTA/SM → ~228KB available
```

**Up-Projection (Triple Buffer)**:
```
bytes_A = 3 × M_tile × (K/2) × sizeof_fp8
bytes_W = 3 × N_tile × (K/2 + pad) × sizeof_fp8
bytes_partial = CALC_WARPS × N_tile × M_tile × sizeof_f32
```

**Down-Projection (Double Buffer)**:
```
bytes_T = 2 × T_TILE × N × sizeof_f32
bytes_Wd = 2 × W_DOWN_TILE × (N + pad) × sizeof_fp8
```

**Tile Size Search**:
- M_tile ∈ {8, 16} (matches m16n8k16 MMA)
- N_tile ∈ {16, 32, 64}
- Prefer largest N_tile that fits SMEM constraint

## T7: Warp Role Partitioning

Warp counts are GPU-dependent:

**H100/H200** (sm_90a, 4 Tensor Cores/SM):
```cpp
static constexpr uint32_t TOTAL_WARP_COUNT = 12;  // 384 threads
static constexpr uint32_t CALC_WARP_COUNT = 8;    // Warps 0-7: MMA
static constexpr uint32_t PREFETCH_WARP_COUNT = 4; // Warps 8-11: async copy
```

**L40S/A100** (sm_89/sm_80, 256 threads/block optimal):
```cpp
static constexpr uint32_t TOTAL_WARP_COUNT = 8;   // 256 threads
static constexpr uint32_t CALC_WARP_COUNT = 6;    // Warps 0-5: MMA
static constexpr uint32_t PREFETCH_WARP_COUNT = 2; // Warps 6-7: async copy
```

**Why 8:4 on H100?** H100 has 4 Tensor Cores/SM. 8 warps saturate TCs, 4 warps keep memory pipeline full.

**Sync Only Calc Warps**:
```cpp
__asm volatile("bar.sync 15, 256;\n");  // Named barrier, 256 threads
```

## T8: Tiny vs Normal Path Specialization

```cpp
if constexpr (Dims::BS <= 8) {
    moe_kernel_BS8(...);  // Everything in registers/SMEM
} else {
    moe_kernel_BS64(...); // Full sorting, triple buffer
}
```

**Tiny Path Characteristics**:
- 1:1 warp-token mapping
- Bitfield expert encoding
- Single-pass weight loading
- No global scratch for routing

## T9: Global Scratchpad Design

```cpp
template <typename Dims>
struct MoEGemmSpec {
    AQ_element activations[BS][K];      // Quantized FP8
    T_element temp[(BS+8) * N];         // Up-projection output FP32
    float topk_weights_scaled[BS];      // Scale-folded gate weights
};
```

**Sizing Formula**:
```
bytes ≈ BS_max × K × 1 + (BS_max + 8) × N × 4 + BS_max × 4
```

## T10: Faster Top‑K Routing (Small‑K) via Sorting-Network / K‑Way Merge (SonicMoE‑style)

**What it is**: replace a slow MoE routing & prepare stage with a small‑K GPU kernel that uses warp‑level primitives and a deterministic tie-break.

This technique is motivated by SonicMoE's router top-k optimization for small `K`.

**Use when (evidence required)**:
- `top_k` is small and fixed (commonly `8` or `16`), and `E_local` is moderate (≤256 is the common "warp-friendly" range).
- Baseline routing is not already fused into the expert GEMM path (or is not already using a specialized small‑K kernel).

**Do not use when**:
- Routing is not in the top‑N kernels by GPU time (you won't buy an E2E win).
- Your model uses nontrivial grouped routing / shared experts semantics you cannot faithfully reproduce (treat as a separate semantics variant and gate explicitly).

**Implementation options (pick one; keep it shape-gated)**:
- **Sorting network / bitonic merge** (SonicMoE-style): good for small, fixed `K`; deterministic; maps well to warp operations.
- **K‑way merge**: each lane pre-sorts a small local list once, then selects `K` winners via repeated warp-reduce where only the winning lane advances its cursor.

**Determinism + semantics constraints (non-negotiable)**:
- Deterministic tie-break (recommended): `(score desc, expert_id asc)`.
- Scoring semantics must match the model:
  - selection function (softmax vs sigmoid(+bias) vs custom)
  - whether weights are renormalized after selection (`norm_topk_prob`)
  - where weights are applied (pre/post activation / folded into scales)

**vLLM integration pattern (recommended)**:
- Add a new routing custom op that returns `(topk_ids, topk_weights)` and "prepare" outputs.
- Dispatch to a small set of specializations keyed by `(top_k, E_local, semantics)`; fall back otherwise.
- Gate with an env var and strict shape checks; keep CUDA-graph safe (`references/cudagraph-safety.md`).

**Prepare (routing → expert-local grouping)**
- If baseline prepare is already fast, keep it and only swap selection.
- If prepare is material, produce (at minimum):
  - `expert_offsets[E_local+1]` (prefix sum over per-expert counts)
  - `pair_indices[total_pairs]` (stable grouping of token-expert pairs by expert)

**Expected profiler signature**
- `nsys` kernel ranking: router/top‑k kernel time decreases materially, or a slow `torch.topk` kernel disappears.
- End-to-end: any win is limited by MoE share `f` (use `references/e2e-delta-math.md` to sanity-check expected delta).

**Stop-ship signals**
- No MoE GPU kernel-time reduction in the target buckets under CUDA graphs.
- Any correctness mismatch beyond declared tolerances (especially routing weights semantics).
- You needed to broaden the dispatch envelope so far that maintenance risk dominates.

## T11: W1 Epilogue Fusion (Activation + Quant) into Expert GEMM (Hybrid Large‑Grid)

**What it is**: fuse activation (e.g., SiLU/SwiGLU) and W2-input quantization (e.g., FP8) into the **W1 expert GEMM epilogue**, so you delete real kernel work and avoid large intermediate writes/reads.

Baseline pattern (common):
1) W1 expert GEMM writes `[M*top_k, 2*N_local]` to global memory
2) activation kernel produces `[M*top_k, N_local]`
3) quant kernel produces FP8 activations (+ scale metadata) for W2
4) W2 expert GEMM consumes FP8 activations

**Use when (evidence required)**:
- `nsys` shows activation and/or quant kernels are present and material, and the intermediate tensor is large enough that removing a round-trip is plausible ROI.
- Baseline W1 kernel does not already fuse the epilogue (verify with kernel list; do not guess).

**Preconditions / constraints**
- Activation can be computed locally per output tile (no global sync).
- Quantization metadata semantics must match the baseline (per-tensor vs per-block/grouped scales, scale layout, saturation rules).
- Epilogue register pressure must stay bounded (avoid spills that slow the GEMM enough to erase the win).

**Implementation sketch (Triton or CUDA)**
- Locate the expert GEMM kernel used for W1 (often called from `fused_experts` or an MoE runner).
- Add a fused-epilogue variant that:
  - computes W1 tile
  - applies activation (SwiGLU/SiLU etc.)
  - quantizes to the exact FP8 format expected by W2 and writes FP8 (+ scales)
- Wire under an env var gate; preserve a fallback.

**Expected profiler signature**
- Activation/quant kernels disappear or shrink substantially; total MoE GPU kernel time decreases under CUDA graphs.
- W1 kernel may get slightly slower, but total should improve; validate with both `nsys` and a small `ncu` spot-check for spills/occupancy regression.

**Stop-ship signals**
- W1 kernel slows enough (regs/spills/occupancy drop) that total MoE kernel time does not improve.
- Any quantization metadata mismatch (silent accuracy loss risk).
- Baseline already had the epilogue fused (no remaining work to delete).

## T12: FlashInfer Kernel Survey + Feasibility Scoring (Attention / Operator Backends)

**What it is**: systematically enumerate which FlashInfer kernel families could apply to your target (decode/prefill attention, paged KV cache, optional variants like cascade/sparse), then score feasibility and expected ROI under production parity.

This technique is about *planning and triage*, not "blindly switch to FlashInfer".

**Why it belongs here**: in many real deployments, attention or KV-cache plumbing dominates end-to-end. If attention is the hotspot, optimizing other components cannot deliver the target win even if they become 0us.

### Step 1: Determine whether FlashInfer is worth considering (evidence gate)

Only do the FlashInfer survey if Stage 1 `nsys` shows attention/KV-cache kernels are a top share in the target regime (prefill or decode).

### Step 1.5: How vLLM typically integrates FlashInfer (preferred path)

In vLLM, FlashInfer is usually consumed via **existing config-backed integration points**, not by directly calling FlashInfer APIs from random model code:

- **`torch.compile` / Inductor pattern-rewrite passes** (best example):
  - Enable via compilation config flags such as:
    - `-cc.pass_config.fuse_allreduce_rms=true`
  - vLLM then registers a pass (e.g., AllReduce fusion) that pattern-matches a known subgraph and replaces it with FlashInfer fused kernels when available.
- **Backend selection via config** (attention/serving):
  - vLLM exposes attention backend controls that may route into FlashInfer (including TRTLLM attention backend selection through FlashInfer).
- **Distributed comm helpers**:
  - Some all-to-all paths can use FlashInfer comm kernels when available.

**Important**: if the FlashInfer kernel family you want is **not already integrated** behind a vLLM config flag / backend selection / compiler pass, then the integration must be done:
- add a wrapper/custom-op or backend dispatch site,
- add shape/dtype/arch guards + a fallback when FlashInfer is missing,
- make it CUDA-graph friendly (workspace reuse, stable shapes per bucket),
- add correctness + performance validation evidence under production parity.

### Step 2: Enumerate "available FlashInfer kernels" (kernel families + variants)

> **Note**: The paths below reference the `gpu-kernel-optimizer` skill which is available in the Codex CLI environment (`.codex/skills/`). If not available, use the upstream FlashInfer repository directly: https://github.com/flashinfer-ai/flashinfer

Use the `gpu-kernel-optimizer` FlashInfer corpus index as the starting point (if available):
- `.codex/skills/gpu-kernel-optimizer/references/corpus.flashinfer.md`

Primary "kernel family" entry points (plan/run + backend selection):
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer/attention.py` (BatchAttention plan/run, paged KV-cache)
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer/decode.py`
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer/prefill.py`
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer/page.py` (paged KV layout helpers)
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer/cascade.py`
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer/sparse.py`

Variant/feature definitions (what knobs exist):
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/include/flashinfer/attention/variants.cuh`
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/include/flashinfer/attention/decode.cuh`
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/include/flashinfer/attention/hopper/*` (SM90)
- `.codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/include/flashinfer/attention/blackwell/*` (SM100)

Practical "gather" recipe (copy/paste):
```bash
# List attention-related python entry points
rg -n "class BatchAttention|BatchPrefill|BatchDecode|determine_attention_backend" \
  .codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/flashinfer

# Jump to CUDA launch sites / bindings
rg -n "batch_(decode|prefill|attention)" \
  .codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/csrc | head

# Find variant/feature switches (masking, sliding window, alibi, logits soft cap)
rg -n "use_(custom_mask|sliding_window|logits_soft_cap|alibi)" \
  .codex/skills/gpu-kernel-optimizer/assets/corpora/flashinfer/include/flashinfer/attention/variants.cuh
```

### Step 3: Compute feasibility (score each kernel family against your target)

For each FlashInfer kernel family you might use (prefill/decode/paged/cascade/sparse), record:
- **Regime**: decode vs prefill, bucket set (BS, seq_len, head_dim)
- **Requirements**: dtype(s), head_dim constraints, KV layout (paged), mask/pos-enc features, arch (SM80/SM90/SM100)
- **Integration cost**:
  - 1: already supported by vLLM backend selection and matches your config
  - 3: minor glue (workspace planning / dispatch / guards)
  - 5: major integration (new backend, API/shape contract changes)
- **CUDA graph friendliness**: plan/run staging, workspace reuse, shape stability across buckets

Score each candidate (0–5):
- **Impact**: time share of the targeted attention/KV kernels in `nsys`
- **Feasibility**: how directly it drops into vLLM for your config (layout/dtype/arch)
- **Risk**: numerical parity, backend compatibility, graph-capture stability

Write the result as candidate OP rows in the Stage 3 opportunity list (Section 3A).

### Step 4: Expected ROI sanity-check

Compute a conservative end-to-end bound:
- If attention share is `s_attn`, a 20% attention speedup cannot yield >`0.2 * s_attn` end-to-end improvement.
- Use `references/e2e-delta-math.md` style reasoning to avoid chasing impossible wins.

## Unique Insights (U1-U6)

### U1: Bit-Packing Expert IDs (BS ≤ 8)
Pack 8 expert IDs into `uint64_t`, use `__popc` for count, `__ffs` for iteration.

### U2: 32-Byte Column Swizzling
```cpp
uint32_t rotate_col_32(uint32_t col, uint32_t row) {
    uint32_t col_base = col & 0xff9f;
    uint32_t col_rot = (col + 0x20 * row) & 0x60;
    return col_base | col_rot;
}
```
Eliminates bank conflicts for columnar reads.

### U3: Fused Gate + Activation in GEMM Reduction
```cpp
// After MMA, fuse SiLU
float x0 = d0 * ts0 * ws0;
float w0 = d2 * ts0 * ws1;
float sig0 = (w0 * x0) / (1 + expf(-x0));
```

### U4: Minimal Global/Shared Split
Only persist what's needed across phases:
- Quantized activations (up → down)
- Temp `[M,N]` (up → down)
- Everything else in SMEM/registers

### U5: Cooperative Grid Sync
```cpp
// Host: cudaLaunchCooperativeKernel(...)
// Device: cooperative_groups::this_grid().sync();  // ~1-2μs vs ~5-10μs launch
```
Requirement: Grid size ≤ SM count.

Use only when per‑expert load is balanced and barriers are amortized; otherwise prefer split kernels.

### U6: Monokernel As Fast Path
Wire monokernel as a conditional fast path and keep fused MoE as a fallback for mismatched dims or regressions.

---

## T14: Split-H for Small Batch SM Utilization (★★★★☆)

**Problem**: For very small batch sizes (BS ≤ 4), standard grid sizing (one block per K-slice) leaves SMs underutilized.

**Example** (Qwen3 on L40S):
```
BS=1, top_k=8: only 8 token-expert pairs
Standard grid: 128 blocks (K/16 slices)
But only ~8-16 pairs to process → most blocks idle
SM utilization: 128 blocks / 142 SMs → each block touches few pairs
```

**Solution**: Use Split-H to have multiple blocks collaborate per (token, expert) pair.

### Configuration

```cpp
template <typename Dims>
struct KernelConfig {
    static constexpr uint32_t SM_COUNT = 142;  // L40S
    static constexpr uint32_t SPLIT_H_THRESHOLD = 4;
    static constexpr uint32_t MAX_SPLIT_FACTOR = 16;

    // Calculate optimal split factor
    __host__ __device__ static constexpr uint32_t get_split_factor(uint32_t batch_size) {
        if (batch_size > SPLIT_H_THRESHOLD) return 1;

        uint32_t total_pairs = batch_size * Dims::TOP_K;
        uint32_t target_blocks = (SM_COUNT * 8) / 10;  // 80% utilization target
        uint32_t split = (target_blocks + total_pairs - 1) / total_pairs;
        return split < MAX_SPLIT_FACTOR ? split : MAX_SPLIT_FACTOR;
    }

    // Dynamic grid size
    __host__ __device__ static constexpr uint32_t get_grid_size(uint32_t batch_size) {
        if (batch_size <= SPLIT_H_THRESHOLD) {
            return batch_size * Dims::TOP_K * get_split_factor(batch_size);
        }
        return STANDARD_GRID_SIZE;  // K / 16
    }
};
```

### Example Calculations

| BS | top_k | Pairs | Standard Grid | Split Factor | Split-H Grid | SM Util |
|----|-------|-------|---------------|--------------|--------------|---------|
| 1 | 8 | 8 | 128 | 14 | 112 | 79% |
| 2 | 8 | 16 | 128 | 7 | 112 | 79% |
| 4 | 8 | 32 | 128 | 4 | 128 | 90% |
| 8 | 8 | 64 | 128 | 1 | 128 | 90% |

### Implementation Pattern

```cpp
template <typename Dims>
__device__ void moe_down_projection_split_h(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const W_element* expert_weights_down,
    const S_element* expert_scales_down,
    uint32_t num_tokens)
{
    uint32_t split_factor = KernelConfig::get_split_factor(num_tokens);

    if (split_factor == 1) {
        // Standard path: one block per K-slice
        moe_down_projection_standard<Dims>(...);
        return;
    }

    // Split-H: multiple blocks per (token, expert) pair
    uint32_t total_pairs = num_tokens * Dims::TOP_K;

    // Map blockIdx to (pair_idx, split_idx)
    uint32_t pair_idx = blockIdx.x / split_factor;
    uint32_t split_idx = blockIdx.x % split_factor;

    if (pair_idx >= total_pairs) return;

    // Each split handles different K-range
    uint32_t k_per_split = Dims::K / split_factor;
    uint32_t k_start = split_idx * k_per_split;
    uint32_t k_end = (split_idx == split_factor - 1) ? Dims::K : (split_idx + 1) * k_per_split;

    // Process this K-slice for this pair
    // IMPORTANT: Output overlaps across blocks; require reduction (atomic or staged)
    for (uint32_t k = k_start; k < k_end; k += W_DOWN_TILE) {
        // ... compute partial result for K-slice ...

        // Atomic or staged accumulation required
        atomicAdd(&scratchpad->output_accum[token_idx * Dims::K + k], partial_result);
    }
}
```

### When to Use Split-H

```python
def should_use_split_h(batch_size: int, top_k: int, sm_count: int) -> bool:
    """
    Split-H beneficial when standard grid underutilizes SMs.
    """
    total_pairs = batch_size * top_k
    standard_utilization = total_pairs / sm_count

    # Below 50% utilization → Split-H helps
    return standard_utilization < 0.5 and batch_size <= 4
```

### Tradeoffs

| Pro | Con |
|-----|-----|
| Better SM utilization for tiny batches | Requires reduction (atomic/staged) |
| Hides memory latency with more blocks | More complex block assignment |
| Can achieve 80%+ SM utilization at BS=1 | Additional coordination overhead |

**Recommendation**: Use as optional optimization. Default to standard grid, enable Split-H via environment variable or when profiling shows underutilization.

```cpp
// Configuration flag
static constexpr bool ENABLE_SPLIT_H = true;

// Runtime check
if constexpr (ENABLE_SPLIT_H) {
    if (batch_size <= SPLIT_H_THRESHOLD) {
        moe_down_projection_split_h<Dims>(...);
        return;
    }
}
moe_down_projection_standard<Dims>(...);
```
