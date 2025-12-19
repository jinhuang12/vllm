# Algorithmic Branching Decisions

This document defines the key algorithmic decisions for MoE monokernel optimization.

## Decision 0: Monokernel Applicability

**When to use monokernel vs stock fused_moe.**

This decision must consider **both** saturation and per‑expert M under uniform routing.
Saturation alone is insufficient for sparse‑M regimes (top_k>1 with large E_global).

**Barrier budget gate**: If the design needs >1–2 grid‑wide `grid.sync` barriers, prefer split‑kernel or token‑major unless M_avg is large and work is balanced.

### Saturation + Uniform‑Routing M_avg

```python
def m_avg_uniform(batch_size: int, top_k: int, e_global: int) -> float:
    """
    Uniform routing estimate:
      M_avg = expected token‑expert pairs per expert (per rank cancels out).
    """
    return (batch_size * top_k) / e_global


def should_use_monokernel(batch_size: int, top_k: int, sm_count: int, bs_threshold: int) -> bool:
    """
    Saturation measures how well the workload fills the GPU:
    - Low saturation (< 0.5): SM underutilization, fusion may help
    - High saturation (>= 0.5): Enough parallelism, stock kernel often OK
    """
    saturation = (batch_size * top_k) / sm_count
    return batch_size <= bs_threshold and saturation < 0.5

# Note: bs_threshold is hardware-specific (see gpu-configs.md).
```

### M_avg Thresholds (Uniform Routing Heuristic)

```
M_avg = BS * top_k / E_global

If M_avg < 8:    expert‑major GEMM is usually under‑utilized → prefer token‑major or split kernels
If 8–16:         hybrid or benchmark (token‑major down‑proj often wins)
If >= 16:        expert‑major may be viable (weight reuse can pay off)

If you have routing traces, prefer **measured per‑expert counts** (p50/p95) over uniform M_avg.
```

### Hardware SM Counts

| GPU | SM Count | Saturation Threshold (BS×K < 0.5×SM) |
|-----|----------|--------------------------------------|
| H100 SXM | 132 | BS × top_k < 66 |
| H200 | 132 | BS × top_k < 66 |
| L40S | 142 | BS × top_k < 71 |
| A100 80GB | 108 | BS × top_k < 54 |

### Examples

```python
# Qwen3-30B-A3B: top_k=8, L40S (142 SMs)
# BS=1: saturation = 1*8/142 = 0.056 → USE MONOKERNEL
# BS=8: saturation = 8*8/142 = 0.45 → USE MONOKERNEL  
# BS=16: saturation = 16*8/142 = 0.90 → USE STOCK

# Llama 4: top_k=1, H200 (132 SMs)
# BS=1: saturation = 1*1/132 = 0.008 → USE MONOKERNEL
# BS=64: saturation = 64*1/132 = 0.48 → USE MONOKERNEL
# BS=128: saturation = 128*1/132 = 0.97 → USE STOCK

# DeepSeek‑like (E_global=256, top_k=8, BS=8)
# M_avg = 8*8/256 = 0.25 → expert‑major will be sparse → prefer token‑major or split kernels

**EP note**: If EP is enabled and dispatch happens before the kernel, use `E_local` for M_avg and ownership decisions.
```

---

## Decision 0b: Ownership Model (Uniform Routing)

Choose **token‑major** vs **expert‑major** based on M_avg.
If routing histograms are available, prefer p50/p95 per‑expert counts over uniform M_avg.

```python
def choose_ownership(batch_size: int, top_k: int, e_global: int) -> str:
    m_avg = m_avg_uniform(batch_size, top_k, e_global)
    if m_avg < 8:
        return "token_major"   # M too small for expert‑major
    if m_avg < 16:
        return "hybrid"        # token‑major down‑proj, expert‑major up‑proj
    return "expert_major"
```

## Decision 0c: Fusion Boundary

Decide single cooperative kernel vs split kernels **after** ownership is chosen.

Guidelines:
- **Token‑major**: prefer **split kernels** (prepare/quantize, up‑proj, down‑proj, convert) to avoid grid.sync barriers.
- **Expert‑major**: cooperative monokernel can be viable if M_avg is large and grid.sync count is low.

## Roofline Justification

**Why monokernel works for low-batch MoE decode.**

### Arithmetic Intensity

```
AI = FLOPs / Bytes_transferred

For FP8 W8A8 MoE GEMM:
- FLOPs per output: 2 × K (multiply-add)
- Bytes per output: K × sizeof(FP8) = K bytes
- AI ≈ 2 FLOPs/byte

For BF16 MoE GEMM:
- FLOPs per output: 2 × K
- Bytes per output: K × sizeof(BF16) = 2K bytes
- AI ≈ 1 FLOP/byte
```

### Ridge Point Comparison

| GPU | Peak FP8 TFLOPs | HBM BW (GB/s) | Ridge Point (FLOPs/byte) |
|-----|-----------------|---------------|--------------------------|
| H100 SXM | 1979 | 3350 | ~590 |
| H200 | 1979 | 4800 | ~412 |
| L40S | 733 | 864 | ~848 |

**MoE decode AI ≈ 1-2 FLOPs/byte**, which is **200-400× below ridge point**.

### Implication

At such low arithmetic intensity, performance is entirely memory-bandwidth limited. The monokernel advantage comes from:

1. **Eliminating HBM round-trips**: Router → SMEM (not HBM) → GEMM
2. **Fusing quantization**: Quantize activations in registers, not via global memory
3. **Cooperative loading**: All SMs share weight loading overhead

**Caveat**: In sparse‑M regimes (low per‑expert counts), kernels can be **latency/occupancy/sync‑bound** rather than bandwidth‑bound. Use roofline as one lens, not the sole decision driver.

---

## Decision A: Output Accumulation Path

**How expert outputs combine into final result.**

### Direct Write (unique ownership)

When only one expert is selected per token, each expert writes directly:

```cpp
// No race condition - each token has exactly one expert
output[token_idx] = expert_output * routing_weight;
```

**Use when**: output elements are uniquely owned (often `top_k == 1` or token‑major K‑slice).

### Conditional Accumulation (output overlap)

When multiple experts contribute to each token, the need for atomics depends on **ownership**.

```cpp
// Expert‑major or Split‑H: multiple blocks write same output location
atomicAdd(&output[token_idx], expert_output * routing_weight);

// Token‑major K‑slice ownership: each block owns unique [token, k] range
output[token_idx * K + k_idx] += expert_output * routing_weight;  // no atomics
```

**Use atomics when**: ownership is expert‑major (or Split‑H) and multiple blocks write same output.
**Avoid atomics when**: token‑major with K‑slice ownership.

### Configuration

```cpp
#if TOP_K == 1
    #define USE_ATOMICS 0
#elif OWNERSHIP_TOKEN_MAJOR
    #define USE_ATOMICS 0
#else
    #define USE_ATOMICS 1
#endif
```

---

## Decision B: Sorter Strategy

**How to organize tokens by expert for coalesced memory access.**

### Sorter Warp Packing

```cpp
coalesce_size = E_local × dtype_bytes  // 1 for FP8, 2 for BF16

if (coalesce_size >= 128) {
    TOKENS_PER_WARP = 1;           // One warp per token (coalescing optimized)
} else {
    TOKENS_PER_WARP = 128 / coalesce_size;  // Pack tokens (latency optimized)
}
```

### Examples

| Model | E_local | dtype | coalesce_size | TOKENS_PER_WARP |
|-------|---------|-------|---------------|-----------------|
| Llama 4 (TP=8) | 16 | FP8 | 16 | 8 |
| Qwen3-30B-A3B | 128 | FP8 | 128 | 1 |
| DeepSeek-V2 (TP=8) | 20 | FP8 | 20 | 6 |
| Mixtral (TP=1) | 8 | BF16 | 16 | 8 |

---

## Decision C: Weight Application Order (CORRECTNESS CRITICAL)

**When to apply routing weights relative to activation function.**

This decision is **critical for numerical correctness**. Applying weights at the wrong stage produces incorrect output that may be hard to detect.

### The Mathematical Difference

Routing weights are part of the **model definition**. You must match where the model applies them.

For any top_k:
```
output = Σᵢ [weightᵢ * activation(expertᵢ_output)]  // Typical MoE definition
```

**Important**: Even when `top_k=1`, `activation(weight * x)` is **not generally equivalent** to `weight * activation(x)` because activation is nonlinear. Only move weights across activation if the model definition explicitly does so (or if weights are constant 1.0).

### Decision Logic (derive from model code)

1. Read the model’s MoE forward path in vLLM.
2. Identify where routing weights are applied:
   - **After activation** (common): `output_i = weight_i * activation(gate_i, up_i)`
   - **Before activation** (model‑specific): weight folded into input/scale **by design**
3. Set `WEIGHT_AFTER_ACTIVATION` accordingly.

### Code Reference (from Qwen3 implementation)

```cpp
// WRONG when model expects post-activation weighting - DO NOT DO THIS:
// float weighted_gate = gate * topk_weight;
// float silu = (weighted_gate * x) / (1 + expf(-x));

// CORRECT when model expects post-activation weighting:
// Step 1: Compute SiLU without weight
float silu_result = (gate * x) / (1.0f + expf(-x));

// Step 2: Apply routing weight AFTER activation
float final_output = silu_result * topk_weight;

// Step 3: Accumulate (atomic only if ownership overlaps)
// Expert‑major / Split‑H:
//   atomicAdd(&output[token_idx], final_output);
// Token‑major K‑slice:
//   output[token_idx * K + k_idx] += final_output;
```

### Llama 4 vs Qwen3 Comparison

| Aspect | Llama 4 (top_k=1) | Qwen3 (top_k=8) |
|--------|-------------------|-----------------|
| Weight timing | Before SiLU | After SiLU |
| Can fold into scale | Yes | No |
| Atomic output | No | Yes (expert‑major), No (token‑major) |
| Code pattern | `silu(x * weight * scale)` | `silu(gate, x) * weight` |

These are model‑specific examples; do not infer weight placement from `top_k` alone.

### Configuration

```cpp
// In kernel config
static constexpr bool WEIGHT_AFTER_ACTIVATION = {true/false};  // from model semantics

// In up-projection reduction
template <bool WeightAfter>
__device__ void apply_activation_and_weight(float gate, float x, float weight) {
    float silu = (gate * x) / (1.0f + expf(-x));
    
    if constexpr (WeightAfter) {
        return silu * weight;
    } else {
        return silu;  // weight folded earlier by model definition
    }
}
```

---

## Decision D: Shared Expert Strategy

**How to handle always-active shared experts.**

### Model Shared Expert Configuration

| Model | Routed Experts | Shared Experts | Total |
|-------|---------------|----------------|-------|
| Qwen3-MoE | 128 (top-8) | 0 | 128 |
| Llama 4 | 16 (top-1) | 1 | 17 |
| DeepSeek-V2/V3 | 160 (top-8) | 2 | 162 |
| Mixtral | 8 (top-2) | 0 | 8 |

### Strategy A: No Shared Experts (Qwen3, Mixtral)

Skip shared expert handling entirely:

```cpp
#define NUM_SHARED_EXPERTS 0
// All blocks work on routed experts only
```

### Strategy B: Sidecar Pattern (Llama 4, DeepSeek)

Reserve a fraction of SM blocks for shared expert computation:

```cpp
#define NUM_SHARED_EXPERTS 1  // or 2 for DeepSeek
#define S_SHARED (SM_COUNT / 10)  // 10% of blocks for shared

// Grid layout:
// [0]: Controller block (routing)
// [1, S*K_ROUT]: Worker blocks (routed experts)
// [S*K_ROUT+1, S*K_ROUT+S_SHARED]: Shared expert blocks
```

**Sidecar sizing**:
- Llama 4: `S_SHARED = SM_COUNT / 10` (10% for 1 shared expert)
- DeepSeek: `S_SHARED = SM_COUNT / 5` (20% for 2 shared experts)

### Strategy C: Sequential (fallback)

Process shared expert after routed experts complete:

```cpp
// Phase 1: All blocks do routed experts
cooperative_groups::this_grid().sync();

// Phase 2: Subset of blocks do shared expert
if (blockIdx.x < S_SHARED) {
    compute_shared_expert(...);
}
```

Use when: Sidecar causes SM imbalance or debugging.

### DeepSeek Shared Expert Pattern (Preview)

DeepSeek-V2/V3 has dedicated shared experts that process ALL tokens, not just routed ones. This requires special handling:

```cpp
// DeepSeek pattern: N routed experts (top-k) + M shared experts (always)
template <typename Dims>
__device__ void moe_with_deepseek_shared_experts(
    const A_element* activations,
    const W_element* routed_weights,      // Shape: [E_routed, 2*N, K]
    const W_element* shared_weights,      // Shape: [E_shared, 2*N, K]
    MoE_SHM<Dims>* shmem,
    MoEGemmSpec<Dims>* scratchpad)
{
    // ========================================
    // Phase 1: Shared experts (ALL tokens)
    // ========================================
    // Unlike routed experts, shared experts process every token
    // No routing decision needed - just GEMM all tokens
    if (blockIdx.x < S_SHARED) {
        uint32_t shared_expert_id = blockIdx.x % Dims::NUM_SHARED_EXPERTS;

        // Process all tokens for this shared expert
        for (uint32_t tok = 0; tok < num_tokens; tok += T_TILE) {
            // Up-projection for shared expert
            shared_up_projection<Dims>(
                activations + tok * Dims::K,
                shared_weights[shared_expert_id],
                scratchpad->shared_temp,
                min(T_TILE, num_tokens - tok));

            // Down-projection with weight=1.0 (shared experts unweighted)
            shared_down_projection<Dims>(
                scratchpad->shared_temp,
                shared_weights[shared_expert_id],
                scratchpad->output_accum,  // Accumulate to same buffer
                1.0f);  // No routing weight for shared
        }
    }

    // ========================================
    // Phase 2: Routed experts (top-k per token)
    // ========================================
    // Grid sync to ensure shared experts complete
    cooperative_groups::this_grid().sync();

    // Standard routing + expert execution
    if (blockIdx.x == 0) {
        topk_route<Dims>(router_logits, num_tokens, shmem);
    }
    __syncthreads();

    // Process routed experts as usual
    moe_up_projection<Dims>(...);
    cooperative_groups::this_grid().sync();
    moe_down_projection<Dims>(...);
}
```

**Key Implementation Details for DeepSeek**:

1. **Separate weight tensors**: Shared experts have their own weights, not in the main expert array
2. **No routing weight**: Shared expert output is NOT multiplied by routing weight
3. **Accumulation order**: Shared + routed outputs sum into same FP32 accumulator
4. **Grid partitioning**: Reserve `NUM_SHARED_EXPERTS * K_SLICES` blocks for shared work

```cpp
// DeepSeek grid layout
constexpr uint32_t SHARED_BLOCKS = Dims::NUM_SHARED_EXPERTS * (Dims::K / K_TILE);
constexpr uint32_t ROUTED_BLOCKS = SM_COUNT - SHARED_BLOCKS;

// Block assignment
if (blockIdx.x < SHARED_BLOCKS) {
    // Process shared expert: blockIdx.x / (K/K_TILE) gives expert ID
    uint32_t shared_id = blockIdx.x / (Dims::K / K_TILE);
    uint32_t k_slice = blockIdx.x % (Dims::K / K_TILE);
    process_shared_expert(shared_id, k_slice, ...);
} else {
    // Process routed experts
    process_routed_experts(blockIdx.x - SHARED_BLOCKS, ...);
}
```

---

## Decision E: Kernel Architecture

**Which kernel pattern to use based on workload.**

See `references/architecture-pattern.md` for detailed implementation.

### Split-H Latency Kernel

**Use when**: `saturation < 0.5`

Multiple blocks collaborate on single token by splitting hidden dimension:

```cpp
// Each block handles H_chunk of hidden dimension
H_chunk = ceil(H / S);  // S = split factor
// Output reduction required across blocks (atomic or staged reduction)
```

### Standard Latency Kernel

**Use when**: `0.5 <= saturation < 1.0`

One block per (token, expert) pair:

```cpp
// Grid: BS × top_k blocks
// Each block: full hidden dimension for one (token, expert)
```

### Stock fused_moe

**Use when**: `saturation >= 1.0` or `BS > 64`

Fall back to vLLM's built-in fused_moe kernel.

---

## Decision F: GEMM Strategy (Per-Pair GEMV vs Expert-Grouped GEMM)

**When is expert grouping worthwhile?**

Expert grouping (sorting tokens by expert, then processing all tokens for each expert together) enables weight reuse but adds overhead. Use the Poisson occupancy model to decide.

### Formula

```python
import math

def expected_weight_reuse(batch_size: int, top_k: int, num_experts: int) -> float:
    """
    Calculate expected weight reuse factor from expert grouping.

    Returns r_max: theoretical max speedup from grouping (1.0 = no benefit).

    Based on Poisson occupancy model for uniform routing.
    """
    P = batch_size * top_k  # Total routed pairs
    λ = P / num_experts     # Expected load per expert (use E_local if EP pre-dispatch)

    if λ < 0.01:  # Edge case: very sparse
        return 1.0

    # Expected active experts (Poisson occupancy: non-empty bins)
    A = num_experts * (1 - math.exp(-λ))

    # Max reuse factor = avg tokens per active expert
    r_max = P / A  # Equivalently: λ / (1 - e^{-λ})

    return r_max
```

### Decision Logic

```python
GROUPING_THRESHOLD = 2.0  # Grouping worthwhile when r_max >= 2

r_max = expected_weight_reuse(batch_size, top_k, num_experts)

if r_max >= GROUPING_THRESHOLD:
    USE_EXPERT_GROUPING = True   # Expert-grouped GEMM
else:
    USE_EXPERT_GROUPING = False  # Per-pair GEMV
```

### Why Threshold = 2.0?

1. **Grouping overhead**: Sorting requires histogram, prefix sum, extra control flow
2. **Break-even point**: r_max < 1.5 means overhead likely exceeds benefit
3. **Tensor Core efficiency**: GEMM wants M>=8-16 per expert; when λ<=1, P(M>=8)≈0
4. **Margin**: r_max >= 2.0 ensures clear win after accounting for overhead

### Worked Examples

| Model | E | k | B_max | λ=Bk/E | r_max | Strategy |
|-------|---|---|-------|--------|-------|----------|
| gpt-oss-120b TP4 | 128 | 4 | 32 | 1.0 | 1.58 | **Per-pair GEMV** |
| Qwen3-30B-A3B | 128 | 8 | 64 | 4.0 | 3.66 | **Grouped-GEMM** |
| Llama 4 Scout | 16 | 1 | 64 | 4.0 | 3.66 | **Grouped-GEMM** |
| Mixtral 8x7B | 8 | 2 | 64 | 16.0 | 15.8 | **Grouped-GEMM** |
| DeepSeek-V2 TP8 | 20 | 6 | 32 | 9.6 | 9.1 | **Grouped-GEMM** |

### Key Insight

For models with **many experts (E>=64) and small top_k (k<=4)**, per-pair GEMV is often correct because:
- λ = Bk/E remains small even at moderate batch sizes
- Most experts visited at most once → no weight reuse opportunity
- Grouping overhead wasted

For models with **few experts (E<=16) or large top_k (k>=6)**, grouped-GEMM wins because:
- λ grows quickly with batch size
- Multiple tokens per expert → significant weight reuse
- Tensor Cores can leverage batched M>1

### With Skewed Routing (Real Histograms)

If you have actual routing distributions (not uniform):

```python
def actual_weight_reuse(expert_histogram: list[int]) -> float:
    """
    Calculate actual reuse from observed routing.

    expert_histogram[e] = number of tokens routed to expert e
    """
    P = sum(expert_histogram)  # Total pairs
    A = sum(1 for count in expert_histogram if count > 0)  # Active experts
    return P / A if A > 0 else 1.0
```

Hot experts (skewed routing) increase actual reuse above Poisson prediction.
Collect histograms during profiling to refine the decision.

### Configuration

```cpp
// In kernel config
template <typename Dims>
struct GEMMStrategy {
    // Calculate at compile time or runtime based on BS
    static constexpr float lambda_max = float(Dims::BS_MAX * Dims::TOP_K) / Dims::NUM_EXPERTS;
    static constexpr float r_max_approx = lambda_max / (1.0f - expf(-lambda_max));
    static constexpr bool USE_EXPERT_GROUPING = (r_max_approx >= 2.0f);
};
```

---

## Decision Summary Flowchart

```
START
  │
  ├─► Calculate saturation = BS × top_k / SM_count
  │
  ├─► saturation >= 0.5 OR BS > 64?
  │     YES → Use stock fused_moe (DONE)
  │     NO  ↓
  │
  ├─► Calculate M_avg = BS × top_k / E_global (uniform routing)
  │
  ├─► M_avg >= M_AVG_THRESHOLD?
  │     YES → Prefer split-kernel or grouped GEMM (DONE)
  │     NO  ↓
  │
  ├─► Decision 0b: ownership overlaps output?
  │     YES → USE_ATOMICS = true (or staged reduction)
  │     NO  → USE_ATOMICS = false (token-major K-slice)
  │
  ├─► Baseline routing share >= 15–20% AND barrier budget <= 2?
  │     YES → Full monokernel is viable (continue)
  │     NO  → Prefer split/hybrid; target expert kernel cost
  │
  ├─► Decision 0c: fusion boundary?
  │     MONO → continue
  │     SPLIT → split routing/quant + GEMM
  │
  ├─► Decision B: coalesce_size = E × dtype_bytes
  │     >= 128 → TOKENS_PER_WARP = 1
  │     < 128  → TOKENS_PER_WARP = 128 / coalesce_size
  │
  ├─► Decision C: weight placement from model semantics
  │     BEFORE → APPLY_WEIGHT = before_activation
  │     AFTER  → APPLY_WEIGHT = after_activation
  │
  ├─► Decision D: num_shared_experts > 0?
  │     NO  → SHARED_STRATEGY = none
  │     YES → SHARED_STRATEGY = sidecar
  │
  ├─► Decision E: saturation < 0.25?
  │     YES → KERNEL = split_h_latency
  │     NO  → KERNEL = standard_latency
  │
  ├─► Decision F: r_max = λ / (1 - e^{-λ}) where λ = BS × top_k / E
  │     r_max >= 2.0 → USE_EXPERT_GROUPING = true (Grouped-GEMM)
  │     r_max < 2.0  → USE_EXPERT_GROUPING = false (Per-pair GEMV)
  │
  ├─► Decision G: output overlap?
  │     YES → USE_FP32_ACCUMULATOR = true (atomic/staged reduction)
  │     NO  → USE_FP32_ACCUMULATOR = false (direct write)
  │
  └─► Proceed to SRAM Tetris (tiling-config.md)
```

---

## Decision G: Multi-Expert Accumulation (top_k > 1)

**Detailed implementation for accumulating outputs from multiple experts.**

This decision expands on Decision 0b for top_k > 1 cases, addressing when FP32 scratchpad accumulation is required and how to map pair indices back to tokens.

### The Problem

When top_k > 1, multiple experts contribute to each token's output:
- Pair indices must map back to token indices for accumulation
- FP32 accumulation is still recommended for numerical stability
- **Atomics are only required when output ownership overlaps** (expert-major or split-H)

### Solution: FP32 Scratchpad Accumulator (when ownership overlaps)

```cpp
// Scratchpad structure for top_k > 1
template <typename Dims>
struct MoEGemmSpec {
    // FP32 accumulator for output (NOT BF16!)
    // Shape: [BS, HIDDEN_STATES] - one accumulator per TOKEN (not per pair)
    float output_accum[Dims::BS * Dims::HIDDEN_STATES];

    // FP8 intermediate after up-projection (gated SiLU applied)
    // Shape: [BS * TOP_K, N] - one per pair
    AQ_element temp[Dims::BS * Dims::TOP_K * Dims::N];

    // Routing weights for applying after activation
    // Shape: [BS * TOP_K]
    float topk_weights[Dims::BS * Dims::TOP_K];
};
```

### Pair-to-Token Index Mapping

```cpp
// CRITICAL: Convert pair index to token index for accumulation
// pair_idx ranges from 0 to BS * TOP_K - 1
// token_idx ranges from 0 to BS - 1

__device__ inline uint32_t pair_to_token(uint32_t pair_idx) {
    return pair_idx / TOP_K;
}

// Example for BS=4, TOP_K=8:
// Pairs 0-7   → Token 0
// Pairs 8-15  → Token 1
// Pairs 16-23 → Token 2
// Pairs 24-31 → Token 3
```

### Down-Projection Accumulation Pattern

```cpp
// From Qwen3 implementation: moe_down_projection.cu
template <typename Dims>
__device__ void moe_down_accumulate_tc(
    MoE_SHM<Dims>* shm,
    MoEGemmSpec<Dims>* scratchpad,
    const uint16_t* pair_indices,  // Current tile's pair indices
    uint32_t k_offset,
    uint32_t num_valid)
{
    // Get pair index from shared memory
    uint16_t pair_idx = pair_indices[thread_row];

    // CRITICAL: Map pair → token for accumulation
    uint16_t token_idx = pair_idx / Dims::TOP_K;

    // ... compute GEMM partial result d0 ...

    // Accumulate to FP32 buffer (multiple experts sum here)
    // Not atomic when blocks own disjoint K-slices
    scratchpad->output_accum[token_idx * Dims::HIDDEN_STATES + k_idx] += d0;
}
```

### Final Conversion Phase

```cpp
// After all experts processed, convert FP32 → BF16 (only if scratchpad used)
template <typename Dims>
__device__ void convert_output_phase(
    MoEGemmSpec<Dims>* scratchpad,
    R_element* output,
    uint32_t num_tokens)
{
    // Grid sync ensures all accumulation complete
    cooperative_groups::this_grid().sync();

    // Parallel conversion
    for (uint32_t tok = blockIdx.x; tok < num_tokens; tok += gridDim.x) {
        for (uint32_t k = threadIdx.x; k < Dims::HIDDEN_STATES; k += blockDim.x) {
            float val = scratchpad->output_accum[tok * Dims::HIDDEN_STATES + k];
            output[tok * Dims::HIDDEN_STATES + k] = __float2bfloat16(val);
        }
    }
}
```

### When to Use atomicAdd vs Direct Write

```cpp
// Decision tree for accumulation method
if (blocks_own_disjoint_k_slices) {
    // Standard grid (one block per K/16 slice)
    // Direct += to FP32 accumulator - no race between blocks
    // Or accumulate in registers and write directly to output
    scratchpad->output_accum[token * K + k] += result;

} else {
    // Split-H or other overlapping schemes
    // Must use atomicAdd to FP32 accumulator
    atomicAdd(&scratchpad->output_accum[token * K + k], result);
}
```

### Memory Overhead

| Configuration | Output Accumulator Size | Notes |
|--------------|------------------------|-------|
| BS=8, K=2048 | 8 × 2048 × 4 = 64 KB | Only if output overlaps |
| BS=64, K=2048 | 64 × 2048 × 4 = 512 KB | Consider streaming |
| BS=1, K=5120 | 1 × 5120 × 4 = 20 KB | Negligible |

### Configuration

```cpp
// In kernel config
template <typename Dims>
struct AccumulationConfig {
    static constexpr bool USE_FP32_ACCUMULATOR = (Dims::OUTPUT_OVERLAP);

    // Scratchpad size for FP32 accumulator
    static constexpr size_t ACCUM_SIZE =
        USE_FP32_ACCUMULATOR ? (Dims::BS * Dims::HIDDEN_STATES * sizeof(float)) : 0;
};
```
