# Algorithmic Branching Decisions

This document defines the key algorithmic decisions for MoE monokernel optimization.

## Decision 0: Monokernel Applicability

**When to use monokernel vs stock fused_moe.**

### Saturation Score

```python
def should_use_monokernel(batch_size: int, top_k: int, sm_count: int) -> bool:
    """
    Calculate saturation score to determine kernel strategy.
    
    Saturation measures how well the workload fills the GPU:
    - Low saturation (< 0.5): SM underutilization, monokernel wins
    - High saturation (>= 0.5): Enough parallelism, stock kernel OK
    """
    saturation = (batch_size * top_k) / sm_count
    
    if batch_size <= 64 and saturation < 0.5:
        return True   # Use monokernel
    else:
        return False  # Use stock fused_moe
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
```

---

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

---

## Decision A: Output Accumulation Path

**How expert outputs combine into final result.**

### Direct Write (top_k == 1)

When only one expert is selected per token, each expert writes directly:

```cpp
// No race condition - each token has exactly one expert
output[token_idx] = expert_output * routing_weight;
```

**Use when**: `top_k == 1` (Llama 4)

### Atomic Accumulation (top_k > 1)

When multiple experts contribute to each token:

```cpp
// Multiple experts write to same output location
atomicAdd(&output[token_idx], expert_output * routing_weight);
```

**Use when**: `top_k > 1` (Qwen3, DeepSeek, Mixtral)

### Configuration

```cpp
#if TOP_K == 1
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

For top_k=1 (single expert per token):
```
output = weight * activation(expert_output)
       = activation(weight * expert_output)  // Equivalent when only one term
```
Order doesn't matter because there's only one expert contribution.

For top_k>1 (multiple experts per token):
```
output = Σᵢ [weightᵢ * activation(expertᵢ_output)]  // CORRECT
       ≠ activation(Σᵢ [weightᵢ * expertᵢ_output])  // WRONG
```
Each expert's activation must be computed independently, THEN weighted and summed.

### Decision Logic

```cpp
if (TOP_K == 1) {
    // Llama 4 pattern: weight can be folded into scale before activation
    // Because there's only one expert, order doesn't affect result
    APPLY_WEIGHT = BEFORE_ACTIVATION;
    
    // In code:
    float scaled = x * topk_weight * quantization_scale;
    float result = silu(scaled);  // Weight already applied
    
} else {  // TOP_K > 1
    // Qwen3/DeepSeek/Mixtral pattern: weight MUST be applied AFTER activation
    // Each expert's contribution: output_i = weight_i * activation(gate_i, up_i)
    APPLY_WEIGHT = AFTER_ACTIVATION;
    
    // In code:
    float activated = silu(gate, x);  // Pure activation, no weight
    float result = activated * topk_weight;  // Weight applied AFTER
}
```

### Code Reference (from Qwen3 implementation)

```cpp
// WRONG for top_k > 1 - DO NOT DO THIS:
// float weighted_gate = gate * topk_weight;
// float silu = (weighted_gate * x) / (1 + expf(-x));

// CORRECT for top_k > 1:
// Step 1: Compute SiLU without weight
float silu_result = (gate * x) / (1.0f + expf(-x));

// Step 2: Apply routing weight AFTER activation
float final_output = silu_result * topk_weight;

// Step 3: Accumulate (atomic for top_k > 1)
atomicAdd(&output[token_idx], final_output);
```

### Llama 4 vs Qwen3 Comparison

| Aspect | Llama 4 (top_k=1) | Qwen3 (top_k=8) |
|--------|-------------------|-----------------|
| Weight timing | Before SiLU | After SiLU |
| Can fold into scale | Yes | No |
| Atomic output | No | Yes |
| Code pattern | `silu(x * weight * scale)` | `silu(gate, x) * weight` |

### Configuration

```cpp
// In kernel config
static constexpr bool WEIGHT_AFTER_ACTIVATION = (TOP_K > 1);

// In up-projection reduction
template <bool WeightAfter>
__device__ void apply_activation_and_weight(float gate, float x, float weight) {
    float silu = (gate * x) / (1.0f + expf(-x));
    
    if constexpr (WeightAfter) {
        return silu * weight;  // top_k > 1
    } else {
        return silu;  // top_k == 1, weight already folded
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
// Output via atomicAdd from all blocks
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
  ├─► Decision A: top_k == 1?
  │     YES → USE_ATOMICS = false
  │     NO  → USE_ATOMICS = true
  │
  ├─► Decision B: coalesce_size = E × dtype_bytes
  │     >= 128 → TOKENS_PER_WARP = 1
  │     < 128  → TOKENS_PER_WARP = 128 / coalesce_size
  │
  ├─► Decision C: top_k == 1?
  │     YES → APPLY_WEIGHT = before_activation
  │     NO  → APPLY_WEIGHT = after_activation  (CRITICAL!)
  │
  ├─► Decision D: num_shared_experts > 0?
  │     NO  → SHARED_STRATEGY = none
  │     YES → SHARED_STRATEGY = sidecar
  │
  ├─► Decision E: saturation < 0.25?
  │     YES → KERNEL = split_h_latency
  │     NO  → KERNEL = standard_latency
  │
  └─► Proceed to SRAM Tetris (tiling-config.md)
```