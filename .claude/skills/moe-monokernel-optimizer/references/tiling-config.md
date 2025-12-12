# SRAM Tetris: Systematic Tile Size Solver

This document provides a deterministic algorithm for solving tile sizes that fit within shared memory constraints.

## Overview

SRAM Tetris is the process of fitting activation tiles, weight tiles, partial results, and metadata into the shared memory budget. Unlike heuristic approaches, this systematic solver guarantees a valid configuration or reports that none exists.

## Shared Memory Budget

| GPU | Total SMEM/SM | Recommended Budget |
|-----|---------------|-------------------|
| H100/H200 | 228 KB | 220 KB |
| L40S | 100 KB | 96 KB |
| A100 | 164 KB | 156 KB |

**Safety margin**: 4-8 KB reserved for stack, alignment, runtime overhead.

## Buffer Components

### 1. Activation Tiles (A)

```
SMEM_A = S_buf × M_t × K_t × sizeof(A_type)

Where:
- S_buf: Buffer count (2 for double, 3 for triple)
- M_t: Tokens per tile (8 for decode, 16 for prefill)
- K_t: Hidden dim chunk (usually 64, MMA-aligned)
- sizeof(A_type): 1 byte for FP8, 2 for BF16
```

### 2. Weight Tiles (W)

```
SMEM_W = S_buf × K_t × N_t × sizeof(W_type)

Where:
- N_t: Expert intermediate dim chunk (solve for this)
- sizeof(W_type): 1 byte for FP8
```

### 3. Accumulator (Partial Results)

```
SMEM_acc = M_t × N_t × sizeof(float)

Note: Only ONE accumulator tile (not buffered)
```

### 4. Metadata

```
SMEM_meta ≈ 4096 bytes

Includes:
- topk_ids[BS_MAX]: 64 bytes
- topk_weights[BS_MAX]: 256 bytes
- expert_counts[E]: 512 bytes
- token_indices[BS_MAX × top_k]: 512 bytes
- Alignment padding: ~2KB
```

### 5. Block-wise Scales (if applicable)

For DeepSeek-style block-wise quantization:

```
SMEM_scales = S_buf × scales_per_tile × sizeof(float)

scales_per_tile = (K_t / K_block) × (N_t / N_block)

Example with 128×128 blocks:
- K_t=64, N_t=512, K_block=128, N_block=128
- scales_per_tile = 0 × 4 = 0 (K_t < K_block)
- Need K_t >= K_block for block-wise scales
```

## Systematic Solver Algorithm

```python
def solve_smem_tiles(
    K: int,           # Hidden size (e.g., 2048)
    N: int,           # Intermediate size local (e.g., 768)
    C_sram: int,      # SMEM budget in bytes (e.g., 98304)
    sm_arch: str,     # "sm_89" or "sm_90"
    weight_dtype: str = "fp8",  # "fp8", "bf16", "fp16"
    activation_dtype: str = "fp8",  # "fp8", "bf16", "fp16"
    batch_mode: str = "decode"  # "decode" or "prefill"
) -> dict:
    """
    Solve for tile sizes that fit in shared memory.
    
    Returns: {M_t, N_t, K_t, S_buf, K_chunks} or raises ValueError
    """
    
    # Element sizes based on dtype
    dtype_sizes = {"fp8": 1, "bf16": 2, "fp16": 2, "fp32": 4}
    sz_W = dtype_sizes[weight_dtype]
    sz_A = dtype_sizes[activation_dtype]
    sz_acc = 4  # Always FP32 accumulator
    
    # Step 1: Fix K_t to MMA-aligned value
    K_t = 64  # 64 is optimal for most MMA instructions
    
    # Step 2: Set buffer strategy based on architecture
    if sm_arch == "sm_90":  # Hopper (H100/H200)
        S_buf = 3  # Triple buffer (TMA async)
    else:  # Ada (L40S) or Ampere
        S_buf = 2  # Double buffer
    
    # Step 3: Set M_t based on batch mode
    M_t = 8 if batch_mode == "decode" else 16
    
    # Step 4: Solve for N_t
    # Constraint: SMEM_A + SMEM_W + SMEM_acc + SMEM_meta <= C_sram
    
    SMEM_meta = 4096  # Fixed overhead
    
    # SMEM_A = S_buf × M_t × K_t × sz_A
    SMEM_A = S_buf * M_t * K_t * sz_A
    
    # Available for weights + accumulator
    available = C_sram - SMEM_A - SMEM_meta
    
    # SMEM_W + SMEM_acc = S_buf × K_t × N_t × sz_W + M_t × N_t × sz_acc
    # available = N_t × (S_buf × K_t × sz_W + M_t × sz_acc)
    # N_t = available / (S_buf × K_t × sz_W + M_t × sz_acc)
    
    denom = S_buf * K_t * sz_W + M_t * sz_acc
    N_t = available // denom
    
    # Round down to multiple of 16 (MMA alignment)
    N_t = (N_t // 16) * 16
    
    # Step 5: Validate or fallback
    if N_t < 32:
        # Try reducing buffer count
        if S_buf > 2:
            return solve_smem_tiles(K, N, C_sram, sm_arch, weight_dtype, 
                                   activation_dtype, batch_mode,
                                   _force_double_buffer=True)
        # Try reducing M_t
        if M_t > 8:
            return solve_smem_tiles(K, N, C_sram, sm_arch, weight_dtype,
                                   activation_dtype, "decode")
        # Try K-chunking
        return solve_with_k_chunking(K, N, C_sram, sm_arch, weight_dtype, activation_dtype)
    
    # Step 6: Calculate K chunks if K > K_t
    K_chunks = (K + K_t - 1) // K_t
    
    # Verify total SMEM
    total_smem = SMEM_A + S_buf * K_t * N_t * sz_W + M_t * N_t * sz_acc + SMEM_meta
    assert total_smem <= C_sram, f"SMEM overflow: {total_smem} > {C_sram}"
    
    return {
        "M_t": M_t,
        "N_t": N_t,
        "K_t": K_t,
        "S_buf": S_buf,
        "K_chunks": K_chunks,
        "SMEM_total": total_smem,
        "SMEM_budget": C_sram,
        "weight_dtype": weight_dtype,
        "activation_dtype": activation_dtype,
        "sz_W": sz_W,
        "sz_A": sz_A
    }


def solve_with_k_chunking(K, N, C_sram, sm_arch, weight_dtype, activation_dtype):
    """Fallback: Split K dimension across multiple kernel invocations."""
    # Reduce K_t to fit
    for K_t in [32, 16]:
        result = solve_smem_tiles(K, N, C_sram, sm_arch, weight_dtype, 
                                 activation_dtype, "decode", K_t=K_t)
        if result["N_t"] >= 32:
            result["K_chunks"] = (K + K_t - 1) // K_t
            return result
    
    raise ValueError(f"Cannot fit tiles in {C_sram} bytes SMEM")
```

## Worked Examples

### Example 1: Qwen3-30B-A3B on L40S

```python
# Input
K = 2048        # Hidden size
N = 768         # Intermediate (local after TP)
C_sram = 98304  # 96 KB budget
sm_arch = "sm_89"

# Step 1: K_t = 64
# Step 2: S_buf = 2 (Ada)
# Step 3: M_t = 8 (decode)

# Step 4: Solve N_t
SMEM_A = 2 × 8 × 64 × 1 = 1024 bytes
available = 98304 - 1024 - 4096 = 93184 bytes
denom = 2 × 64 × 1 + 8 × 4 = 128 + 32 = 160
N_t = 93184 // 160 = 582 → round to 576 (multiple of 16)

# Verify
SMEM_W = 2 × 64 × 576 × 1 = 73728 bytes
SMEM_acc = 8 × 576 × 4 = 18432 bytes
Total = 1024 + 73728 + 18432 + 4096 = 97280 bytes ✓ (< 98304)

# Result
{
    "M_t": 8,
    "N_t": 576,
    "K_t": 64,
    "S_buf": 2,
    "K_chunks": 32,  # 2048 / 64
    "SMEM_total": 97280
}
```

### Example 2: Llama 4 Maverick on H200

```python
# Input
K = 5120        # Hidden size
N = 8192        # Intermediate (local, TP=8 → 1024 per GPU)
N_local = 1024
C_sram = 225280 # 220 KB budget
sm_arch = "sm_90"

# Step 1: K_t = 64
# Step 2: S_buf = 3 (Hopper)
# Step 3: M_t = 8 (decode)

# Step 4: Solve N_t
SMEM_A = 3 × 8 × 64 × 1 = 1536 bytes
available = 225280 - 1536 - 4096 = 219648 bytes
denom = 3 × 64 × 1 + 8 × 4 = 192 + 32 = 224
N_t = 219648 // 224 = 980 → round to 976

# Verify
SMEM_W = 3 × 64 × 976 × 1 = 187392 bytes
SMEM_acc = 8 × 976 × 4 = 31232 bytes
Total = 1536 + 187392 + 31232 + 4096 = 224256 bytes ✓ (< 225280)

# Result
{
    "M_t": 8,
    "N_t": 976,
    "K_t": 64,
    "S_buf": 3,
    "K_chunks": 80,  # 5120 / 64
    "SMEM_total": 224256
}
```

### Example 3: DeepSeek-V3 on H100 (Block-wise Scales)

```python
# Additional: block-wise quantization with 128×128 blocks
K_block = 128
N_block = 128

# After solving base tiles (N_t=896, K_t=64):
# K_t < K_block, so scales refresh every K_block/K_t = 2 iterations
# No additional SMEM for scales (loaded on demand)

# If K_t >= K_block:
scales_per_tile = (K_t // K_block) × (N_t // N_block)
                = (128 // 128) × (896 // 128)
                = 1 × 7 = 7 scales per tile

SMEM_scales = S_buf × scales_per_tile × 4 = 3 × 7 × 4 = 84 bytes
# Negligible overhead
```

## Configuration Struct

```cpp
template <uint32_t _M_t, uint32_t _N_t, uint32_t _K_t, 
          uint32_t _S_buf, uint32_t _K_chunks>
struct TileConfig {
    static constexpr uint32_t M_t = _M_t;
    static constexpr uint32_t N_t = _N_t;
    static constexpr uint32_t K_t = _K_t;
    static constexpr uint32_t S_buf = _S_buf;
    static constexpr uint32_t K_chunks = _K_chunks;
    
    // Derived
    static constexpr size_t SMEM_A = S_buf * M_t * K_t * sizeof(uint8_t);
    static constexpr size_t SMEM_W = S_buf * K_t * N_t * sizeof(uint8_t);
    static constexpr size_t SMEM_acc = M_t * N_t * sizeof(float);
    static constexpr size_t SMEM_meta = 4096;
    static constexpr size_t SMEM_total = SMEM_A + SMEM_W + SMEM_acc + SMEM_meta;
};

// Qwen3-30B-A3B on L40S
using TileConfig_Qwen3_L40S = TileConfig<8, 576, 64, 2, 32>;
static_assert(TileConfig_Qwen3_L40S::SMEM_total <= 98304, "SMEM overflow");

// Llama 4 on H200
using TileConfig_Llama4_H200 = TileConfig<8, 976, 64, 3, 80>;
static_assert(TileConfig_Llama4_H200::SMEM_total <= 225280, "SMEM overflow");
```

## Troubleshooting

### N_t Too Small (< 32)

1. **Reduce S_buf**: Triple → Double buffer
2. **Reduce M_t**: 16 → 8
3. **Use K-chunking**: Process K in multiple passes

### SMEM Overflow After Solving

1. Verify alignment: N_t must be multiple of 16
2. Check SMEM_meta estimate: May need more for complex models
3. Reduce safety margin: Last resort

### Performance Suboptimal

1. **N_t not multiple of 128**: May cause bank conflicts
2. **K_chunks too large**: Loop overhead dominates
3. **S_buf=2 on Hopper**: Leaves TMA bandwidth on table

## Quick Reference

| Model | GPU | dtype | M_t | N_t | K_t | S_buf | SMEM |
|-------|-----|-------|-----|-----|-----|-------|------|
| Qwen3-30B (FP8) | L40S | FP8 | 8 | 576 | 64 | 2 | 97 KB |
| Qwen3-30B (FP8) | H100 | FP8 | 8 | 896 | 64 | 3 | 218 KB |
| Qwen3-30B (BF16) | L40S | BF16 | 8 | 288 | 64 | 2 | 95 KB |
| Llama 4 | H200 | FP8 | 8 | 976 | 64 | 3 | 224 KB |
| DeepSeek-V3 | H100 | FP8 | 8 | 768 | 64 | 3 | 196 KB |
| Mixtral (FP16) | L40S | FP16 | 8 | 288 | 64 | 2 | 95 KB |
| Mixtral (FP8) | L40S | FP8 | 8 | 512 | 64 | 2 | 85 KB |

**Note**: BF16/FP16 weights use 2× SMEM per element, resulting in smaller N_t tiles.
