# Kernel Architecture Patterns

This document describes the kernel architectures used in MoE monokernel optimization.

## Pattern Selection (Monokernel vs Split)

Use ownership and M_avg to pick an architecture:

- Token-major ownership + low M_avg: monokernel can work well and avoid atomics.
- Expert-major ownership or high M_avg: prefer split-kernel or grouped GEMM.
- In-kernel routing: only required when routing must be fused; otherwise route outside and pass top-k data in.

## Token-Major Ownership Pattern

Use when you can precompute top-k (outside the kernel) and want to avoid atomics.

### Grid Layout

```
Blocks: (token, k-slice) ownership
Each block owns one token and a K-slice of the output.
```

### Execution

```cpp
// Pseudocode: each block owns token t and K-slice [k0, k1)
for (int expert_i = 0; expert_i < TOP_K; ++expert_i) {
    // load expert id + weight for (t, expert_i)
    // compute up_proj + activation + down_proj for owned K-slice
    // accumulate in registers or FP32 scratchpad (no atomics)
}
// write final output for this token and K-slice
```

This pattern removes output overlap and reduces grid.sync barriers in the down-projection path.

---

## Split-Kernel Pattern

Separate routing/quantization from GEMM when M_avg is high or ownership is expert-major.

Benefits:
- No controller block or grid.sync inside GEMM kernel
- Enables grouped GEMM and better weight reuse
- Plays well with CUDA graphs and torch.compile

Typical flow:
1. Router kernel: compute top-k ids and weights (optional sorting)
2. Prepare kernel: build packed token/expert lists
3. GEMM kernel: grouped by expert or token-major K-slice

---

## Hybrid Pattern (Expert‑major Up, Token‑major Down)

Use when up‑projection benefits from expert grouping but down‑projection suffers from atomics/overlap.

Typical flow:
- Up‑proj: expert‑major grouped GEMM (weight reuse)
- Down‑proj: token‑major K‑slice ownership (no atomics)

This pattern often outperforms a fully cooperative monokernel for top_k>1 with EP enabled.

---

## When NOT to Default to Full Monokernel

Avoid full monokernel when any of the following hold:
- Routing/prepare share is small (< ~15% of combined‑graph time)
- Barrier budget would exceed 1–2 grid.sync
- EP reduces M_avg and routing is imbalanced
- Cooperative launch limit is too low for the kernel resources

## Controller-Worker Pattern

The monokernel uses a **controller-worker** architecture where one block coordinates routing while others compute expert GEMMs.
Use this only when routing must happen inside the kernel and barrier budget allows it.

### Grid Layout

```
Block 0:           Controller (routing + sorting)
Blocks 1..N:       Workers (expert computation)
Blocks N+1..N+S:   Shared expert blocks (if applicable)
```

### Controller Block (blockIdx.x == 0)

Responsibilities:
1. Compute top-k expert selection from router logits
2. Build expert histogram (token counts per expert)
3. Signal workers via ready flag

```cpp
template <typename Dims>
__device__ void controller_block(
    const __nv_bfloat16* router_logits,
    uint32_t num_tokens,
    MoE_SHM<Dims>* shmem)
{
    // Step 1: Top-k selection
    compute_topk<Dims>(router_logits, num_tokens, 
                       shmem->topk_ids, shmem->topk_weights);
    __syncthreads();
    
    // Step 2: Build expert histogram
    build_expert_histogram<Dims>(shmem->topk_ids, num_tokens,
                                 shmem->expert_counts);
    __syncthreads();
    
    // Step 3: Signal workers
    __threadfence();  // Ensure writes visible to other blocks
    if (threadIdx.x == 0) {
        atomicExch(&shmem->ready_flag, 1);
    }
}
```

### Worker Blocks (blockIdx.x >= 1)

Responsibilities:
1. Wait for controller ready signal
2. Compute assigned expert slice
3. Accumulate to output

```cpp
template <typename Dims>
__device__ void worker_block(
    uint32_t block_idx,  // blockIdx.x - 1
    const W_element* weights,
    const S_element* scales,
    MoE_SHM<Dims>* shmem,
    R_element* output)
{
    // Step 1: Spin until controller ready
    if (threadIdx.x == 0) {
        while (atomicAdd(&shmem->ready_flag, 0) == 0) {
            // Spin - could add backoff
        }
    }
    __syncthreads();
    
    // Step 2: Determine assignment
    // For Split-H: block handles portion of hidden dim
    // For standard: block handles one (token, expert) pair
    uint32_t expert_slot = block_idx % Dims::K_ROUT;
    uint32_t slice_idx = block_idx / Dims::K_ROUT;
    
    // Step 3: Compute expert GEMM slice
    compute_expert_slice<Dims>(
        expert_slot, slice_idx,
        weights, scales, shmem, output);
}
```

### Synchronization

```cpp
// Global ready flag in scratchpad memory (not SMEM)
struct Scratchpad {
    uint32_t ready_flag;           // Controller → workers
    uint32_t phase_complete_count; // Workers → next phase
};

// Wait for all workers to complete a phase
__device__ void wait_phase_complete(Scratchpad* scratch, uint32_t expected) {
    if (threadIdx.x == 0) {
        atomicAdd(&scratch->phase_complete_count, 1);
    }
    cooperative_groups::this_grid().sync();
}
```

---

## Split-H Latency Kernel

**When to use**: Low saturation (BS × top_k < 0.5 × SM_count)

Multiple blocks collaborate on each token by splitting the hidden dimension.

### Split Factor Calculation

```python
def calculate_split_factor(batch_size: int, top_k: int, sm_count: int) -> int:
    """
    Calculate how many blocks should collaborate per (token, expert).
    
    Goal: Fill all SMs while maintaining efficiency.
    """
    total_work_items = batch_size * top_k
    
    # Want ~80% SM utilization for efficiency
    target_blocks = int(sm_count * 0.8)
    
    # Split factor = blocks per (token, expert)
    S = max(1, target_blocks // total_work_items)
    
    # Round to power of 2 for clean division
    S = 1 << (S.bit_length() - 1)
    
    return min(S, 16)  # Cap at 16 to limit atomicAdd contention
```

### Grid Configuration

```cpp
// For Split-H kernel
constexpr uint32_t S = SPLIT_FACTOR;           // e.g., 4
constexpr uint32_t K_ROUT = TOP_K;             // e.g., 8
constexpr uint32_t WORKER_BLOCKS = S * K_ROUT; // e.g., 32

// Total grid (cooperative launch)
constexpr uint32_t GRID_SIZE = 1 + WORKER_BLOCKS + S_SHARED;
// Example: 1 + 32 + 14 = 47 blocks for Qwen3 on L40S
```

### Hidden Dimension Chunking

```cpp
// Each worker handles H_chunk elements of hidden dimension
constexpr uint32_t H = HIDDEN_SIZE;        // e.g., 2048
constexpr uint32_t H_CHUNK = H / S;        // e.g., 512

// Worker block assignment
uint32_t h_start = slice_idx * H_CHUNK;
uint32_t h_end = min(h_start + H_CHUNK, H);
```

### Output Accumulation

With Split-H, multiple blocks write to same output location:

```cpp
// All S blocks for same (token, expert) reduce to output
template <typename Dims>
__device__ void accumulate_output(
    uint32_t token_idx,
    float* partial_result,
    R_element* output)
{
    // Convert FP32 partial to BF16 and atomicAdd (or staged reduction)
    for (uint32_t i = threadIdx.x; i < Dims::K; i += blockDim.x) {
        __nv_bfloat16 val = __float2bfloat16(partial_result[i]);
        atomicAdd(&output[token_idx * Dims::K + i], val);
    }
}
```

---

## Shared Expert Sidecar

For models with always-active shared experts (Llama 4, DeepSeek).

### Sidecar Grid Layout

```cpp
// Reserve blocks for shared expert computation
constexpr uint32_t S_SHARED = SM_COUNT / 10;  // 10% for 1 shared expert
                                               // 20% for 2 shared experts

// Grid layout:
// Block 0:                    Controller
// Blocks [1, S*K_ROUT]:       Routed expert workers
// Blocks [S*K_ROUT+1, end]:   Shared expert workers

constexpr uint32_t GRID_SIZE = 1 + S * K_ROUT + S_SHARED;
```

### Shared Expert Block Logic

```cpp
template <typename Dims>
__device__ void shared_expert_block(
    uint32_t shared_block_idx,  // 0 to S_SHARED-1
    const W_element* shared_weights,
    const S_element* shared_scales,
    uint32_t num_tokens,
    const A_element* activations,
    R_element* output)
{
    // Shared expert processes ALL tokens (not just top-k selected)
    // Split work across S_SHARED blocks
    
    uint32_t tokens_per_block = (num_tokens + S_SHARED - 1) / S_SHARED;
    uint32_t token_start = shared_block_idx * tokens_per_block;
    uint32_t token_end = min(token_start + tokens_per_block, num_tokens);
    
    for (uint32_t t = token_start; t < token_end; t++) {
        // Compute shared expert output for token t
        compute_shared_expert_gemm<Dims>(
            t, activations, shared_weights, shared_scales, output);
    }
}
```

### Detecting Shared Expert Blocks

```cpp
template <typename Dims>
__global__ void moe_kernel(...) {
    // Determine block role
    if (blockIdx.x == 0) {
        controller_block<Dims>(...);
    } 
    else if (blockIdx.x <= Dims::S * Dims::K_ROUT) {
        worker_block<Dims>(blockIdx.x - 1, ...);
    }
    else if constexpr (Dims::NUM_SHARED > 0) {
        uint32_t shared_idx = blockIdx.x - 1 - Dims::S * Dims::K_ROUT;
        shared_expert_block<Dims>(shared_idx, ...);
    }
}
```

### Shared Expert Weight Layout

```cpp
// Weights are stored with shared experts at the end
// For Llama 4: E=16 routed + 1 shared = 17 total
// Shared expert weights at index E_ROUTED

const W_element* shared_weights_up = 
    expert_weights_up + E_ROUTED * weight_stride;
const W_element* shared_weights_down = 
    expert_weights_down + E_ROUTED * weight_stride;
```

---

## Cooperative Launch Requirements

All monokernel architectures require cooperative kernel launch:

```cpp
// Host-side launch
void launch_moe_monokernel(...) {
    // Verify grid fits on device
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    assert(GRID_SIZE <= num_sms);
    
    // Cooperative launch
    void* args[] = { &activations, &router_logits, ... };
    cudaLaunchCooperativeKernel(
        (void*)moe_kernel<Dims>,
        dim3(GRID_SIZE),
        dim3(BLOCK_SIZE),
        args,
        SMEM_SIZE);
}
```

### Grid Size Constraints

| GPU | Max Cooperative Grid | Typical Monokernel Grid |
|-----|---------------------|------------------------|
| H100 | 132 | ~80-100 |
| H200 | 132 | ~80-100 |
| L40S | 142 | ~90-110 |

---

## Architecture Selection Flowchart

```
START
  │
  ├─► Calculate saturation = BS × top_k / SM
  │
  ├─► saturation < 0.25?
  │     YES → Use Split-H with S = SM / (BS × top_k)
  │     NO  ↓
  │
  ├─► saturation < 0.5?
  │     YES → Use Split-H with S = 2 or 4
  │     NO  ↓
  │
  ├─► saturation < 1.0?
  │     YES → Use Standard (1 block per token×expert)
  │     NO  → Use stock fused_moe
  │
  └─► Add shared expert sidecar if num_shared > 0
```
