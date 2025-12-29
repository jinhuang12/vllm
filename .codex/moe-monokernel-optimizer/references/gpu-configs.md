# GPU Configuration Reference

These are **hardware guardrails**, not decomposition decisions. Use them to bound feasibility (SMEM, cooperative launch, occupancy), then choose ownership/fusion based on M_avg and routing.

## Contents
- How to Use This in Phase 1/2
- Supported Architectures
- Shared Memory and Register Constraints
- Cooperative Kernel Constraints
- Rules of Thumb

## Search anchors
SM arch, SMEM, registers, cooperative grid, occupancy, M_avg, EP.

## How to Use This in Phase 1/2
- Check cooperative launch limits and SM count when considering monokernel.
- Use SMEM budget to validate tile feasibility.
- Do **not** pick ownership or tiling solely from this table.

## Supported Architectures

| GPU | SM Arch | SMEM/SM | FP8 Support | TMA | Heuristic BS Threshold (assumes low M_avg) |
|-----|---------|---------|-------------|-----|-------------------------------------------|
| H100/H200 | sm_90a | 228 KB | ✓ | ✓ | BS ≤ 128 |
| L40S | sm_89 | 100 KB | ✓ | ✗ | BS ≤ 32 |
| A100 | sm_80 | 164 KB | ✗ | ✗ | BS ≤ 64 |
| A10 | sm_86 | 100 KB | ✗ | ✗ | BS ≤ 32 |
| RTX 4090 | sm_89 | 100 KB | ✓ | ✗ | BS ≤ 32 |
| RTX 3090 | sm_86 | 100 KB | ✗ | ✗ | BS ≤ 32 |

## Monokernel Zone Explanation

The **Monokernel Zone** is the batch size range where fused monokernel beats standard CUTLASS/Triton grouped GEMM **when M_avg is small**.

**Always compute**: `M_avg = BS * top_k / E_local` (use E_global only if EP is not pre‑dispatch). If `M_avg >= 8`, monokernel is often a poor fit; prefer split‑kernel or expert‑grouped GEMM.

**Why thresholds differ by hardware:**
- **H200 (3.35 TB/s)**: High bandwidth means monokernel's fusion benefits persist to larger batches
- **L40S (864 GB/s)**: Lower bandwidth means dense kernels catch up faster as batch grows

**Recommended approach**: Benchmark BS=1 to BS=128, find crossover point.

## TMA (Tensor Memory Accelerator)

**Available on**: Hopper (sm_90a) only

**Benefits**:
- Hardware-accelerated async copy
- Automatic address generation
- Better than `cp.async` for large tiles

**Usage**:
```cpp
#if __CUDA_ARCH__ >= 900
    // Use TMA for prefetch
    __nv_tma_desc_t tma_desc;
    // ... setup TMA descriptor
    cuda::memcpy_async(dst, src, tma_desc, pipe);
#else
    // Fall back to cp.async
    cuda::memcpy_async(dst, src, cuda::aligned_size_t<16>(bytes), pipe);
#endif
```

## AWS Instance Mapping

| Instance | GPUs | GPU Type | Monokernel Zone | Recommended For |
|----------|------|----------|-----------------|-----------------|
| g6e.12xlarge | 4 | L40S | BS ≤ 32 | Small MoE (E≤64) |
| g6e.24xlarge | 4 | L40S | BS ≤ 32 | Small MoE (E≤64) |
| g6e.48xlarge | 8 | L40S | BS ≤ 32 | Medium MoE |
| p5.48xlarge | 8 | H100 | BS ≤ 128 | Large MoE (E≤256) |
| p5e.48xlarge | 8 | H200 | BS ≤ 128 | Large MoE, long context |

## SMEM Budget Calculation

For monokernel, target occupancy is typically 1 CTA/SM:

```
SMEM_available = SMEM_per_SM - margin
margin = 4-8 KB for header structures
```

### H100/H200 (228 KB/SM)
- Available: ~220 KB
- Can fit large tiles for K≤8192, N≤4096
- **Use**: Triple buffering + TMA

### L40S (100 KB/SM)
- Available: ~92 KB
- Requires smaller tiles or K-chunking
- **Use**: Double buffering + cp.async

**L40S-Specific Optimizations** (from Qwen3 implementation):
- SM_COUNT: 142 (highest among common inference GPUs)
- FP8 Support: Yes (sm_89)
- TMA: No (requires sm_90+)
- Recommended warp config: 6 calc + 2 prefetch (8 total)
- Block size: 256 threads optimal
- Split-H threshold: BS ≤ 4

**cp.async Pattern for L40S** (no TMA):
```cpp
// Use cuda::memcpy_async with explicit 16-byte alignment
const auto shape16 = cuda::aligned_size_t<16>(16);
cuda::memcpy_async(&dest, &source, shape16, pipeline);

// Commit and wait
pipeline.producer_commit();
cuda::pipeline_consumer_wait_prior<0>(pipeline);
```

### A100 (164 KB/SM)
- Available: ~156 KB
- Good middle ground
- No FP8 → use BF16 kernels
- **Use**: Triple buffering + cp.async

## CUDA Graph Compatibility

For CUDA graph capture, batch sizes must be fixed. Use power-of-2 bucketing:

```python
BATCH_BUCKETS = [1, 2, 4, 8, 16, 32, 64, 128]

def get_bucket(actual_bs: int) -> int:
    """Pad to nearest power of 2."""
    for bucket in BATCH_BUCKETS:
        if actual_bs <= bucket:
            return bucket
    return BATCH_BUCKETS[-1]
```

Instantiate separate compiled kernels for each bucket:
```python
# During model initialization
self.monokernel_variants = {
    bs: torch.compile(moe_monokernel_fn, fullgraph=True)
    for bs in BATCH_BUCKETS if bs <= MONOKERNEL_THRESHOLD
}
```

## Cooperative Launch Constraints

Cooperative launch requires **grid size ≤ max active blocks** for the kernel, which depends on SM resources (registers, SMEM, block size).  
SM count is a rough upper bound, but **not the true limit**.

Guidance:
- Use `cudaOccupancyMaxActiveBlocksPerMultiprocessor` (or Nsight Compute occupancy) to determine max active blocks.
- Ensure `gridDim.x <= max_active_blocks_per_SM * SM_count`.

## Split-H Thresholds by GPU

Split-H optimization targets small batches where standard grid underutilizes SMs. Thresholds vary by SM count and typical workload:

| GPU | SM Count | Split-H Threshold | Target Utilization | Notes |
|-----|----------|-------------------|-------------------|-------|
| H100/H200 | 132 | BS ≤ 4 | 80% | High BW means standard often sufficient |
| L40S | 142 | BS ≤ 4 | 80% | Reference: Qwen3 implementation |
| A100 | 108 | BS ≤ 4 | 80% | Lower SM count, less benefit |
| RTX 4090 | 128 | BS ≤ 4 | 80% | Consumer card, similar to H100 |

**Split Factor Calculation**:
```python
def get_split_factor(batch_size: int, top_k: int, sm_count: int) -> int:
    if batch_size > 4:
        return 1  # No split for larger batches

    total_pairs = batch_size * top_k
    target_blocks = int(sm_count * 0.8)  # 80% utilization
    split = (target_blocks + total_pairs - 1) // total_pairs
    return min(split, 16)  # Cap at 16 to limit atomicAdd contention
```

## cp.async vs TMA Decision

| GPU | Architecture | Async Memory | Recommendation |
|-----|--------------|--------------|----------------|
| H100/H200 | sm_90a | TMA | Use TMA for large tiles (>16KB) |
| L40S | sm_89 | cp.async only | Double buffer with 16-byte aligned copies |
| RTX 4090 | sm_89 | cp.async only | Same as L40S |
| A100 | sm_80 | cp.async only | Triple buffer compensates for no TMA |

## Quick Reference Commands

```bash
# H100/H200 (with TMA)
--gpu-arch sm_90a --smem-per-sm 228 --monokernel-threshold 128

# L40S
--gpu-arch sm_89 --smem-per-sm 100 --monokernel-threshold 32

# A100
--gpu-arch sm_80 --smem-per-sm 164 --monokernel-threshold 64
```
