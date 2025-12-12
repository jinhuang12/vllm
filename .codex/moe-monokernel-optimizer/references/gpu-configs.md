# GPU Configuration Reference

## Supported Architectures

| GPU | SM Arch | SMEM/SM | FP8 Support | TMA | Monokernel BS Threshold |
|-----|---------|---------|-------------|-----|-------------------------|
| H100/H200 | sm_90a | 228 KB | ✓ | ✓ | BS ≤ 128 |
| L40S | sm_89 | 100 KB | ✓ | ✗ | BS ≤ 32 |
| A100 | sm_80 | 164 KB | ✗ | ✗ | BS ≤ 64 |
| A10 | sm_86 | 100 KB | ✗ | ✗ | BS ≤ 32 |
| RTX 4090 | sm_89 | 100 KB | ✓ | ✗ | BS ≤ 32 |
| RTX 3090 | sm_86 | 100 KB | ✗ | ✗ | BS ≤ 32 |

## Monokernel Zone Explanation

The **Monokernel Zone** is the batch size range where fused monokernel beats standard CUTLASS/Triton grouped GEMM.

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

Grid size must be ≤ SM count for cooperative kernel launch:

| GPU | SM Count | Max Grid Size |
|-----|----------|---------------|
| H100 SXM | 132 | 132 |
| H100 PCIe | 114 | 114 |
| H200 | 132 | 132 |
| L40S | 142 | 142 |
| A100 40GB | 108 | 108 |
| A100 80GB | 108 | 108 |

## Quick Reference Commands

```bash
# H100/H200 (with TMA)
--gpu-arch sm_90a --smem-per-sm 228 --monokernel-threshold 128

# L40S
--gpu-arch sm_89 --smem-per-sm 100 --monokernel-threshold 32

# A100
--gpu-arch sm_80 --smem-per-sm 164 --monokernel-threshold 64
```