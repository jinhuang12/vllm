# Plan Stage Prompt Template

Research the {{SYMBOL_NAME}} kernel in vLLM and document:

## Kernel Analysis

- **Location**: Full file path to kernel implementation (e.g., `vllm/model_executor/layers/fused_moe/fused_moe.py`)
- **Type**: [Triton | CUDA | Hybrid]
- **Primary Function**: {entry_point_function_name} that launches the kernel
- **Wrapper Function**: {python_wrapper_function_name} that handles tensor prep

## Dependency Map

For each required dependency, list:
- Import path (e.g., `from vllm.utils import xyz`)
- Function/class name
- Source file location in vLLM repo
- Why it's needed (brief explanation)
- Transitive dependencies (if any)

Example format:
```
1. vllm.utils.reshape_and_cache
   - Source: vllm/utils/tensor_ops.py:45
   - Needed for: Output tensor preparation
   - Transitive: None

2. vllm.model_executor.layers.activation.SiluAndMul
   - Source: vllm/model_executor/layers/activation.py:12
   - Needed for: Activation function in expert FFN
   - Transitive: Imports torch.nn.functional (OK - external)
```

If NO dependencies exist (self-contained kernel), explicitly state:
```
No vLLM dependencies - kernel is self-contained
```

## Input/Output Contract

### Input Tensors
For each input tensor, specify:
- **Name**: Parameter name in function signature
- **Shape**: Dimensions with semantic meaning
  - Example: `(batch_size, seq_len, hidden_size)` where:
    - batch_size: Number of sequences in batch
    - seq_len: Sequence length (tokens)
    - hidden_size: Model hidden dimension
- **Dtype**: Expected data type (float16, bfloat16, float32, int32, etc.)
- **Device**: CPU or CUDA
- **Meaning**: Semantic purpose (e.g., "input activations", "expert weights", "attention scores")

### Metadata Requirements
For kernel-specific metadata (beyond primary tensors):
- **For MoE kernels**: topk_ids (int32), topk_weights (float32/16), expert_ids
- **For PagedAttention**: block_tables (int32), context_lens (int32), seq_lens (int32)
- **For Quantization**: scales (float32), zero_points (uint8/int8), group_indices
- **For Grouped Operations**: group_ids, num_groups

### Output Tensors
For each output tensor:
- **Shape**: Output dimensions (may depend on input shapes)
- **Dtype**: Output data type
- **Meaning**: What the output represents

### In-Place Operations
List any tensors modified in-place (if applicable):
- Which tensors are modified
- What modifications occur
- Why in-place is necessary (performance, memory)

## Kernel Configuration

### Triton Decorators (if Triton kernel)
- **@triton.jit**: Present? Any special parameters?
- **@triton.autotune**: Configurations being tested?
  - BLOCK_M values
  - BLOCK_N values
  - BLOCK_K values (for matmul)
  - num_warps values
  - num_stages values

### Launch Parameters
- **Grid dimensions**: How are grid dims calculated?
  - Example: `grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))`
- **Block dimensions**: Thread block size
  - Example: `block = (BLOCK_M, BLOCK_N)`
- **Shared memory**: Amount of shared memory used (if known)

### Compile-Time Constants
List tl.constexpr values:
- BLOCK_M, BLOCK_N, BLOCK_K
- HEAD_DIM (for attention)
- GROUP_SIZE (for quantization)
- NUM_EXPERTS (for MoE)

### Hardware-Specific Settings
For {{HARDWARE}} ({{HARDWARE}}):
- Compute capability requirements (e.g., SM80 for A100, SM90 for H200)
- Tensor core usage (if applicable)
- Special features (FP8 tensor cores for H200, etc.)
- Memory bandwidth considerations

## Benchmark Strategy

### Realistic Dimension Ranges
For {{MODEL}} on {{HARDWARE}} with {{PRECISION}}:

**Batch Sizes**:
- Quick mode: [1, 16]
- Full mode: [1, 2, 4, 8, 16, 32, 64, 128]
- Rationale: Cover range from single-sample inference to large batches

**Sequence Lengths**:
- Quick mode: [512, 2048]
- Full mode: [128, 256, 512, 1024, 2048, 4096, 8192]
- Rationale: Cover typical inference (128-1024) and long-context (2048-8192)

**Hidden Dimensions**:
- Quick mode: [4096]
- Full mode: [2048, 4096, 8192, 16384]
- Rationale: Cover model sizes from 7B (2048/4096) to 70B+ (8192+)

### Kernel-Specific Sweep Parameters

**For Attention Kernels**:
- seq_len: [128, 256, 512, 1024, 2048, 4096]
- num_heads: [32, 64, 96, 128]
- head_dim: [64, 128, 256]

**For MoE Kernels**:
- num_experts: [8, 16, 32, 64, 160]
- top_k: [1, 2, 4, 8]
- expert_capacity: [computed based on tokens and top_k]
- intermediate_size: [2560, 5120, 10240]

**For Matmul/GEMM Kernels**:
- M (batch * seq_len): [128, 512, 2048, 8192]
- N (output dim): [2048, 4096, 8192]
- K (input dim): [2048, 4096, 8192]

**For Quantization Kernels**:
- group_size: [64, 128, 256]
- num_bits: [4, 8] (if applicable)

### Expected Performance Ballpark
Provide rough estimates for validation:
- **Latency**: X ms for typical config (B=4, S=2048, H=4096)
- **Throughput**: Y samples/sec or Y tokens/sec
- **TFLOPS**: Z TFLOPS for compute-bound kernels
- **Memory Bandwidth**: W GB/s for memory-bound kernels

Example:
```
For fused_moe_kernel on H200 with B=4, S=2048, experts=160, top_k=2:
- Latency: ~2-5 ms (rough estimate)
- Throughput: ~800-2000 samples/sec
- Expected to be compute-bound with 70-80% TFLOP/s utilization
```

## Reference Implementation Strategy

### PyTorch Equivalent
Identify the closest PyTorch operation:

**For Attention**:
```python
torch.nn.functional.scaled_dot_product_attention(query, key, value, attn_mask=None)
```

**For MoE**:
```python
# No direct equivalent - implement with explicit routing:
output = torch.zeros_like(input)
for batch_idx in range(batch_size):
    for expert_idx in topk_ids[batch_idx]:
        weight = topk_weights[batch_idx, expert_idx]
        expert_out = expert_forward(input[batch_idx], expert_idx)
        output[batch_idx] += weight * expert_out
```

**For Activation Functions**:
```python
torch.nn.functional.silu(x)  # For SiLU
torch.nn.functional.gelu(x)  # For GELU
x * torch.sigmoid(x)          # For Swish
```

**For Matmul**:
```python
torch.matmul(input, weight.T) + bias  # For linear layers
torch.bmm(input, weight)              # For batched matmul
```

**For LayerNorm**:
```python
torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps=1e-5)
```

### If No Direct Equivalent

Describe algorithmic approach:
1. What mathematical operation the kernel performs
2. How to implement it with basic PyTorch ops (loops, indexing, etc.)
3. Known limitations of the reference (performance, numerical precision)

Add comment in generated code:
```python
# Reference may be slow - for correctness only
# This implementation uses explicit loops for clarity
```

## Gotchas & Edge Cases

Document known issues and considerations:

### Numerical Instabilities
- FP16 underflow in softmax (large negative values)
- FP8 quantization errors exceeding typical tolerances
- Accumulation errors in long sequences
- Recommended tolerances: rtol=1e-3, atol=1e-5 for FP16

### Hardware-Specific Behaviors
- Tensor core usage requirements (input alignment, size constraints)
- Warp-level reduction assumptions
- Shared memory bank conflicts
- L2 cache behavior for large tensors

### Memory Constraints
- MoE expert capacity limits (OOM for large num_experts × batch_size)
- PagedAttention block table size limits
- Maximum sequence length constraints
- Memory fragmentation in long runs

### vLLM-Specific Optimizations
- Custom CUDA streams for overlapping
- In-place operations to reduce memory
- Fused operations (e.g., SiluAndMul instead of separate ops)
- Quantization-aware kernels (FP8 inputs but FP32 accumulation)

### Edge Cases to Test
- Batch size = 1 (single sample)
- Sequence length = 1 (decoder-only, no context)
- Empty batches (if applicable)
- Maximum dimensions (test OOM boundaries)
- Non-divisible dimensions (not perfectly aligned to block sizes)

## Execution Strategy Notes

### Build Order
1. **Start with dependencies**: Inline all helpers first
2. **Extract kernel**: Get the core kernel code
3. **Create wrapper**: Add Python wrapper with proper grid/block logic
4. **Add invocation**: Create invocation_example() for testing
5. **Generate benchmark**: Create full benchmark suite

### Common Pitfalls
- Forgetting to inline transitive dependencies
- Incorrect grid/block dimension calculation
- Missing `torch.cuda.synchronize()` before timing
- Using `requires_grad=True` on benchmark tensors
- Hardcoding device indices instead of using `torch.cuda.current_device()`

### Testing Strategy
1. **Local execution**: Test bundle with `python bundle.py`
2. **Quick validation**: Run `--quick --validate` first
3. **Full sweep**: Only after validation passes
4. **Remote verification**: Deploy to p5e-cmh for hardware validation

---

**Save this plan to**: `/tmp/plan_{{SYMBOL_NAME}}.md`

After creating this plan, the bundle generation process will use it as the blueprint for:
- Identifying dependencies to inline
- Generating appropriate test inputs
- Creating realistic benchmark sweeps
- Implementing correct reference operations
- Setting appropriate numerical tolerances
