# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a fast and easy-to-use library for LLM inference and serving. It provides state-of-the-art serving throughput using PagedAttention for efficient KV cache management, continuous batching, and optimized CUDA kernels.

## Key Commands

### Build and Installation

```bash
pip install -e .                              # Install from source
python setup.py build_ext --inplace           # Build C++ extensions only
```

This repo has it's own virtual environment in `.venv/`, please use activate it before running any python commands 

### Testing

```bash
pytest tests/                                              # Run all tests
pytest tests/path/to/test_file.py                         # Run specific test file
pytest tests/path/to/test_file.py::test_function_name     # Run specific test
pytest -v tests/                                           # Verbose output
pytest -k "pattern" tests/                                 # Run tests matching pattern
.buildkite/scripts/rerun-test.sh tests/path/to/test.py::test_name  # Debug flaky tests
```

### Linting and Formatting

```bash
pip install -r requirements/lint.txt && pre-commit install  # Setup
pre-commit run --all-files                                  # Run all checks
./tools/mypy.sh                                             # Type checking
```

### Running vLLM

```bash
vllm serve <model_name>                                     # Start API server
vllm bench {serve,latency,throughput}                       # Benchmarking CLI
```

### Benchmarking

```bash
python benchmarks/benchmark_throughput.py --model <model>   # Throughput benchmark
python benchmarks/benchmark_latency.py --model <model>      # Latency benchmark
python benchmarks/benchmark_serving.py --model <model>      # Serving benchmark
```

## Architecture Overview

### V1 Engine Architecture (Primary)

The V1 architecture (`vllm/v1/`) is the primary engine implementation. V0 is deprecated.

**Engine Core Components** (`vllm/v1/engine/`):
- **EngineCore** (`core.py`): Inner loop containing scheduler, model executor, and structured output manager
- **Processor** (`processor.py`): Transforms raw inputs → EngineCoreRequests via validation, tokenization
- **OutputProcessor** (`output_processor.py`): Converts EngineCoreOutputs → RequestOutput for users
- **AsyncLLM** (`async_llm.py`): Async interface for serving

**Three-Stage Execution Loop**:
1. **Schedule**: Select requests for decode/prefill from waiting/running queues
2. **Forward Pass**: Run model and sample tokens
3. **Postprocess**: Update requests, check stop conditions, clean up finished requests

**Request Lifecycle**:
- Requests enter scheduler's waiting queue with `WAITING` status
- Move to `RUNNING` during execution
- Scheduler uses FCFS or priority policies

### KV Cache and Memory Management

**Paged Attention**:
- Block-based KV cache with 16-token blocks by default
- `KVCacheManager` (`vllm/v1/core/kv_cache_manager.py`): Maintains `free_block_queue` - pool of available blocks
- Block size calculation: `2 * block_size * num_kv_heads * head_size * dtype_bytes`

**Prefix Caching**:
- Hash-based block identification for prompt prefix reuse
- Splits prompts into 16-token chunks with `cached_block_hash_to_block` mapping

### Executor Hierarchy

- **UniProcExecutor** (`vllm/v1/executor/uniproc_executor.py`): Single-GPU execution
- **MultiProcExecutor** (`vllm/v1/executor/multiproc_executor.py`): Multi-GPU with `rpc_broadcast_mq` coordination
- Supports tensor parallelism (TP) and pipeline parallelism (PP)

### Key Directories

- `vllm/v1/`: Primary V1 engine implementation
- `vllm/v1/core/sched/`: Scheduler implementations
- `vllm/v1/attention/backends/`: V1 attention backends (FlashAttention, FlashInfer, MLA, etc.)
- `vllm/model_executor/models/`: Model implementations
- `vllm/attention/`: Attention mechanisms including PagedAttention
- `csrc/`: C++/CUDA kernels
- `tests/`: Test suite
- `benchmarks/`: Performance tools

### Model Support

Models in `vllm/model_executor/models/` include:
- Model architecture implementation
- Weight loading logic
- Forward pass implementation
- Parallelism strategy support

Registration in `vllm/model_executor/models/registry.py` maps HuggingFace architecture names to vLLM implementations.

## Development Patterns

### Adding New Models

1. Create model file in `vllm/model_executor/models/`
2. Register in `vllm/model_executor/models/registry.py`
3. Add tests in `tests/models/`

### Testing Practices

- Use `pytest` fixtures from `tests/conftest.py`
- `@pytest.mark.parametrize` for multiple configurations
- `@pytest.mark.distributed` for distributed tests
- `@pytest.mark.skip_v1` for V1-incompatible tests

### CUDA Kernel Development

- CUDA kernels in `csrc/`
- Python bindings in `vllm/_custom_ops.py`
- Incremental compilation is configured via `CMakeUserPresets.json` ([docs](https://docs.vllm.ai/en/latest/contributing/incremental_build/))

**After editing `csrc/` code:**
```bash
cmake --build --preset release --target install
```

**After adding new `csrc/` files:**
```bash
cmake --preset release && cmake --build --preset release --target install
```

Python changes take effect immediately (editable install).

## Important Configuration

### Environment Variables

```bash
# V1 Engine
VLLM_USE_V1="1"                          # Use V1 engine (default)
VLLM_ENABLE_V1_MULTIPROCESSING="0"       # Disable V1 multiprocessing

# Attention
VLLM_ATTENTION_BACKEND=<backend>         # Force specific attention backend

# Debugging
VLLM_LOGGING_LEVEL=DEBUG                 # Enable debug logging

# Hardware
CUDA_VISIBLE_DEVICES=<ids>               # Control GPU visibility
VLLM_CPU_ONLY=1                          # CPU-only mode
```

### Config Classes (`vllm/config.py`)

- `VllmConfig`: Main configuration container
- `ModelConfig`: Model-specific settings
- `ParallelConfig`: Parallelism configuration
- `SchedulerConfig`: Scheduling parameters
- `CacheConfig`: KV cache configuration

## Performance Features

### Chunked Prefill

Splits long prompts into smaller chunks to prevent monopolization. Controlled by `long_prefill_token_threshold`.

### Speculative Decoding

V1 supports n-gram, EAGLE, and Medusa methods. n-gram: matches last `prompt_lookup_max` tokens against prior sequence.

### CUDA Graphs

Reduces kernel launch overhead when `--enforce-eager` not used. Captured during worker initialization after KV cache setup.

## Kernel Development Reference

This section provides detailed information for CUDA kernel extraction, benchmarking, and development.

### CUDA Kernel Directory Map

```
csrc/
├── torch_bindings.cpp          # Main ops registration (33KB, ~100+ ops)
├── ops.h                       # Forward declarations for all kernel functions
├── dispatch_utils.h            # Type dispatch macros (VLLM_DISPATCH_*)
├── cuda_compat.h               # CUDA/HIP compatibility layer
├── cache.h                     # KV cache structures
│
├── attention/                   # Attention mechanisms
│   ├── paged_attention_v1.cu   # Original PagedAttention
│   ├── paged_attention_v2.cu   # Optimized PagedAttention
│   ├── merge_attn_states.cu    # Merge split-KV results
│   └── mla/                    # Multi-head Latent Attention (SM100)
│       └── sm100_cutlass_mla_kernel.cu
│
├── moe/                        # Mixture of Experts
│   ├── torch_bindings.cpp      # MoE ops registration
│   ├── moe_ops.h               # MoE kernel signatures
│   ├── topk_softmax_kernels.cu # Expert selection (TopK)
│   ├── moe_align_sum_kernels.cu # Block alignment & summation
│   ├── grouped_topk_kernels.cu # Grouped TopK routing
│   ├── moe_permute_unpermute_op.cu # Expert shuffling
│   └── moe_wna16.cu            # WNA16 GEMM
│
├── quantization/               # Quantization schemes (20+ subdirs)
│   ├── awq/                    # AWQ: INT4 weights, FP16 activations
│   ├── gptq/                   # GPTQ: Per-group INT4 quantization
│   ├── gptq_marlin/            # GPTQ with Marlin GEMM
│   ├── gptq_allspark/          # AllSpark W8A16 GPTQ
│   ├── marlin/                 # Marlin GEMM framework (dense & sparse)
│   ├── gguf/                   # GGML format quantization
│   ├── w8a8/                   # 8-bit weights & activations
│   │   ├── cutlass/            # CUTLASS-based W8A8
│   │   └── fp8/common.cu       # FP8 quantization ops
│   ├── fp4/                    # NVFP4 block-scaled floating point
│   ├── cutlass_w4a8/           # CUTLASS W4A8 implementation
│   ├── machete/                # Mixed-precision GEMM (Hopper)
│   ├── hadamard/               # Hadamard transforms
│   ├── aqlm/                   # AQLM quantization
│   ├── qqq/                    # QQQ quantization
│   ├── squeezellm/             # SqueezeLLM quantization
│   ├── exl2/                   # EXL2 quantization
│   └── compressed_tensors/     # Compressed tensor support
│
├── rocm/                       # AMD/ROCm backend
│   ├── torch_bindings.cpp      # ROCm ops registration
│   ├── attention.cu            # ROCm attention
│   └── skinny_gemms.cu         # Skinny GEMM variants
│
├── cpu/                        # CPU backend
│   ├── torch_bindings.cpp      # CPU ops registration
│   └── sgl-kernels/            # CPU-optimized kernels
│
├── core/                       # Core utilities
│   ├── registration.h          # TORCH_LIBRARY_EXPAND macros
│   └── math.hpp                # Math utilities
│
├── mamba/                      # Mamba architecture (SSM)
│   └── mamba_ssm/              # Selective Scan implementation
│
├── cutlass_extensions/         # CUTLASS framework extensions
├── sparse/                     # Sparse tensor operations
├── quickreduce/                # QuickReduce all-reduce
│
└── Root-level kernels:
    ├── activation_kernels.cu   # SiLU, GELU, etc.
    ├── cache_kernels.cu        # KV cache ops (56KB, largest file)
    ├── layernorm_kernels.cu    # LayerNorm, RMS normalization
    ├── layernorm_quant_kernels.cu # Fused LayerNorm + quantization
    ├── pos_encoding_kernels.cu # Position encoding (RoPE)
    ├── sampler.cu              # Sampling operations
    └── custom_all_reduce.cu    # All-reduce collectives
```

### torch.ops Namespace Reference

| Namespace | Binding File | Operations |
|-----------|--------------|------------|
| `torch.ops._C` | `csrc/torch_bindings.cpp` | attention, activation, layernorm, cache, quant |
| `torch.ops._moe_C` | `csrc/moe/torch_bindings.cpp` | topk_softmax, moe_align_block_size, shuffle_rows |
| `torch.ops._rocm_C` | `csrc/rocm/torch_bindings.cpp` | ROCm-specific attention |
| `torch.ops._C_cache_ops` | `csrc/torch_bindings.cpp` | reshape_and_cache, copy_blocks |

### Header Dependencies for CUDA Extraction

**Core Headers (almost always needed):**
- `csrc/core/registration.h` - `TORCH_LIBRARY_EXPAND`, `REGISTER_EXTENSION` macros
- `csrc/dispatch_utils.h` - Type dispatch macros:
  - `VLLM_DISPATCH_FLOATING_TYPES` - float, half, bfloat16
  - `VLLM_DISPATCH_FP8_TYPES` - FP8 variants
  - `VLLM_DISPATCH_QUANT_TYPES` - FP8, int8
- `csrc/cuda_compat.h` - CUDA/HIP compatibility (`WARP_SIZE`, `VLLM_LDG`, `VLLM_SHFL_*_SYNC`)
- `csrc/ops.h` - Forward declarations (~100+ kernel signatures)

**Domain-Specific Headers:**
- `csrc/cache.h` - KV cache structures
- `csrc/moe/moe_ops.h` - MoE kernel signatures
- `csrc/quantization/marlin/marlin.cuh` - Marlin GEMM
- `csrc/attention/dtype_fp8.cuh` - FP8 attention utilities

**System Headers (don't copy - provided by PyTorch/CUDA):**
- `<torch/extension.h>`, `<ATen/...>`, `<c10/...>`
- `<cuda...>`, `<cub/...>`, `<thrust/...>`

### Test & Benchmark File Locations

**Test files** (`tests/kernels/`):
- `attention/` - 23 test files
- `moe/` - 28 test files (key: `test_moe.py`)
- `quantization/` - 31 test files
- `core/` - 12 test files (layernorm, activation, pos_encoding)

**Benchmark files** (`benchmarks/kernels/`):
- `benchmark_*.py` - 33 files
- Key: `benchmark_moe.py`, `benchmark_paged_attention.py`, `benchmark_activation.py`

**Utility files:**
- `tests/kernels/utils.py` - Core test utilities (opcheck, make_qkv, etc.)
- `tests/kernels/quant_utils.py` - Quantization helpers
- `benchmarks/kernels/utils.py` - Benchmark infrastructure

### Python-to-CUDA Call Flow

```
Python Entry Point (vllm/attention/ops/, vllm/model_executor/layers/)
    ↓
Wrapper Function (vllm/_custom_ops.py)
    ↓
torch.ops Call (torch.ops._C.operation_name)
    ↓
Op Registration (csrc/torch_bindings.cpp: ops.def + ops.impl)
    ↓
CUDA Implementation (csrc/*.cu or csrc/subdir/*.cu)
```

### Key Operations Quick Reference

**Attention Operations:**

| Operation | Python Entry | torch.ops | CUDA Source |
|-----------|-------------|-----------|-------------|
| PagedAttention V1 | `vllm/attention/ops/paged_attn.py` | `_C.paged_attention_v1` | `csrc/attention/paged_attention_v1.cu` |
| PagedAttention V2 | `vllm/attention/ops/paged_attn.py` | `_C.paged_attention_v2` | `csrc/attention/paged_attention_v2.cu` |
| Merge Attn States | `vllm/_custom_ops.py` | `_C.merge_attn_states` | `csrc/attention/merge_attn_states.cu` |

**MoE Operations:**

| Operation | Python Entry | torch.ops | CUDA Source |
|-----------|-------------|-----------|-------------|
| TopK Softmax | `vllm/model_executor/layers/fused_moe/` | `_moe_C.topk_softmax` | `csrc/moe/topk_softmax_kernels.cu` |
| MoE Align Block | `vllm/_custom_ops.py` | `_moe_C.moe_align_block_size` | `csrc/moe/moe_align_sum_kernels.cu` |
| Shuffle Rows | `vllm/_custom_ops.py` | `_moe_C.shuffle_rows` | `csrc/moe/moe_permute_unpermute_op.cu` |

**Normalization Operations:**

| Operation | Python Entry | torch.ops | CUDA Source |
|-----------|-------------|-----------|-------------|
| RMS Norm | `vllm/model_executor/layers/layernorm.py` | `_C.rms_norm` | `csrc/layernorm_kernels.cu` |
| Fused Add RMS Norm | `vllm/_custom_ops.py` | `_C.fused_add_rms_norm` | `csrc/layernorm_kernels.cu` |

**Quantization Operations:**

| Operation | Python Entry | torch.ops | CUDA Source |
|-----------|-------------|-----------|-------------|
| FP8 Quant | `vllm/_custom_ops.py` | `_C.scaled_fp8_quant` | `csrc/quantization/w8a8/fp8/common.cu` |
| GPTQ GEMM | `vllm/_custom_ops.py` | `_C.gptq_gemm` | `csrc/quantization/gptq/` |
| AWQ GEMM | `vllm/_custom_ops.py` | `_C.awq_gemm` | `csrc/quantization/awq/` |
| Marlin GEMM | `vllm/_custom_ops.py` | `_C.marlin_gemm` | `csrc/quantization/marlin/` |
| CUTLASS Scaled MM | `vllm/_custom_ops.py` | `_C.cutlass_scaled_mm` | `csrc/quantization/w8a8/cutlass/` |

**Cache Operations:**

| Operation | Python Entry | torch.ops | CUDA Source |
|-----------|-------------|-----------|-------------|
| Reshape & Cache | `vllm/_custom_ops.py` | `_C_cache_ops.reshape_and_cache` | `csrc/cache_kernels.cu` |
| Copy Blocks | `vllm/_custom_ops.py` | `_C_cache_ops.copy_blocks` | `csrc/cache_kernels.cu` |

**Activation Operations:**

| Operation | Python Entry | torch.ops | CUDA Source |
|-----------|-------------|-----------|-------------|
| SiLU and Mul | `vllm/model_executor/layers/activation.py` | `_C.silu_and_mul` | `csrc/activation_kernels.cu` |
| GELU and Mul | `vllm/model_executor/layers/activation.py` | `_C.gelu_and_mul` | `csrc/activation_kernels.cu` |
| Rotary Embedding | `vllm/_custom_ops.py` | `_C.rotary_embedding` | `csrc/pos_encoding_kernels.cu` |

### Test Patterns Quick Reference

**Tensor Creation** (from `tests/kernels/moe/test_moe.py`):
```python
a = torch.randn((m, k), device="cuda", dtype=dtype) / 10  # Scale to avoid overflow
w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
score = torch.randn((m, e), device="cuda", dtype=dtype)
```

**Validation Pattern** (from `tests/kernels/core/test_layernorm.py`):
```python
@torch.inference_mode()
def test_kernel(num_tokens, hidden_size, dtype):
    ref_out = layer.forward_native(x)  # Reference
    out = layer(x)                      # Optimized kernel
    torch.testing.assert_close(out, ref_out, atol=1e-2, rtol=1e-2)
    opcheck(torch.ops._C.rms_norm, (out, x, weight, epsilon))
```

**Benchmark Timing** (from `benchmarks/kernels/`):
```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(num_iters):
    output = kernel_fn(*args)
end_event.record()
torch.cuda.synchronize()
elapsed_ms = start_event.elapsed_time(end_event) / num_iters
```

**Tolerance by Dtype:**
- `torch.float32`: atol=1e-3, rtol=1e-3
- `torch.float16`: atol=1e-2, rtol=1e-2
- `torch.bfloat16`: atol=1e-2, rtol=1e-2

## LLM Council Usage Policy

For complex development tasks (especially CUDA kernel development), **proactively** invoke the `llm-council` skill to get a second opinion from Gemini and Codex. Each deliberation round takes **~20-30 minutes** (Gemini and Codex do deep codebase exploration), but this catches issues that would cost hours to debug.

### When to Use llm-council

| Trigger | When | Why |
|---------|------|-----|
| **After major phases** | Completing constraints, planning, or implementation phases | Validate approach before investing more time |
| **Before finalizing plans** | After creating optimization plans, architectural designs | Catch flawed assumptions early |
| **After 3 failed attempts** | Same bug/error persisting despite multiple fixes | Fresh perspective beats spinning |
| **Before committing code** | After generating significant CUDA/C++ implementations | External review catches subtle bugs |
| **Uncertain decisions** | Multiple valid approaches, unclear trade-offs | Get diverse opinions on best path |

### How to Invoke

The llm-council skill is model-invoked. Simply state your intent:
- "Let me get a second opinion on this plan from the council"
- "I'll consult Gemini and Codex before proceeding"
- "This has failed 3 times - time to get external feedback"

Or explicitly: "Use the llm-council skill to review [topic]"

### Integration with Task Subagents

When spawning Task subagents for implementation, include council checkpoints:
- After completing each major stage, have the orchestrator invoke llm-council
- For blocked tasks, the escalation path should include council consultation
- Task prompts can include: "If stuck after 3 attempts, document blocker for council review"

### Cost-Benefit

| Cost | Benefit |
|------|---------|
| ~20-30 min per deliberation round | Catches architectural flaws early |
| ~2-5K tokens per round | Avoids hours of debugging wrong approach |
| Requires CLI setup | Diverse perspectives (3 different models) |

**Rule of thumb**: If you're about to spend >2 hours on implementation, invest ~30 min getting council feedback first.

### IMPORTANT: Background Execution Required

Because each deliberation round takes 20-30 minutes and the Bash tool's max timeout is 10 minutes, **ALWAYS run deliberation in background mode**:

```
Bash tool with:
  command: "bash .claude/skills/llm-council/scripts/run_deliberation.sh 1 3"
  run_in_background: true

# Check results with TaskOutput when complete
```

See `.claude/skills/llm-council/SKILL.md` for detailed execution patterns.
