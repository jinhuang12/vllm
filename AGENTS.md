# AGENTS.md

This file provides guidance to Codex when working with code in this repository.

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