# vLLM Project Context

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). It is designed to be fast, flexible, and easy to use, supporting a wide range of hardware (NVIDIA GPUs, AMD GPUs, Intel CPUs/GPUs, TPUs, etc.) and popular open-source models.

**Key Features:**
*   **PagedAttention:** Efficient memory management for attention key/value cache.
*   **Continuous Batching:** High throughput by batching incoming requests.
*   **Broad Hardware Support:** CUDA, ROCm, TPU, CPU, etc.
*   **Quantization Support:** GPTQ, AWQ, INT4, INT8, FP8.
*   **MoE Monokernel Architecture:** A reference implementation (demonstrated on Llama 4) for fusing MoE operations into a single cooperative kernel. Serves as a guide for optimizing any MoE model on H200+ hardware for low-latency regimes.
*   **OpenAI-Compatible API:** Easy integration with existing tools.

## Architecture & Structure

*   **`vllm/`**: Main Python source code.
    *   `engine/`: Core inference engine logic.
    *   `model_executor/`: Model execution and kernel invocation.
    *   `entrypoints/`: API server and CLI entry points.
*   **`csrc/`**: C++ and CUDA/HIP source code for custom kernels (attention, activation, etc.).
    *   `moe/moe_monokernel/`: Reference implementation for monolithic MoE kernel fusion patterns.
*   **`cmake/`**: CMake build configuration files.
*   **`examples/`**: Usage examples and templates.
*   **`tests/`**: Comprehensive test suite using `pytest`.
*   **`requirements/`**: Python dependency files for different platforms.

## Building & Installation

The project uses `setuptools` with a custom `CMakeExtension` to build C++/CUDA extensions.

**Prerequisites:**
*   Python 3.10 - 3.13
*   CUDA toolkit (if building for NVIDIA GPUs) or ROCm (for AMD)
    *   **Note:** The MoE Monokernel reference requires CUDA >= 12.0 and targets Compute Capability 9.0a (Hopper).
*   PyTorch

**Install from Source:**
```bash
# Install dependencies (select the appropriate requirement file, e.g., cuda.txt, rocm.txt, cpu.txt)
pip install -r requirements/cuda.txt

# Build and install
pip install .

# For development (editable install)
pip install -e .
```

**Build Options:**
*   `MAX_JOBS`: Control the number of compilation jobs.
*   `NVCC_THREADS`: Control concurrent NVCC threads.
*   `VLLM_TARGET_DEVICE`: Force target device (e.g., `cuda`, `rocm`, `cpu`, `tpu`).

## Development Conventions

**Code Style & Linting:**
*   **Python:** The project uses `ruff` for linting and formatting. Configuration is in `pyproject.toml`.
    *   Run linting: `ruff check .`
    *   Run formatting: `ruff format .`
*   **Type Checking:** Uses `mypy`. Configured in `pyproject.toml`.
    *   Run type check: `mypy .`
*   **Spell Checking:** Uses `typos`. Configured in `pyproject.toml`.

**Testing:**
*   Framework: `pytest`
*   Configuration: `pyproject.toml` defines markers (e.g., `slow_test`, `cpu_model`, `distributed`).
*   Run tests: `pytest tests/`

**Contribution:**
*   C++ extensions are built via CMake.
*   New models should be added to `vllm/model_executor/models/`.
*   Follow the existing patterns for kernel integration in `csrc/`.

## Optimization Guides

Detailed architecture and optimization guides are located in `optimization-guides/`.
*   **High-Performance MoE Guide:** See `optimization-guides/MOE_MONOKERNEL_OPTIMIZATION_GUIDE.md`. This guide details how to adapt the 'Monokernel' pattern—fusing routing, quantization, and GEMMs—to optimize *any* supported MoE model for specific batch-size regimes (e.g., BS ≤ 64) on Hopper architectures.
