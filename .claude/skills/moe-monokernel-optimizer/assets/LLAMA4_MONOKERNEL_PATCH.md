# MoE Monokernel for Llama 4 in vLLM

The version of vLLM used is 0.10.2.

## Full Implementation

```
Subject: [PATCH] Implement MoE Monokernel for Llama4 Scout and Maverick

---
 CMakeLists.txt                                |  20 +
 csrc/moe/moe_monokernel/moe_wrapper.cu        |  87 ++
 csrc/moe/moe_monokernel/src/moe.cu            | 235 +++++
 .../moe_monokernel/src/moe_down_projection.cu | 665 ++++++++++++++
 csrc/moe/moe_monokernel/src/moe_interface.h   | 110 +++
 csrc/moe/moe_monokernel/src/moe_internal.h    | 286 ++++++
 csrc/moe/moe_monokernel/src/moe_prepare.cu    | 501 ++++++++++
 csrc/moe/moe_monokernel/src/moe_routing.cu    | 267 ++++++
 .../moe_monokernel/src/moe_scale_inputs.cu    | 370 ++++++++
 .../moe_monokernel/src/moe_up_projection.cu   | 854 ++++++++++++++++++
 csrc/moe/moe_monokernel/src/ptx_utils.h       | 301 ++++++
 csrc/moe/moe_ops.h                            |  59 +-
 csrc/moe/torch_bindings.cpp                   |  29 +
 vllm/_custom_ops.py                           |  88 ++
 .../compressed_tensors_moe.py                 |  33 +-
 15 files changed, 3902 insertions(+), 3 deletions(-)
 create mode 100644 csrc/moe/moe_monokernel/moe_wrapper.cu
 create mode 100644 csrc/moe/moe_monokernel/src/moe.cu
 create mode 100644 csrc/moe/moe_monokernel/src/moe_down_projection.cu
 create mode 100644 csrc/moe/moe_monokernel/src/moe_interface.h
 create mode 100644 csrc/moe/moe_monokernel/src/moe_internal.h
 create mode 100644 csrc/moe/moe_monokernel/src/moe_prepare.cu
 create mode 100644 csrc/moe/moe_monokernel/src/moe_routing.cu
 create mode 100644 csrc/moe/moe_monokernel/src/moe_scale_inputs.cu
 create mode 100644 csrc/moe/moe_monokernel/src/moe_up_projection.cu
 create mode 100644 csrc/moe/moe_monokernel/src/ptx_utils.h

diff --git a/CMakeLists.txt b/CMakeLists.txt
index bcbd1b52a..e1430a7d2 100644
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -811,6 +811,26 @@ if(VLLM_GPU_LANG STREQUAL "CUDA")
   set(VLLM_MOE_WNA16_SRC
     "csrc/moe/moe_wna16.cu")
 
+  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
+    set(SRCS
+       "csrc/moe/moe_monokernel/moe_wrapper.cu")
+    set_gencode_flags_for_srcs(
+      SRCS "${SRCS}"
+      CUDA_ARCHS "9.0a")
+    list(APPEND VLLM_MOE_EXT_SRC "${SRCS}")
+    message(STATUS "Building optimized MoE kernel for archs: 9.0a")
+  else()
+    if (NOT ${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER 12.0)
+      message(STATUS "Not building moe _wrapper.cu as CUDA Compiler version is "
+                     "not >= 12.0, we recommend upgrading to CUDA 12.0 or "
+                     "later if you intend on running FP8 quantized models on "
+                     "Hopper.")
+    else()
+      message(STATUS "Not building optimized MoE kernel as no compatible archs found "
+                     "in CUDA target architectures")
+    endif()
+  endif()
+
   set_gencode_flags_for_srcs(
     SRCS "${VLLM_MOE_WNA16_SRC}"
     CUDA_ARCHS "${CUDA_ARCHS}")
diff --git a/csrc/moe/moe_monokernel/moe_wrapper.cu b/csrc/moe/moe_monokernel/moe_wrapper.cu
new file mode 100644
index 000000000..38769ee35
--- /dev/null
+++ b/csrc/moe/moe_monokernel/moe_wrapper.cu
@@ -0,0 +1,87 @@
+#include <torch/all.h>
+#include <ATen/cuda/CUDAContext.h>
+#include <c10/cuda/CUDAGuard.h>
+
+#include <ATen/ATen.h>
+#include <ATen/cuda/Atomic.cuh>
+
+#include "cuda_utils.h"
+
+#include "src/moe.cu"
+
+/**
+ * @brief Macro that expands to a kernel call wrapper for moe_kernel with specified @p dims
+ *
+ * moe_kernel() needs to be instantiated with different parameters. In the past, dispatching
+ * to the different kernels on the device or on the host within C++ code generated problems
+ * in vLLM and PyTorch. So these wrappers are to move this dispatch to Python.
+ * For each instantiated kernel, we generate one host invocation function.
+ */
+#define MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(name, dims) \
+    void name( \
+        const torch::Tensor& activations_in, \
+        const torch::Tensor& router_logits, \
+        const torch::Tensor& expert_weights_up, \
+        const torch::Tensor& expert_scales_up, \
+        const torch::Tensor& expert_weights_down, \
+        const torch::Tensor& expert_scales_down, \
+        torch::Tensor& activations_out, \
+        torch::Tensor& scratchpad) \
+    { \
+        /* Check if the input tensors are on the GPU. */ \
+        TORCH_CHECK(activations_in.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(router_logits.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(expert_weights_up.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(expert_scales_up.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(expert_weights_down.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(expert_scales_down.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(activations_out.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+        TORCH_CHECK(scratchpad.is_cuda(), "Optimized MoE kernel must be called with CUDA tensors only."); \
+ \
+        /* Get raw data pointers from the PyTorch tensors. */ \
+        const auto* activations_in_ptr = activations_in.data_ptr<at::BFloat16>(); \
+        const auto* router_logits_ptr = router_logits.data_ptr<at::BFloat16>(); \
+        const auto* expert_weights_up_ptr = expert_weights_up.data_ptr<at::Float8_e4m3fn>(); \
+        const auto* expert_scales_up_ptr = expert_scales_up.data_ptr<float>(); \
+        const auto* expert_weights_down_ptr = expert_weights_down.data_ptr<at::Float8_e4m3fn>(); \
+        const auto* expert_scales_down_ptr = expert_scales_down.data_ptr<float>(); \
+        auto* activations_out_ptr = activations_out.data_ptr<at::BFloat16>(); \
+        char* scratchpad_ptr = reinterpret_cast<char*>(scratchpad.data_ptr<float>()); \
+ \
+        using namespace moe_monokernel; \
+        const uint32_t num_tokens = activations_in.size(0); \
+        const uint32_t num_experts = expert_weights_up.size(0); \
+        const size_t shmem_size = get_moe_max_shmem_size(); \
+        const size_t scratchpad_size = scratchpad.nbytes(); \
+ \
+        void *kernel_args[] = { \
+            (void *)&activations_in_ptr, \
+            (void *)&num_tokens, \
+            (void *)&router_logits_ptr, \
+            (void *)&expert_weights_up_ptr, \
+            (void *)&expert_scales_up_ptr, \
+            (void *)&expert_weights_down_ptr, \
+            (void *)&expert_scales_down_ptr, \
+            (void *)&activations_out_ptr, \
+            (void *)&scratchpad_ptr, \
+            (void *)&scratchpad_size, \
+            (void *)&shmem_size \
+        }; \
+        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); \
+        CUDA_CHECK(cudaFuncSetAttribute(moe_kernel<dims>, \
+                                        cudaFuncAttributeMaxDynamicSharedMemorySize, \
+                                        shmem_size)); \
+        CUDA_CHECK(cudaLaunchCooperativeKernel( \
+                moe_kernel<dims>, \
+                dims::KernelConfig::GRID_SIZE, \
+                dims::KernelConfig::BLOCK_SIZE, \
+                kernel_args, \
+                shmem_size, \
+                stream)); \
+    }
+
+MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS8_E16_TP8_impl, moe_monokernel::Dims_BS8_E16_TP8)
+MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS64_E16_TP8_impl, moe_monokernel::Dims_BS64_E16_TP8)
+
+MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS8_E128_TP8_impl, moe_monokernel::Dims_BS8_E128_TP8)
+MOEMONOKERNEL_WRAPPER_IMPLEMENTATION(moe_monokernel_BS64_E128_TP8_impl, moe_monokernel::Dims_BS64_E128_TP8)
diff --git a/csrc/moe/moe_monokernel/src/moe.cu b/csrc/moe/moe_monokernel/src/moe.cu
new file mode 100644
index 000000000..91a41240a
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe.cu
@@ -0,0 +1,235 @@
+/**
+ * This is the main file of the MoE monokernel.
+ * It is designed so that you just need to build this file. It includes all relevant implementations.
+ * For documentation of the main entry function moe_kernel, see moe_interface.h
+ */
+
+#include <cooperative_groups.h>
+#include <cstdint>
+
+#include "moe_interface.h"
+
+#define INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#include "moe_down_projection.cu"
+#include "moe_internal.h"
+#include "moe_prepare.cu"
+#include "moe_scale_inputs.cu"
+#include "moe_up_projection.cu"
+#include "moe_routing.cu"
+#undef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+
+namespace moe_monokernel {
+
+template <typename Dims>
+__device__ void moe_kernel_BS64(
+    const A_element* __restrict__ activations_in,
+    std::uint32_t batch_size,
+    const __nv_bfloat16* __restrict__ router_logits,
+    const W_element* __restrict expert_weights_up,
+    const S_element* __restrict expert_scales_up,
+    const W_element* __restrict expert_weights_down,
+    const S_element* __restrict expert_scales_down,
+    R_element* __restrict activations_out,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    if (is_calc_warp<Dims>()) {
+        top1_BS64<Dims>(router_logits, batch_size, shmem);
+    }
+    __syncthreads();
+    prepare_moe_BSx_Ey<Dims>(batch_size, shmem);
+    __syncthreads();
+    assert(shmem->experts[shmem->expert_count-1].last_token == batch_size);
+
+    moe_scale_activation_BSx<Dims>(
+        activations_in,
+        batch_size,
+        spec,
+        shmem);
+
+#ifdef DEBUG_MOE
+    if (blockIdx.x == 0) {
+        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
+            spec->token_indexes[i] = shmem->token_indexes[i];
+        }
+    }
+#endif
+
+    moe_up_projection<Dims>(
+        expert_weights_up,
+        expert_scales_up,
+        spec,
+        shmem);
+    cooperative_groups::this_grid().sync();
+    moe_down_projection<Dims>(
+        batch_size,
+        expert_weights_down,
+        expert_scales_down,
+        activations_out,
+        spec,
+        shmem);
+}
+
+template <typename Dims>
+__device__ void moe_kernel_BS8(
+    const A_element* __restrict__ activations_in,
+    std::uint32_t batch_size,
+    const __nv_bfloat16* __restrict__ router_logits,
+    const W_element* __restrict expert_weights_up,
+    const S_element* __restrict expert_scales_up,
+    const W_element* __restrict expert_weights_down,
+    const S_element* __restrict expert_scales_down,
+    R_element* __restrict activations_out,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    static_assert(Dims::BS <= 8);
+
+    using CoreDims = MoECoreDims<Dims>;
+
+    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
+    if (is_prefetch_warp<Dims>()) {
+        // Prefetch activations for rescaling
+        const std::uint32_t warp = get_prefetch_warp<Dims>();
+        for (std::uint32_t token = warp; token < batch_size; token += CoreDims::PREFETCH_WARP_COUNT) {
+            moe_fetch_activation_async<Dims>(
+                activations_in + token * Dims::HIDDEN_STATES,
+                shmem->u.tiny.w[0].orig[token],
+                pipe);
+        }
+    } else {
+        top1_BS8<Dims>(router_logits, batch_size, shmem);
+        sync_calc_threads<Dims>();
+        prepare_moe_BS8<Dims>(batch_size, shmem);
+    }
+
+    cuda::pipeline_consumer_wait_prior<0>(pipe);
+    __syncthreads();
+    if (is_prefetch_warp<Dims>()) {
+        //
+        // Prefetch first expert weights
+        //
+        pipe.producer_acquire();
+
+        // bring in first W tile
+        moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
+            expert_weights_up,
+            shmem->expert_ids & 0xff,
+            shmem->u.tiny.w[1].up,
+            pipe);
+
+        pipe.producer_commit();
+    }
+    else {
+        //
+        // Rescale activations
+        //
+        const std::uint32_t warp = get_calc_warp<Dims>();
+        if (warp < batch_size) {
+            moe_scale_activation_BS8<Dims>(
+                shmem->u.tiny.w[0].orig[warp],
+                (AQ_element *)shmem->u.tiny.a.up[warp],
+                shmem->topk_weights[warp]);
+        }
+    }
+
+    __syncthreads();
+
+#ifdef DEBUG_MOE
+    if (blockIdx.x == 0) {
+        for (int i = threadIdx.x; i < batch_size; i += blockDim.x) {
+            spec->token_indexes[i] = shmem->token_indexes[i];
+        }
+    }
+#endif
+
+    std::uint32_t w_index = moe_up_projection_tiny<Dims>(
+        expert_weights_up,
+        expert_scales_up,
+        expert_weights_down,
+        expert_scales_down,
+        1,
+        spec,
+        shmem,
+        pipe);
+    cooperative_groups::this_grid().sync();
+    moe_down_projection_tiny<Dims>(
+        batch_size,
+        expert_weights_down,
+        expert_scales_down,
+        w_index,
+        activations_out,
+        spec,
+        shmem,
+        pipe);
+}
+
+template <typename Dims>
+__global__ void moe_kernel(
+    const A_element* __restrict__ activations_in,
+    std::uint32_t token_count,
+    const __nv_bfloat16 *__restrict__ router_logits,
+    const W_element* __restrict__ expert_weights_up,
+    const S_element* __restrict__ expert_scales_up,
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    R_element* __restrict__ activations_out,
+    void* __restrict__ scratchpad,
+    size_t scratchpad_size,
+    size_t shmem_size)
+{
+    // we require 8 warps per SM and assume X to be the only relevant dimension
+    assert(MoECoreDims<Dims>::THREADS_PER_WARP == 32);
+    assert(blockDim.x == Dims::KernelConfig::BLOCK_SIZE);
+    assert(blockDim.y == 1);
+    assert(blockDim.z == 1);
+
+    assert(gridDim.x == Dims::KernelConfig::GRID_SIZE);
+    assert(gridDim.y == 1);
+    assert(gridDim.z == 1);
+
+    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
+
+    assert(token_count <= Dims::BS);
+    assert(token_count > 0);
+
+    assert((uintptr_t) scratchpad % alignof(MoEGemmSpec<Dims>) == 0);
+    assert(scratchpad_size >= get_moe_scratchpad_size<Dims>());
+    MoEGemmSpec<Dims> *spec = reinterpret_cast<MoEGemmSpec<Dims> *>(scratchpad);
+
+    assert(shmem_size >= get_moe_shmem_size<Dims>());
+
+    extern __shared__ char shmem_buffer[];
+    MoE_SHM<Dims>* shmem = reinterpret_cast<MoE_SHM<Dims>*>(shmem_buffer);
+
+    if constexpr (Dims::BS <= 8) {
+        moe_kernel_BS8(
+                activations_in,
+                token_count,
+                router_logits,
+                expert_weights_up,
+                expert_scales_up,
+                expert_weights_down,
+                expert_scales_down,
+                activations_out,
+                spec,
+                shmem);
+    } else {
+        moe_kernel_BS64(
+                activations_in,
+                token_count,
+                router_logits,
+                expert_weights_up,
+                expert_scales_up,
+                expert_weights_down,
+                expert_scales_down,
+                activations_out,
+                spec,
+                shmem);
+    }
+}
+
+} // namespace moe_monokernel
diff --git a/csrc/moe/moe_monokernel/src/moe_down_projection.cu b/csrc/moe/moe_monokernel/src/moe_down_projection.cu
new file mode 100644
index 000000000..1d8dece85
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_down_projection.cu
@@ -0,0 +1,665 @@
+
+#pragma once
+#ifndef MOE_DOWN_PROJECTION_CU
+#define MOE_DOWN_PROJECTION_CU
+
+#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#error Do not include this file directly.
+#endif
+
+#include <cuda.h>
+#include <cuda/pipeline>
+#include <cuda_fp8.h>
+#include <stdio.h>
+
+#include "moe_interface.h"
+#include "moe_internal.h"
+#include "ptx_utils.h"
+
+///////////////////////////////////////////////////////////////////////////////
+//
+// Design Considerations
+//
+// * The smallest matrix dimensions that the Tensor Cores support is 16 x 8 x K
+//   - 8 is a reasonable amount of input tokens we can process at once per expert,
+//     if there are <= 64 tokens and >= 16 experts.
+//   - Weight matrix is 5120 rows, distributed over 128 SMs, gives 40 rows/SM
+//   - run 2 MMA iterations with 16 rows each
+//   - duplicate the remaining 8 rows when running the 3rd iteration and filter
+//     the output (don't overwrite results of other SM!)
+//
+// * Max. 220kB of the 224kB Shared Memory are available for matrix tiles
+//   - token/'temp' tiles are 32kB each, weight tiles are 40kB each
+//   - plenty to space for double buffering both: fetching the next tile of expert
+//     weights and / or token activations while processing the current one.
+//
+// * Reading matrix data from Shared Memory accesses the same columns in different
+//   rows simultaneously.
+//   - kernel uses tile rows lengths with an extra 16 byte padding,
+//     mapping the same column for all 8 consecutive rows onto to different banks.
+//   - downside: prevents Global->Shared Memory transfers from being fully coalesced.
+//
+///////////////////////////////////////////////////////////////////////////////
+
+namespace moe_monokernel {
+
+/**
+ * @brief Initiate the copy of expert weights and scales from Global to Shared Memory
+ *
+ * This device function issues the asynchronous data copy requests for a tile of 
+ * expert weights and corresponding weights. The copy operations will be queued
+ * in the given @a pipe, which the caller must use to wait for their completion.
+ * 
+ * While the expert is selected by @a id, the tile to copy is implicitly selected
+ * by the @c blockIdx. The result is stored in the tile @a w_index within @a shm.
+ *
+ * @note Like all prefetching functions, this function must only be called by
+ *       threads in prefetch warps.
+ *
+ * @param expert_weights_down Pointer weights array of shape [NUM_EXPERTS, HIDDEN_STATES, N]
+ *                            in expert, row-major order.
+ *                            Individual elements are in __nv_fp8_e4m3 format.
+ *                            Stored in Global Memory.
+ * @param expert_scales_down Pointer scales array of shape [NUM_EXPERTS, HIDDEN_STATES]
+ *                           in row-major order.
+ *                           Individual elements are in __nv_fp8_e4m3 format.
+ *                           Stored in Global Memory.
+ * @param id Expert index within @a expert_weights_down and @a expert_scales_down.
+ * @param shm Shared Memory struct to store the result to.
+ * @param w_index Index of tile to use within @a shm.
+ * @param pipe Asynchronous completion pipe to use.
+ */
+template <typename Dims>
+__device__ inline void moe_request_down_expert(
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    std::uint32_t id,
+    typename MoE_SHM<Dims>::U::Gemm2Data* shm,
+    std::uint32_t w_index,
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    const unsigned base_row = blockIdx.x * MoECoreDims<Dims>::W_DOWN_TILE;
+
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned d_thread = threadIdx.x % (2 * CoreDims::THREADS_PER_WARP);
+    const unsigned d_warp = get_prefetch_warp<Dims>() / 2;
+
+    // request W tile
+    {
+        const unsigned chunk_size = 16;
+        static_assert(Dims::N <= 2 * CoreDims::THREADS_PER_WARP * chunk_size);
+
+        // Compiler on H200 barfs out on plain FP8 transfers as it fails to propagate alignment guarantees:
+        // just case to a 32-bit value; we know all data to be 16-byte aligned.
+        const OpaqueElement* weights = (const OpaqueElement *)(expert_weights_down + id * Dims::N * Dims::HIDDEN_STATES + base_row * Dims::N);
+        for (unsigned row = d_warp, i = 0; i < CoreDims::W_DOWN_TILE / (CoreDims::PREFETCH_WARP_COUNT / 2); row += CoreDims::PREFETCH_WARP_COUNT / 2, i++) {
+            unsigned col = d_thread * chunk_size;
+            // "clever" condition to allow for compile-time optimization (becomes no-op on H200)
+            if (Dims::N == 2 * CoreDims::THREADS_PER_WARP * chunk_size || col < Dims::N) {
+                copy128(shm->w[w_index][row][col],
+                        weights[(row * Dims::N + col) / sizeof(OpaqueElement)],
+                        pipe);
+            }
+        }
+    }
+
+    // request Scale tile
+    if (d_warp == 0) {
+        const unsigned chunk_size = 16 / sizeof(*expert_scales_down);
+        if (d_thread < CoreDims::W_DOWN_TILE / chunk_size) {
+            copy128(shm->scale[w_index][chunk_size * d_thread],
+                    expert_scales_down[id * Dims::HIDDEN_STATES + base_row + chunk_size * d_thread],
+                    pipe);
+        }
+    }
+}
+
+/**
+ * @brief Initiate the copy of expert weights and scales from Global to Shared Memory for 'Tiny' kernel.
+ *
+ * This device function issues the asynchronous data copy requests for a tile of 
+ * expert weights and corresponding weights. The copy operations will be queued
+ * in the given @a pipe, which the caller must use to wait for their completion.
+ * 
+ * While the expert is selected by @a id, the tile to copy is implicitly selected
+ * by the @c blockIdx. The result is stored in the tile @a w_index within @a shm.
+ *
+ * @note Like all prefetching functions, this function must only be called by
+ *       threads in prefetch warps.
+ *
+ * @param expert_weights_down Pointer weights array of shape [NUM_EXPERTS, HIDDEN_STATES, N]
+ *                            in expert, row-major order.
+ *                            Individual elements are in __nv_fp8_e4m3 format.
+ *                            Stored in Global Memory.
+ * @param expert_scales_down Pointer scales array of shape [NUM_EXPERTS, HIDDEN_STATES]
+ *                           in row-major order.
+ *                           Individual elements are in __nv_fp8_e4m3 format.
+ *                           Stored in Global Memory.
+ * @param id Expert index within @a expert_weights_down and @a expert_scales_down.
+ * @param shm Shared Memory struct to store the result to.
+ * @param w_index Index of tile to use within @a shm.
+ * @param pipe Asynchronous completion pipe to use.
+ */
+template <typename Dims>
+__device__ inline void moe_request_down_expert_tiny(
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    std::uint32_t id,
+    typename MoE_SHM<Dims>::U::TinyData* shm,
+    std::uint32_t w_index,
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    const unsigned base_row = blockIdx.x * MoECoreDims<Dims>::W_DOWN_TILE;
+
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned d_thread = threadIdx.x % (2 * CoreDims::THREADS_PER_WARP);
+    const unsigned d_warp = get_prefetch_warp<Dims>() / 2;
+
+    // request W tile
+    {
+        const unsigned chunk_size = 16;
+        static_assert(Dims::N <= 2 * CoreDims::THREADS_PER_WARP * chunk_size);
+
+        // Compiler on H200 barfs out on plain FP8 transfers as it fails to propagate alignment guarantees:
+        // just case to a 32-bit value; we know all data to be 16-byte aligned.
+        const OpaqueElement* weights = (const OpaqueElement *)(expert_weights_down + id * Dims::N * Dims::HIDDEN_STATES + base_row * Dims::N);
+        for (unsigned row = d_warp, i = 0; i < CoreDims::W_DOWN_TILE / (CoreDims::PREFETCH_WARP_COUNT / 2); row += CoreDims::PREFETCH_WARP_COUNT / 2, i++) {
+            unsigned col = d_thread * chunk_size;
+            // "clever" condition to allow for compile-time optimization (becomes no-op on H200)
+            if (Dims::N == 2 * CoreDims::THREADS_PER_WARP * chunk_size || col < Dims::N) {
+                copy128(shm->w[w_index].down[row][col],
+                        weights[(row * Dims::N + col) / sizeof(OpaqueElement)],
+                        pipe);
+            }
+        }
+    }
+
+    // request Scale tile
+    if (d_warp == 0) {
+        const unsigned chunk_size = 16 / sizeof(*expert_scales_down);
+        if (d_thread < CoreDims::W_DOWN_TILE / chunk_size) {
+            copy128(shm->scale[w_index][chunk_size * d_thread],
+                    expert_scales_down[id * Dims::HIDDEN_STATES + base_row + chunk_size * d_thread],
+                    pipe);
+        }
+    }
+}
+
+/**
+ * @brief Initiate the copy of up-projection output ('temp tokens') from Global to Shared Memory
+ *
+ * This device function issues the asynchronous data copy requests for a tile of 
+ * token activations. The copy operations will be queued in the given @a pipe,
+ * which the caller must use to wait for their completion.
+ *
+ * The number of tokens copied is limited by the capacity of @a dest as well as
+ * the remaining range of tokens specified by @a expert and the starting offset
+ * @a a_row. 
+ *
+ * @note Like all prefetching functions, this function must only be called by
+ *       threads in prefetch warps.
+ *
+ * @param source Pointer activation array of shape [BS, N] in row-major order.
+ *               Individual elements are in FP32 format.
+ *               Stored in Global Memory.
+ * @param expert Specified the range of rows relevant for the current expert.
+ * @param a_row Offset of the first row within the range of @a expert to copy.
+ * @param dest Shared Memory tile to store the copy.
+ * @param pipe Asynchronous completion pipe to use.
+ */
+template <typename Dims, std::size_t Rows, std::size_t Cols>
+__device__ inline void moe_request_temp_token(
+    const T_element* __restrict__ source,
+    const ExpertRef& expert,
+    unsigned a_row,
+    T_element (&dest)[Rows][Cols],
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_prefetch_warp<Dims>();
+
+    // async transfers are 16 bytes / thread
+    const unsigned chunk_size = 16 / sizeof(*source);
+
+    const T_element* t = &source[expert.first_token * Dims::N];
+    unsigned int a_rows = expert.last_token - expert.first_token;
+    for (unsigned i = warp; i < Rows; i += CoreDims::PREFETCH_WARP_COUNT) {
+        if (i + a_row < a_rows) {
+            for (unsigned col = thread * chunk_size; col < Cols; col += CoreDims::THREADS_PER_WARP * chunk_size) {
+                copy128(dest[i][col], t[(i + a_row) * Dims::N + col], pipe);
+            }
+        }
+    }
+}
+
+/**
+ * @brief Run partial the MMAs on all warps in parallel.
+ *
+ * This device function multiplies the given @a temps tile of activations with
+ * the tile of @a weights, multiplies the result with the respective @a scale
+ * and stores it in @a partial_result.
+ * 
+ * The output can be filtered, i.e. @a store_row0 and @a store_row1 control whether
+ * the results for the respective tokens shall be written.  This allows the called
+ * to optionally interleave the result of multiple experts by selectively suppressing
+ * results in the output.
+ *
+ * @note Like all calculation functions, this function must only be called by
+ *       threads in calculation warps.
+ *
+ * @param weights Weights tile. Elements are in __nv_fp8_e4m3 format.
+ * @param scale Scaling factor for each row of @a weights.
+ * @param temps Activations tile to multiply with @a weights.
+ * @param store_row0 If @c false , suppress the output of the first token.
+ * @param store_row1 If @c false , suppress the output of the second token.
+ * @param partial_result .
+ */
+template <typename Dims, std::size_t Rows, std::size_t Cols, std::size_t OutCols>
+__device__ inline void moe_down_mult(
+    const W_element __restrict__ (&weights)[Rows][Cols],
+    const float* __restrict__ scale,
+    const T_element __restrict__ (&temps)[Dims::N],
+    bool store_row0,
+    bool store_row1,
+    float (&partial_result)[Rows / 2 + MoECoreDims<Dims>::CALC_WARP_COUNT / 2][OutCols])
+{
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_calc_warp<Dims>();
+
+    for (unsigned w_row = 0; w_row < CoreDims::W_DOWN_TILE; w_row += CoreDims::W_DOWN_MMA_TILE) {
+
+        // init accumulators
+        float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
+
+        // run partial scalar products
+        for (unsigned base_col = warp * CoreDims::K_TILE, i = 0; i < Dims::N / CoreDims::BLOCK_STRIDE; base_col += CoreDims::BLOCK_STRIDE, i++) {
+
+            unsigned far_row = (Rows % 16 == 8 && w_row + 8 == Rows) ? w_row : w_row + 8;
+            __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&weights[w_row   + thread / 4][base_col + 4 * (thread % 4) +   0];
+            __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&weights[far_row + thread / 4][base_col + 4 * (thread % 4) +   0];
+            __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&weights[w_row   + thread / 4][base_col + 4 * (thread % 4) +  16];
+            __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&weights[far_row + thread / 4][base_col + 4 * (thread % 4) +  16];
+
+            float4 b0 = *(float4*)&temps[base_col + 4 * (thread % 4) +  0];
+            float4 b1 = *(float4*)&temps[base_col + 4 * (thread % 4) + 16];
+
+            mma_fp8_tf32(d0, d1, d2, d3, w0, w1, w2, w3, b0, b1, d0, d1, d2, d3);
+        }
+
+        // load scales (quick-ish as they are stored in SHM)
+        float ws0 = scale[(thread / 4) + w_row + 0];
+        float ws1 = scale[(thread / 4) + w_row + 8];
+
+        d0 *= ws0;
+        d1 *= ws0;
+        d2 *= ws1;
+        d3 *= ws1;
+
+        if (store_row0) {
+            partial_result[w_row / 2 + warp][thread +  0] = d0;
+            partial_result[w_row / 2 + warp][thread + 64] = d2;
+        }
+        if (store_row1) {
+            partial_result[w_row / 2 + warp][thread + 32] = d1;
+            partial_result[w_row / 2 + warp][thread + 96] = d3;
+        }
+    }
+}
+
+/**
+ * @brief Performs the MMA result reduction.
+ *
+ * This device function sums up the partial scalar products created by all warps
+ * and stores the results in Global Memory.  The tile to be written is
+ * implicitly determined by the @c blockIdx.
+ * 
+ * The output can be filtered, i.e. @a store_row0 and @a store_row1 control whether
+ * the results for the respective rows @a row0 and @a row1 shall be written.  This
+ * allows the called to always process data at the full tile size and simply suppress
+ * superfluous results in the output.
+ *
+ * @param partial_result Array of MMA results from all warps of shape [20, 4, THREADS]
+ *                       in row-major order. Individual elements are in FP32 format.
+ * @param store_row0 Specifies if result in @a row0 shall be stored.
+ * @param store_row1 Specifies if result in @a row1 shall be stored.
+ * @param row0 Row to store the scalar products for the first token.
+ * @param row1 Row to store the scalar products for the second token.
+ * @param result Pointer to the output array of shape [BS, N] in row-major order.
+ *               Individual elements are in FP32 format.
+ */
+template <typename Dims, std::size_t Rows, std::size_t Cols>
+__device__ inline void moe_down_reduction(
+    const float (&partial_result)[Rows][Cols],
+    bool store_row0,
+    bool store_row1,
+    unsigned row0,
+    unsigned row1,
+    R_element* __restrict__ result)
+{
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_any_warp<Dims>();
+
+    // starting row to process
+    const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;
+
+    // reduction and output
+    for (unsigned w_row = warp * CoreDims::W_DOWN_MMA_TILE; w_row < CoreDims::W_DOWN_TILE; w_row += CoreDims::W_DOWN_MMA_TILE * CoreDims::TOTAL_WARP_COUNT) {
+
+        float d0 = partial_result[w_row / 2][thread +  0];
+        float d1 = partial_result[w_row / 2][thread + 32];
+        float d2 = partial_result[w_row / 2][thread + 64];
+        float d3 = partial_result[w_row / 2][thread + 96];
+
+        // combine results
+        for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
+
+            d0 += partial_result[w_row / 2 + i][thread +  0];
+            d1 += partial_result[w_row / 2 + i][thread + 32];
+            d2 += partial_result[w_row / 2 + i][thread + 64];
+            d3 += partial_result[w_row / 2 + i][thread + 96];
+        }
+
+        // write final result. Only write valid lines
+        if (store_row0) {
+            result[row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row + 0] = d0;
+            if (CoreDims::W_DOWN_TILE % 16 == 0 || w_row + 8 < CoreDims::W_DOWN_TILE) {
+                result[row0 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row + 8] = d2;
+            }
+        }
+        if (store_row1) {
+            result[row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row + 0] = d1;
+            if (CoreDims::W_DOWN_TILE % 16 == 0 || w_row + 8 < CoreDims::W_DOWN_TILE) {
+                result[row1 * Dims::HIDDEN_STATES + (thread / 4) + base_row + w_row + 8] = d3;
+            }
+        }
+    }
+}
+
+/**
+ * @brief Standard kernel for the second GEMM ("down projection").
+ *
+ * This device function processes @c BS tokens, grouped by expert in internal batches of 8 tokens.
+ * The experts to use and the respective list of tokes for each of them is given by @a spec.
+ *
+ * Activations are taken from temporary storage in @a spec and all non-expert data is taken from
+ * @a shmem. Outputs are in the same order as the input to "up projection".
+ *
+ * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS, HIDDEN_STATES, N]
+ *                          in expert, row-major order. Individual elements are in __nv_fp8_e4m3 format.
+ *                          Stored in Global Memory.
+ * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS, HIDDEN_STATES]
+ *                         in row-major order. Individual elements are in FP32 format.
+ *                         Stored in Global Memory.
+ * @param result Global Memory array of shape [BS, HIDDEN_STATES] in row-major order, receiving
+ *               the output.  Individual elements are in __nv_bfloat16 format.
+ * @param spec Global Memory struct containing the scaled input token activations.
+ * @param shmem Shared Memory struct containing the expert<=>token mapping, activation weights
+ *              and will be uses as local scratch pad store for faster operation.
+ */
+template <typename Dims>
+__device__ inline void moe_down_projection_normal(
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    R_element* __restrict__ result,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem
+)
+{
+    using CoreDims = MoECoreDims<Dims>;
+    using MoE_SHM = MoE_SHM<Dims>;
+
+    // position within the block
+    const unsigned thread = get_thread<Dims>();
+
+    // required for async copies
+    cuda::pipeline<cuda::thread_scope_thread> pipe   = cuda::make_pipeline();
+    const auto                                shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
+
+    // commonly used values
+    typename MoE_SHM::U::Gemm2Data* shm = &shmem->u.gemm2;
+    const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;
+    std::uint32_t expert_count = shmem->expert_count;
+
+    const ExpertRef& expert = shmem->experts[0];
+    std::uint32_t id = expert.id;
+
+    assert(expert_count > 0);
+
+    // request first W tile
+    if (is_prefetch_warp<Dims>()) {
+        pipe.producer_acquire();
+        moe_request_down_expert<Dims>(expert_weights_down,
+            expert_scales_down,
+            id,
+            shm,
+            0,
+            pipe);
+
+        // request first T tile
+        moe_request_temp_token<Dims>(spec->temp, expert, 0, shm->t[0], pipe);
+        pipe.producer_commit();
+    }
+
+    // current tile indexes
+    // will be toggled to 0 at the start of the first loop iterations
+    std::uint32_t t_index = 1;
+    std::uint32_t w_index = 1;
+
+    // loop over all experts
+    for (std::uint32_t e = 0; e < expert_count; ++e) {
+        const ExpertRef& expert = shmem->experts[e];
+        unsigned int a_rows = expert.last_token - expert.first_token;
+        w_index ^= 1;
+
+        // process all activations for this set of weights
+        for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::T_TILE) {
+            t_index ^= 1;
+
+            // wait for currently needed data to come in
+            cuda::pipeline_consumer_wait_prior<0>(pipe);
+            __syncthreads();
+
+            // request next T, S and W tiles as needed
+            if (is_prefetch_warp<Dims>()) {
+                pipe.producer_acquire();
+                if (e + 1 < expert_count && a_row == 0) {
+                    // request first W tile
+                    moe_request_down_expert<Dims>(expert_weights_down,
+                        expert_scales_down,
+                        shmem->experts[e + 1].id,
+                        shm,
+                        w_index ^ 1,
+                        pipe);
+                }
+                if (a_row + CoreDims::T_TILE < a_rows) {
+                    // request the next T tile for same expert
+                    moe_request_temp_token<Dims>(spec->temp, expert, a_row + CoreDims::T_TILE, shm->t[t_index ^ 1], pipe);
+                }
+                else if (e + 1 < expert_count) {
+                    // request the first T tile for next expert
+                    moe_request_temp_token<Dims>(spec->temp, shmem->experts[e + 1], 0, shm->t[t_index ^ 1], pipe);
+                }
+                pipe.producer_commit();
+            }
+            else {
+                // matrix multiplication step
+                static_assert(CoreDims::W_DOWN_TILE % 8 == 0);
+                moe_down_mult<Dims>(
+                    shm->w[w_index],
+                    shm->scale[w_index],
+                    shm->t[t_index][thread / 4],
+                    true,
+                    true,
+                    shm->partial_result);
+            }
+
+            __syncthreads();
+
+            // reduction and output
+            moe_down_reduction<Dims>(
+                shm->partial_result,
+                a_row + (thread % 4) * 2 < a_rows,
+                a_row + (thread % 4) * 2  + 1 < a_rows,
+                shmem->token_indexes[expert.first_token + a_row + (thread % 4) * 2],
+                shmem->token_indexes[expert.first_token + a_row + (thread % 4) * 2 + 1],
+                result);
+        }
+
+        __syncthreads();
+    }
+}
+
+/**
+ * @brief 'Tiny' kernel for the second GEMM ("down projection").
+ *
+ * This device function processes up to 8 tokens, store in @a spec.
+ * The experts to use with each token is given by @a shm, the function will apply them to all
+ * and only filter the output accordingly.
+ * The weights and scales for the first expert have already been prefetched into @a shmem as well. 
+ *
+ * All non-expert data is taken from our temporary storage in @a shmem
+ * and results will be written to @a result. Output order is the input token order.
+ *
+ * @param token_count Number of input tokens.
+ * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS, HIDDEN_STATES, N]
+ *                          in expert, row-major order. Individual elements are in __nv_fp8_e4m3 format.
+ *                          Stored in Global Memory.
+ * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS, HIDDEN_STATES]
+ *                         in row-major order. Individual elements are in FP32 format.
+ *                         Stored in Global Memory.
+ * @param w_index Index of the tile in @a shmem that contains the prefetched weight data
+ *                for the first expert.
+ * @param result Global Memory array of shape [BS, HIDDEN_STATES] in row-major order, receiving
+ *               the output.  Individual elements are in __nv_bfloat16 format.
+ * @param spec Global Memory struct containing the token activations.
+ * @param shmem Shared Memory struct containing the expert<=>token mapping, activation weights
+ *              and will be uses as local scratch pad store for faster operation.
+ * @param pipe Asynchronous completion pipe to use with prefetching.
+ */
+template <typename Dims>
+__device__ inline void moe_down_projection_tiny(
+    std::uint32_t token_count,
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    std::uint32_t w_index,
+    R_element* __restrict__ result,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem,
+    cuda::pipeline<cuda::thread_scope_thread>& pipe
+)
+{
+    using CoreDims = MoECoreDims<Dims>;
+    using MoE_SHM = MoE_SHM<Dims>;
+
+    assert(w_index == 0 || w_index == 1);
+
+    // position within the block
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_any_warp<Dims>();
+
+    // commonly used values
+    typename MoE_SHM::U::TinyData* shm = &shmem->u.tiny;
+    const unsigned base_row = blockIdx.x * CoreDims::W_DOWN_TILE;
+    std::uint32_t expert_count = shmem->expert_count;
+    std::uint64_t expert_mask = shmem->expert_mask;
+    std::uint64_t expert_ids = shmem->expert_ids;
+
+    assert(expert_count > 0);
+    std::uint32_t id = expert_ids & 0xff;
+
+    // request data for the first expert
+    pipe.producer_acquire();
+
+    // request T (W has already been requested by up-projection)
+    if (warp < token_count) {
+        for (unsigned col = thread * 4, i = 0; i < Dims::N / (CoreDims::THREADS_PER_WARP * 4); i++, col += CoreDims::THREADS_PER_WARP * 4) {
+            copy128(shm->a.down[warp][col], spec->temp[warp * Dims::N + col], pipe);
+        }
+    }
+
+    pipe.producer_commit();
+
+    // loop over all experts
+    for (std::uint32_t e = 0; e < expert_count; ++e) {
+        id = expert_ids & 0xff;
+        expert_ids >>= 8;
+
+        // wait for currently needed data to come in
+        cuda::pipeline_consumer_wait_prior<0>(pipe);
+        __syncthreads();
+
+        // request next S and W tiles as needed
+        if (is_prefetch_warp<Dims>()) {
+            pipe.producer_acquire();
+            if (e + 1 < expert_count) {
+                moe_request_down_expert_tiny<Dims>(expert_weights_down,
+                    expert_scales_down,
+                    expert_ids & 0xff,
+                    shm,
+                    w_index ^ 1,
+                    pipe);
+            }
+            pipe.producer_commit();
+        }
+        else {
+            // matrix multiplication step
+            static_assert(CoreDims::W_DOWN_TILE % 8 == 0);
+            std::uint32_t row0 = (thread % 4) * 2 + 0;
+            std::uint32_t row1 = (thread % 4) * 2 + 1;
+            bool store_row0 = (expert_mask >> (row0 * 8) & 0xff) == id;
+            bool store_row1 = (expert_mask >> (row1 * 8) & 0xff) == id;
+
+            moe_down_mult<Dims>(
+                shm->w[w_index].down,
+                shm->scale[w_index],
+                shm->a.down[thread / 4],
+                store_row0,
+                store_row1,
+                shm->partial_result.down);
+        }
+
+        __syncthreads();
+        w_index ^= 1;
+    }
+
+    // reduction and output
+    std::uint32_t row0 = (thread % 4) * 2 + 0;
+    std::uint32_t row1 = (thread % 4) * 2 + 1;
+    moe_down_reduction<Dims>(
+        shm->partial_result.down,
+        row0 < token_count,
+        row1 < token_count,
+        row0,
+        row1,
+        result);
+}
+
+/**
+ * @brief Forwards to @c moe_down_projection_normal.
+ *
+ * @see moe_down_projection_normal
+ */
+template <typename Dims>
+__device__ void moe_down_projection(
+    std::uint32_t token_count,
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    R_element* __restrict__ result,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    static_assert(Dims::BS > 8, "Tiny is handled by its own kernel. Do not use moe_down_projection for BS<=8");
+    moe_down_projection_normal<Dims>(expert_weights_down, expert_scales_down, result, spec, shmem);
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/moe_interface.h b/csrc/moe/moe_monokernel/src/moe_interface.h
new file mode 100644
index 000000000..239639273
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_interface.h
@@ -0,0 +1,110 @@
+#ifndef MOE_INTERFACE_H
+#define MOE_INTERFACE_H
+
+#pragma once
+
+#include <cuda_bf16.h>
+#include <cuda_fp8.h>
+#include <cstdint>
+
+namespace moe_monokernel {
+
+template <uint32_t m, uint32_t n, uint32_t k, uint32_t num_experts>
+struct MoEDimensions {
+    static constexpr uint32_t HIDDEN_STATES = k;
+    static constexpr uint32_t K = k;
+    static constexpr uint32_t N = n;
+    static constexpr uint32_t BS = m;
+    static constexpr uint32_t M = m;
+    static constexpr uint32_t NUM_EXPERTS = num_experts;
+
+    struct KernelConfig {
+        static constexpr std::uint32_t GRID_SIZE = (2*N) / 16;
+        static constexpr std::uint32_t BLOCK_SIZE = 384;
+    };
+};
+
+// Pre-defined dimensions for Llama4 Scout and Maverick
+using Dims_BS8_E16_TP8 = MoEDimensions<8, 1024, 5120, 16>;
+using Dims_BS64_E16_TP8 = MoEDimensions<64, 1024, 5120, 16>;
+using Dims_BS8_E128_TP8 = MoEDimensions<8, 1024, 5120, 128>;
+using Dims_BS64_E128_TP8 = MoEDimensions<64, 1024, 5120, 128>;
+
+using W_element = __nv_fp8_e4m3;    // expert weights
+using A_element = __nv_bfloat16;    // activations as they go into the GEMM
+using AQ_element = __nv_fp8_e4m3;   // activations after quantization
+using S_element = float;            // scaling factors
+using R_element = __nv_bfloat16;    // MoE output
+
+/**
+ * @brief Returns the maximum amount of shared memory necessary to run moe_kernel()
+ */
+constexpr size_t get_moe_max_shmem_size();
+
+/**
+ * @brief Returns the maximum amount of global scratchpad memory to run moe_kernel()
+ */
+constexpr size_t get_moe_max_scratchpad_size();
+
+/**
+ * @brief W8A8 MoE kernel
+ *
+ * This function implements a W8A8 Mixture-of-Experts kernel.
+ * It routes each input token to the top 1 expert as determined by the @p routing_logits .
+ *
+ * Inputs:
+ * Activations are provided as bfloat16. They are quantized to FP8 E4M3 before the matrix multiplies.
+ * Expert weights are provided as FP8 E4M3.
+ *
+ * This function is templatized. For best performance, instantiate all template parameters with the respective parameters that you pass at runtime.
+ * For the batch-size, use at least two instantiations: One for BS=8 and one for BS=64.
+ * Instances of this kernel can handle runtime token counts <tt>token_count <= Dims::BS</tt>, i.e. you need to instantiate the kernel with at least
+ * the number of runtime tokens. This can be achieved e.g. with a switch-case statement.
+ *
+ * This is a cooperative kernel. It needs to be launched via cudaLaunchCooperativeKernel().
+ * This kernel needs at least get_moe_max_shmem_size() shared memory. Set it via
+ * <tt>
+ * cudaFuncSetAttribute(moe_kernel<KernelDims>, cudaFuncAttributeMaxDynamicSharedMemorySize, get_moe_max_shmem_size()));
+ * </tt>
+ * before calling!
+ *
+ * On top of the shared memory, the kernel stores temporaries in global device memory. This memory needs to be allocated
+ * by the user. It has to be at least get_moe_max_scratchpad_size() Bytes.
+ * 
+ * @note All input tensors are considered to be row-major and contiguous! I.e. no padding and the stride is the product of the trailing dimensions.
+ *
+ * In the parameter descriptions, we use the following shorthand constants:
+ * - The batch size M, @c Dims::BS
+ * - The number of experts E, @c Dims::NUM_EXPERTS
+ * - The number of hidden states K, @c Dims::HIDDEN_STATES
+ * - The up-projection dimension N, @c Dims::N
+ *
+ * @param [in] activations_in Input activations. Shape: [M, K]
+ * @param [in] token_count Number of active tokens
+ * @param [in] router_logits Result of routing matrix multiply. Determines which expert each token is routed to. Shape: [M, E]
+ * @param [in] expert_weights_up Scales for the weights of the down projection. Shape: [E, 2*N, K]
+ * @param [in] expert_scales_up Scales for the weights of the down projection. Shape: [E, 2*N]
+ * @param [in] expert_weights_down Scales for the weights of the down projection. Shape: [E, K, N]
+ * @param [in] expert_scales_down Scales for the weights of the down projection. Shape: [E, K]
+ * @param [out] activations_out Pointer to the output buffer. Shape: [M, K]
+ * @param [out] scratchpad Global memory to use for temporary data
+ * @param [in] scratchpad_size Size of the scratchpad. The kernel uses it to check if the caller allocated enough storage.
+ * @param [in] shmem_size Size of the shared memory. The kernel uses it to check if the caller allocated enough storage.
+ */
+template <typename Dims>
+__global__ extern void moe_kernel(
+    const A_element* __restrict__ activations_in,
+    std::uint32_t token_count,
+    const __nv_bfloat16 *__restrict__ router_logits,
+    const W_element* __restrict expert_weights_up,
+    const S_element* __restrict expert_scales_up,
+    const W_element* __restrict expert_weights_down,
+    const S_element* __restrict expert_scales_down,
+    R_element* __restrict activations_out,
+    void* __restrict__ scratchpad,
+    size_t scratchpad_size,
+    size_t shmem_size);
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/moe_internal.h b/csrc/moe/moe_monokernel/src/moe_internal.h
new file mode 100644
index 000000000..ae19eb2ea
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_internal.h
@@ -0,0 +1,286 @@
+
+#pragma once
+#ifndef MOE_INTERNAL_H
+#define MOE_INTERNAL_H
+
+#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#error Do not include this file directly.
+#endif
+
+#include "moe_interface.h"
+
+namespace moe_monokernel {
+
+using T_element = float; //< Type of GEMM1 (up projection) result as well as sigmoid
+using OpaqueElement = std::uint32_t; //< Auxiliary 32-bit type used to generate better assembly code in loads
+
+/**
+ * @brief Offets into the @c token_indexes field
+ *
+ * This is an offset array. To find all the tokens that belong to expert @c id :
+ * <tt>
+ * for (int i = first_token; i < last_token; i++) {
+ *    int token_index = token_indexes[i];
+ * }
+ */
+struct ExpertRef
+{
+    std::uint16_t first_token;
+    std::uint16_t last_token;
+    std::uint32_t id;
+};
+
+/**
+ * @brief Scratchpad memory for use within the monokernel.
+ *
+ * Place in global memory.
+ *
+ */
+template <typename Dims>
+struct MoEGemmSpec
+{
+#ifdef DEBUG_MOE
+    // Debug information passed out. The actual token_indexes are stored in shared memory.
+    std::int32_t token_indexes[Dims::BS];
+    T_element gemm1[(Dims::BS + 8) * 2 * Dims::N];
+#endif
+    AQ_element activations[Dims::BS][Dims::HIDDEN_STATES]; //< Quantized activations
+    T_element temp[(Dims::BS + 8) * Dims::N]; //< Up projection result
+    float topk_weights_scaled[Dims::BS]; //< topk_weights multiplied with the activation quantization
+};
+
+
+// Maximum supported dimensions for shared memory and scratchpad allocation sizes
+#if USE_SMALL_SETUP
+// SHM limits batch size to ~2k
+using Dims_Max = MoEDimensions<1024, 256, 1024, 128>;
+#else
+using Dims_Max = MoEDimensions<1024, 1024, 5120, 128>;
+#endif
+
+/**
+ * @brief contains various constants used within the MoE monokernel.
+ */
+template <typename Dims>
+struct MoECoreDims {
+    using MoEDims = Dims;
+
+    // GPU configuration.
+    static constexpr std::uint32_t THREADS_PER_WARP = 32;
+    static constexpr std::uint32_t TOTAL_WARP_COUNT = Dims::KernelConfig::BLOCK_SIZE / THREADS_PER_WARP;
+    static constexpr std::uint32_t CALC_WARP_COUNT = 8;
+    static constexpr std::uint32_t PREFETCH_WARP_COUNT = TOTAL_WARP_COUNT - CALC_WARP_COUNT;
+
+    // MMA 1 matrix tile dimensions.
+    static constexpr std::uint32_t A_TILE =     8;
+    static constexpr std::uint32_t W_UP_TILE = 16;
+    static constexpr std::uint32_t K_TILE =    32;
+
+    // GEMM 2 matrix tile dimensions.
+    static constexpr std::uint32_t W_DOWN_MMA_TILE = 16;
+    static constexpr std::uint32_t W_DOWN_TILE = Dims::HIDDEN_STATES / Dims::KernelConfig::GRID_SIZE;
+    static constexpr std::uint32_t T_TILE = 8;
+
+    static constexpr std::uint32_t W_DIM = 2 * Dims::N;
+
+    static constexpr unsigned BLOCK_STRIDE = CALC_WARP_COUNT * K_TILE;
+
+    static constexpr unsigned PADDING = 32;     // this works *slightly* better than 16 due to reduced L2 transfers
+    static constexpr unsigned K_DIM_PADDED_A = Dims::HIDDEN_STATES;
+    static constexpr unsigned K_DIM_PADDED_W = Dims::HIDDEN_STATES;
+    static constexpr unsigned K_DIM_HALF_PADDED_A = Dims::HIDDEN_STATES / 2;
+    static constexpr unsigned K_DIM_HALF_PADDED_W = Dims::HIDDEN_STATES / 2;
+};
+
+// 1 tile per warp
+// 20 warps x 2 params x 1k = 20k pre-fetch
+template <typename Dims>
+struct MoE_SHM {
+    using CoreDims = MoECoreDims<Dims>;
+    union U {
+        struct SortData {
+            std::uint32_t counters[Dims::NUM_EXPERTS][CoreDims::THREADS_PER_WARP];
+            std::uint32_t total_counts[Dims::NUM_EXPERTS];
+        } sorting;
+        struct RescaleData {
+            A_element a[CoreDims::CALC_WARP_COUNT][Dims::HIDDEN_STATES];
+        } rescale;
+        struct Gemm1Data {
+            // prefetch & process tile in 2 halves
+            AQ_element a[3][CoreDims::A_TILE][CoreDims::K_DIM_HALF_PADDED_A];
+            W_element w[3][CoreDims::W_UP_TILE][CoreDims::K_DIM_HALF_PADDED_W];
+            T_element partial_result[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];
+        } gemm1;
+        // GEMM2 uses the same data structure as Tiny
+        struct TinyData {
+            // input activations for up- and down-projection
+            union {
+                AQ_element up[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];
+                T_element down[CoreDims::T_TILE][Dims::N];
+            } a;
+
+            // prefetch & process tile in 2 halves
+            union {
+                A_element orig[CoreDims::T_TILE][CoreDims::K_DIM_PADDED_A];      // input activations to be scaled 
+                W_element up[CoreDims::W_UP_TILE][CoreDims::K_DIM_PADDED_W];     // up-projection weights
+                W_element down[CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)]; // down-projection weights
+            } w[2];
+
+            // down-projection scales
+            S_element scale[2][CoreDims::W_DOWN_TILE + CoreDims::PADDING];
+
+            // scratch pad
+            union {
+                T_element up[CoreDims::CALC_WARP_COUNT][CoreDims::W_UP_TILE * CoreDims::T_TILE];
+                T_element down[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2][CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
+            } partial_result;
+        } tiny;
+        struct Gemm2Data {
+            // prefetch 1 tile ahead
+            T_element t[2][CoreDims::T_TILE][Dims::N];
+            W_element w[2][CoreDims::W_DOWN_TILE][Dims::N + CoreDims::PADDING / sizeof(W_element)];
+            S_element scale[2][CoreDims::W_DOWN_TILE + CoreDims::PADDING];
+            T_element partial_result[CoreDims::W_DOWN_TILE / 2 + CoreDims::CALC_WARP_COUNT / 2][CoreDims::W_DOWN_MMA_TILE * CoreDims::T_TILE];
+        } gemm2;
+    } u;
+
+    static_assert(Dims::NUM_EXPERTS < 255, "Number of experts too high, cannot store as uint8 anymore.");
+    alignas(uint64_t) uint8_t topk_ids[Dims::BS < 8? 8: Dims::BS];
+    std::uint16_t token_indexes[Dims::BS + CoreDims::PADDING];
+    S_element topk_weights[Dims::BS];
+    ExpertRef experts[Dims::NUM_EXPERTS];
+
+    // 8 packed 8-bit expert id values.
+    // 0xff for "unused"
+    // only value if token count <= 8
+    std::uint64_t expert_mask;
+    std::uint64_t expert_ids;
+    std::uint32_t expert_count;
+};
+
+/**
+ * @brief Returns the amount of shared memory necessary to run @c moe_kernel with template parameter @p Dims
+ */
+template <typename Dims>
+__device__ __host__ constexpr size_t get_moe_shmem_size()
+{
+    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
+    return sizeof(MoE_SHM<Dims>);
+}
+
+constexpr size_t get_moe_max_shmem_size()
+{
+    return sizeof(MoE_SHM<Dims_Max>);
+}
+
+/**
+ * @brief Returns the amount of global scratchpad memory necessary to run moe_kernel() with template parameter @p Dims
+ */
+template <typename Dims>
+__device__ __host__ constexpr size_t get_moe_scratchpad_size()
+{
+    static_assert(Dims::M <= Dims_Max::M, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::N <= Dims_Max::N, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::K <= Dims_Max::K, "Dimension larger than the maximum supported dimension.");
+    static_assert(Dims::NUM_EXPERTS <= Dims_Max::NUM_EXPERTS, "Dimension larger than the maximum supported dimension.");
+    return sizeof(MoEGemmSpec<Dims>);
+}
+
+constexpr size_t get_moe_max_scratchpad_size()
+{
+    return sizeof(MoEGemmSpec<Dims_Max>);
+}
+
+template <typename Dims>
+inline __device__ bool is_calc_warp()
+{
+    using CoreDims = MoECoreDims<Dims>;
+    return threadIdx.x < CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
+}
+
+template <typename Dims>
+inline __device__ bool is_prefetch_warp()
+{
+    using CoreDims = MoECoreDims<Dims>;
+    return threadIdx.x >= CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP;
+}
+
+template <typename Dims>
+inline __device__ unsigned get_thread()
+{
+    using CoreDims = MoECoreDims<Dims>;
+    return threadIdx.x % CoreDims::THREADS_PER_WARP;
+}
+
+template <typename Dims>
+inline __device__ unsigned get_any_warp()
+{
+    using CoreDims = MoECoreDims<Dims>;
+    return threadIdx.x / CoreDims::THREADS_PER_WARP;
+}
+
+template <typename Dims>
+inline __device__ unsigned get_calc_warp()
+{
+    using CoreDims = MoECoreDims<Dims>;
+    assert(is_calc_warp<Dims>());
+    return threadIdx.x / CoreDims::THREADS_PER_WARP;
+}
+
+template <typename Dims>
+inline __device__ unsigned get_prefetch_warp()
+{
+    using CoreDims = MoECoreDims<Dims>;
+    assert(is_prefetch_warp<Dims>());
+    return threadIdx.x / CoreDims::THREADS_PER_WARP - CoreDims::CALC_WARP_COUNT;
+}
+
+/**
+ * @brief Synchronizes the first 256 threads of the calling CUDA block
+ *
+ * This is a collective operation that needs to be called by all of the first 256 threads in each CUDA block.
+ *
+ */
+template <typename Dims>
+__device__ __forceinline__ void sync_calc_threads()
+{
+    // First 256 threads
+    using CoreDims = MoECoreDims<Dims>;
+    static_assert(CoreDims::CALC_WARP_COUNT * CoreDims::THREADS_PER_WARP == 256, "Adapt the thread number if sync_calc_threads");
+    __asm volatile("bar.sync  15, 256;\n");
+}
+
+/**
+ * @brief Computes the maximum value within a warp
+ *
+ * This is a collective operation. Each thread in a warp needs to call it.
+ * The resulting maximum value is returned on all threads.
+ *
+ */
+__device__ static inline float warp_reduce_max_float(float value)
+{
+    for (int i = 16; i >= 1; i /= 2) {
+        value = fmaxf(__shfl_xor_sync(0xffffffff, value, i, 32), value);
+    }
+    return value;
+}
+
+/**
+ * @brief Reinterprets the bit-pattern of @p x to type @p To
+ */
+template <typename To, typename From>
+__device__ static __forceinline__ To type_pun(From x)
+{
+    static_assert(sizeof(To) == sizeof(From), "Types of different size");
+    To y;
+    // This memcpy is optimized out by NVCC
+    memcpy(&y, &x, sizeof(From));
+    return y;
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/moe_prepare.cu b/csrc/moe/moe_monokernel/src/moe_prepare.cu
new file mode 100644
index 000000000..cb746c614
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_prepare.cu
@@ -0,0 +1,501 @@
+
+#pragma once
+#ifndef MOE_PREPARE_CU
+#define MOE_PREPARE_CU
+
+#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#error Do not include this file directly.
+#endif
+
+#include <cstdint>
+
+#include "moe_internal.h"
+
+#ifndef __SIZEOF_INT128__
+static_assert(false, "This module currently needs int128. You're host compiler does not support it.")
+#endif
+
+#define FULL_MASK 0xFFFFFFFFU
+
+namespace moe_monokernel {
+
+// We use an uint128 to store 16 uint8
+typedef __uint128_t uint8x16_t;
+
+/**
+ * @brief 16-byte allreduce summation within a warp
+ *
+ * Sums a 16-byte integer across all threads and returns the sum.
+ * This operation is collective and needs to be called by all threads within a warp.
+ *
+ */
+__device__ static inline uint8x16_t allreduce_sum_across_warp(uint8x16_t val)
+{
+    uint64_t val_lo = val & 0xFFFFFFFFFFFFFFFFU;
+    uint64_t val_hi = val >> 64;
+    for (int offset = 16; offset > 0; offset /= 2) {
+        val_lo += __shfl_xor_sync(FULL_MASK, val_lo, offset, 32);
+        val_hi += __shfl_xor_sync(FULL_MASK, val_hi, offset, 32);
+    }
+    return val_lo | ((uint8x16_t) val_hi << 64);
+}
+
+/** 
+ * @brief Prefix sum of all uint8s in an uint8x16_t
+ *
+ * Computes a prefix sum over 16 uint8. First element is the least significant byte.
+ */
+__device__ static inline uint8x16_t prefix_sum_over_bytes(uint8x16_t val)
+{
+    val += val << 8;
+    val += val << 16;
+    val += val << 32;
+    val += val << 64;
+    return val;
+}
+
+/**
+ * @brief Prepares the MoE computation for batch size 8 and 16 experts for the matrix multiply functions.
+ *
+ * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
+ * Specifically, it fills:
+ * - @c shm->expert_mask
+ * - @c shm->expert_ids
+ * - @c shm->expert_count
+ *
+ * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
+ *
+ * @param batch_size The batch size (active elements in the topk_ids array)
+ * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
+ *
+ */
+template <typename Dims>
+__device__ static void prepare_moe_BS8_E16(
+    std::uint32_t  batch_size,
+    MoE_SHM<Dims>* __restrict__ shm)
+{
+    static_assert(Dims::NUM_EXPERTS <= 16, "This function is only for up to 16 experts");
+    static_assert(Dims::BS <= 8, "This function is only for up to batch size 8");
+    assert(batch_size <= 8);
+
+    std::uint64_t packed_unique_topkids = 0;
+    
+    // Initialize the SHM in every SM.
+    // One thread in each warp does this so that we do not need a block-wide synchronization before accessing the first expert
+    if (threadIdx.x % 32 == 0) {
+        uint32_t expert_count = 0;
+        
+        // Packing is a no-op since topk_ids are stored as uint8 already
+        uint64_t packed_topkids = *(uint64_t*) shm->topk_ids;
+
+        uint32_t t0 = shm->topk_ids[0];
+        uint32_t t1 = shm->topk_ids[1];
+        uint32_t t2 = shm->topk_ids[2];
+        uint32_t t3 = shm->topk_ids[3];
+        uint32_t t4 = shm->topk_ids[4];
+        uint32_t t5 = shm->topk_ids[5];
+        uint32_t t6 = shm->topk_ids[6];
+        uint32_t t7 = shm->topk_ids[7];
+
+        // To determine the unique experts, we use a bitfield of expert ids.
+        // Bit i is 1 <==> expert i appears in topk_ids
+        uint32_t expert_bitset = 0;
+        expert_bitset |= 1 << t0;
+        expert_bitset |= 1 << t1;
+        expert_bitset |= 1 << t2;
+        expert_bitset |= 1 << t3;
+        expert_bitset |= 1 << t4;
+        expert_bitset |= 1 << t5;
+        expert_bitset |= 1 << t6;
+        expert_bitset |= 1 << t7;
+
+        expert_count = __popc(expert_bitset);
+
+        // Extract the up to 8 unique experts from the bitfield, by finding the up to 8 bits that are set to 1.
+        //
+        // Only the first "expert_count" of these are valid, the other eX values are 0xFFFFFFFF
+        // __ffs(0) - 1 == 0xFFFFFFFF
+        uint32_t e0 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e1 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e2 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e3 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e4 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e5 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e6 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        expert_bitset &= expert_bitset - 1;
+        uint32_t e7 = __ffs(expert_bitset) - 1; /* ffs starts counting at 1 */
+        assert((expert_bitset & (expert_bitset - 1)) == 0);
+
+        // Pack the expert ids into bytes
+        e1 <<= 8*1;
+        e2 <<= 8*2;
+        e3 <<= 8*3;
+        e5 <<= 8*1;
+        e6 <<= 8*2;
+        e7 <<= 8*3;
+        uint32_t lo_packed_unique_topkids = (e0 | e1) | (e2 | e3);
+        uint32_t hi_packed_unique_topkids = (e4 | e5) | (e6 | e7);
+        packed_unique_topkids = lo_packed_unique_topkids | ((uint64_t) hi_packed_unique_topkids << 32);
+        
+        if (threadIdx.x == 0) {
+            shm->expert_mask = packed_topkids;
+            shm->expert_ids = packed_unique_topkids;
+            shm->expert_count = expert_count;
+        }
+    }
+}
+
+/**
+ * @brief Prepares the MoE computation for batch size 8 and 128 experts for the matrix multiply functions.
+ *
+ * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
+ * Specifically, it fills:
+ * - @c shm->expert_mask
+ * - @c shm->expert_ids
+ * - @c shm->expert_count
+ *
+ * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
+ *
+ * This function *does not* support a higher number of experts than 128.
+ *
+ * @param batch_size The batch size (active elements in the topk_ids array)
+ * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
+ *
+ */
+template <typename Dims>
+__device__ static void prepare_moe_BS8_E128(
+    std::uint32_t  batch_size,
+    MoE_SHM<Dims>* __restrict__ shm)
+{
+    static_assert(Dims::NUM_EXPERTS <= 128, "This function is only for up to 128 experts");
+    static_assert(Dims::BS <= 8, "This function is only for up to batch size 8");
+    assert(batch_size <= 8);
+
+    std::uint64_t packed_unique_topkids = 0;
+    
+    // Initialize the SHM in every SM.
+    // One thread in each warp does this so that we do not need a block-wide synchronization before accessing the first expert
+    // For explanation of the method, see prepare_moe_BS8_E16
+    if (threadIdx.x % 32 == 0) {
+        uint64_t packed_topkids = *(uint64_t*) shm->topk_ids;
+
+        uint32_t t0 = shm->topk_ids[0];
+        uint32_t t1 = shm->topk_ids[1];
+        uint32_t t2 = shm->topk_ids[2];
+        uint32_t t3 = shm->topk_ids[3];
+        uint32_t t4 = shm->topk_ids[4];
+        uint32_t t5 = shm->topk_ids[5];
+        uint32_t t6 = shm->topk_ids[6];
+        uint32_t t7 = shm->topk_ids[7];
+
+        __uint128_t expert_bitset = 0;
+        expert_bitset |= __uint128_t(1) << t0;
+        expert_bitset |= __uint128_t(1) << t1;
+        expert_bitset |= __uint128_t(1) << t2;
+        expert_bitset |= __uint128_t(1) << t3;
+        expert_bitset |= __uint128_t(1) << t4;
+        expert_bitset |= __uint128_t(1) << t5;
+        expert_bitset |= __uint128_t(1) << t6;
+        expert_bitset |= __uint128_t(1) << t7;
+
+        uint64_t b0, b1;
+        b0 = expert_bitset & 0xFFFFFFFFFFFFFFFFU;
+        b1 = expert_bitset >> 64;
+        uint32_t expert_count = __popcll(b0) + __popcll(b1);
+
+        uint32_t addend = 0;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e0 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e1 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e2 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e3 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e4 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e5 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e6 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        b0 &= b0 - 1;
+        if (b0 == 0) { b0 = b1; addend = 64; }
+        uint32_t e7 = __ffsll(b0) - 1 + addend; /* ffs starts counting at 1 */
+        // Note: (b0 & (b0 - 1)) == 0 does not hold here.
+        // If expert_count == 7, b1 is copied into b0 again before the last __ffsll
+
+        e1 <<= 8*1;
+        e2 <<= 8*2;
+        e3 <<= 8*3;
+        e5 <<= 8*1;
+        e6 <<= 8*2;
+        e7 <<= 8*3;
+        uint32_t lo_packed_unique_topkids = (e0 | e1) | (e2 | e3);
+        uint32_t hi_packed_unique_topkids = (e4 | e5) | (e6 | e7);
+        packed_unique_topkids = lo_packed_unique_topkids | ((uint64_t) hi_packed_unique_topkids << 32);
+        
+        if (threadIdx.x == 0) {
+            shm->expert_mask = packed_topkids;
+            shm->expert_ids = packed_unique_topkids;
+            shm->expert_count = expert_count;
+        }
+    }
+}
+
+/**
+ * @brief Prepares the MoE computation for batch size 64 and 16 experts for the matrix multiply functions.
+ *
+ * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
+ * Specifically, it fills:
+ * - @c shm->expert_count
+ * - @c shm->experts
+ * - @c shm->token_indexes
+ *
+ * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
+ *
+ * @param batch_size The batch size (active elements in the topk_ids array)
+ * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
+ *
+ */
+template <typename Dims>
+__device__ static void prepare_moe_BS64_E16(
+    std::uint32_t batch_size,
+    MoE_SHM<Dims>* __restrict__ shm)
+{
+    static_assert(Dims::NUM_EXPERTS <= 16, "This function is only for up to 16 experts");
+    static_assert(Dims::BS <= 64, "This function is only for up to batch size 64");
+    assert(batch_size <= 64);
+
+    constexpr int warp_size = 32;
+    const int thread_idx = get_thread<Dims>();
+    const int warp_idx = get_any_warp<Dims>();
+
+    // One byte per 16 experts suffices as long as Dims::BS<256
+    static_assert(Dims::BS < 256);
+    uint8x16_t token_count_local = 0;
+
+    // Every thread does batch_size / warp_size tokens
+    for (std::uint32_t i = thread_idx; i < batch_size; i += warp_size) {
+        std::uint32_t expert_id = shm->topk_ids[i];
+        token_count_local += (uint8x16_t(1) << (expert_id*8));
+    }
+
+    const uint8x16_t token_counts = allreduce_sum_across_warp(token_count_local);
+    const uint8x16_t token_prefixes = prefix_sum_over_bytes(token_counts);
+
+    // Every thread is now responsible for one expert.
+    static_assert(Dims::NUM_EXPERTS <= warp_size); // Need at least one thread per expert in each warp
+    // Because Dims::NUM_EXPERTS==16, upper half of the threads has 0
+    uint32_t token_count = (token_counts >> (8*thread_idx)) & 0xFF;
+    uint32_t token_prefix = (token_prefixes >> (8*thread_idx)) & 0xFF;
+
+    uint32_t nonzero_mask = __ballot_sync(FULL_MASK, token_count != 0);
+    uint32_t num_nonzero_elements = __popc(nonzero_mask);
+    // "inverse_permut_last" is the source lane for compacting "token_count".
+    uint32_t inverse_permut_last = __fns(nonzero_mask, 0, thread_idx + 1);
+    uint32_t inverse_permut_first = __fns(nonzero_mask, 0, thread_idx); // Index of token start is the previous thread's end index.
+    uint32_t token_start = __shfl_sync(FULL_MASK, token_prefix, inverse_permut_first);
+    uint32_t token_end = __shfl_sync(FULL_MASK, token_prefix, inverse_permut_last);
+
+    // Writing is done on the first warp
+    if (warp_idx == 0) {
+        if (thread_idx == 0) {
+            token_start = 0; // Correct the token start index of expert 0
+            shm->expert_count = num_nonzero_elements;
+        }
+        if (thread_idx < num_nonzero_elements) {
+            shm->experts[thread_idx].id = inverse_permut_last;
+            shm->experts[thread_idx].first_token = token_start;
+            shm->experts[thread_idx].last_token = token_end;
+        }
+    }
+    //
+    // MoE_SHM::token_indexes is the permutation that sorts topk_ids.
+    //
+    if constexpr (Dims::BS <= warp_size) { // Can be handled completely within a single warp
+        uint32_t eid;
+        if (thread_idx < batch_size)
+            eid = shm->topk_ids[thread_idx];
+        else
+            eid = Dims::NUM_EXPERTS;
+        // Prefix, where all tokens with expert eid need to go
+        uint32_t sorting_prefix = (token_prefixes >> (8*(eid-1))) & 0xFF;
+        // Branchless: if (eid == 0) sorting_prefix = 0;
+        sorting_prefix = sorting_prefix & ((eid == 0) - 1);
+
+        // For threads with the same sorting_prefix:
+        // We define here that their threadIdx determines the relative token order
+        uint32_t mask_of_threads_with_same_prefix = __match_any_sync(FULL_MASK, sorting_prefix);
+        // Relative order == number of one bits in mask before its position
+        uint32_t lower_idx_mask = (1 << thread_idx) - 1;
+        uint32_t rank = __popc(mask_of_threads_with_same_prefix & lower_idx_mask);
+        uint32_t target_lane_id = sorting_prefix + rank;
+
+        if (warp_idx == 0 && thread_idx < batch_size) {
+            shm->token_indexes[target_lane_id] = thread_idx;
+        }
+    } else if constexpr (Dims::BS <= 2*warp_size) { // Can be handled by two warps
+        if (warp_idx < 2) {
+            uint32_t eid;
+            if (thread_idx + warp_size * warp_idx < batch_size)
+                eid = shm->topk_ids[thread_idx + warp_size * warp_idx];
+            else
+                eid = Dims::NUM_EXPERTS;
+            // Prefix, where all tokens with expert eid need to go
+            uint32_t sorting_prefix = (token_prefixes >> (8*(eid-1))) & 0xFF;
+            // Branchless: if (eid == 0) sorting_prefix = 0;
+            sorting_prefix = sorting_prefix & ((eid == 0) - 1);
+
+            // For threads with the same sorting_prefix:
+            // We define here that their threadIdx determines the relative token order
+            // Tokens that are processed by threads from the first warp are sorted upwards, otherwise downwards.
+            uint32_t mask_of_threads_with_same_prefix = __match_any_sync(FULL_MASK, sorting_prefix);
+            // Relative order == number of one bits in mask before its position
+            uint32_t lower_idx_mask = (1 << thread_idx) - 1;
+            uint32_t rank = __popc(mask_of_threads_with_same_prefix & lower_idx_mask);
+
+            uint32_t target_lane_id;
+            if (warp_idx == 0) {
+                // Warp 0 goes from bottom to top
+                target_lane_id = sorting_prefix + rank;
+            } else {
+                // warp 1 goes from top to bottom
+                uint32_t next_sorting_prefix = (token_prefixes >> (8*eid)) & 0xFF;
+                target_lane_id = next_sorting_prefix - rank - 1;
+            }
+            if (thread_idx + warp_size * warp_idx < batch_size) {
+                shm->token_indexes[target_lane_id] = thread_idx + warp_idx * warp_size;
+            }
+        }
+    }
+}
+
+/**
+ * @brief Prepares the MoE computation for batch size 64 and 128 experts for the matrix multiply functions.
+ *
+ * Reads @c shm->topk_ids and sets up the necessary data structures and state for @c moe_up_projection and @c moe_down_projection .
+ * Specifically, it fills:
+ * - @c shm->expert_count
+ * - @c shm->experts
+ * - @c shm->token_indexes
+ *
+ * The computation is done redundantly on each CUDA block such that the result can be stored in shared memory.
+ *
+ * This function should support any number of experts, but is only tested for up to 128.
+ *
+ * @param batch_size The batch size (active elements in the topk_ids array)
+ * @param shm Pointer to shared memory struct to read inputs from and write outputs to.
+ *
+ */
+template <typename Dims>
+__device__ static void prepare_moe_BSx_Ey(
+    std::uint32_t batch_size,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    using CoreDims = MoECoreDims<Dims>;
+    using MoE_SHM = MoE_SHM<Dims>;
+
+    typename MoE_SHM::U::SortData* shm = &shmem->u.sorting;
+
+    auto& counters = shm->counters;
+    auto& total_counts = shm->total_counts;
+
+    // Implements a Radix sort on the first warp of each CUDA block.
+    if (threadIdx.x < CoreDims::THREADS_PER_WARP) {
+        // initialize
+        for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
+            counters[e][threadIdx.x] = 0;
+        }
+
+        // count inputs
+        for (unsigned i = threadIdx.x; i < batch_size; i += CoreDims::THREADS_PER_WARP) {
+            counters[shmem->topk_ids[i]][threadIdx.x]++;
+        }
+
+        __syncwarp();
+
+        // sum up. counters become offsets
+        for (unsigned e = threadIdx.x; e < Dims::NUM_EXPERTS; e += CoreDims::THREADS_PER_WARP) {
+            std::uint32_t sum = 0;
+            for (unsigned i = 0; i < CoreDims::THREADS_PER_WARP; ++i) {
+                std::uint32_t prior = sum;
+                sum += counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP];
+                counters[e][(i + threadIdx.x) % CoreDims::THREADS_PER_WARP] = prior;
+            }
+            total_counts[e] = sum;
+        }
+
+        __syncwarp();
+
+        // global offsets and expert ranges
+        if (threadIdx.x == 0) {
+            std::uint32_t sum = 0;
+            std::uint32_t expert_count = 0;
+
+            for (unsigned e = 0; e < Dims::NUM_EXPERTS; ++e) {
+                std::uint32_t local_count = total_counts[e];
+
+                if (local_count > 0) {
+                    std::uint32_t prior = sum;
+                    total_counts[e] = prior;
+                    sum += local_count;
+
+                    shmem->experts[expert_count].first_token = prior;
+                    shmem->experts[expert_count].last_token = sum;
+                    shmem->experts[expert_count].id = e;
+                    expert_count++;
+                }
+            }
+
+            shmem->expert_count = expert_count;
+        }
+
+        __syncwarp();
+
+        // write index order
+        std::uint16_t* ordered = shmem->token_indexes;
+        for (unsigned i = threadIdx.x; i < batch_size; i += CoreDims::THREADS_PER_WARP) {
+            std::int32_t e = shmem->topk_ids[i];
+            unsigned offset = counters[e][threadIdx.x];
+            unsigned index = total_counts[e] + offset;
+            counters[e][threadIdx.x] = offset + 1;
+
+            ordered[index] = (std::uint16_t)i;
+        }
+    }
+}
+
+/**
+ * @brief Dispatch function for @c prepare_moe_BS8_E16 and @c prepare_moe_BS8_E128 .
+ */
+template <typename Dims>
+__device__ __forceinline__ void prepare_moe_BS8(
+    std::uint32_t  batch_size,
+    MoE_SHM<Dims>* __restrict__ shm)
+{
+    static_assert(Dims::NUM_EXPERTS <= 128, "This function is only for up to 128 experts");
+    static_assert(Dims::BS <= 8, "This function is only for up to batch size 8");
+    assert(batch_size <= 8);
+
+    if constexpr (Dims::NUM_EXPERTS <= 16) {
+        prepare_moe_BS8_E16(batch_size, shm);
+    } else {
+        prepare_moe_BS8_E128(batch_size, shm);
+    }
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/moe_routing.cu b/csrc/moe/moe_monokernel/src/moe_routing.cu
new file mode 100644
index 000000000..aee9f48f1
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_routing.cu
@@ -0,0 +1,267 @@
+
+#pragma once
+#ifndef MOE_GATING_CU
+#define MOE_GATING_CU
+
+#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#error Do not include this file directly.
+#endif
+
+#include <cstdint>
+#include <cfloat>
+#include <cuda_bf16.h>
+
+#include "moe_internal.h"
+
+namespace moe_monokernel {
+
+/**
+ * @brief Determines the thread with the lowest index that has a specific value.
+ *
+ * This device function checks, within a warp, which threads have their @p haystack value
+ * equal to the specified @p needle value. If there are multiple such threads, returns
+ * the one with the lowest thread index.
+ *
+ * @param haystack The value to compare against the needle for the calling thread.
+ * @param needle The value to search for among the threads in the warp.
+ * @return uint32_t Thread index
+ */
+__device__ static inline uint32_t warp_who_has(float haystack, float needle)
+{
+    uint32_t mask = __ballot_sync(0xFFFFFFFFU, haystack == needle);
+    assert(mask > 0);
+    return __ffs(mask) - 1; // Nvidia starts counting bits at 1
+}
+
+/**
+ * @brief Computes the sigmoid activation function for a given input.
+ *
+ * @param x The input value.
+ * @return The sigmoid of the input value.
+ */
+__device__ static inline float sigmoid(float x)
+{
+    return 1.0f / (1.0f + std::exp(-x));
+}
+
+/**
+ * @brief Selects the top-1 expert for each of up to 64 tokens based on router logits.
+ *
+ * This device function processes a batch of up to 64 tokens, each with up to 16 experts.
+ * It takes as input a pointer to the router logits (in bfloat16 format) and determines,
+ * for each token, the expert with the highest routing score.
+ *
+ * The selection is done on each CUDA block redundantly such that the result can be placed in shared memory.
+ *
+ * @param router_logits Pointer to the input router logits array of shape [num_tokens, experts] in row-major order.
+ *                      Individual elements are in __nv_bfloat16 format.
+ * @param num_tokens Number of tokens
+ * @param shmem Shared Memory struct to store the result to.
+ */
+template <typename Dims>
+__device__ static void top1_BS64_E16(const __nv_bfloat16 *__restrict__ router_logits,
+                uint32_t num_tokens,
+                MoE_SHM<Dims>* shmem)
+{
+    static_assert(Dims::NUM_EXPERTS <= 16, "Dispatch to incorrect imlementation");
+    static_assert(Dims::BS <= 64, "Dispatch to incorrect imlementation");
+    static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX, "Batch size or number of experts too high for uint32 indices.");
+
+    uint32_t thread_idx = threadIdx.x;
+
+    // On all SMs in parallel: Every thread does one token
+    for (uint32_t tokidx = thread_idx; tokidx < num_tokens; tokidx += 256) {
+        float max_value = -FLT_MAX;
+        uint32_t max_index = 0;
+        for (uint32_t idx = 0; idx < Dims::NUM_EXPERTS; idx++) {
+            uint32_t index = tokidx * Dims::NUM_EXPERTS + idx; // Make NVCC produce simpler array indexing code
+            float value = (float) router_logits[index];
+            // Branchless version of:
+            // if (max_value < value) {
+            //     max_index = idx;
+            //     max_value = value;   
+            // }
+            max_value = fmaxf(max_value, value);
+            int is_new = max_value == value;
+            max_index = max_index * (1 - is_new) + idx * is_new;
+        }
+
+        shmem->topk_ids[tokidx] = (uint8_t) max_index;
+        shmem->topk_weights[tokidx] = sigmoid(max_value);
+    }
+}
+
+/**
+ * @brief Selects the top-1 expert for each of up to 64 tokens based on router logits.
+ *
+ * This device function processes a batch of up to 64 tokens, each with up to 128 experts.
+ * It takes as input a pointer to the router logits (in bfloat16 format) and determines,
+ * for each token, the expert with the highest routing score.
+ *
+ * It should work for any number of experts, but is tested for performance for up to 128 experts only.
+ *
+ * The selection is done on each CUDA block redundantly such that the result can be placed in shared memory.
+ *
+ * @param router_logits Pointer to the input router logits array of shape [num_tokens, experts] in row-major order.
+ *                      Individual elements are in __nv_bfloat16 format.
+ * @param num_tokens Number of tokens
+ * @param shmem Shared Memory struct to store the result to.
+ */
+template <typename Dims>
+__device__ static void top1_BS64_E128(const __nv_bfloat16 *__restrict__ router_logits,
+                uint32_t num_tokens,
+                MoE_SHM<Dims>* shmem)
+{
+    static_assert(Dims::BS <= 64, "Dispatch to incorrect imlementation");
+    // This function does one token with multiple threads.
+    // The following constants define the per-token parallelism:
+    constexpr uint32_t NUM_EXPERTS_PER_THREAD = 16; // Best performance for BS=64
+    constexpr uint32_t NUM_THREADS_PER_TOKEN = Dims::NUM_EXPERTS / NUM_EXPERTS_PER_THREAD;
+    constexpr uint32_t NUM_TOKENS_PER_WARP = 32 / NUM_THREADS_PER_TOKEN;
+
+    static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX, "Batch size or number of experts too high for uint32 indices.");
+    static_assert(Dims::NUM_EXPERTS % NUM_THREADS_PER_TOKEN == 0, "Number of experts must be divisible.");
+
+    uint32_t thread_idx = threadIdx.x;
+
+    // Ensure participation of whole warps in the loop
+    int padded_num_tokens = (num_tokens + NUM_TOKENS_PER_WARP - 1) / NUM_TOKENS_PER_WARP * NUM_TOKENS_PER_WARP;
+
+    // On all SMs in parallel: NUM_THREADS_PER_TOKEN threads together do one token
+    for (uint32_t tokidx = thread_idx / NUM_THREADS_PER_TOKEN; tokidx < padded_num_tokens; tokidx += 256 / NUM_THREADS_PER_TOKEN) {
+        // Check really full warps participate
+        assert(__activemask() == 0xFFFFFFFFU);
+        
+        float max_value = -FLT_MAX;
+        uint32_t max_index = 0;
+
+        if (tokidx < num_tokens) {
+            const uint32_t part_idx = thread_idx % NUM_THREADS_PER_TOKEN;
+            for (uint32_t idx = part_idx * NUM_EXPERTS_PER_THREAD; idx < (part_idx + 1) * NUM_EXPERTS_PER_THREAD; idx++) {
+                uint32_t index = tokidx * Dims::NUM_EXPERTS + idx; // Make NVCC produce simpler array indexing code
+                float value = (float) router_logits[index];
+                // Branchless version of:
+                // if (max_value < value) {
+                //     max_index = idx;
+                //     max_value = value;   
+                // }
+                max_value = fmaxf(max_value, value);
+                int is_new = max_value == value;
+                max_index = max_index * (1 - is_new) + idx * is_new;
+            }
+        }
+
+        // In each warp: synchronize between those threads that do the same token
+        float quad_max_value = max_value;
+        for (int i = 1; i < NUM_THREADS_PER_TOKEN; i *= 2)
+            quad_max_value = fmaxf(quad_max_value, __shfl_xor_sync(0xFFFFFFFFU, quad_max_value, i, 32));
+
+        // ATTENTION: Here, potentially multiple threads can write to the same shmem values.
+        // But the writes themselves are atomic and if multiple write, they have the same max_value.
+        // In this case, one of the maximum topk_values is chosen indeterministically.
+        if (quad_max_value == max_value) {
+            shmem->topk_ids[tokidx] = (uint8_t) max_index;
+            shmem->topk_weights[tokidx] = 1.0f / (1.0f + std::exp(-max_value));
+        }
+    }
+}
+
+/**
+ * @brief Selects the top-1 expert for each of up to 64 tokens based on router logits.
+ *
+ * It takes as input a pointer to the router logits (in bfloat16 format) and determines,
+ * for each token, the expert with the highest routing score.
+ *
+ * The selection is done on each CUDA block redundantly such that the result can be placed in shared memory.
+ *
+ * @param router_logits Pointer to the input router logits array of shape [num_tokens, experts] in row-major order.
+ *                      Individual elements are in __nv_bfloat16 format.
+ * @param num_tokens Number of tokens
+ * @param shmem Shared Memory struct to store the result to.
+ */
+template <typename Dims>
+__device__ __forceinline__ void top1_BS64(const __nv_bfloat16 *__restrict__ router_logits,
+                uint32_t num_tokens,
+                MoE_SHM<Dims>* shmem)
+{
+    static_assert(Dims::BS <= 64, "Dispatch to incorrect imlementation");
+    if constexpr (Dims::NUM_EXPERTS <= 16) {
+        top1_BS64_E16(router_logits, num_tokens, shmem);
+    } else {
+        top1_BS64_E128(router_logits, num_tokens, shmem);
+    }
+}
+
+/**
+ * @brief Selects the top-1 expert for each of up to 8 tokens based on router logits.
+ *
+ * This device function processes a batch of up to 8 tokens, each with up to 128 experts.
+ * It takes as input a pointer to the router logits (in bfloat16 format) and determines,
+ * for each token, the expert with the highest routing score.
+ *
+ * It should work for any number of experts, but is tested for performance for up to 128 experts only.
+ *
+ * The selection is done on each CUDA block redundantly such that the result can be placed in shared memory.
+ * In case, num_tokens is smaller than 8, pads topk_ids to 8 by setting the remaining elements to 0xFF.
+ *
+ * @param router_logits Pointer to the input router logits array of shape [num_tokens, experts] in row-major order.
+ *                      Individual elements are in __nv_bfloat16 format.
+ * @param num_tokens Number of tokens
+ * @param shmem Shared Memory struct to store the result to.
+ */
+template <typename Dims>
+__device__ void top1_BS8(const __nv_bfloat16 *__restrict__ router_logits,
+                uint32_t num_tokens,
+                MoE_SHM<Dims>* shmem)
+{
+    static_assert(Dims::BS <= 8, "Dispatch to incorrect imlementation");
+    static_assert(Dims::BS * Dims::NUM_EXPERTS < UINT32_MAX, "Batch size or number of experts too high for uint32 indices.");
+
+    uint32_t warp_idx = get_calc_warp<Dims>();
+    uint32_t thread_idx_within_warp = get_thread<Dims>();
+
+    // On all SMs in parallel: Every warp does one token
+    if (warp_idx < num_tokens) {
+        float max_value = -FLT_MAX;
+        uint32_t max_index = 0;
+        for (uint32_t idx = thread_idx_within_warp; idx < Dims::NUM_EXPERTS; idx += 32) {
+            uint32_t index = warp_idx * Dims::NUM_EXPERTS + idx; // Make NVCC produce simpler array indexing code
+            float value = (float) router_logits[index];
+            // if (std::isfinite(value)) {
+                if constexpr (Dims::NUM_EXPERTS > 32) {
+                    if (max_value < value) {
+                        max_index = idx;
+                        max_value = value;   
+                    }
+                } else {
+                    max_index = idx;
+                    max_value = value;
+                }
+            // }
+        }
+
+        // Warp reduction
+        float warpmax_value = warp_reduce_max_float(max_value);
+        assert(max_value <= warpmax_value);
+        // We need this in case several threads have the exact same float.
+        // In this case, we cannot simply use "warpmax_value == max_value" as condition for the writing below.
+        uint32_t max_thread = warp_who_has(max_value, warpmax_value);
+
+        // Thread that has max value writes
+        if (thread_idx_within_warp == max_thread) {
+            assert(warpmax_value == max_value);
+            assert(max_index < Dims::NUM_EXPERTS);
+            shmem->topk_ids[warp_idx] = (uint8_t) max_index;
+            shmem->topk_weights[warp_idx] = 1.0f / (1.0f + std::exp(-max_value));
+        }
+    } else if (warp_idx < 8) {
+        // Padding elements
+        if (thread_idx_within_warp == 0) {
+            shmem->topk_ids[warp_idx] = (uint8_t) 0xFF;
+        }
+    }
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/moe_scale_inputs.cu b/csrc/moe/moe_monokernel/src/moe_scale_inputs.cu
new file mode 100644
index 000000000..717b9668a
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_scale_inputs.cu
@@ -0,0 +1,370 @@
+
+#pragma once
+#ifndef MOE_SCALE_INPUTS_CU
+#define MOE_SCALE_INPUTS_CU
+
+#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#error Do not include this file directly.
+#endif
+
+#include <cstdint>
+#include <cstring>
+
+#include <cuda/pipeline>
+#include <cuda_bf16.h>
+#include <cooperative_groups.h>
+
+#include "moe_interface.h"
+#include "moe_internal.h"
+#include "ptx_utils.h"
+
+namespace moe_monokernel {
+
+/** 
+ * @brief Sets NaNs (positive or negative, any payload) to 0.0 (bit-pattern all 0).
+ *
+ * Values other than NaN remain unchanged.
+ */
+__device__ static __forceinline__ __nv_bfloat162 mask_NaNs_to_zero(__nv_bfloat162 xs)
+{
+    return type_pun<__nv_bfloat162>(type_pun<uint32_t>(xs) & __heq2_mask(xs, xs));
+}
+
+// Internal linkage
+namespace {
+
+/**
+ * @brief Struct for keeping a chunk of 8 bfloat16 in registers.
+ */
+struct BF16x8
+{
+    float4 raw; // Storage for 8 BFloat16 values. Never accessed as fp32.
+
+    /**
+     * @brief Loads a BF16x8 from memory.
+     *
+     * The address @p a must be BF16x8 aligned.
+     */
+    __device__ static BF16x8 load(const A_element* a) {
+        assert(reinterpret_cast<uintptr_t>(a) % 16 == 0);
+        BF16x8 val { *reinterpret_cast<const float4*>(a) };
+        return val;
+    }
+
+    /**
+     * @brief Stores a BF16x8 to memory.
+     *
+     * The address @p a must be BF16x8 aligned.
+     */
+    __device__ void store_to(A_element* a) {
+        assert(reinterpret_cast<uintptr_t>(a) % 16 == 0);
+        *reinterpret_cast<float4*>(a) = raw;
+    }
+
+    /**
+     * @brief Returns bfloat16 0 and 1 as pair.
+     */
+    __device__ __nv_bfloat162 first_pair() const {
+        return type_pun<__nv_bfloat162>(raw.x);
+    }
+    /**
+     * @brief Returns bfloat16 2 and 3 as pair.
+     */
+    __device__ __nv_bfloat162 second_pair() const {
+        return type_pun<__nv_bfloat162>(raw.y);
+    }
+    /**
+     * @brief Returns bfloat16 5 and 5 as pair.
+     */
+    __device__ __nv_bfloat162 third_pair() const {
+        return type_pun<__nv_bfloat162>(raw.z);
+    }
+    /**
+     * @brief Returns bfloat16 6 and 7 as pair.
+     */
+    __device__ __nv_bfloat162 fourth_pair() const {
+        return type_pun<__nv_bfloat162>(raw.w);
+    }
+
+    /**
+     * @brief Converts 8 Bfloat16 values to FP8 E4M3 in accordance with vLLM's MoE activation quantization.
+     *
+     * Scales, clamps (incl. NaN replacement), rounds and converts each BFloat16 to FP8 E4M3.
+     * Scaling is done with float accuracy.
+     * Clamping and saturation are implemented via __NVSATFINITE semantics of the FP8 conversion.
+     *
+     * @param scale Scaling factor to use.
+     * @returns Eight FP8 E4M3 packed into a uint64_t
+     */
+    __device__ uint64_t to_fp8x8(float scale) const {
+        // We do not need to actually clamp. Clamping is handled implicitly by the satfinite semantics of the float->e4m3 conversion.
+        // We only need to swallow NaNs. Here, we set them to 0.
+        __nv_bfloat162 bf0 = mask_NaNs_to_zero(first_pair());
+        __nv_bfloat162 bf1 = mask_NaNs_to_zero(second_pair());
+        __nv_bfloat162 bf2 = mask_NaNs_to_zero(third_pair());
+        __nv_bfloat162 bf3 = mask_NaNs_to_zero(fourth_pair());
+
+        float2 f0 = __bfloat1622float2(bf0);
+        float2 f1 = __bfloat1622float2(bf1);
+        float2 f2 = __bfloat1622float2(bf2);
+        float2 f3 = __bfloat1622float2(bf3);
+
+        __nv_fp8x4_e4m3 converted0{float4{f0.x * scale, f0.y * scale, f1.x * scale, f1.y * scale}};
+        __nv_fp8x4_e4m3 converted1{float4{f2.x * scale, f2.y * scale, f3.x * scale, f3.y * scale}};
+
+        return type_pun<uint32_t>(converted0) | ((uint64_t) type_pun<uint32_t>(converted1) << 32);
+    }
+};
+
+} // namespace
+
+/**
+ * @brief Fetches all activations for a single token from global to shared memory.
+ *
+ * Fetches all @c Dims::HIDDEN_STATES activations for a single token asynchronously from global to shared memory.
+ * Before accessing the shared memory values, wait for the transfer via @c pipe .
+ *
+ * @param source Pointer to the first activation of the token (in global memory).
+ * @param dest Pointer to the first activation of the token (in shared memory).
+ * @param pipe CUDA pipeline to execute the transfer in
+ */
+template <typename Dims>
+__device__ void moe_fetch_activation_async(
+    const A_element* __restrict__ source,
+    A_element* __restrict dest,
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    using CoreDims = MoECoreDims<Dims>;
+
+    const std::uint32_t thread = get_thread<Dims>();
+    const std::uint32_t warp = get_any_warp<Dims>(); // we run this at the beginning of our kernel with 1 warp per input token
+    const std::uint32_t thread_chunk_size = 16 / sizeof(*source);
+    const std::uint32_t chunk_size = CoreDims::THREADS_PER_WARP * thread_chunk_size;
+
+    pipe.producer_acquire();
+    for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES; k += chunk_size) {
+        copy128(dest[rotate_col_32(k, warp)], source[k], pipe);
+    }
+    pipe.producer_commit();
+}
+
+/**
+ * @brief Scales activation values in shared memory.
+ *
+ * It is used only for BS8. The function expects that the activations have been transferred to shared memory already.
+ * Fetch them with @c moe_scale_activation_BS8 well before this call (sychronization required).
+ *
+ * Each warp scales one token, i.e. do not call this function on more warps than tokens.
+ * For the details of the conversion, @see BF16x8::to_fp8x8
+ * All the activations for one token are scaled to [-FP8_MAX, FP8_MAX] range.
+ * The inverse of this scaling factor is multiplied to @p topk_weight .
+ *
+ *
+ * @param [in] activation_in Pointer to the first activation of the token in shared memory. Must be aligned to 16 Byte.
+ * @param [out] activation_out Pointer to the first quantized activation of the token in shared memory. Must be aligned to 8 Byte.
+ * @param [inout] topk_weight Adjusted by the inverse scaling that is used to scale the activations.
+ */
+template <typename Dims>
+__device__ void moe_scale_activation_BS8(
+    const A_element* __restrict__ activation_in,
+    AQ_element* __restrict__ activation_out,
+    float& __restrict__ topk_weight)
+{
+    static_assert(Dims::BS <= 8, "This function is only for use with BS tup to 8");
+    assert((uintptr_t) activation_in != (uintptr_t) activation_out);
+
+    static_assert(Dims::HIDDEN_STATES * sizeof(A_element) % 16 == 0, "Next token activation will not be properly aligned.");
+    static_assert(Dims::HIDDEN_STATES  % 8 == 0, "Next quantized token activation will not be properly aligned.");
+
+    assert((uintptr_t) activation_in % 16 == 0);
+    assert((uintptr_t) activation_out % 8 == 0);
+
+    using CoreDims = MoECoreDims<Dims>;
+
+    const std::uint32_t thread = get_thread<Dims>();
+    const std::uint32_t thread_chunk_size = sizeof(BF16x8) / sizeof(*activation_in);
+    const std::uint32_t chunk_size = CoreDims::THREADS_PER_WARP * thread_chunk_size;
+
+    constexpr float FP8_MAX = 448.f;
+    constexpr float FP8_MAX_INV = 1.0f / 448.f;
+
+    // find max
+    __nv_bfloat162 m0{0.0f, 0.0f}, m1{0.0f, 0.0f}, m2{0.0f, 0.0f}, m3{0.0f, 0.0f};
+    for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES; k += chunk_size) {
+        BF16x8 chunk = BF16x8::load(activation_in + k);
+
+        m0 = __hmax2(m0, __habs2(chunk.first_pair()));
+        m1 = __hmax2(m1, __habs2(chunk.second_pair()));
+        m2 = __hmax2(m2, __habs2(chunk.third_pair()));
+        m3 = __hmax2(m3, __habs2(chunk.fourth_pair()));
+    }
+
+    m0 = __hmax2(__hmax2(m0, m1), __hmax2(m2, m3));
+    float m = (float) __hmax(m0.x, m0.y);
+    m = warp_reduce_max_float(m);
+
+    // avoid 0/0 (or inf)
+    if (m < __FLT_MIN__) {
+        m = 1.f;
+    }
+    float scale = m * FP8_MAX_INV;
+    float inv_scale = FP8_MAX / m;
+
+    assert((uintptr_t) activation_out % 8 == 0);
+    uint64_t *activation_out8 = reinterpret_cast<uint64_t*>(activation_out);
+
+    // scale, clamp (incl. NaN replacement), round and convert to FP8 range
+    for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES; k += chunk_size) {
+        BF16x8 chunk = BF16x8::load(activation_in + k);
+        activation_out8[k/8] = chunk.to_fp8x8(inv_scale);
+    }
+
+    // fold scaling factors
+    if (thread == 0) {
+        topk_weight = topk_weight * scale;
+    }
+}
+
+
+namespace detail {
+
+/**
+ * @brief Scales activation values.
+ *
+ * The function expects that the activations in global memory.
+ *
+ * Each warp scales one token, i.e. do not call this function on more warps than tokens.
+ * For the details of the conversion, @see BF16x8::to_fp8x8
+ * All the activations for one token are scaled to [-FP8_MAX, FP8_MAX] range.
+ * The inverse of this scaling factor is multiplied to @p topk_weight .
+ *
+ * The resulting @p topk_weight_scaled and @p temp out are assumed to be in shared memory.
+ *
+ * @param [in] activation_in Pointer to the first activation of the token in globale memory. Must be aligned to 16 Byte.
+ * @param [out] temp Temorary scratchpad shared memory storage. Must be aligned to 16 Byte.
+ * @param [out] activation_out Pointer to the first quantized activation of the token in global memory. Must be aligned to 8 Byte.
+ * @param [in] topk_weight Original token weight
+ * @param [out] topk_weight_scaled Adjusted token weight. Multiplied with the inverse scaling that is used to scale the activations.
+ */
+template <typename Dims>
+__device__ static void moe_scale_activation_BSx_chunk(
+    const A_element* __restrict__ activation_in,
+    A_element* __restrict__ temp,
+    AQ_element* __restrict__ activation_out,
+    const float topk_weight,
+    float& __restrict__ topk_weight_scaled)
+{  
+    assert((uintptr_t) activation_in != (uintptr_t) temp);
+    assert((uintptr_t) activation_out != (uintptr_t) temp);
+    assert((uintptr_t) activation_in != (uintptr_t) activation_out);
+    using CoreDims = MoECoreDims<Dims>;
+
+    const std::uint32_t thread = get_thread<Dims>();
+    const std::uint32_t thread_chunk_size = sizeof(BF16x8) / sizeof(*activation_in);
+    const std::uint32_t chunk_size = CoreDims::THREADS_PER_WARP * thread_chunk_size;
+
+    constexpr float FP8_MAX = 448.f;
+    constexpr float FP8_MAX_INV = 1.0f / 448.f;
+
+    // copy to SHM and find max
+    __nv_bfloat162 m0{0.0f, 0.0f}, m1{0.0f, 0.0f}, m2{0.0f, 0.0f}, m3{0.0f, 0.0f};
+    for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES; k += chunk_size) {
+        BF16x8 chunk = BF16x8::load(activation_in + k);
+        chunk.store_to(&temp[k]);
+
+        m0 = __hmax2(m0, __habs2(chunk.first_pair()));
+        m1 = __hmax2(m1, __habs2(chunk.second_pair()));
+        m2 = __hmax2(m2, __habs2(chunk.third_pair()));
+        m3 = __hmax2(m3, __habs2(chunk.fourth_pair()));
+    }
+
+    m0 = __hmax2(__hmax2(m0, m1), __hmax2(m2, m3));
+    float m = (float) __hmax(m0.x, m0.y);
+    m = warp_reduce_max_float(m);
+
+    // avoid 0/0 (or inf)
+    if (m < __FLT_MIN__) {
+        m = 1.f;
+    }
+    float scale = m * FP8_MAX_INV;
+    float inv_scale = FP8_MAX / m;
+
+    assert((uintptr_t) activation_out % 8 == 0);
+    uint64_t *activation_out8 = reinterpret_cast<uint64_t*>(activation_out);
+
+    // scale, clamp (incl. NaN replacement), round and convert to FP8 range
+    for (std::uint32_t k = thread * thread_chunk_size; k < Dims::HIDDEN_STATES; k += chunk_size) {
+        BF16x8 chunk = BF16x8::load(activation_in + k);
+        activation_out8[k/8] = chunk.to_fp8x8(inv_scale);
+    }
+
+    // fold scaling factors
+    if (thread == 0) {
+        topk_weight_scaled = topk_weight * scale;
+    }
+}
+
+} // namespace detail
+
+
+/**
+ * @brief Scales the activations of all tokens.
+ *
+ * The function expects that the activations in global memory.
+ * This function is collective and needs to be called by all threads in all CUDA blocks.
+ *
+ * For the details of the conversion, @see BF16x8::to_fp8x8
+ * All the activations for one token are scaled to [-FP8_MAX, FP8_MAX] range.
+ * The inverse of this scaling factor is multiplied to the @c topk_weight in @p shmem .
+ * The resulting scaled activations are placed in @c spec->activations .
+ *
+ *
+ * @param [in] activation_in Pointer to the first activation of the first token in globale memory. Must be aligned to 16 Byte.
+ * @param [out] token_count Number of active tokens
+ * @param [out] spec Provides storage for the result.
+ * @param [out] shmem Provides storage for the result topk_weight.
+ */
+template <typename Dims>
+__device__ void moe_scale_activation_BSx(
+    const A_element* __restrict__ activations_in,
+    std::uint32_t token_count,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    static_assert(Dims::BS > 8, "BS=8 is handled by its own kernel. Do not use moe_scale_inputs for BS<=8");
+    static_assert(Dims::HIDDEN_STATES * sizeof(A_element) % 16 == 0, "Next token activation will not be properly aligned.");
+    static_assert(Dims::HIDDEN_STATES  % 8 == 0, "Next quantized token activation will not be properly aligned.");
+
+    assert((uintptr_t) activations_in % 16 == 0);
+    assert((uintptr_t) spec->activations % 8 == 0);
+    
+    using CoreDims = MoECoreDims<Dims>;
+
+    if (is_calc_warp<Dims>()) {
+        const std::uint32_t global_warp_count = gridDim.x * CoreDims::CALC_WARP_COUNT;
+        const std::uint32_t warp = get_calc_warp<Dims>();
+        const std::uint32_t global_warp = blockIdx.x * CoreDims::CALC_WARP_COUNT + warp;
+
+        for (std::uint32_t i = global_warp; i < token_count; i += global_warp_count) {
+            detail::moe_scale_activation_BSx_chunk<Dims>(
+                activations_in + i * Dims::HIDDEN_STATES,
+                shmem->u.rescale.a[warp],
+                spec->activations[i],
+                shmem->topk_weights[i],
+                spec->topk_weights_scaled[i]);
+        }
+    }
+
+    // spec->topk_weights_scaled is written to by different blocks, make them available to all blocks
+    cooperative_groups::this_grid().sync();
+
+    for (uint32_t i = threadIdx.x; i < token_count; i += blockDim.x) {
+        shmem->topk_weights[i] = spec->topk_weights_scaled[i];
+    }
+
+    __syncthreads();
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/moe_up_projection.cu b/csrc/moe/moe_monokernel/src/moe_up_projection.cu
new file mode 100644
index 000000000..fc53c74a9
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/moe_up_projection.cu
@@ -0,0 +1,854 @@
+
+#pragma once
+#ifndef MOE_UP_PROJECTION_CU
+#define MOE_UP_PROJECTION_CU
+
+#ifndef INSIDE_MOE_MONOKERNEL_IMPLEMENTATION
+#error Do not include this file directly.
+#endif
+
+#include <cuda.h>
+#include <cuda/pipeline>
+#include <cuda_fp8.h>
+#include <stdio.h>
+
+#include "ptx_utils.h"
+#include "moe_interface.h"
+#include "moe_internal.h"
+#include "moe_down_projection.cu"
+
+///////////////////////////////////////////////////////////////////////////////
+//
+// Design Considerations
+//
+// * The smallest matrix dimensions that the Tensor Cores support is 16 x 8 x K
+//   - 8 is a reasonable amount of input tokens we can process at once per expert,
+//     if there are <= 64 tokens and >= 16 experts.
+//   - 16 rows of the weight matrix per SM matches the 2048 = 2*N rows we have with TP=8
+//   - with K=5120, input token activation tile is 40kB, a weight tile is 80kB
+//
+// * Max. 220kB of the 224kB Shared Memory are available for matrix tiles
+//   - 'tiny' kernel holds 1 fixed token tile and 2 weight tiles (200kB total),
+//     so we use double-buffering for 'tiny': fetching the next expert weights while
+//     processing the current one.
+//   - 'normal' kernel needs to also prefetch input tokens, hence we use tiles
+//     that only span HALF of K, but have now enough space for 3x20kB input token
+//     tiles plus 3x40kB of weight tiles. We use triple-buffering to ensure that
+//     data is likely to arrive early enough: 1 tile in MMA processing, 1 coming in,
+//     1 being currently requested from Global Memory.
+//
+// * Reading matrix data from Shared Memory accesses the same columns in different
+//   rows simultaneously. We prevent bank conflicts in different ways:
+//   - 'normal' kernel uses tile rows lengths with an extra 16 byte padding,
+//     mapping the same column for all 8 consecutive rows onto to different banks.
+//   - downside: prevents Global->Shared Memory transfers from being fully coalesced.
+//   - 'tiny' kernel has larger tile sizes that make it more easy to be memory bound.
+//     We don't pad the row 'swizzle' the data inside them. Experiments found that
+//     32-byte swizzles be the sweet spot, eliminating non-coalesced transfers at the
+//     expense of 50% Shared Memory bank conflicts.
+//
+///////////////////////////////////////////////////////////////////////////////
+
+namespace moe_monokernel {
+
+/**
+ * @brief Initiate the copy of token activations from Global to Shared Memory
+ *
+ * This device function issues the asynchronous data copy requests for a tile of 
+ * token activation vectors. The copy operations will be queued in the given @a pipe,
+ * which the caller must use to wait for their completion.
+ * 
+ * Token addressing is indirect, using @a token_indexes to specify which activation
+ * vectors to copy from @a source. If @a max_count is less than the number of tokens
+ * that fit into the given Shared Memory tile at @a dest, only the content for the
+ * first @a max_count tokens will be replaced.
+ *
+ * @note Like all prefetching functions, this function must only be called by
+ *       threads in prefetch warps.
+ *
+ * @param source Pointer token activation array of shape [BS, HIDDEN_STATES] in row-major
+ *               order. Individual elements are quantized in __nv_fp8_e4m3 format.
+ *               Tile offsets within the source rows are expressed by shifting the
+ *               pointer by the respective number of columns.
+ *               Stored in Global Memory.
+ * @param token_indexes Row indexes within @a source of the tokens to fetch.
+ * @param dest Shared Memory struct (tile of token activations) to store the result to.
+ * @param max_count Maximum number of tokens to copy. Implicitly limited to @a dest capacity.
+ * @param pipe Asynchronous completion pipe to use.      
+ */
+template <typename Dims>
+__device__ inline void moe_request_input_tokens(
+    const AQ_element* __restrict__ source,
+    const std::uint16_t* __restrict__ token_indexes,
+    AQ_element (&dest)[MoECoreDims<Dims>::A_TILE][MoECoreDims<Dims>::K_DIM_HALF_PADDED_A],
+    std::uint32_t max_count,
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_prefetch_warp<Dims>();
+
+    // async transfers are 16 bytes / thread
+    const unsigned chunk_size = 16 / sizeof(*source);
+
+    for (unsigned row = warp; row < CoreDims::A_TILE; row += CoreDims::PREFETCH_WARP_COUNT) {
+        if (row < max_count) {
+            std::uint32_t row_idx = token_indexes[row];
+            const AQ_element* a = source + row_idx * Dims::HIDDEN_STATES;
+            for (unsigned col = thread * chunk_size; col < Dims::HIDDEN_STATES / 2; col += CoreDims::THREADS_PER_WARP * chunk_size) {
+                unsigned dest_col = rotate_col_32(col, row);
+                copy128(dest[row][dest_col], a[col], pipe);
+            }
+        }
+    }
+}
+
+/**
+ * @brief Initiate the copy of expert weights from Global to Shared Memory
+ *
+ * This device function issues the asynchronous data copy requests for a tile of 
+ * expert weights. The copy operations will be queued in the given @a pipe,
+ * which the caller must use to wait for their completion.
+ * 
+ * While the expert is selected by @a id, the tile to copy is implicitly selected
+ * by the @c blockIdx. Two ranges of rows will be copied: @a Rows/2 rows from the
+ * first @c N rows of weights, followed by @a Rows/2 rows of the second @c N rows
+ * of weights.
+ * 
+ * To allow for differences in Shared Memory data layout, @a CopyCols template
+ * parameter specifies the number of elements to copy per row.
+ *
+ * @note Like all prefetching functions, this function must only be called by
+ *       threads in prefetch warps.
+ *
+ * @tparam CopyCols Number of elements to copy in each row.
+ * @param source Pointer token weights array of shape [NUM_EXPERTS, 2*N, HIDDEN_STATES]
+ *               in expert, row-major order. Tile offsets within the source rows are
+ *               expressed by shifting the pointer by the respective number of columns.
+ *               Individual elements are in __nv_fp8_e4m3 format.
+ *               Stored in Global Memory.
+ * @param id Expert index within @a source.
+ * @param dest Shared Memory struct (tile of token activations) to store the result to.
+ * @param pipe Asynchronous completion pipe to use.
+ */
+template <typename Dims, std::size_t CopyCols, std::size_t Rows, std::size_t Cols>
+__device__ inline void moe_request_up_expert(
+    const W_element* __restrict__ source,
+    std::uint32_t id,
+    W_element (&dest)[Rows][Cols],
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    static_assert(CopyCols <= Cols);
+
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_prefetch_warp<Dims>();
+    const unsigned chunk_size = 16 / sizeof(*source);
+
+    // starting row to process
+    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
+    const unsigned item_cols_per_iteration = CoreDims::THREADS_PER_WARP * chunk_size;
+
+    // Each row within the weights matrix needs to be multiple of the copy size
+    static_assert(Dims::HIDDEN_STATES % item_cols_per_iteration == 0);
+    const OpaqueElement* weights = (const OpaqueElement *)(source + id * 2 * Dims::N * Dims::HIDDEN_STATES);
+
+    // bring W tile
+    // Even warps fetch 8 rows from lower N weight rows, odd warps fetch 8 rows from upper N
+    // Curb unrolling for smaller and slightly faster kernel code
+    #pragma unroll 1
+    for (unsigned row = warp / 2; row < CoreDims::W_UP_TILE / 2; row += CoreDims::PREFETCH_WARP_COUNT / 2) {
+        for (unsigned col = 0; col <  CopyCols; col += item_cols_per_iteration) {
+            unsigned source_col = thread;
+            unsigned is_upper = warp & 1;
+
+            copy128(dest[row + is_upper * CoreDims::W_UP_TILE / 2][rotate_col_32(col + source_col * chunk_size, row)],
+                    weights[((row + base_row + is_upper * Dims::N) * Dims::HIDDEN_STATES + col + source_col * chunk_size) / sizeof(OpaqueElement)],
+                    pipe);
+        }
+    }
+}
+
+/**
+ * @brief Performs the MMA result reduction and sigmoid step of up-projection.
+ *
+ * This device function sums up the partial scalar products created by all warps,
+ * applied the respective weight and token activation scales, calculates the sigmoid,
+ * and finally stores the results in Global Memory.  The tile to be written is
+ * implicitly determined by the @c blockIdx.
+ * 
+ * The output can be filtered, i.e. @a store_row0 and @a store_row1 control whether
+ * the results for the respective rows @a row0 and @a row1 shall be written.  This
+ * allows the called to always process data at the full tile size and simply suppress
+ * superfluous results in the output.
+ *
+ * @note This function is supposed to be called by warp 0 only.
+ *
+ * @param partial_result Array of MMA results from all warps of shape [WARPS, 4, THREADS]
+ *                       in row-major order. Individual elements are in FP32 format.
+ * @param d0 First element of the MMA result of warp 0.
+ * @param d1 Second element of the MMA result of warp 0.
+ * @param d2 Third element of the MMA result of warp 0.
+ * @param d3 Forth element of the MMA result of warp 0.
+ * @param ws0 First weight scale for the respective expert.
+ * @param ws1 Second weight scale for the respective expert.
+ * @param ts0 First token activation scale.
+ * @param ts1 Second token activation scale.
+ * @param store_row0 Specifies if result in @a row0 shall be stored.
+ * @param store_row1 Specifies if result in @a row1 shall be stored.
+ * @param row0 Row to store the scalar products for the first token.
+ * @param row1 Row to store the scalar products for the second token.
+ * @param result Pointer to the output array of shape [BS, N] in row-major order.
+ *               Individual elements are in FP32 format.
+ */
+template <typename Dims, std::size_t Rows, std::size_t Cols>
+__device__ inline void moe_up_reduction(
+    const float (&partial_result)[Rows][Cols],
+    float d0, float d1, float d2, float d3,
+    float ws0, float ws1,
+    float ts0, float ts1,
+    bool store_row0,
+    bool store_row1,
+    unsigned row0,
+    unsigned row1,
+#ifdef DEBUG_MOE
+    float* __restrict__ gemm1,
+#endif
+    T_element* __restrict__ result)
+{
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+
+    // starting row to process
+    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
+
+    // combine results
+    for (unsigned i = 1; i < CoreDims::CALC_WARP_COUNT; ++i) {
+
+        d0 += partial_result[i][thread +  0];
+        d1 += partial_result[i][thread + 32];
+        d2 += partial_result[i][thread + 64];
+        d3 += partial_result[i][thread + 96];
+    }
+
+    // for debugging purposes
+#ifdef DEBUG_MOE
+    if (store_row0) {
+        gemm1[row0 * 2 * Dims::N + (thread / 4) + base_row +       0] = d0 * ts0 * ws0;
+        gemm1[row0 * 2 * Dims::N + (thread / 4) + base_row + Dims::N] = d2 * ts0 * ws1;
+    }
+    if (store_row1) {
+        gemm1[row1 * 2 * Dims::N + (thread / 4) + base_row +       0] = d1 * ts1 * ws0;
+        gemm1[row1 * 2 * Dims::N + (thread / 4) + base_row + Dims::N] = d3 * ts1 * ws1;
+    }
+#endif
+
+    // apply weights and store as temp
+    // x: columns 0 ..  7
+    // w: columns 8 .. 15
+    float x0 = d0 * ts0 * ws0;
+    float x1 = d1 * ts1 * ws0;
+    float w0 = d2 * ts0 * ws1;
+    float w1 = d3 * ts1 * ws1;
+
+    float sig0 = (w0 * x0) / (1 + expf(-x0));
+    float sig1 = (w1 * x1) / (1 + expf(-x1));
+
+    // write to temporary buffer
+    if (store_row0) {
+        result[row0 * Dims::N + (thread / 4) + base_row] = sig0;
+    }
+    if (store_row1) {
+        result[row1 * Dims::N + (thread / 4) + base_row] = sig1;
+    }
+}
+
+/**
+ * @brief Performs the MMA reduction and sigmoid step of up-projection for 'Tiny' kernels.
+ *
+ * This device function sums up the partial scalar products created by all warps,
+ * applied the respective weight and token activation scales, calculates the sigmoid,
+ * and finally stores the results in Global Memory.  The tile to be written is
+ * implicitly determined by the @c blockIdx.
+ *
+ * @note This function is supposed to be called by warps 0 and 1 only.
+ *
+ * @param partial_result Array of MMA results from all warps of shape [WARPS, 4, THREADS]
+ *                       in row-major order. Individual elements are in FP32 format.
+ * @param ws0 First weight scale for the respective expert.
+ * @param ws1 Second weight scale for the respective expert.
+ * @param ts Token activation scale.
+ * @param row Row to store the scalar products for the token.
+ * @param result Pointer to the output array of shape [BS, N] in row-major order.
+ *               Individual elements are in FP32 format.
+ */
+template <typename Dims, std::size_t Rows, std::size_t Cols>
+__device__ inline void moe_up_reduction_tiny(
+    const float (&partial_result)[Rows][Cols],
+    float ws0, float ws1,
+    float ts,
+    unsigned row,
+#ifdef DEBUG_MOE
+    float* __restrict__ gemm1,
+#endif
+    T_element* __restrict__ result)
+{
+    // position within the block
+    using CoreDims = MoECoreDims<Dims>;
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_calc_warp<Dims>();
+
+    // starting row to process
+    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
+
+    // combine results, reduce dependency chain on dX
+    float d0 = partial_result[0][thread + warp * 32 +  0] + partial_result[1][thread + warp * 32 +  0];
+    float d2 = partial_result[0][thread + warp * 32 + 64] + partial_result[1][thread + warp * 32 + 64];
+
+    for (unsigned i = 2; i < CoreDims::CALC_WARP_COUNT; i += 2) {
+
+        d0 += partial_result[i][thread + warp * 32 +  0] + partial_result[i + 1][thread + warp * 32 +  0];
+        d2 += partial_result[i][thread + warp * 32 + 64] + partial_result[i + 1][thread + warp * 32 + 64];
+    }
+
+    // for debugging purposes
+#ifdef DEBUG_MOE
+    gemm1[row * 2 * Dims::N + (thread / 4) + base_row +       0] = d0 * ts * ws0;
+    gemm1[row * 2 * Dims::N + (thread / 4) + base_row + Dims::N] = d2 * ts * ws1;
+#endif
+
+    // write to temporary buffer
+    float x0 = d0 * ts * ws0;
+    float w0 = d2 * ts * ws1;
+    float sig0 = (w0 * x0) / (1 + expf(-x0));
+    result[row * Dims::N + (thread / 4) + base_row] = sig0;
+}
+
+/**
+ * @brief Standard kernel for the first GEMM ("up projection"), combined with a sigmoid reduction.
+ *
+ * This device function processes @c BS tokens, grouped by expert in internal batches of 8 tokens.
+ * The experts to use and the respective list of tokes for each of them is given by @a shmem.
+ *
+ * All non-expert data is taken from our temporary storage in either @a spec or @a shmem
+ * and results will be written to @a spec. Outputs are grouped by expert.
+ *
+ * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS, 2*N, HIDDEN_STATES]
+ *                          in expert, row-major order. Individual elements are in __nv_fp8_e4m3 format.
+ *                          Stored in Global Memory.
+ * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS, 2*N]
+ *                         in row-major order. Individual elements are in FP32 format.
+ *                         Stored in Global Memory.
+ * @param spec Global Memory struct containing the scaled input token activations.
+ *             It will also receive the output of this function.
+ * @param shmem Shared Memory struct containing the expert<=>token mapping, activation weights
+ *              and will be uses as local scratch pad store for faster operation.
+ */
+template <typename Dims>
+__device__ inline void moe_up_projection_normal(
+    const W_element* __restrict__ expert_weights_up,
+    const S_element* __restrict__ expert_scales_up,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    using CoreDims = MoECoreDims<Dims>;
+    using MoE_SHM = MoE_SHM<Dims>;
+
+    // position within the block
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_any_warp<Dims>();
+
+    // required for async copies
+    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
+
+    // starting row to process
+    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
+
+    // some commonly used values
+    typename MoE_SHM::U::Gemm1Data* shm = &shmem->u.gemm1;
+    std::uint32_t expert_count = shmem->expert_count;
+    const AQ_element* activations = spec->activations[0];
+
+    // some paranoia bounds-checking
+#ifndef NDEBUG
+    assert(expert_count <= Dims::BS);
+    assert(shmem->experts[0].first_token == 0);
+    assert(shmem->experts[expert_count-1].last_token <= Dims::BS);
+    if (threadIdx.x < expert_count) {
+        assert(shmem->experts[threadIdx.x].last_token > shmem->experts[threadIdx.x].first_token);
+        assert(threadIdx.x == 0 || shmem->experts[threadIdx.x - 1].last_token == shmem->experts[threadIdx.x].first_token);
+    }
+#endif
+
+    // triple-buffering: queue in the first 2 tiles of both, token activations and weights
+    if (is_prefetch_warp<Dims>()) {
+        const ExpertRef& expert = shmem->experts[0];
+
+        // bring in first A and W tiles
+        pipe.producer_acquire();
+        moe_request_input_tokens<Dims>(
+            activations,
+            &shmem->token_indexes[expert.first_token],
+            shm->a[0],
+            expert.last_token,
+            pipe);
+        moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
+            expert_weights_up,
+            expert.id,
+            shm->w[0],
+            pipe);
+
+        pipe.producer_commit();
+
+        // request second A and W tiles
+        pipe.producer_acquire();
+        moe_request_input_tokens<Dims>(
+            activations  + Dims::HIDDEN_STATES / 2,
+            &shmem->token_indexes[expert.first_token],
+            shm->a[1],
+            expert.last_token,
+            pipe);
+        moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
+            expert_weights_up + Dims::HIDDEN_STATES / 2,
+            expert.id,
+            shm->w[1],
+            pipe);
+
+        pipe.producer_commit();
+    }
+
+    // track current positions within the triple-buffers
+    std::uint32_t t_index_read = 0;
+    std::uint32_t t_index_write = 2;
+    std::uint32_t w_index_read = 0;
+    std::uint32_t w_index_write = 2;
+
+    // loop over all experts
+    for (std::uint32_t e = 0; e < expert_count; ++e) {
+        // some commonly used pointers, counts and IDs
+        const ExpertRef& expert = shmem->experts[e];
+        std::uint32_t id = expert.id;
+        const S_element* scales = expert_scales_up + id * 2 * Dims::N;
+        unsigned int a_rows = expert.last_token - expert.first_token;
+        T_element* temp = &spec->temp[expert.first_token * Dims::N];
+
+        // weight scales
+        float ws0 = scales[base_row + thread / 4          ];
+        float ws1 = scales[base_row + thread / 4 + Dims::N];
+
+        // process all activations for this set of weights
+        for (unsigned a_row = 0; a_row < a_rows; a_row += CoreDims::A_TILE) {
+
+            // init accumulators
+            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
+
+            // wait for currently needed data to come in
+            cuda::pipeline_consumer_wait_prior<1>(pipe);
+            __syncthreads();
+
+            if (is_prefetch_warp<Dims>()) {
+                // request next T and W tiles as needed
+                pipe.producer_acquire();
+                if (e + 1 < expert_count && a_row == 0) {
+                    // there will be another expert -> request its weights
+                    moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
+                        expert_weights_up,
+                        shmem->experts[e + 1].id,
+                        shm->w[w_index_write],
+                        pipe);
+                    w_index_write = w_index_write == 2 ? 0 : w_index_write + 1;
+                }
+
+                if (a_row + CoreDims::A_TILE < a_rows) {
+                    // request the next T tile for same expert
+                    moe_request_input_tokens<Dims>(
+                        activations,
+                        &shmem->token_indexes[expert.first_token + CoreDims::A_TILE + a_row],
+                        shm->a[t_index_write],
+                        a_rows - CoreDims::A_TILE - a_row,
+                        pipe);
+                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
+                }
+                else if (e + 1 < expert_count) {
+                    // request the first T tile for next expert
+                    const ExpertRef& expert = shmem->experts[e + 1];
+                    moe_request_input_tokens<Dims>(
+                        activations,
+                        &shmem->token_indexes[expert.first_token],
+                        shm->a[t_index_write],
+                        expert.last_token - expert.first_token,
+                        pipe);
+                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
+                }
+                pipe.producer_commit();
+            }
+            else {
+                // first half of the row: run partial scalar products on the tiles
+                for (unsigned base_col = warp * CoreDims::K_TILE; base_col < Dims::HIDDEN_STATES / 2; base_col += CoreDims::BLOCK_STRIDE) {
+
+                    unsigned row = 0 + thread / 4;
+                    unsigned col = 4 * (thread % 4);
+
+                    __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col +  0, row)];
+                    __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col +  0, row)];
+                    __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col + 16, row)];
+                    __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col + 16, row)];
+
+                    __nv_fp8x4_e4m3 a02 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(base_col + col +  0, row)]);
+                    __nv_fp8x4_e4m3 a13 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(base_col + col + 16, row)]);
+
+                    mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
+                }
+            }
+
+            // we will now write to the same tile that we just consumed.
+            // Make sure everyone is done reading it.
+            __syncthreads();
+
+            w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
+            t_index_read = t_index_read == 2 ? 0 : t_index_read + 1;
+
+            // wait for currently needed data to come in
+            cuda::pipeline_consumer_wait_prior<1>(pipe);
+            __syncthreads();
+
+            if (is_prefetch_warp<Dims>()) {
+                // request next T and W tiles as needed
+                pipe.producer_acquire();
+                if (a_row + CoreDims::A_TILE < a_rows) {
+                    // request the next T tile for same expert
+                    moe_request_input_tokens<Dims>(
+                        activations + Dims::HIDDEN_STATES / 2,
+                        &shmem->token_indexes[expert.first_token + CoreDims::A_TILE + a_row],
+                        shm->a[t_index_write],
+                        a_rows - CoreDims::A_TILE - a_row,
+                        pipe);
+                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
+                }
+                else if (e + 1 < expert_count) {
+                    // request the first T and W tile for next expert
+                    const ExpertRef& expert = shmem->experts[e + 1];
+                    moe_request_input_tokens<Dims>(
+                        activations + Dims::HIDDEN_STATES / 2,
+                        &shmem->token_indexes[expert.first_token],
+                        shm->a[t_index_write],
+                        expert.last_token - expert.first_token,
+                        pipe);
+                    t_index_write = t_index_write == 2 ? 0 : t_index_write + 1;
+
+                    moe_request_up_expert<Dims, Dims::HIDDEN_STATES / 2>(
+                        expert_weights_up + Dims::HIDDEN_STATES / 2,
+                        expert.id,
+                        shm->w[w_index_write],
+                        pipe);
+                    w_index_write = w_index_write == 2 ? 0 : w_index_write + 1;
+                }
+                pipe.producer_commit();
+            }
+            else {
+                // second half of the row: run partial scalar products on the tiles
+                for (unsigned base_col = warp * CoreDims::K_TILE; base_col < Dims::HIDDEN_STATES / 2; base_col += CoreDims::BLOCK_STRIDE) {
+
+                    unsigned row = 0 + thread / 4;
+                    unsigned col = 4 * (thread % 4);
+
+                    __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col +  0, row)];
+                    __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col +  0, row)];
+                    __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 0][rotate_col_32(base_col + col + 16, row)];
+                    __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index_read][row + 8][rotate_col_32(base_col + col + 16, row)];
+
+                    __nv_fp8x4_e4m3 a02 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(base_col + col +  0, row)]);
+                    __nv_fp8x4_e4m3 a13 = *(__nv_fp8x4_e4m3*)(&shm->a[t_index_read][row][rotate_col_32(base_col + col + 16, row)]);
+
+                    mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
+                }
+
+                // store partial results
+                shm->partial_result[warp][thread +  0] = d0;
+                shm->partial_result[warp][thread + 32] = d1;
+                shm->partial_result[warp][thread + 64] = d2;
+                shm->partial_result[warp][thread + 96] = d3;
+            }
+
+            // make sure all partial results have been written and weights are not needed anymore
+            __syncthreads();
+
+            w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
+            t_index_read = t_index_read == 2 ? 0 : t_index_read + 1;
+
+            if (a_row + CoreDims::A_TILE < a_rows) {
+                w_index_read = w_index_read == 2 ? 0 : w_index_read + 1;
+            }
+
+            if (warp == 0) {
+
+                // reading the topK_weights from valid row / token indexes
+                std::uint32_t row0 = a_row + (thread % 4) * 2 + 0;
+                std::uint32_t row1 = a_row + (thread % 4) * 2 + 1;
+                const std::uint16_t* token_indexes = &shmem->token_indexes[expert.first_token];
+
+                std::uint32_t ts0_idx = token_indexes[row0 < a_rows ? row0 : 0];
+                std::uint32_t ts1_idx = token_indexes[row1 < a_rows ? row1 : 0];
+                float ts0 = shmem->topk_weights[ts0_idx];
+                float ts1 = shmem->topk_weights[ts1_idx];
+
+                // combine results
+                moe_up_reduction<Dims>(
+                    shm->partial_result,
+                    d0, d1, d2, d3,
+                    ws0, ws1,
+                    ts0, ts1,
+                    row0 < a_rows,
+                    row1 < a_rows,
+                    row0,
+                    row1,
+#ifdef DEBUG_MOE
+                    &spec->gemm1[expert.first_token * 2 * Dims::N],
+#endif
+                    temp);
+            }
+        }
+    }
+}
+
+/**
+ * @brief 'Tiny' kernel for the first GEMM ("up projection"), combined with a sigmoid reduction.
+ *
+ * This device function processes up to 8 tokens, already prefetched into @a shmem.
+ * The experts to use with each token is also given by @a shmem, the function will apply them to all
+ * and only filter the output accordingly.
+ * The weights for the first expert have already been prefetched into @a shmem as well. 
+ *
+ * All non-expert data is taken from our temporary storage in @a shmem
+ * and results will be written to @a spec. Output order is the input token order.
+ *
+ * @param expert_weights_up Pointer token weights array of shape [NUM_EXPERTS, 2*N, HIDDEN_STATES]
+ *                          in expert, row-major order. Individual elements are in __nv_fp8_e4m3 format.
+ *                          Stored in Global Memory.
+ * @param expert_scales_up Pointer weights scales array of shape [NUM_EXPERTS, 2*N]
+ *                         in row-major order. Individual elements are in FP32 format.
+ *                         Stored in Global Memory.
+ * @param expert_weights_down Pointer down-projection weights array of shape [NUM_EXPERTS, HIDDEN_STATES, N]
+ *                            in expert, row-major order. Individual elements are in __nv_fp8_e4m3 format.
+ *                            Used for prefetching on behalf of down-projection only.
+ *                            Stored in Global Memory.
+ * @param expert_scales_down Pointer down-projection scales array of shape [NUM_EXPERTS, HIDDEN_STATES]
+ *                           in row-major order. Individual elements are in FP32 format.
+ *                           Used for prefetching on behalf of down-projection only.
+ *                           Stored in Global Memory.
+ * @param w_index Index of the tile in @a shmem that contains the prefetched weight data
+ *                for the first expert.
+ * @param spec Global Memory struct containing the scaled input token activations.
+ *             It will also receive the output of this function.
+ * @param shmem Shared Memory struct containing the expert<=>token mapping, activation weights
+ *              and will be uses as local scratch pad store for faster operation.
+ * @param pipe Asynchronous completion pipe to use with prefetching.
+ */
+template <typename Dims>
+__device__ inline std::uint32_t moe_up_projection_tiny(
+    const W_element* __restrict__ expert_weights_up,
+    const S_element* __restrict__ expert_scales_up,
+    const W_element* __restrict__ expert_weights_down,
+    const S_element* __restrict__ expert_scales_down,
+    std::uint32_t w_index,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem,
+    cuda::pipeline<cuda::thread_scope_thread>& pipe)
+{
+    using CoreDims = MoECoreDims<Dims>;
+    using MoE_SHM = MoE_SHM<Dims>;
+
+    // position within the block
+    const unsigned thread = get_thread<Dims>();
+    const unsigned warp = get_any_warp<Dims>();
+
+    // starting row to process
+    const unsigned base_row = blockIdx.x * CoreDims::W_UP_TILE / 2;
+
+    // commonly used values
+    typename MoE_SHM::U::TinyData* shm = &shmem->u.tiny;
+    std::uint32_t expert_count = shmem->expert_count;
+    std::uint64_t expert_mask = shmem->expert_mask;
+    std::uint64_t expert_ids = shmem->expert_ids;
+
+    assert(expert_count >= 1);
+
+    // loop over all experts, except the last one
+    for (std::uint32_t e = 0; e < expert_count - 1; ++e) {
+        std::uint32_t id = expert_ids & 0xff;
+        expert_ids >>= 8;
+
+        // weight scales
+        const S_element* scales = expert_scales_up + id * 2 * Dims::N;
+        float ws0 = scales[base_row + thread / 4          ];
+        float ws1 = scales[base_row + thread / 4 + Dims::N];
+
+        // wait for currently needed data to come in
+        cuda::pipeline_consumer_wait_prior<0>(pipe);
+        __syncthreads();
+
+        if (is_prefetch_warp<Dims>()) {
+            // request next W tile
+            pipe.producer_acquire();
+            moe_request_up_expert<Dims, Dims::HIDDEN_STATES>(
+                expert_weights_up,
+                expert_ids & 0xff,
+                shm->w[w_index ^ 1].up,
+                pipe);
+            pipe.producer_commit();
+        }
+        else {
+            // init accumulators
+            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
+
+            // run partial scalar products on the tiles
+            for (unsigned base_col = warp * CoreDims::K_TILE, i = 0; i < Dims::HIDDEN_STATES / CoreDims::BLOCK_STRIDE; ++i, base_col += CoreDims::BLOCK_STRIDE) {
+
+                unsigned row = 0 + thread / 4;
+                unsigned col = 4 * (thread % 4);
+
+                __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 0][rotate_col_32(base_col + col +  0, row)];
+                __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 8][rotate_col_32(base_col + col +  0, row)];
+                __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 0][rotate_col_32(base_col + col + 16, row)];
+                __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 8][rotate_col_32(base_col + col + 16, row)];
+
+                __nv_fp8x4_e4m3 a02 = *(__nv_fp8x4_e4m3*)(&shm->a.up[row][rotate_col_32(base_col + col +  0, row)]);
+                __nv_fp8x4_e4m3 a13 = *(__nv_fp8x4_e4m3*)(&shm->a.up[row][rotate_col_32(base_col + col + 16, row)]);
+
+                mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
+            }
+
+            // store partial results
+            shm->partial_result.up[warp][thread +  0] = d0;
+            shm->partial_result.up[warp][thread + 32] = d1;
+            shm->partial_result.up[warp][thread + 64] = d2;
+            shm->partial_result.up[warp][thread + 96] = d3;
+        }
+
+        // make sure all partial results have been written and weights are not needed anymore
+        __syncthreads();
+        w_index ^= 1;
+
+        // we only have work for 2 warps
+        if (warp < 2) {
+            // filter output
+            std::uint32_t row = (thread % 4) * 2 + warp;
+            bool store_row = (expert_mask >> (row * 8) & 0xff) == id;
+
+            // combine results
+            if (store_row) {
+                moe_up_reduction_tiny<Dims>(
+                    shm->partial_result.up,
+                    ws0, ws1,
+                    shmem->topk_weights[row],
+                    row,
+#ifdef DEBUG_MOE
+                    spec->gemm1,
+#endif
+                    spec->temp);
+            }
+        }
+    }
+
+    // Last expert: prefetch the first expert weights for down-projection
+    {
+        std::uint32_t id = expert_ids & 0xff;
+
+        // weight scales
+        const S_element* scales = expert_scales_up + id * 2 * Dims::N;
+        float ws0 = scales[base_row + thread / 4          ];
+        float ws1 = scales[base_row + thread / 4 + Dims::N];
+
+        // wait for currently needed data to come in
+        cuda::pipeline_consumer_wait_prior<0>(pipe);
+        __syncthreads();
+
+        if (is_prefetch_warp<Dims>()) {
+            //prefetch data for down-projection
+            pipe.producer_acquire();
+            moe_request_down_expert_tiny<Dims>(expert_weights_down,
+                expert_scales_down,
+                shmem->expert_ids & 0xff,
+                shm,
+                w_index ^ 1,
+                pipe);
+            pipe.producer_commit();
+        }
+        else {
+            // init accumulators
+            float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
+
+            // run partial scalar products on the tiles
+            for (unsigned base_col = warp * CoreDims::K_TILE, i = 0; i < Dims::HIDDEN_STATES / CoreDims::BLOCK_STRIDE; ++i, base_col += CoreDims::BLOCK_STRIDE) {
+
+                unsigned row = thread / 4;
+                unsigned col = 4 * (thread % 4);
+
+                __nv_fp8x4_e4m3 w0 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 0][rotate_col_32(base_col + col +  0, row)];
+                __nv_fp8x4_e4m3 w1 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 8][rotate_col_32(base_col + col +  0, row)];
+                __nv_fp8x4_e4m3 w2 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 0][rotate_col_32(base_col + col + 16, row)];
+                __nv_fp8x4_e4m3 w3 = *(__nv_fp8x4_e4m3*)&shm->w[w_index].up[row + 8][rotate_col_32(base_col + col + 16, row)];
+
+                __nv_fp8x4_e4m3 a02 = *(__nv_fp8x4_e4m3*)(&shm->a.up[row][rotate_col_32(base_col + col +  0, row)]);
+                __nv_fp8x4_e4m3 a13 = *(__nv_fp8x4_e4m3*)(&shm->a.up[row][rotate_col_32(base_col + col + 16, row)]);
+
+                mma_fp8_fp8(d0, d1, d2, d3, w0, w1, w2, w3, a02, a13, d0, d1, d2, d3);
+            }
+
+            // store partial results
+            shm->partial_result.up[warp][thread +  0] = d0;
+            shm->partial_result.up[warp][thread + 32] = d1;
+            shm->partial_result.up[warp][thread + 64] = d2;
+            shm->partial_result.up[warp][thread + 96] = d3;
+        }
+
+        // make sure all partial results have been written and weights are not needed anymore
+        __syncthreads();
+        w_index ^= 1;
+
+        if (warp < 2) {
+            // filter output
+            std::uint32_t row = (thread % 4) * 2 + warp;
+            bool store_row = (expert_mask >> (row * 8) & 0xff) == id;
+
+            // combine results
+            if (store_row) {
+                moe_up_reduction_tiny<Dims>(
+                    shm->partial_result.up,
+                    ws0, ws1,
+                    shmem->topk_weights[row],
+                    row,
+#ifdef DEBUG_MOE
+                    spec->gemm1,
+#endif
+                    spec->temp);
+            }
+        }
+    }
+
+    return w_index;
+}
+
+
+/**
+ * @brief Forwards to @c moe_up_projection_normal.
+ *
+ * @see moe_up_projection_normal
+ */
+template <typename Dims>
+__device__ void moe_up_projection(
+    const W_element* __restrict__ expert_weights_up,
+    const S_element* __restrict__ expert_scales_up,
+    MoEGemmSpec<Dims>* __restrict__ spec,
+    MoE_SHM<Dims>* __restrict__ shmem)
+{
+    static_assert(Dims::BS > 8, "Tiny is handled by its own kernel. Do not use moe_up_projection for BS<=8");
+    moe_up_projection_normal<Dims>(expert_weights_up, expert_scales_up, spec, shmem);
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_monokernel/src/ptx_utils.h b/csrc/moe/moe_monokernel/src/ptx_utils.h
new file mode 100644
index 000000000..51eb5eba3
--- /dev/null
+++ b/csrc/moe/moe_monokernel/src/ptx_utils.h
@@ -0,0 +1,301 @@
+#ifndef PTX_UTILS_H
+#define PTX_UTILS_H
+
+#pragma once
+
+#include <cuda.h>
+#include <cuda/pipeline>
+#include <cuda_bf16.h>
+#include <cuda_fp8.h>
+
+namespace moe_monokernel {
+
+__device__ static inline void
+mma_fp8_fp8(float& d0, float& d1, float& d2, float& d3,
+            __nv_fp8x4_e4m3 const& a0, __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2, __nv_fp8x4_e4m3 const& a3,
+            __nv_fp8x4_e4m3 const& b02, __nv_fp8x4_e4m3 const& b13,
+            float const& c0, float const& c1, float const& c2, float const& c3)
+{
+#define X2U(x) reinterpret_cast<const unsigned&>(x)
+    asm volatile(
+        "{"
+        ".reg .b16 lo0, lo1, lo2, lo3;\n"
+        ".reg .b16 hi0, hi1, hi2, hi3;\n"
+        ".reg .b16 bh0, bh1, bh2, bh3;\n"
+        ".reg .b32 al0, al1, al2, al3;\n"
+        ".reg .b32 ah0, ah1, ah2, ah3;\n"
+        ".reg .b32 b0, b1, b2, b3;\n"
+        ".reg .b32 t0, t1, t2, t3;\n"
+        "mov.b32 {lo0, hi0}, %4;\n"
+        "mov.b32 {lo1, hi1}, %5;\n"
+        "mov.b32 {lo2, hi2}, %6;\n"
+        "mov.b32 {lo3, hi3}, %7;\n"
+        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
+        "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
+        "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
+        "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
+        "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
+        "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
+        "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
+        "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
+        "mov.b32 {bh0, bh2}, %8;\n"
+        "mov.b32 {bh1, bh3}, %9;\n"
+        "cvt.rn.f16x2.e4m3x2 b0, bh0;\n"
+        "cvt.rn.f16x2.e4m3x2 b1, bh1;\n"
+        "cvt.rn.f16x2.e4m3x2 b2, bh2;\n"
+        "cvt.rn.f16x2.e4m3x2 b3, bh3;\n"
+        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
+        "{t0, t1, t2, t3}, "
+        "{al0, al1, al2, al3}, "
+        "{b0, b1}, "
+        "{%10, %11, %12, %13};\n"
+        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
+        "{%0, %1, %2, %3}, "
+        "{ah0, ah1, ah2, ah3}, "
+        "{b2, b3}, "
+        "{t0, t1, t2, t3};\n"
+        "}\n"
+    :  "=f"(d0),      "=f"(d1),      "=f"(d2),      "=f"(d3)
+    :   "r"(X2U(a0)),  "r"(X2U(a1)),  "r"(X2U(a2)),  "r"(X2U(a3)),
+        "r"(X2U(b02)), "r"(X2U(b13)),
+        "f"(c0),       "f"(c1),       "f"(c2),       "f"(c3)
+    );
+
+#undef X2U
+}
+
+__device__ static inline void
+mma_fp8_f16(float         & d0, float         & d1, float         & d2, float         & d3,
+             __nv_fp8x4_e4m3 const& a0, __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2, __nv_fp8x4_e4m3 const& a3,
+             __half2 const& b0, __half2 const& b1, __half2 const& b2, __half2 const& b3,
+             float const   & c0, float const   & c1, float const   & c2, float const   & c3)
+{
+#define X2U(x) reinterpret_cast<const unsigned&>(x)
+    asm volatile(
+        "{"
+        ".reg .b16 lo0, lo1, lo2, lo3;\n"
+        ".reg .b16 hi0, hi1, hi2, hi3;\n"
+        ".reg .b32 al0, al1, al2, al3;\n"
+        ".reg .b32 ah0, ah1, ah2, ah3;\n"
+        ".reg .b32 t0, t1, t2, t3;\n"
+        "mov.b32 {lo0, hi0}, %4;\n"
+        "mov.b32 {lo1, hi1}, %5;\n"
+        "mov.b32 {lo2, hi2}, %6;\n"
+        "mov.b32 {lo3, hi3}, %7;\n"
+        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
+        "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
+        "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
+        "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
+        "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
+        "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
+        "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
+        "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
+        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
+        "{t0, t1, t2, t3}, "
+        "{al0, al1, al2, al3}, "
+        "{%8, %9}, "
+        "{%12, %13, %14, %15};\n"
+        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
+        "{%0, %1, %2, %3}, "
+        "{ah0, ah1, ah2, ah3}, "
+        "{%10, %11}, "
+        "{t0, t1, t2, t3};\n"
+        "}\n"
+    :  "=f"(d0),      "=f"(d1),      "=f"(d2),      "=f"(d3)
+    :   "r"(X2U(a0)),  "r"(X2U(a1)),  "r"(X2U(a2)),  "r"(X2U(a3)),
+        "r"(X2U(b0)),  "r"(X2U(b1)),  "r"(X2U(b2)),  "r"(X2U(b3)),
+        "f"(c0),       "f"(c1),       "f"(c2),       "f"(c3)
+    );
+
+#undef X2U
+}
+
+__device__ static inline void
+mma_fp8_bf16(float         & d0, float         & d1, float         & d2, float         & d3,
+              __nv_fp8x4_e4m3 const& a0, __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2, __nv_fp8x4_e4m3 const& a3,
+              __nv_bfloat162  const& b0, __nv_bfloat162  const& b1, __nv_bfloat162  const& b2, __nv_bfloat162  const& b3,
+              float const   & c0, float const   & c1, float const   & c2, float const   & c3)
+{
+#define X2U(x) reinterpret_cast<const unsigned&>(x)
+    asm volatile(
+        "{"
+        ".reg .b16 lo0, lo1, lo2, lo3;\n"
+        ".reg .b16 hi0, hi1, hi2, hi3;\n"
+        ".reg .b16 b0, b1, b2, b3;\n"
+        ".reg .b16 b4, b5, b6, b7;\n"
+        ".reg .b16 b8, b9, b10, b11;\n"
+        ".reg .b16 b12, b13, b14, b15;\n"
+        ".reg .b32 al0, al1, al2, al3;\n"
+        ".reg .b32 ah0, ah1, ah2, ah3;\n"
+        ".reg .b32 t0, t1, t2, t3;\n"
+        "mov.b32 {lo0, hi0}, %4;\n"
+        "mov.b32 {lo1, hi1}, %5;\n"
+        "mov.b32 {lo2, hi2}, %6;\n"
+        "mov.b32 {lo3, hi3}, %7;\n"
+        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
+        "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
+        "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
+        "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
+        "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
+        "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
+        "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
+        "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
+        "mov.b32 {b0, b1}, al0;\n"
+        "mov.b32 {b2, b3}, ah0;\n"
+        "mov.b32 {b4, b5}, al1;\n"
+        "mov.b32 {b6, b7}, ah1;\n"
+        "mov.b32 {b8, b9}, al2;\n"
+        "mov.b32 {b10, b11}, ah2;\n"
+        "mov.b32 {b12, b13}, al3;\n"
+        "mov.b32 {b14, b15}, ah3;\n"
+        "cvt.rn.bf16.f16 b0, b0;\n"
+        "cvt.rn.bf16.f16 b1, b1;\n"
+        "cvt.rn.bf16.f16 b2, b2;\n"
+        "cvt.rn.bf16.f16 b3, b3;\n"
+        "cvt.rn.bf16.f16 b4, b4;\n"
+        "cvt.rn.bf16.f16 b5, b5;\n"
+        "cvt.rn.bf16.f16 b6, b6;\n"
+        "cvt.rn.bf16.f16 b7, b7;\n"
+        "cvt.rn.bf16.f16 b8, b8;\n"
+        "cvt.rn.bf16.f16 b9, b9;\n"
+        "cvt.rn.bf16.f16 b10, b10;\n"
+        "cvt.rn.bf16.f16 b11, b11;\n"
+        "cvt.rn.bf16.f16 b12, b12;\n"
+        "cvt.rn.bf16.f16 b13, b13;\n"
+        "cvt.rn.bf16.f16 b14, b14;\n"
+        "cvt.rn.bf16.f16 b15, b15;\n"
+        "mov.b32 al0, {b0, b1};\n"
+        "mov.b32 ah0, {b2, b3};\n"
+        "mov.b32 al1, {b4, b5};\n"
+        "mov.b32 ah1, {b6, b7};\n"
+        "mov.b32 al2, {b8, b9};\n"
+        "mov.b32 ah2, {b10, b11};\n"
+        "mov.b32 al3, {b12, b13};\n"
+        "mov.b32 ah3, {b14, b15};\n"
+        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
+        "{t0, t1, t2, t3}, "
+        "{al0, al1, al2, al3}, "
+        "{%8, %9}, "
+        "{%12, %13, %14, %15};\n"
+        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
+        "{%0, %1, %2, %3}, "
+        "{ah0, ah1, ah2, ah3}, "
+        "{%10, %11}, "
+        "{t0, t1, t2, t3};\n"
+        "}\n"
+    :  "=f"(d0),      "=f"(d1),      "=f"(d2),      "=f"(d3)
+    :   "r"(X2U(a0)),  "r"(X2U(a1)),  "r"(X2U(a2)),  "r"(X2U(a3)),
+        "r"(X2U(b0)),  "r"(X2U(b1)),  "r"(X2U(b2)),  "r"(X2U(b3)),
+        "f"(c0),       "f"(c1),       "f"(c2),       "f"(c3)
+    );
+
+#undef X2U
+}
+
+__device__ static inline void
+mma_fp8_tf32(float         & d0, float         & d1, float         & d2, float         & d3,
+             __nv_fp8x4_e4m3 const& a0, __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2, __nv_fp8x4_e4m3 const& a3,
+             float4  const& b0, float4  const& b1,
+             float const   & c0, float const   & c1, float const   & c2, float const   & c3)
+{
+#define X2U(x) reinterpret_cast<const unsigned&>(x)
+    asm volatile(
+        "{"
+        ".reg .b16 lo0, lo1, lo2, lo3;\n"
+        ".reg .b16 hi0, hi1, hi2, hi3;\n"
+        ".reg .b16 h0, h1, h2, h3;\n"
+        ".reg .b16 h4, h5, h6, h7;\n"
+        ".reg .b16 h8, h9, h10, h11;\n"
+        ".reg .b16 h12, h13, h14, h15;\n"
+        ".reg .b32 w0, w1, w2, w3;\n"
+        ".reg .b32 w4, w5, w6, w7;\n"
+        ".reg .b32 w8, w9, w10, w11;\n"
+        ".reg .b32 w12, w13, w14, w15;\n"
+        ".reg .b32 al0, al1, al2, al3;\n"
+        ".reg .b32 ah0, ah1, ah2, ah3;\n"
+        ".reg .b32 t0, t1, t2, t3;\n"
+        "mov.b32 {lo0, hi0}, %4;\n"
+        "mov.b32 {lo1, hi1}, %5;\n"
+        "mov.b32 {lo2, hi2}, %6;\n"
+        "mov.b32 {lo3, hi3}, %7;\n"
+        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
+        "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
+        "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
+        "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
+        "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
+        "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
+        "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
+        "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
+        "mov.b32 {h0, h1}, al0;\n"
+        "mov.b32 {h2, h3}, ah0;\n"
+        "mov.b32 {h4, h5}, al1;\n"
+        "mov.b32 {h6, h7}, ah1;\n"
+        "mov.b32 {h8, h9}, al2;\n"
+        "mov.b32 {h10, h11}, ah2;\n"
+        "mov.b32 {h12, h13}, al3;\n"
+        "mov.b32 {h14, h15}, ah3;\n"
+        "cvt.f32.f16 w0, h0;\n"
+        "cvt.f32.f16 w1, h1;\n"
+        "cvt.f32.f16 w2, h2;\n"
+        "cvt.f32.f16 w3, h3;\n"
+        "cvt.f32.f16 w4, h4;\n"
+        "cvt.f32.f16 w5, h5;\n"
+        "cvt.f32.f16 w6, h6;\n"
+        "cvt.f32.f16 w7, h7;\n"
+        "cvt.f32.f16 w8, h8;\n"
+        "cvt.f32.f16 w9, h9;\n"
+        "cvt.f32.f16 w10, h10;\n"
+        "cvt.f32.f16 w11, h11;\n"
+        "cvt.f32.f16 w12, h12;\n"
+        "cvt.f32.f16 w13, h13;\n"
+        "cvt.f32.f16 w14, h14;\n"
+        "cvt.f32.f16 w15, h15;\n"
+        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
+        "{t0, t1, t2, t3}, "
+        "{w0, w4, w8, w12}, "
+        "{%8, %12}, "
+        "{%16, %17, %18, %19};\n"
+        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
+        "{t0, t1, t2, t3}, "
+        "{w1, w5, w9, w13}, "
+        "{%9, %13}, "
+        "{t0, t1, t2, t3};\n"
+        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
+        "{t0, t1, t2, t3}, "
+        "{w2, w6, w10, w14}, "
+        "{%10, %14}, "
+        "{t0, t1, t2, t3};\n"
+        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
+        "{%0, %1, %2, %3}, "
+        "{w3, w7, w11, w15}, "
+        "{%11, %15}, "
+        "{t0, t1, t2, t3};\n"
+        "}\n"
+    :  "=f"(d0),       "=f"(d1),       "=f"(d2),       "=f"(d3)
+    :   "r"(X2U(a0)),   "r"(X2U(a1)),   "r"(X2U(a2)),   "r"(X2U(a3)),
+        "r"(X2U(b0.x)), "r"(X2U(b0.y)), "r"(X2U(b0.z)), "r"(X2U(b0.w)),
+        "r"(X2U(b1.x)), "r"(X2U(b1.y)), "r"(X2U(b1.z)), "r"(X2U(b1.w)),
+        "f"(c0),        "f"(c1),        "f"(c2),        "f"(c3)
+    );
+
+#undef X2U
+}
+
+template<typename Target, typename Source>
+__device__ static inline
+void copy128(Target& dest, const Source& source, cuda::pipeline<cuda::thread_scope_thread>& pipeline)
+{
+    const auto shape4 = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
+    cuda::memcpy_async(&dest, &source, shape4, pipeline);
+}
+
+__device__ inline std::uint32_t rotate_col_32(std::uint32_t col, std::uint32_t row)
+{
+    std::uint32_t col_base = col & 0xff9f;
+    std::uint32_t col_rot = (col + 0x20 * row) & 0x60;
+    return col_base | col_rot;
+}
+
+} // namespace moe_monokernel
+
+#endif
diff --git a/csrc/moe/moe_ops.h b/csrc/moe/moe_ops.h
index 661730c96..c727c07b9 100644
--- a/csrc/moe/moe_ops.h
+++ b/csrc/moe/moe_ops.h
@@ -28,4 +28,61 @@ bool moe_permute_unpermute_supported();
 
 void shuffle_rows(const torch::Tensor& input_tensor,
                   const torch::Tensor& dst2src_map,
-                  torch::Tensor& output_tensor);
\ No newline at end of file
+                  torch::Tensor& output_tensor);
+
+#ifndef USE_ROCM
+void moe_monokernel_BS8_E16_TP8_impl(
+    const torch::Tensor& activations_in,
+    const torch::Tensor& router_logits,
+    const torch::Tensor& expert_weights_up,
+    const torch::Tensor& expert_scales_up,
+    const torch::Tensor& expert_weights_down,
+    const torch::Tensor& expert_scales_down,
+    torch::Tensor& activations_out,
+    torch::Tensor& gemmspec);
+void moe_monokernel_BS8_E16_TP4_impl(
+    const torch::Tensor& activations_in,
+    const torch::Tensor& router_logits,
+    const torch::Tensor& expert_weights_up,
+    const torch::Tensor& expert_scales_up,
+    const torch::Tensor& expert_weights_down,
+    const torch::Tensor& expert_scales_down,
+    torch::Tensor& activations_out,
+    torch::Tensor& gemmspec);
+void moe_monokernel_BS64_E16_TP8_impl(
+    const torch::Tensor& activations_in,
+    const torch::Tensor& router_logits,
+    const torch::Tensor& expert_weights_up,
+    const torch::Tensor& expert_scales_up,
+    const torch::Tensor& expert_weights_down,
+    const torch::Tensor& expert_scales_down,
+    torch::Tensor& activations_out,
+    torch::Tensor& gemmspec);
+void moe_monokernel_BS64_E16_TP4_impl(
+    const torch::Tensor& activations_in,
+    const torch::Tensor& router_logits,
+    const torch::Tensor& expert_weights_up,
+    const torch::Tensor& expert_scales_up,
+    const torch::Tensor& expert_weights_down,
+    const torch::Tensor& expert_scales_down,
+    torch::Tensor& activations_out,
+    torch::Tensor& gemmspec);
+void moe_monokernel_BS8_E128_TP8_impl(
+    const torch::Tensor& activations_in,
+    const torch::Tensor& router_logits,
+    const torch::Tensor& expert_weights_up,
+    const torch::Tensor& expert_scales_up,
+    const torch::Tensor& expert_weights_down,
+    const torch::Tensor& expert_scales_down,
+    torch::Tensor& activations_out,
+    torch::Tensor& gemmspec);
+void moe_monokernel_BS64_E128_TP8_impl(
+    const torch::Tensor& activations_in,
+    const torch::Tensor& router_logits,
+    const torch::Tensor& expert_weights_up,
+    const torch::Tensor& expert_scales_up,
+    const torch::Tensor& expert_weights_down,
+    const torch::Tensor& expert_scales_down,
+    torch::Tensor& activations_out,
+    torch::Tensor& gemmspec);
+#endif
diff --git a/csrc/moe/torch_bindings.cpp b/csrc/moe/torch_bindings.cpp
index 7e49f68f6..918df3995 100644
--- a/csrc/moe/torch_bindings.cpp
+++ b/csrc/moe/torch_bindings.cpp
@@ -22,6 +22,35 @@ TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, m) {
       "                     Tensor! num_tokens_post_pad) -> ()");
   m.impl("moe_align_block_size", torch::kCUDA, &moe_align_block_size);
 
+#ifndef USE_ROCM
+  // Moe Monokernel for Llama4 Scout - TP8
+  m.def("moe_monokernel_BS8_E16_TP8(Tensor activations_in,"
+  "Tensor router_logits,"
+  "Tensor expert_weights_up, Tensor expert_scales_up,"
+  "Tensor expert_weights_down, Tensor expert_scales_down,"
+  "Tensor! activations_out, Tensor! scratchpad) -> ()");
+  m.impl("moe_monokernel_BS8_E16_TP8", torch::kCUDA, &moe_monokernel_BS8_E16_TP8_impl);
+  m.def("moe_monokernel_BS64_E16_TP8(Tensor activations_in,"
+  "Tensor router_logits,"
+  "Tensor expert_weights_up, Tensor expert_scales_up,"
+  "Tensor expert_weights_down, Tensor expert_scales_down,"
+  "Tensor! activations_out, Tensor! scratchpad) -> ()");
+  m.impl("moe_monokernel_BS64_E16_TP8", torch::kCUDA, &moe_monokernel_BS64_E16_TP8_impl);
+  // Moe Monokernel for Llama4 Maverick - TP8
+  m.def("moe_monokernel_BS8_E128_TP8(Tensor activations_in,"
+  "Tensor router_logits,"
+  "Tensor expert_weights_up, Tensor expert_scales_up,"
+  "Tensor expert_weights_down, Tensor expert_scales_down,"
+  "Tensor! activations_out, Tensor! scratchpad) -> ()");
+  m.impl("moe_monokernel_BS8_E128_TP8", torch::kCUDA, &moe_monokernel_BS8_E128_TP8_impl);
+  m.def("moe_monokernel_BS64_E128_TP8(Tensor activations_in,"
+  "Tensor router_logits,"
+  "Tensor expert_weights_up, Tensor expert_scales_up,"
+  "Tensor expert_weights_down, Tensor expert_scales_down,"
+  "Tensor! activations_out, Tensor! scratchpad) -> ()");
+  m.impl("moe_monokernel_BS64_E128_TP8", torch::kCUDA, &moe_monokernel_BS64_E128_TP8_impl);
+#endif
+
 #ifndef USE_ROCM
   m.def(
       "moe_wna16_gemm(Tensor input, Tensor! output, Tensor b_qweight, "
diff --git a/vllm/_custom_ops.py b/vllm/_custom_ops.py
index 0d556053f..92179aca8 100644
--- a/vllm/_custom_ops.py
+++ b/vllm/_custom_ops.py
@@ -97,6 +97,94 @@ def paged_attention_v2(
         blocksparse_block_size, blocksparse_head_sliding_step)
 
 
+def moe_monokernel(activations_in: torch.Tensor,
+                   router_logits: torch.Tensor,
+                   expert_weights_up: torch.Tensor,
+                   expert_scales_up: torch.Tensor,
+                   expert_weights_down: torch.Tensor,
+                   expert_scales_down: torch.Tensor,
+                   scratchpad: torch.Tensor
+) -> torch.Tensor:
+    if not current_platform.is_cuda():
+        raise NotImplementedError(
+            "The optimized moe kernel is only available on CUDA platforms")
+    
+    # The following asserts check all the assumptions of the MoE monokernel.
+    # They are only evaluated at graph creation time and do not cause a runtime overhead (except in eager mode)
+
+    # Ensure appropriate dimensions
+    assert activations_in.dim() == 2
+    assert router_logits.dim() == 2
+    assert expert_weights_up.dim() == 3
+    assert expert_scales_up.dim() == 3
+    assert expert_weights_down.dim() == 3
+    assert expert_scales_down.dim() == 3
+
+    # Ensure non-padded tensors (avoid dealing with strides)
+    assert activations_in.is_contiguous()
+    assert router_logits.is_contiguous()
+    assert expert_weights_up.is_contiguous()
+    assert expert_scales_up.is_contiguous()
+    assert expert_weights_down.is_contiguous()
+    assert expert_scales_down.is_contiguous()
+
+    # Ensure proper sizes
+    TP = torch.distributed.get_world_size()
+    E, M, N, K = router_logits.size(1), activations_in.size(0), 2*8192//TP, 5120
+    assert router_logits.size() == (M, E), f"size is: {router_logits.size()}"
+    assert expert_weights_up.size() == (E, N, K), f"size is: {expert_weights_up.size()}"
+    assert expert_scales_up.size() == (E, N, 1), f"size is: {expert_scales_up.size()}"
+    assert expert_weights_down.size() == (E, K, N//2), f"size is: {expert_weights_down.size()}"
+    assert expert_scales_down.size() == (E, K, 1), f"size is: {expert_scales_down.size()}"
+
+    # Ensure proper types
+    assert activations_in.dtype is torch.bfloat16, f"type of x is: {activations_in.type()}"
+    assert router_logits.dtype is torch.bfloat16, f"type of x is: {router_logits.type()}"
+    assert expert_weights_up.dtype is torch.float8_e4m3fn, f"type of x is: {expert_weights_up.type()}"
+    assert expert_scales_up.dtype is torch.float32, f"type of x is: {expert_scales_up.type()}"
+    assert expert_weights_down.dtype is torch.float8_e4m3fn, f"type of x is: {expert_weights_down.type()}"
+    assert expert_scales_down.dtype is torch.float32, f"type of x is: {expert_scales_down.type()}"
+    
+    # Moe_monokernel is called "in-line" by default, i.e. activations_in are replaced with the result. No tensor allocation needed.
+
+    # Dispatch to the appropriate kernel call wrapper
+    # We do the dispatch in Python and not C++, so that the PyTorch Inductor sees different function calls.
+    assert M <= 64
+    assert TP == 8
+    if E == 16:
+        if M <= 8:
+            torch.ops._moe_C.moe_monokernel_BS8_E16_TP8(activations_in, router_logits, expert_weights_up, expert_scales_up,
+                                                        expert_weights_down, expert_scales_down, activations_in, scratchpad)
+        else:
+            torch.ops._moe_C.moe_monokernel_BS64_E16_TP8(activations_in, router_logits, expert_weights_up, expert_scales_up,
+                                                         expert_weights_down, expert_scales_down, activations_in, scratchpad)
+    elif E == 128:
+        if M <= 8:
+            torch.ops._moe_C.moe_monokernel_BS8_E128_TP8(activations_in, router_logits, expert_weights_up, expert_scales_up,
+                                                         expert_weights_down, expert_scales_down, activations_in, scratchpad)
+        else:
+            torch.ops._moe_C.moe_monokernel_BS64_E128_TP8(activations_in, router_logits, expert_weights_up, expert_scales_up,
+                                                          expert_weights_down, expert_scales_down, activations_in, scratchpad)
+    return activations_in
+
+def moe_monokernel_fake(activations_in: torch.Tensor,
+                        router_logits: torch.Tensor,
+                        expert_weights_up: torch.Tensor,
+                        expert_scales_up: torch.Tensor,
+                        expert_weights_down: torch.Tensor,
+                        expert_scales_down: torch.Tensor,
+                        gemmspec: torch.Tensor
+) -> torch.Tensor:
+    return activations_in
+
+from vllm.utils import direct_register_custom_op
+direct_register_custom_op(
+    op_name="moe_monokernel",
+    op_func=moe_monokernel,
+    mutates_args=[],
+    fake_impl=moe_monokernel_fake,
+)
+
 def paged_attention_rocm(
     out: torch.Tensor,
     exp_sum: torch.Tensor,
diff --git a/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py b/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
index 8ca8249e6..a55032820 100644
--- a/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
+++ b/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py
@@ -490,10 +490,21 @@ class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):
         # cutlass path
         self.is_fp8_w8a8_sm100 = quant_config._is_fp8_w8a8_sm100(
             self.weight_quant, self.input_quant)
-        self.use_cutlass = (quant_config._is_fp8_w8a8_sm90(
-            self.weight_quant, self.input_quant) or self.is_fp8_w8a8_sm100)
+        #self.use_cutlass = (quant_config._is_fp8_w8a8_sm90(
+        #    self.weight_quant, self.input_quant) or self.is_fp8_w8a8_sm100)
+        self.use_cutlass = False
         self.disable_expert_map = False
 
+        # Scratchpad for MoE monokernel.
+        # (more than) enough space for 
+        # BS+8 x N             fp32
+        # BS                   fp32
+        # BS   x HIDDEN_STATES fp16
+        # with BS = 1024:  4MB + <1MB + 10MB < 4M x 4byte
+        self.moe_monokernel_scratchpad = torch.empty((1024, 4096),
+                                                     dtype=torch.float32,
+                                                     device=f"cuda:{torch.distributed.get_rank()}")
+
     def create_weights(self, layer: torch.nn.Module, num_experts: int,
                        hidden_size: int, intermediate_size_per_partition: int,
                        params_dtype: torch.dtype, **extra_weight_attrs):
@@ -764,6 +775,24 @@ class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):
             raise NotImplementedError(
                 "EPLB not supported for "
                 "`CompressedTensorsW8A8Fp8MoEMethod` yet.")
+        
+        # MoE monokernel for BS <= 64
+        # Supports Llama4 Maverick & Scout with TP = 8
+        E, M, N, K = global_num_experts, x.size(0), layer.w13_weight.size(1), x.size(1)
+        if (E == 16 or E == 128) and M <= 64 and K == 5120 and N == 2048:
+            # Migrate scratchpad to appropriate device
+            if self.moe_monokernel_scratchpad.device != x.device:
+                self.moe_monokernel_scratchpad = self.moe_monokernel_scratchpad.to(x.device)
+            return torch.ops.vllm.moe_monokernel(
+                x,
+                router_logits,
+                layer.w13_weight,
+                layer.w13_weight_scale,
+                layer.w2_weight,
+                layer.w2_weight_scale,
+                self.moe_monokernel_scratchpad
+            )
+
 
         topk_weights, topk_ids = FusedMoE.select_experts(
             hidden_states=x,
-- 
2.51.0
```