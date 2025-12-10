#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#pragma once

#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace moe_monokernel {

/**
 * @brief FP8 to FP8 matrix multiply-accumulate (m16n8k32)
 *
 * Performs two chained m16n8k16 MMAs for K=32 with FP8 E4M3 inputs.
 * Converts FP8 to FP16 in registers before the MMA.
 */
__device__ static inline void
mma_fp8_fp8(float& d0, float& d1, float& d2, float& d3,
            __nv_fp8x4_e4m3 const& a0, __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2, __nv_fp8x4_e4m3 const& a3,
            __nv_fp8x4_e4m3 const& b02, __nv_fp8x4_e4m3 const& b13,
            float const& c0, float const& c1, float const& c2, float const& c3)
{
#define X2U(x) reinterpret_cast<const unsigned&>(x)
    asm volatile(
        "{"
        ".reg .b16 lo0, lo1, lo2, lo3;\n"
        ".reg .b16 hi0, hi1, hi2, hi3;\n"
        ".reg .b16 bh0, bh1, bh2, bh3;\n"
        ".reg .b32 al0, al1, al2, al3;\n"
        ".reg .b32 ah0, ah1, ah2, ah3;\n"
        ".reg .b32 b0, b1, b2, b3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {lo0, hi0}, %4;\n"
        "mov.b32 {lo1, hi1}, %5;\n"
        "mov.b32 {lo2, hi2}, %6;\n"
        "mov.b32 {lo3, hi3}, %7;\n"
        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
        "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
        "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
        "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
        "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
        "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
        "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
        "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
        "mov.b32 {bh0, bh2}, %8;\n"
        "mov.b32 {bh1, bh3}, %9;\n"
        "cvt.rn.f16x2.e4m3x2 b0, bh0;\n"
        "cvt.rn.f16x2.e4m3x2 b1, bh1;\n"
        "cvt.rn.f16x2.e4m3x2 b2, bh2;\n"
        "cvt.rn.f16x2.e4m3x2 b3, bh3;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{t0, t1, t2, t3}, "
        "{al0, al1, al2, al3}, "
        "{b0, b1}, "
        "{%10, %11, %12, %13};\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{ah0, ah1, ah2, ah3}, "
        "{b2, b3}, "
        "{t0, t1, t2, t3};\n"
        "}\n"
    :  "=f"(d0),      "=f"(d1),      "=f"(d2),      "=f"(d3)
    :   "r"(X2U(a0)),  "r"(X2U(a1)),  "r"(X2U(a2)),  "r"(X2U(a3)),
        "r"(X2U(b02)), "r"(X2U(b13)),
        "f"(c0),       "f"(c1),       "f"(c2),       "f"(c3)
    );

#undef X2U
}

/**
 * @brief FP8 to TF32 matrix multiply-accumulate for down-projection
 *
 * Converts FP8 activations to TF32 and multiplies with TF32 weights.
 */
__device__ static inline void
mma_fp8_tf32(float         & d0, float         & d1, float         & d2, float         & d3,
             __nv_fp8x4_e4m3 const& a0, __nv_fp8x4_e4m3 const& a1, __nv_fp8x4_e4m3 const& a2, __nv_fp8x4_e4m3 const& a3,
             float4  const& b0, float4  const& b1,
             float const   & c0, float const   & c1, float const   & c2, float const   & c3)
{
#define X2U(x) reinterpret_cast<const unsigned&>(x)
    asm volatile(
        "{"
        ".reg .b16 lo0, lo1, lo2, lo3;\n"
        ".reg .b16 hi0, hi1, hi2, hi3;\n"
        ".reg .b16 h0, h1, h2, h3;\n"
        ".reg .b16 h4, h5, h6, h7;\n"
        ".reg .b16 h8, h9, h10, h11;\n"
        ".reg .b16 h12, h13, h14, h15;\n"
        ".reg .b32 w0, w1, w2, w3;\n"
        ".reg .b32 w4, w5, w6, w7;\n"
        ".reg .b32 w8, w9, w10, w11;\n"
        ".reg .b32 w12, w13, w14, w15;\n"
        ".reg .b32 al0, al1, al2, al3;\n"
        ".reg .b32 ah0, ah1, ah2, ah3;\n"
        ".reg .b32 t0, t1, t2, t3;\n"
        "mov.b32 {lo0, hi0}, %4;\n"
        "mov.b32 {lo1, hi1}, %5;\n"
        "mov.b32 {lo2, hi2}, %6;\n"
        "mov.b32 {lo3, hi3}, %7;\n"
        "cvt.rn.f16x2.e4m3x2 al0, lo0;\n"
        "cvt.rn.f16x2.e4m3x2 ah0, hi0;\n"
        "cvt.rn.f16x2.e4m3x2 al1, lo1;\n"
        "cvt.rn.f16x2.e4m3x2 ah1, hi1;\n"
        "cvt.rn.f16x2.e4m3x2 al2, lo2;\n"
        "cvt.rn.f16x2.e4m3x2 ah2, hi2;\n"
        "cvt.rn.f16x2.e4m3x2 al3, lo3;\n"
        "cvt.rn.f16x2.e4m3x2 ah3, hi3;\n"
        "mov.b32 {h0, h1}, al0;\n"
        "mov.b32 {h2, h3}, ah0;\n"
        "mov.b32 {h4, h5}, al1;\n"
        "mov.b32 {h6, h7}, ah1;\n"
        "mov.b32 {h8, h9}, al2;\n"
        "mov.b32 {h10, h11}, ah2;\n"
        "mov.b32 {h12, h13}, al3;\n"
        "mov.b32 {h14, h15}, ah3;\n"
        "cvt.f32.f16 w0, h0;\n"
        "cvt.f32.f16 w1, h1;\n"
        "cvt.f32.f16 w2, h2;\n"
        "cvt.f32.f16 w3, h3;\n"
        "cvt.f32.f16 w4, h4;\n"
        "cvt.f32.f16 w5, h5;\n"
        "cvt.f32.f16 w6, h6;\n"
        "cvt.f32.f16 w7, h7;\n"
        "cvt.f32.f16 w8, h8;\n"
        "cvt.f32.f16 w9, h9;\n"
        "cvt.f32.f16 w10, h10;\n"
        "cvt.f32.f16 w11, h11;\n"
        "cvt.f32.f16 w12, h12;\n"
        "cvt.f32.f16 w13, h13;\n"
        "cvt.f32.f16 w14, h14;\n"
        "cvt.f32.f16 w15, h15;\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{t0, t1, t2, t3}, "
        "{w0, w4, w8, w12}, "
        "{%8, %12}, "
        "{%16, %17, %18, %19};\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{t0, t1, t2, t3}, "
        "{w1, w5, w9, w13}, "
        "{%9, %13}, "
        "{t0, t1, t2, t3};\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{t0, t1, t2, t3}, "
        "{w2, w6, w10, w14}, "
        "{%10, %14}, "
        "{t0, t1, t2, t3};\n"
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0, %1, %2, %3}, "
        "{w3, w7, w11, w15}, "
        "{%11, %15}, "
        "{t0, t1, t2, t3};\n"
        "}\n"
    :  "=f"(d0),       "=f"(d1),       "=f"(d2),       "=f"(d3)
    :   "r"(X2U(a0)),   "r"(X2U(a1)),   "r"(X2U(a2)),   "r"(X2U(a3)),
        "r"(X2U(b0.x)), "r"(X2U(b0.y)), "r"(X2U(b0.z)), "r"(X2U(b0.w)),
        "r"(X2U(b1.x)), "r"(X2U(b1.y)), "r"(X2U(b1.z)), "r"(X2U(b1.w)),
        "f"(c0),        "f"(c1),        "f"(c2),        "f"(c3)
    );

#undef X2U
}

/**
 * @brief Asynchronous 128-bit (16-byte) copy from global to shared memory
 *
 * Uses cuda::memcpy_async for efficient asynchronous memory transfer.
 * IMPORTANT: Both source and destination addresses MUST be 16-byte aligned!
 *
 * The alignment requirement is enforced by the calling code:
 * - moe_request_input_tokens: col = thread * 16 (aligned)
 * - moe_request_up_expert: col_start = multiple of 128 (aligned)
 * - moe_request_down_activations: col = vec * 4 floats = 16 bytes (aligned)
 * - moe_request_down_expert: col = vec * 16 (aligned)
 */
template<typename Target, typename Source>
__device__ static inline
void copy128(Target& dest, const Source& source, cuda::pipeline<cuda::thread_scope_thread>& pipeline)
{
#ifdef MOE_MONOKERNEL_DEBUG
    // Debug mode: verify 16-byte alignment
    assert((reinterpret_cast<uintptr_t>(&dest) & 0xF) == 0 && "copy128: dest not 16-byte aligned");
    assert((reinterpret_cast<uintptr_t>(&source) & 0xF) == 0 && "copy128: source not 16-byte aligned");
#endif

    // Use async copy with 16-byte alignment
    const auto shape16 = cuda::aligned_size_t<16>(16);
    cuda::memcpy_async(&dest, &source, shape16, pipeline);
}

/**
 * @brief 32-byte column swizzling for bank conflict mitigation
 *
 * Rotates column index based on row to map same column to different banks.
 */
__device__ inline std::uint32_t rotate_col_32(std::uint32_t col, std::uint32_t row)
{
    std::uint32_t col_base = col & 0xff9f;
    std::uint32_t col_rot = (col + 0x20 * row) & 0x60;
    return col_base | col_rot;
}

} // namespace moe_monokernel

#endif
