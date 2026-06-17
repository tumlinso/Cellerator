#pragma once

namespace cellerator::quantized::extreme_backend {

__device__ __forceinline__ float ptx_mul_f32(float lhs, float rhs) {
    float out = 0.0f;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(lhs), "f"(rhs));
    return out;
}

__device__ __forceinline__ float ptx_sub_f32(float lhs, float rhs) {
    float out = 0.0f;
    asm("sub.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(lhs), "f"(rhs));
    return out;
}

__device__ __forceinline__ float ptx_fma_f32(float a, float b, float c) {
    float out = 0.0f;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__device__ __forceinline__ float ptx_rcp_nr1_f32(float den) {
    float recip = 0.0f;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(recip) : "f"(den));
    const float err = ptx_sub_f32(1.0f, ptx_mul_f32(den, recip));
    return ptx_fma_f32(recip, err, recip);
}

__device__ __forceinline__ int ptx_round_rni_s32(float value) {
    int out = 0;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(out) : "f"(value));
    return out;
}

__device__ __forceinline__ float ptx_cvt_f32_u32(unsigned int value) {
    float out = 0.0f;
    asm("cvt.rn.f32.u32 %0, %1;" : "=f"(out) : "r"(value));
    return out;
}

__device__ __forceinline__ unsigned int ptx_clamp_code(unsigned int max_code, int rounded) {
    int lower_bounded = 0;
    int upper_bounded = 0;

    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.lt.s32 p, %1, 0;\n\t"
        "selp.s32 %0, 0, %1, p;\n\t"
        "}"
        : "=r"(lower_bounded)
        : "r"(rounded));

    asm(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.gt.s32 p, %1, %2;\n\t"
        "selp.s32 %0, %2, %1, p;\n\t"
        "}"
        : "=r"(upper_bounded)
        : "r"(lower_bounded), "r"(static_cast<int>(max_code)));

    return static_cast<unsigned int>(upper_bounded);
}

} // namespace cellerator::quantized::extreme_backend
