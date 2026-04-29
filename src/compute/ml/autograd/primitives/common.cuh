#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace cellerator::compute::autograd::primitives {

__device__ inline float load_f16_as_f32(const __half *values, std::uint32_t idx) {
    return __half2float(values[idx]);
}

__device__ inline float dot_f32(const float *lhs, const float *rhs, std::int64_t count) {
    float accum = 0.0f;
    for (std::int64_t i = 0; i < count; ++i) accum += lhs[i] * rhs[i];
    return accum;
}

} // namespace cellerator::compute::autograd::primitives
