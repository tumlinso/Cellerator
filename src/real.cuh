#pragma once

#include <cstddef>

#include <cuda_fp16.h>

namespace matrix {

typedef __half Real;

__host__ __device__ __forceinline__ float real_to_float(Real value) {
    return __half2float(value);
}

__host__ __device__ __forceinline__ Real real_from_float(float value) {
    return __float2half(value);
}

template<typename ValueT>
__host__ __device__ __forceinline__ void require_fp_storage() {
    static_assert(sizeof(ValueT) == 2 || sizeof(ValueT) == 4,
                  "matrix values must be 16-bit or 32-bit floating storage");
}

template<typename ValueT>
__host__ __device__ __forceinline__ std::size_t value_bytes() {
    require_fp_storage<ValueT>();
    return sizeof(ValueT);
}

} // namespace matrix
