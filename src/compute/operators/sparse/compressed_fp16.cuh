#pragma once

#include <Cellerator/tensor/base/compressed.hh>

#include <cuda_fp16.h>

namespace cellerator::compute::sparse::detail {

// Both CUDA half and external binary16 buffers use the same two storage bytes.
// Access the representation without aliasing either scalar object type.
__device__ __forceinline__ __half load_fp16(const void *values, uint64_t i) {
    const auto *bytes = static_cast<const unsigned char *>(__builtin_assume_aligned(values, alignof(uint16_t)));
    uint16_t bits;
    __builtin_memcpy(&bits, bytes + i * sizeof(bits), sizeof(bits));
    return __ushort_as_half(bits);
}

__device__ __forceinline__ void store_fp16(void *values, uint64_t i, __half value) {
    auto *bytes = static_cast<unsigned char *>(__builtin_assume_aligned(values, alignof(uint16_t)));
    const uint16_t bits = __half_as_ushort(value);
    __builtin_memcpy(bytes + i * sizeof(bits), &bits, sizeof(bits));
}

template<class Index, class Value>
bool valid_csr_fp16(const tensor::base::CompressedView<Index, Value> &matrix) {
    const auto &s = matrix.structure;
    constexpr uint64_t index_max = Index(-1) > Index(0) ? uint64_t(Index(-1))
        : sizeof(Index) == sizeof(int64_t) ? uint64_t(INT64_MAX) : uint64_t(INT32_MAX);
    // The row-count kernel uses a 32-bit grid-stride counter with at most
    // 4096 blocks of 256 threads. Leave room for its last increment.
    constexpr uint64_t row_max = UINT32_MAX - (4096u * 256u - 1u);
    if (s.axis != tensor::base::CompressedAxis::row || s.rows > row_max
        || s.rows > index_max || s.cols > index_max || s.cols > UINT32_MAX
        || s.nnz > index_max || s.nnz > SIZE_MAX / sizeof(Index)
        || s.major_ptr == nullptr || uintptr_t(s.major_ptr) % alignof(Index) != 0) return false;
    return s.nnz == 0 || (s.rows != 0 && s.cols != 0 && s.minor_idx != nullptr
        && matrix.values != nullptr && uintptr_t(s.minor_idx) % alignof(Index) == 0
        && uintptr_t(matrix.values) % alignof(uint16_t) == 0);
}

} // namespace cellerator::compute::sparse::detail
