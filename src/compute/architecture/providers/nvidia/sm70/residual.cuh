#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class residual_apply_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    cuda_failure = 2u
};

// Row-owned CSR in the projection's persistent destination order. The kernel
// adds exact residual contributions to caller-owned FP32 accumulation; it does
// not allocate, synchronize, canonicalize, or apply the operation epilogue.
struct residual_apply_request_v1 {
    const std::uint32_t *row_offsets = nullptr;
    std::uint32_t row_count = 0u;
    const std::uint32_t *column_indices = nullptr;
    std::uint64_t edge_count = 0u;
    const __half *edge_values = nullptr;
    const __half *dense_rhs = nullptr;
    std::uint32_t source_count = 0u;
    std::uint32_t dense_width = 0u;
    float *accumulation = nullptr;
    cudaStream_t stream = nullptr;
};

residual_apply_status_v1 enqueue_row_owned_residual_v1(
    const residual_apply_request_v1 &request) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
