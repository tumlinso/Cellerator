#include "residual.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {
namespace {

__global__ void row_owned_residual_kernel_v1(
    const std::uint32_t *row_offsets,
    std::uint32_t row_count,
    const std::uint32_t *column_indices,
    std::uint64_t edge_count,
    const __half *edge_values,
    const __half *dense_rhs,
    std::uint32_t source_count,
    std::uint32_t dense_width,
    float *accumulation) {
    const std::uint64_t output_index = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    const std::uint64_t output_count =
        static_cast<std::uint64_t>(row_count) * dense_width;
    if (output_index >= output_count) return;
    const std::uint32_t row = static_cast<std::uint32_t>(
        output_index / dense_width);
    const std::uint32_t dense_column = static_cast<std::uint32_t>(
        output_index % dense_width);
    const std::uint32_t begin = row_offsets[row];
    const std::uint32_t end = row_offsets[row + 1u];
    if (begin > end || end > edge_count) return;

    float sum = 0.0f;
    for (std::uint32_t edge = begin; edge < end; ++edge) {
        const std::uint32_t source = column_indices[edge];
        if (source >= source_count) return;
        sum += __half2float(edge_values[edge]) * __half2float(
            dense_rhs[static_cast<std::size_t>(source) * dense_width
                + dense_column]);
    }
    accumulation[output_index] += sum;
}

} // namespace

residual_apply_status_v1 enqueue_row_owned_residual_v1(
    const residual_apply_request_v1 &request) noexcept {
    if (request.row_offsets == nullptr || request.row_count == 0u
        || request.column_indices == nullptr || request.edge_count == 0u
        || request.edge_count > std::numeric_limits<std::uint32_t>::max()
        || request.edge_values == nullptr || request.dense_rhs == nullptr
        || request.source_count == 0u || request.dense_width == 0u
        || request.accumulation == nullptr)
        return residual_apply_status_v1::invalid_argument;
    const std::uint64_t output_count =
        static_cast<std::uint64_t>(request.row_count) * request.dense_width;
    if (output_count > std::numeric_limits<std::uint32_t>::max())
        return residual_apply_status_v1::invalid_argument;
    constexpr std::uint32_t block_size = 256u;
    const std::uint32_t grid_size = static_cast<std::uint32_t>(
        (output_count + block_size - 1u) / block_size);
    row_owned_residual_kernel_v1<<<grid_size, block_size, 0u, request.stream>>>(
        request.row_offsets, request.row_count, request.column_indices,
        request.edge_count, request.edge_values, request.dense_rhs,
        request.source_count, request.dense_width, request.accumulation);
    return cudaGetLastError() == cudaSuccess
        ? residual_apply_status_v1::success
        : residual_apply_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
