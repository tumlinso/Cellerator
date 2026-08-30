#include "transpose_cover.cc"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class transpose_relation_apply_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    cuda_failure = 2u
};

// The cover and value plane are immutable-address prepared inputs. Each cover
// record has already been validated against its logical edge before upload.
// One thread owns one dX element and accumulates every matching MMA or residual
// contribution exactly once, so this correctness path needs no atomics.
struct transpose_relation_apply_request_v1 {
    const target_edge_placement_v1 *transpose_cover = nullptr;
    const __half *logical_edge_values = nullptr;
    std::uint64_t logical_edge_count = 0u;
    const __half *destination_gradient = nullptr;
    std::uint32_t destination_count = 0u;
    std::uint32_t source_count = 0u;
    std::uint32_t dense_width = 0u;
    float *source_gradient = nullptr;
    cudaStream_t stream = nullptr;
};

namespace {

__global__ void transpose_relation_apply_kernel_v1(
    const target_edge_placement_v1 *transpose_cover,
    const __half *logical_edge_values,
    std::uint32_t logical_edge_count,
    const __half *destination_gradient,
    std::uint32_t destination_count,
    std::uint32_t source_count,
    std::uint32_t dense_width,
    float *source_gradient) {
    const std::uint64_t output_index = static_cast<std::uint64_t>(blockIdx.x)
        * blockDim.x + threadIdx.x;
    const std::uint64_t output_count =
        static_cast<std::uint64_t>(source_count) * dense_width;
    if (output_index >= output_count) return;
    const std::uint32_t source = static_cast<std::uint32_t>(
        output_index / dense_width);
    const std::uint32_t dense_column = static_cast<std::uint32_t>(
        output_index % dense_width);
    float sum = 0.0f;
    for (std::uint32_t edge = 0u; edge < logical_edge_count; ++edge) {
        const target_edge_placement_v1 placement = transpose_cover[edge];
        const std::uint32_t original_source = placement.destination_group
                * projection::mma_group_extent_limit_v1
            + placement.row;
        if (original_source != source) continue;
        const std::uint32_t original_destination = placement.source_group
                * projection::mma_group_extent_limit_v1
            + placement.column;
        if (original_destination >= destination_count) continue;
        sum += __half2float(logical_edge_values[edge]) * __half2float(
            destination_gradient[
                static_cast<std::size_t>(original_destination) * dense_width
                + dense_column]);
    }
    source_gradient[output_index] = sum;
}

} // namespace

transpose_relation_apply_status_v1 enqueue_transpose_relation_apply_v1(
    const transpose_relation_apply_request_v1 &request) noexcept {
    if (request.transpose_cover == nullptr
        || request.logical_edge_values == nullptr
        || request.logical_edge_count == 0u
        || request.logical_edge_count
            > std::numeric_limits<std::uint32_t>::max()
        || request.destination_gradient == nullptr
        || request.destination_count == 0u || request.source_count == 0u
        || request.dense_width == 0u || request.source_gradient == nullptr)
        return transpose_relation_apply_status_v1::invalid_argument;
    const std::uint64_t output_count =
        static_cast<std::uint64_t>(request.source_count) * request.dense_width;
    if (output_count > std::numeric_limits<std::uint32_t>::max())
        return transpose_relation_apply_status_v1::invalid_argument;
    constexpr std::uint32_t block_size = 256u;
    const std::uint32_t grid_size = static_cast<std::uint32_t>(
        (output_count + block_size - 1u) / block_size);
    transpose_relation_apply_kernel_v1<<<grid_size, block_size, 0u,
        request.stream>>>(request.transpose_cover, request.logical_edge_values,
        static_cast<std::uint32_t>(request.logical_edge_count),
        request.destination_gradient, request.destination_count,
        request.source_count, request.dense_width, request.source_gradient);
    return cudaGetLastError() == cudaSuccess
        ? transpose_relation_apply_status_v1::success
        : transpose_relation_apply_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
