#include "contract_on_support_projection.cc"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

inline constexpr bool cover_native_normalize_empirical_required_v1 = true;

struct cover_native_partition_v1 {
    projection::physical_region_kind_v1 region_kind =
        projection::physical_region_kind_v1::residual;
    std::uint32_t selected_begin = 0u;
    std::uint32_t selected_count = 0u;
};

enum class cover_native_normalize_status_v1 : std::uint8_t {
    success = 0u, invalid_argument = 1u, cuda_failure = 2u
};

struct cover_native_normalize_request_v1 {
    const support_projection_edge_v1 *selected_edges = nullptr;
    std::uint32_t selected_edge_count = 0u;
    const cover_native_partition_v1 *partitions = nullptr;
    std::uint32_t partition_count = 0u;
    const float *logical_edge_values = nullptr;
    std::uint32_t logical_edge_count = 0u;
    float *logical_edge_output = nullptr;
    cudaStream_t stream = nullptr;
};

namespace {
__global__ void cover_native_normalize_kernel_v1(
    const support_projection_edge_v1 *edges,
    const cover_native_partition_v1 *partitions,
    std::uint32_t partition_count,
    const float *values,
    std::uint32_t logical_count,
    float *output) {
    const std::uint32_t partition_index = blockIdx.x;
    if (partition_index >= partition_count || threadIdx.x != 0u) return;
    const cover_native_partition_v1 partition = partitions[partition_index];
    float maximum = -CUDART_INF_F;
    for (std::uint32_t local = 0u; local < partition.selected_count; ++local) {
        const auto edge = edges[partition.selected_begin + local];
        if (edge.region_kind != partition.region_kind
            || edge.stable_output_index >= logical_count) return;
        maximum = fmaxf(maximum, values[edge.stable_output_index]);
    }
    float denominator = 0.0f;
    for (std::uint32_t local = 0u; local < partition.selected_count; ++local) {
        const auto edge = edges[partition.selected_begin + local];
        denominator += expf(values[edge.stable_output_index] - maximum);
    }
    for (std::uint32_t local = 0u; local < partition.selected_count; ++local) {
        const auto edge = edges[partition.selected_begin + local];
        output[edge.stable_output_index] =
            expf(values[edge.stable_output_index] - maximum) / denominator;
    }
}
}

cover_native_normalize_status_v1 enqueue_cover_native_normalize_v1(
    const cover_native_normalize_request_v1 &request) noexcept {
    if (request.selected_edges == nullptr || request.selected_edge_count == 0u
        || request.partitions == nullptr || request.partition_count == 0u
        || request.logical_edge_values == nullptr
        || request.logical_edge_count == 0u
        || request.logical_edge_output == nullptr)
        return cover_native_normalize_status_v1::invalid_argument;
    if (cudaMemsetAsync(request.logical_edge_output, 0,
            request.logical_edge_count * sizeof(float), request.stream)
        != cudaSuccess)
        return cover_native_normalize_status_v1::cuda_failure;
    cover_native_normalize_kernel_v1<<<request.partition_count, 1u, 0u,
        request.stream>>>(request.selected_edges, request.partitions,
        request.partition_count, request.logical_edge_values,
        request.logical_edge_count, request.logical_edge_output);
    return cudaGetLastError() == cudaSuccess
        ? cover_native_normalize_status_v1::success
        : cover_native_normalize_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
