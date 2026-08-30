#include "contract_on_support.cu"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class edge_value_gradient_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    cuda_failure = 2u
};

struct edge_value_gradient_request_v1 {
    const support_logical_edge_v1 *logical_edges = nullptr;
    const projection::projection_value_map_v1 *physical_value_map = nullptr;
    std::uint64_t logical_edge_count = 0u;
    const __half *source_activation = nullptr;
    std::uint32_t source_count = 0u;
    const __half *destination_gradient = nullptr;
    std::uint32_t destination_count = 0u;
    std::uint32_t dense_width = 0u;
    float *logical_edge_gradient = nullptr;
    cudaStream_t stream = nullptr;
};

namespace {

__global__ void edge_value_gradient_kernel_v1(
    const support_logical_edge_v1 *logical_edges,
    const projection::projection_value_map_v1 *physical_value_map,
    std::uint32_t edge_count,
    const __half *source_activation,
    std::uint32_t source_count,
    const __half *destination_gradient,
    std::uint32_t destination_count,
    std::uint32_t dense_width,
    float *logical_edge_gradient) {
    const std::uint32_t physical_index = blockIdx.x;
    if (physical_index >= edge_count) return;
    const auto map = physical_value_map[physical_index];
    std::uint32_t logical_index = edge_count;
    for (std::uint32_t candidate = 0u; candidate < edge_count; ++candidate)
        if (logical_edges[candidate].logical_edge_id.value
            == map.logical_edge_id.value) {
            logical_index = candidate;
            break;
        }
    if (logical_index == edge_count) return;
    const auto edge = logical_edges[logical_index];
    if (edge.source_index >= source_count
        || edge.destination_index >= destination_count)
        return;
    float partial = 0.0f;
    for (std::uint32_t column = threadIdx.x; column < dense_width;
        column += blockDim.x)
        partial += __half2float(source_activation[
            static_cast<std::size_t>(edge.source_index) * dense_width + column])
            * __half2float(destination_gradient[
                static_cast<std::size_t>(edge.destination_index) * dense_width
                + column]);
    __shared__ float reduction[128];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t stride = 64u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u)
        logical_edge_gradient[logical_index] = reduction[0];
}

} // namespace

edge_value_gradient_status_v1 enqueue_edge_value_gradient_v1(
    const edge_value_gradient_request_v1 &request) noexcept {
    if (request.logical_edges == nullptr || request.physical_value_map == nullptr
        || request.logical_edge_count == 0u
        || request.logical_edge_count > std::numeric_limits<std::uint32_t>::max()
        || request.source_activation == nullptr || request.source_count == 0u
        || request.destination_gradient == nullptr
        || request.destination_count == 0u || request.dense_width == 0u
        || request.logical_edge_gradient == nullptr)
        return edge_value_gradient_status_v1::invalid_argument;
    if (cudaMemsetAsync(request.logical_edge_gradient, 0,
            request.logical_edge_count * sizeof(float), request.stream)
        != cudaSuccess)
        return edge_value_gradient_status_v1::cuda_failure;
    edge_value_gradient_kernel_v1<<<
        static_cast<std::uint32_t>(request.logical_edge_count), 128u, 0u,
        request.stream>>>(request.logical_edges, request.physical_value_map,
        static_cast<std::uint32_t>(request.logical_edge_count),
        request.source_activation, request.source_count,
        request.destination_gradient, request.destination_count,
        request.dense_width, request.logical_edge_gradient);
    return cudaGetLastError() == cudaSuccess
        ? edge_value_gradient_status_v1::success
        : edge_value_gradient_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
