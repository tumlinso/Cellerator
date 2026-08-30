#include "contract_on_support_projection.cc"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace cellerator::compute::architecture::providers::nvidia::sm70 {

enum class contract_on_support_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    cuda_failure = 2u
};

struct contract_on_support_request_v1 {
    const support_logical_edge_v1 *logical_edges = nullptr;
    std::uint64_t logical_edge_count = 0u;
    const support_projection_edge_v1 *selected_edges = nullptr;
    std::uint64_t selected_edge_count = 0u;
    const __half *source_features = nullptr;
    std::uint32_t source_count = 0u;
    const __half *destination_features = nullptr;
    std::uint32_t destination_count = 0u;
    std::uint32_t dense_width = 0u;
    float *logical_edge_output = nullptr;
    cudaStream_t stream = nullptr;
};

namespace {

__global__ void contract_on_support_kernel_v1(
    const support_logical_edge_v1 *logical_edges,
    std::uint32_t logical_edge_count,
    const support_projection_edge_v1 *selected_edges,
    std::uint32_t selected_edge_count,
    const __half *source_features,
    std::uint32_t source_count,
    const __half *destination_features,
    std::uint32_t destination_count,
    std::uint32_t dense_width,
    float *logical_edge_output) {
    const std::uint32_t selected_index = blockIdx.x;
    if (selected_index >= selected_edge_count) return;
    const support_projection_edge_v1 selected = selected_edges[selected_index];
    const std::uint32_t logical_index = selected.stable_output_index;
    if (logical_index >= logical_edge_count) return;
    const support_logical_edge_v1 edge = logical_edges[logical_index];
    if (edge.logical_edge_id.value != selected.logical_edge_id.value
        || edge.source_index >= source_count
        || edge.destination_index >= destination_count)
        return;
    float partial = 0.0f;
    for (std::uint32_t column = threadIdx.x; column < dense_width;
        column += blockDim.x) {
        partial += __half2float(source_features[
            static_cast<std::size_t>(edge.source_index) * dense_width + column])
            * __half2float(destination_features[
                static_cast<std::size_t>(edge.destination_index) * dense_width
                + column]);
    }
    __shared__ float reduction[128];
    reduction[threadIdx.x] = partial;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u; stride /= 2u) {
        if (threadIdx.x < stride)
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0u) logical_edge_output[logical_index] = reduction[0];
}

} // namespace

contract_on_support_status_v1 enqueue_contract_on_support_v1(
    const contract_on_support_request_v1 &request) noexcept {
    if (request.logical_edges == nullptr || request.logical_edge_count == 0u
        || request.logical_edge_count > std::numeric_limits<std::uint32_t>::max()
        || request.selected_edges == nullptr || request.selected_edge_count == 0u
        || request.selected_edge_count > request.logical_edge_count
        || request.source_features == nullptr || request.source_count == 0u
        || request.destination_features == nullptr
        || request.destination_count == 0u || request.dense_width == 0u
        || request.logical_edge_output == nullptr)
        return contract_on_support_status_v1::invalid_argument;
    if (cudaMemsetAsync(request.logical_edge_output, 0,
            request.logical_edge_count * sizeof(float), request.stream)
        != cudaSuccess)
        return contract_on_support_status_v1::cuda_failure;
    contract_on_support_kernel_v1<<<
        static_cast<std::uint32_t>(request.selected_edge_count), 128u, 0u,
        request.stream>>>(request.logical_edges,
        static_cast<std::uint32_t>(request.logical_edge_count),
        request.selected_edges,
        static_cast<std::uint32_t>(request.selected_edge_count),
        request.source_features, request.source_count,
        request.destination_features, request.destination_count,
        request.dense_width, request.logical_edge_output);
    return cudaGetLastError() == cudaSuccess
        ? contract_on_support_status_v1::success
        : contract_on_support_status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70
