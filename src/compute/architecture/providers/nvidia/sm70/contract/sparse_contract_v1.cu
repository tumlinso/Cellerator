#include <Cellerator/compute/architecture/providers/nvidia/sm70/contract/contract_portfolio_v1.cuh>

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace cellerator::compute::architecture::providers::nvidia::sm70::contract {
namespace {

__device__ float edge_term(const dense_pair_v1 dense, const edge_ref_v1 edge,
    std::uint32_t component) {
    const std::size_t source_offset =
        static_cast<std::size_t>(edge.source_local) * dense.dense_width;
    const std::size_t destination_offset =
        static_cast<std::size_t>(edge.destination_local) * dense.dense_width;
    return __half2float(dense.source[source_offset + component])
        * __half2float(dense.destination[destination_offset + component]);
}

__device__ std::uint32_t output_index(const edge_ref_v1 edge,
    std::uint32_t projection_index, output_order_v1 order) {
    return order == output_order_v1::projection_native
        ? projection_index : edge.logical_output_local;
}

__global__ void thread_per_edge_kernel(support_view_v1 support,
    dense_pair_v1 dense, output_order_v1 order, float *output) {
    const std::uint32_t edge_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge_index >= support.local_edge_count) return;
    const edge_ref_v1 edge = support.edges[edge_index];
    float sum = 0.0f;
    for (std::uint32_t component = 0u; component < dense.dense_width;
        ++component)
        sum = fmaf(__half2float(dense.source[
                       static_cast<std::size_t>(edge.source_local)
                           * dense.dense_width + component]),
            __half2float(dense.destination[
                static_cast<std::size_t>(edge.destination_local)
                    * dense.dense_width + component]), sum);
    output[output_index(edge, edge_index, order)] = sum;
}

__global__ void warp_per_edge_kernel(support_view_v1 support,
    dense_pair_v1 dense, output_order_v1 order, float *output) {
    const std::uint32_t warp_in_block = threadIdx.x / 32u;
    const std::uint32_t lane = threadIdx.x % 32u;
    const std::uint32_t edge_index = blockIdx.x * (blockDim.x / 32u)
        + warp_in_block;
    if (edge_index >= support.local_edge_count) return;
    const edge_ref_v1 edge = support.edges[edge_index];
    float sum = 0.0f;
    for (std::uint32_t component = lane; component < dense.dense_width;
        component += 32u)
        sum += edge_term(dense, edge, component);
    for (std::uint32_t offset = 16u; offset != 0u; offset /= 2u)
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    if (lane == 0u)
        output[output_index(edge, edge_index, order)] = sum;
}

__global__ void cooperative_edge_kernel(support_view_v1 support,
    dense_pair_v1 dense, output_order_v1 order, float *output) {
    const std::uint32_t edge_index = blockIdx.x;
    if (edge_index >= support.local_edge_count) return;
    const edge_ref_v1 edge = support.edges[edge_index];
    float sum = 0.0f;
    for (std::uint32_t component = threadIdx.x;
        component < dense.dense_width; component += blockDim.x)
        sum += edge_term(dense, edge, component);
    __shared__ float partial[128];
    partial[threadIdx.x] = sum;
    cg::thread_block group = cg::this_thread_block();
    group.sync();
    for (std::uint32_t stride = blockDim.x / 2u; stride != 0u;
        stride /= 2u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        group.sync();
    }
    if (threadIdx.x == 0u)
        output[output_index(edge, edge_index, order)] = partial[0];
}

} // namespace

status_v1 validate_launch_v1(const launch_request_v1 &request) noexcept {
    const support_view_v1 &support = request.support;
    if (support.edges == nullptr || support.local_edge_count == 0u
        || support.source_count == 0u || support.destination_count == 0u
        || request.dense.source == nullptr || request.dense.destination == nullptr
        || request.dense.dense_width == 0u || request.output == nullptr)
        return status_v1::invalid_argument;
    if (request.output_order != output_order_v1::logical_edge
        && request.output_order != output_order_v1::projection_native)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_sparse_v1(const launch_request_v1 &request) noexcept {
    if (validate_launch_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    constexpr std::uint32_t threads = 128u;
    switch (request.candidate) {
        case sparse_candidate_v1::thread_per_edge:
            thread_per_edge_kernel<<<
                (request.support.local_edge_count + threads - 1u) / threads,
                threads, 0u, request.stream>>>(request.support, request.dense,
                request.output_order, request.output);
            break;
        case sparse_candidate_v1::warp_per_edge:
            warp_per_edge_kernel<<<
                (request.support.local_edge_count + 3u) / 4u, threads, 0u,
                request.stream>>>(request.support, request.dense,
                request.output_order, request.output);
            break;
        case sparse_candidate_v1::cooperative_group:
            cooperative_edge_kernel<<<request.support.local_edge_count, threads,
                0u, request.stream>>>(request.support, request.dense,
                request.output_order, request.output);
            break;
        default: return status_v1::unsupported;
    }
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::contract
