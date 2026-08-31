#include <Cellerator/compute/operation/fusion/fusion_kernels_v1.cuh>

#include <cmath>
#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

constexpr std::uint32_t threads = 256u;

__global__ void value_pack_kernel(pack_apply_request_v1 request) {
    const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge < request.local_edge_count)
        request.projection_edge_values[request.logical_to_projection[edge]] =
            request.logical_edge_values[edge];
}

template<bool FusedPack>
__global__ void edge_apply_kernel(pack_apply_request_v1 request) {
    const std::uint32_t edge_index = blockIdx.x;
    if (edge_index >= request.local_edge_count) return;
    const relation_edge_v1 edge = request.logical_edges[edge_index];
    const std::uint32_t projection = request.logical_to_projection[edge_index];
    const float weight = FusedPack ? request.logical_edge_values[edge_index]
        : request.projection_edge_values[projection];
    if constexpr (FusedPack)
        if (threadIdx.x == 0u)
            request.projection_edge_values[projection] = weight;
    for (std::uint32_t component = threadIdx.x;
        component < request.component_count; component += blockDim.x)
        atomicAdd(&request.destination[
                      static_cast<std::size_t>(edge.destination_local)
                          * request.component_count + component],
            weight * request.source[
                static_cast<std::size_t>(edge.source_local)
                    * request.component_count + component]);
}

__device__ float row_sum(const apply_epilogue_request_v1 &request,
    std::uint32_t destination, std::uint32_t component) {
    float sum = 0.0f;
    for (std::uint32_t edge_index =
             request.destination_row_offsets[destination];
        edge_index < request.destination_row_offsets[destination + 1u];
        ++edge_index) {
        const row_edge_v1 edge = request.edges[edge_index];
        sum = fmaf(request.projection_edge_values[edge.projection_slot_local],
            request.source[static_cast<std::size_t>(edge.source_local)
                    * request.component_count + component], sum);
    }
    return sum;
}

__device__ float epilogue(const apply_epilogue_request_v1 &request,
    std::size_t position, float accumulation) {
    float value = request.alpha * accumulation;
    if (request.prior_destination != nullptr)
        value = fmaf(request.beta, request.prior_destination[position], value);
    if (request.bias != nullptr) value += request.bias[position % request.component_count];
    return request.relu ? fmaxf(value, 0.0f) : value;
}

template<int Mode>
__global__ void row_apply_epilogue_kernel(apply_epilogue_request_v1 request) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count =
        request.destination_count * request.component_count;
    if (linear >= count) return;
    const std::uint32_t destination = linear / request.component_count;
    const std::uint32_t component = linear % request.component_count;
    if constexpr (Mode == 0)
        request.accumulation_workspace[linear] =
            row_sum(request, destination, component);
    else if constexpr (Mode == 1)
        request.destination[linear] = epilogue(request, linear,
            request.accumulation_workspace[linear]);
    else
        request.destination[linear] = epilogue(request, linear,
            row_sum(request, destination, component));
}

} // namespace

status_v1 validate_pack_apply_request_v1(
    const pack_apply_request_v1 &request) noexcept {
    if (request.logical_edges == nullptr
        || request.logical_to_projection == nullptr
        || request.logical_edge_values == nullptr
        || request.projection_edge_values == nullptr || request.source == nullptr
        || request.destination == nullptr || request.local_edge_count == 0u
        || request.source_count == 0u || request.destination_count == 0u
        || request.component_count == 0u || request.structure_id == 0u
        || request.structure_epoch == 0u || request.value_generation == 0u
        || request.global_edge_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.local_edge_count)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_value_pack_unfused_v1(
    const pack_apply_request_v1 &request) noexcept {
    if (validate_pack_apply_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    value_pack_kernel<<<(request.local_edge_count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_apply_from_packed_unfused_v1(
    const pack_apply_request_v1 &request) noexcept {
    if (validate_pack_apply_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    edge_apply_kernel<false><<<request.local_edge_count, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_value_pack_apply_fused_v1(
    const pack_apply_request_v1 &request) noexcept {
    if (validate_pack_apply_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    edge_apply_kernel<true><<<request.local_edge_count, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 validate_apply_epilogue_request_v1(
    const apply_epilogue_request_v1 &request) noexcept {
    if (request.destination_row_offsets == nullptr || request.edges == nullptr
        || request.projection_edge_values == nullptr || request.source == nullptr
        || request.destination == nullptr || request.edge_count == 0u
        || request.source_count == 0u || request.destination_count == 0u
        || request.component_count == 0u
        || request.destination_count
            > std::numeric_limits<std::uint32_t>::max()
                / request.component_count
        || !std::isfinite(request.alpha)
        || !std::isfinite(request.beta) || request.reserved[0] != 0u
        || request.reserved[1] != 0u || request.reserved[2] != 0u)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_relation_apply_unfused_v1(
    const apply_epilogue_request_v1 &request) noexcept {
    if (validate_apply_epilogue_request_v1(request) != status_v1::success
        || request.accumulation_workspace == nullptr)
        return status_v1::invalid_argument;
    const std::uint32_t count = request.destination_count * request.component_count;
    row_apply_epilogue_kernel<0><<<(count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_epilogue_unfused_v1(
    const apply_epilogue_request_v1 &request) noexcept {
    if (validate_apply_epilogue_request_v1(request) != status_v1::success
        || request.accumulation_workspace == nullptr)
        return status_v1::invalid_argument;
    const std::uint32_t count = request.destination_count * request.component_count;
    row_apply_epilogue_kernel<1><<<(count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_apply_epilogue_fused_v1(
    const apply_epilogue_request_v1 &request) noexcept {
    if (validate_apply_epilogue_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    const std::uint32_t count = request.destination_count * request.component_count;
    row_apply_epilogue_kernel<2><<<(count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::fusion
