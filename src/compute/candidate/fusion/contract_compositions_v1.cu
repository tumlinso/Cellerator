#include <Cellerator/compute/operation/fusion/fusion_kernels_v1.cuh>

#include <cmath>
#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

__device__ float contract_edge(const contract_composition_request_v1 &request,
    std::uint32_t edge_index) {
    const contract_edge_v1 edge = request.edges[edge_index];
    float value = 0.0f;
    for (std::uint32_t component = 0u; component < request.dense_width;
        ++component)
        value = fmaf(request.source[
                         static_cast<std::size_t>(edge.source_local)
                             * request.dense_width + component],
            request.destination[
                static_cast<std::size_t>(edge.destination_local)
                    * request.dense_width + component], value);
    return value;
}

template<int Mode>
__global__ void contract_map_kernel(contract_composition_request_v1 request) {
    const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= request.edge_count) return;
    if constexpr (Mode == 0)
        request.contraction_workspace[edge] = contract_edge(request, edge);
    else {
        const float contraction = Mode == 1
            ? request.contraction_workspace[edge] : contract_edge(request, edge);
        float mapped = fmaf(request.map_scale, contraction, request.map_bias);
        if (request.per_edge_gate != nullptr)
            mapped *= request.per_edge_gate[edge];
        request.mapped_output[edge] = mapped;
    }
}

template<bool Fused>
__global__ void contract_segment_kernel(
    contract_composition_request_v1 request) {
    const std::uint32_t segment = blockIdx.x;
    if (segment >= request.segment_count || threadIdx.x != 0u) return;
    float sum = 0.0f;
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t edge = request.segment_offsets[segment];
        edge < request.segment_offsets[segment + 1u]; ++edge) {
        const float value = Fused ? contract_edge(request, edge)
            : request.contraction_workspace[edge];
        sum += value;
        maximum = fmaxf(maximum, value);
    }
    request.segment_sum_output[segment] = sum;
    request.segment_maximum_output[segment] = maximum;
}

} // namespace

status_v1 validate_contract_composition_request_v1(
    const contract_composition_request_v1 &request) noexcept {
    if (request.edges == nullptr || request.source == nullptr
        || request.destination == nullptr || request.contraction_workspace == nullptr
        || request.mapped_output == nullptr || request.segment_offsets == nullptr
        || request.segment_sum_output == nullptr
        || request.segment_maximum_output == nullptr || request.edge_count == 0u
        || request.source_count == 0u || request.destination_count == 0u
        || request.dense_width == 0u || request.segment_count == 0u
        || !std::isfinite(request.map_scale)
        || !std::isfinite(request.map_bias) || request.structure_epoch == 0u
        || request.value_generation == 0u || request.global_edge_begin
            > std::numeric_limits<std::uint64_t>::max() - request.edge_count)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_contraction_unfused_v1(
    const contract_composition_request_v1 &request) noexcept {
    if (validate_contract_composition_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    constexpr std::uint32_t threads = 256u;
    contract_map_kernel<0><<<(request.edge_count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_edge_map_unfused_v1(
    const contract_composition_request_v1 &request) noexcept {
    if (validate_contract_composition_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    constexpr std::uint32_t threads = 256u;
    contract_map_kernel<1><<<(request.edge_count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_contract_edge_map_fused_v1(
    const contract_composition_request_v1 &request) noexcept {
    if (validate_contract_composition_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    constexpr std::uint32_t threads = 256u;
    contract_map_kernel<2><<<(request.edge_count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_segment_statistic_unfused_v1(
    const contract_composition_request_v1 &request) noexcept {
    if (validate_contract_composition_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    contract_segment_kernel<false><<<request.segment_count, 1u, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_contract_segment_statistic_fused_v1(
    const contract_composition_request_v1 &request) noexcept {
    if (validate_contract_composition_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    contract_segment_kernel<true><<<request.segment_count, 1u, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::fusion
