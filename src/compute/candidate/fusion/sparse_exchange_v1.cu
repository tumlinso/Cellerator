#include <Cellerator/compute/operation/fusion/fusion_kernels_v1.cuh>

#include <cmath>
#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

constexpr std::uint32_t threads = 256u;

__device__ float contraction(const sparse_exchange_request_v1 &request,
    std::uint32_t edge_index) {
    const contract_edge_v1 edge = request.edges[edge_index];
    float score = 0.0f;
    for (std::uint32_t component = 0u;
        component < request.contraction_width; ++component)
        score = fmaf(request.source_key[
                         static_cast<std::size_t>(edge.source_local)
                             * request.contraction_width + component],
            request.destination_query[
                static_cast<std::size_t>(edge.destination_local)
                    * request.contraction_width + component], score);
    return score;
}

__device__ float mapped(const sparse_exchange_request_v1 &request,
    std::uint32_t edge_index) {
    float value = fmaf(request.map_scale, contraction(request, edge_index),
        request.map_bias);
    if (request.per_edge_gate != nullptr)
        value *= request.per_edge_gate[edge_index];
    return value;
}

template<int Stage>
__global__ void edge_stage_kernel(sparse_exchange_request_v1 request) {
    const std::uint32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
    if (edge >= request.edge_count) return;
    if constexpr (Stage == 0)
        request.contraction_workspace[edge] = contraction(request, edge);
    else {
        float value = fmaf(request.map_scale,
            request.contraction_workspace[edge], request.map_bias);
        if (request.per_edge_gate != nullptr)
            value *= request.per_edge_gate[edge];
        request.mapped_workspace[edge] = value;
    }
}

__global__ void normalize_kernel(sparse_exchange_request_v1 request) {
    const std::uint32_t destination = blockIdx.x;
    if (destination >= request.destination_count || threadIdx.x != 0u) return;
    const std::uint32_t begin = request.destination_segment_offsets[destination];
    const std::uint32_t end = request.destination_segment_offsets[destination + 1u];
    float maximum = -__int_as_float(0x7f800000);
    for (std::uint32_t edge = begin; edge < end; ++edge)
        maximum = fmaxf(maximum, request.mapped_workspace[edge]);
    float denominator = 0.0f;
    for (std::uint32_t edge = begin; edge < end; ++edge)
        denominator += expf(request.mapped_workspace[edge] - maximum);
    for (std::uint32_t edge = begin; edge < end; ++edge)
        request.normalized_workspace[edge] =
            expf(request.mapped_workspace[edge] - maximum) / denominator;
}

template<int Mode>
__global__ void exchange_apply_kernel(sparse_exchange_request_v1 request) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = request.destination_count * request.value_width;
    if (linear >= count) return;
    const std::uint32_t destination = linear / request.value_width;
    const std::uint32_t component = linear % request.value_width;
    const std::uint32_t begin = request.destination_segment_offsets[destination];
    const std::uint32_t end = request.destination_segment_offsets[destination + 1u];
    float maximum = -__int_as_float(0x7f800000);
    float denominator = 0.0f;
    if constexpr (Mode != 0) {
        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const float score = Mode == 1 ? request.mapped_workspace[edge]
                : mapped(request, edge);
            maximum = fmaxf(maximum, score);
        }
        for (std::uint32_t edge = begin; edge < end; ++edge) {
            const float score = Mode == 1 ? request.mapped_workspace[edge]
                : mapped(request, edge);
            denominator += expf(score - maximum);
        }
    }
    float output = 0.0f;
    for (std::uint32_t edge = begin; edge < end; ++edge) {
        const float weight = Mode == 0 ? request.normalized_workspace[edge]
            : expf((Mode == 1 ? request.mapped_workspace[edge]
                              : mapped(request, edge)) - maximum)
                / denominator;
        const std::uint32_t source = request.edges[edge].source_local;
        output = fmaf(weight, request.source_value[
            static_cast<std::size_t>(source) * request.value_width + component],
            output);
    }
    request.destination_output[linear] = output;
}

} // namespace

status_v1 validate_sparse_exchange_request_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    if (request.edges == nullptr || request.destination_segment_offsets == nullptr
        || request.source_key == nullptr || request.destination_query == nullptr
        || request.source_value == nullptr
        || request.contraction_workspace == nullptr
        || request.mapped_workspace == nullptr
        || request.normalized_workspace == nullptr
        || request.destination_output == nullptr || request.edge_count == 0u
        || request.source_count == 0u || request.destination_count == 0u
        || request.contraction_width == 0u || request.value_width == 0u
        || request.destination_count
            > std::numeric_limits<std::uint32_t>::max() / request.value_width
        || !std::isfinite(request.map_scale)
        || !std::isfinite(request.map_bias) || request.structure_id == 0u
        || request.structure_epoch == 0u || request.value_generation == 0u)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_exchange_contraction_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    if (validate_sparse_exchange_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    edge_stage_kernel<0><<<(request.edge_count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_exchange_map_gate_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    if (validate_sparse_exchange_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    edge_stage_kernel<1><<<(request.edge_count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_exchange_normalization_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    if (validate_sparse_exchange_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    normalize_kernel<<<request.destination_count, 1u, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

template<int Mode>
status_v1 enqueue_apply(const sparse_exchange_request_v1 &request) noexcept {
    if (validate_sparse_exchange_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    const std::uint32_t count = request.destination_count * request.value_width;
    exchange_apply_kernel<Mode><<<(count + threads - 1u) / threads,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_exchange_apply_unfused_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    return enqueue_apply<0>(request);
}

status_v1 enqueue_normalize_apply_fused_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    return enqueue_apply<1>(request);
}

status_v1 enqueue_sparse_exchange_fused_v1(
    const sparse_exchange_request_v1 &request) noexcept {
    return enqueue_apply<2>(request);
}

} // namespace cellerator::compute::operation::fusion
