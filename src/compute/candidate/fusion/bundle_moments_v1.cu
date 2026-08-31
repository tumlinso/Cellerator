#include <Cellerator/compute/operation/fusion/fusion_kernels_v1.cuh>

#include <limits>

namespace cellerator::compute::operation::fusion {
namespace {

constexpr std::uint32_t threads = 256u;

template<bool Fused>
__global__ void bundle_materialize_kernel(bundle_output_request_v1 request) {
    const std::uint32_t output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= request.local_output_count) return;
    if constexpr (!Fused)
        for (std::uint32_t bundle = 0u; bundle < request.bundle_count; ++bundle)
            request.materialized_workspace[
                static_cast<std::size_t>(bundle) * request.local_output_count
                    + output] = request.bundle_partial_outputs[
                static_cast<std::size_t>(bundle) * request.local_output_count
                    + output];
    else {
        float sum = 0.0f;
        for (std::uint32_t bundle = 0u; bundle < request.bundle_count; ++bundle)
            sum += request.bundle_partial_outputs[
                static_cast<std::size_t>(bundle) * request.local_output_count
                    + output];
        request.shared_destination[output] = sum;
    }
}

__global__ void bundle_accumulate_kernel(bundle_output_request_v1 request) {
    const std::uint32_t output = blockIdx.x * blockDim.x + threadIdx.x;
    if (output >= request.local_output_count) return;
    float sum = 0.0f;
    for (std::uint32_t bundle = 0u; bundle < request.bundle_count; ++bundle)
        sum += request.materialized_workspace[
            static_cast<std::size_t>(bundle) * request.local_output_count
                + output];
    request.shared_destination[output] = sum;
}

template<int Mode>
__global__ void moments_kernel(relation_moments_request_v1 request) {
    const std::uint32_t linear = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t count = request.destination_count * request.component_count;
    if (linear >= count) return;
    const std::uint32_t destination = linear / request.component_count;
    const std::uint32_t component = linear % request.component_count;
    float first = 0.0f;
    float second = 0.0f;
    for (std::uint32_t edge_index =
             request.destination_row_offsets[destination];
        edge_index < request.destination_row_offsets[destination + 1u];
        ++edge_index) {
        const row_edge_v1 edge = request.edges[edge_index];
        const float value = request.source[
            static_cast<std::size_t>(edge.source_local)
                * request.component_count + component];
        const float weight = request.edge_values[edge.projection_slot_local];
        if constexpr (Mode != 1) first = fmaf(weight, value, first);
        if constexpr (Mode != 0) second = fmaf(weight, value * value, second);
    }
    if constexpr (Mode != 1) request.first_moment[linear] = first;
    if constexpr (Mode != 0) request.second_moment[linear] = second;
}

} // namespace

status_v1 validate_bundle_output_request_v1(
    const bundle_output_request_v1 &request) noexcept {
    if (request.bundle_partial_outputs == nullptr
        || request.materialized_workspace == nullptr
        || request.shared_destination == nullptr || request.bundle_count == 0u
        || request.local_output_count == 0u
        || request.destination_order_id == 0u
        || request.global_output_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.local_output_count)
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_bundle_materialize_unfused_v1(
    const bundle_output_request_v1 &request) noexcept {
    if (validate_bundle_output_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    bundle_materialize_kernel<false><<<
        (request.local_output_count + threads - 1u) / threads, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_shared_destination_accumulate_unfused_v1(
    const bundle_output_request_v1 &request) noexcept {
    if (validate_bundle_output_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    bundle_accumulate_kernel<<<
        (request.local_output_count + threads - 1u) / threads, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_bundle_shared_destination_fused_v1(
    const bundle_output_request_v1 &request) noexcept {
    if (validate_bundle_output_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    bundle_materialize_kernel<true><<<
        (request.local_output_count + threads - 1u) / threads, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 validate_relation_moments_request_v1(
    const relation_moments_request_v1 &request) noexcept {
    if (request.destination_row_offsets == nullptr || request.edges == nullptr
        || request.edge_values == nullptr || request.source == nullptr
        || request.first_moment == nullptr || request.second_moment == nullptr
        || request.edge_count == 0u || request.source_count == 0u
        || request.destination_count == 0u || request.component_count == 0u
        || request.destination_count
            > std::numeric_limits<std::uint32_t>::max()
                / request.component_count
        || request.structure_epoch == 0u || request.value_generation == 0u)
        return status_v1::invalid_argument;
    return status_v1::success;
}

template<int Mode>
status_v1 enqueue_moments(const relation_moments_request_v1 &request) noexcept {
    if (validate_relation_moments_request_v1(request) != status_v1::success)
        return status_v1::invalid_argument;
    const std::uint32_t count = request.destination_count * request.component_count;
    moments_kernel<Mode><<<(count + threads - 1u) / threads, threads, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
        : status_v1::cuda_failure;
}

status_v1 enqueue_first_moment_unfused_v1(
    const relation_moments_request_v1 &request) noexcept {
    return enqueue_moments<0>(request);
}

status_v1 enqueue_second_moment_unfused_v1(
    const relation_moments_request_v1 &request) noexcept {
    return enqueue_moments<1>(request);
}

status_v1 enqueue_relation_moments_fused_v1(
    const relation_moments_request_v1 &request) noexcept {
    return enqueue_moments<2>(request);
}

} // namespace cellerator::compute::operation::fusion
