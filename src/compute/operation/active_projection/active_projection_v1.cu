#include <Cellerator/compute/operation/active_projection/active_projection_v1.cuh>

#include <cmath>

namespace cellerator::compute::operation::active_projection {
namespace {

bool valid_identity(const projection_identity_v1 &identity) noexcept {
    return identity.structure_id != 0u && identity.structure_epoch != 0u
        && identity.activity.generation != 0u
        && identity.activity.scope <= activity_scope_v1::per_instance
        && (identity.activity.scope == activity_scope_v1::shared
            || identity.activity.owner_id != 0u);
}

__global__ void compact_copy_kernel(compact_copy_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t compact = first; compact < request.active_count;
         compact += stride)
        request.compact_output[compact] =
            request.full_input[request.logical_indices[compact]];
}

__global__ void zero_kernel(float *output, std::uint32_t count) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t item = first; item < count; item += stride)
        output[item] = 0.0f;
}

__global__ void compact_scatter_kernel(compact_scatter_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t compact = first; compact < request.active_count;
         compact += stride)
        request.full_output[request.logical_indices[compact]] =
            request.compact_input[compact];
}

__global__ void weighted_primal_kernel(weighted_response_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t compact = first; compact < request.primal_count;
         compact += stride) {
        const std::uint32_t logical = request.primal_indices[compact];
        request.primal_output[logical] =
            request.weights[logical] * request.input[logical];
    }
}

__global__ void parameter_response_kernel(weighted_response_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t compact = first; compact < request.derivative_count;
         compact += stride) {
        const std::uint32_t logical = request.derivative_indices[compact];
        request.parameter_response[logical] = request.input[logical];
    }
}

std::uint32_t grid_size(std::uint32_t count) noexcept {
    constexpr std::uint32_t threads = 256u;
    constexpr std::uint32_t maximum_blocks = 65535u;
    const std::uint32_t required = (count + threads - 1u) / threads;
    return required < maximum_blocks ? required : maximum_blocks;
}

} // namespace

status_v1 validate_approximate_drop_policy_v1(
    const approximate_drop_policy_v1 &policy) noexcept {
    if (!policy.enabled) {
        return policy.provenance == approximation_provenance_v1::exact
            ? status_v1::success : status_v1::invalid_argument;
    }
    if (policy.provenance == approximation_provenance_v1::exact
        || !std::isfinite(policy.threshold) || policy.threshold < 0.0f)
        return status_v1::invalid_argument;
    if (policy.provenance == approximation_provenance_v1::bounded)
        return std::isfinite(policy.absolute_error_bound)
                && policy.absolute_error_bound >= 0.0f
            ? status_v1::success : status_v1::invalid_argument;
    if (policy.provenance == approximation_provenance_v1::empirical)
        return policy.empirical_sample_count != 0u ? status_v1::success
                                                   : status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 build_exact_projection_v1(const projection_build_request_v1 &request,
    projection_map_v1 *primal, projection_map_v1 *derivative) noexcept {
    if (!valid_identity(request.identity) || request.primal_active == nullptr
        || request.derivative_active == nullptr || request.full_count == 0u
        || primal == nullptr || derivative == nullptr)
        return status_v1::invalid_argument;
    std::uint32_t primal_count = 0u;
    std::uint32_t derivative_count = 0u;
    for (std::uint32_t logical = 0u; logical < request.full_count; ++logical) {
        if (request.primal_active[logical] != 0u) {
            if (request.primal_indices == nullptr
                || primal_count == request.primal_capacity)
                return status_v1::insufficient_capacity;
            request.primal_indices[primal_count++] = logical;
        }
        if (request.derivative_active[logical] != 0u) {
            if (request.derivative_indices == nullptr
                || derivative_count == request.derivative_capacity)
                return status_v1::insufficient_capacity;
            request.derivative_indices[derivative_count++] = logical;
        }
    }
    *primal = {request.identity, request.full_count, primal_count};
    *derivative = {request.identity, request.full_count, derivative_count};
    return status_v1::success;
}

status_v1 build_approximate_primal_projection_v1(
    const approximate_projection_build_request_v1 &request,
    projection_map_v1 *primal, projection_map_v1 *derivative) noexcept {
    if (request.primal_coefficients == nullptr
        || validate_approximate_drop_policy_v1(request.policy)
            != status_v1::success)
        return status_v1::invalid_argument;
    if (!request.policy.enabled)
        return build_exact_projection_v1(request.exact, primal, derivative);

    // The caller's mutable mask is never changed: approximate admission is a
    // separate map.  Derivative support always follows exact derivative input.
    std::uint32_t primal_count = 0u;
    std::uint32_t derivative_count = 0u;
    const projection_build_request_v1 &exact = request.exact;
    if (!valid_identity(exact.identity) || exact.primal_active == nullptr
        || exact.derivative_active == nullptr || exact.full_count == 0u
        || primal == nullptr || derivative == nullptr)
        return status_v1::invalid_argument;
    for (std::uint32_t logical = 0u; logical < exact.full_count; ++logical) {
        if (exact.primal_active[logical] != 0u
            && std::fabs(request.primal_coefficients[logical])
                >= request.policy.threshold) {
            if (exact.primal_indices == nullptr || primal_count == exact.primal_capacity)
                return status_v1::insufficient_capacity;
            exact.primal_indices[primal_count++] = logical;
        }
        if (exact.derivative_active[logical] != 0u) {
            if (exact.derivative_indices == nullptr
                || derivative_count == exact.derivative_capacity)
                return status_v1::insufficient_capacity;
            exact.derivative_indices[derivative_count++] = logical;
        }
    }
    *primal = {exact.identity, exact.full_count, primal_count};
    *derivative = {exact.identity, exact.full_count, derivative_count};
    return status_v1::success;
}

status_v1 enqueue_compact_copy_v1(const compact_copy_request_v1 &request) noexcept {
    if (request.full_input == nullptr || request.logical_indices == nullptr
        || request.compact_output == nullptr || request.active_count == 0u)
        return status_v1::invalid_argument;
    compact_copy_kernel<<<grid_size(request.active_count), 256u, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
                                               : status_v1::cuda_failure;
}

status_v1 enqueue_compact_scatter_v1(
    const compact_scatter_request_v1 &request) noexcept {
    if (request.compact_input == nullptr || request.logical_indices == nullptr
        || request.full_output == nullptr || request.full_count == 0u
        || request.active_count == 0u || request.active_count > request.full_count)
        return status_v1::invalid_argument;
    zero_kernel<<<grid_size(request.full_count), 256u, 0u, request.stream>>>(
        request.full_output, request.full_count);
    compact_scatter_kernel<<<grid_size(request.active_count), 256u, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
                                               : status_v1::cuda_failure;
}

status_v1 enqueue_weighted_response_v1(
    const weighted_response_request_v1 &request) noexcept {
    if (request.input == nullptr || request.weights == nullptr
        || request.primal_output == nullptr || request.parameter_response == nullptr
        || request.primal_indices == nullptr || request.derivative_indices == nullptr
        || request.primal_count == 0u || request.derivative_count == 0u)
        return status_v1::invalid_argument;
    weighted_primal_kernel<<<grid_size(request.primal_count), 256u, 0u,
        request.stream>>>(request);
    parameter_response_kernel<<<grid_size(request.derivative_count), 256u, 0u,
        request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess ? status_v1::success
                                               : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::active_projection
