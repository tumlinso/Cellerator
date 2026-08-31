#include <Cellerator/compute/operation/edge/edge_operations_v1.cuh>

#include <cmath>
#include <limits>

namespace cellerator::compute::operation::edge {
namespace {

__device__ float apply_map(float value, map_kind_v1 map,
    map_parameters_v1 parameters) {
    switch (map) {
        case map_kind_v1::identity: return value;
        case map_kind_v1::affine:
            return fmaf(parameters.first, value, parameters.second);
        case map_kind_v1::clamp:
            return fminf(fmaxf(value, parameters.first), parameters.second);
        case map_kind_v1::absolute: return fabsf(value);
        case map_kind_v1::exponential: return expf(value);
        case map_kind_v1::logarithm: return logf(value);
        case map_kind_v1::reciprocal: return 1.0f / value;
    }
    return value;
}

__global__ void edge_map_kernel(edge_map_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t edge = first; edge < request.edges.local_edge_count;
        edge += stride) {
        const float mapped = apply_map(request.input[edge], request.map,
            request.parameters);
        if (request.gate == gate_kind_v1::per_edge_multiplicative)
            request.output[edge] = mapped
                * static_cast<const float *>(request.per_edge_gate)[edge];
        else if (request.gate == gate_kind_v1::per_edge_predicate)
            request.output[edge] =
                static_cast<const std::uint8_t *>(request.per_edge_gate)[edge]
                != 0u ? mapped : 0.0f;
        else
            request.output[edge] = mapped;
    }
}

} // namespace

status_v1 validate_edge_map_request_v1(
    const edge_map_request_v1 &request) noexcept {
    if (request.edges.local_edge_count == 0u || request.input == nullptr
        || request.output == nullptr || request.structure_id == 0u
        || request.structure_epoch == 0u || request.value_generation == 0u
        || request.edges.global_edge_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.edges.local_edge_count)
        return status_v1::invalid_argument;
    if (request.map > map_kind_v1::reciprocal
        || request.gate > gate_kind_v1::per_edge_predicate)
        return status_v1::unsupported;
    if ((request.gate == gate_kind_v1::none)
            != (request.per_edge_gate == nullptr))
        return status_v1::invalid_argument;
    if (!std::isfinite(request.parameters.first)
        || !std::isfinite(request.parameters.second)
        || (request.map == map_kind_v1::clamp
            && request.parameters.first > request.parameters.second))
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_edge_map_v1(const edge_map_request_v1 &request) noexcept {
    const status_v1 validation = validate_edge_map_request_v1(request);
    if (validation != status_v1::success) return validation;
    constexpr std::uint32_t threads = 256u;
    constexpr std::uint32_t maximum_blocks = 65535u;
    const std::uint32_t required =
        (request.edges.local_edge_count + threads - 1u) / threads;
    const std::uint32_t blocks = required < maximum_blocks
        ? required : maximum_blocks;
    edge_map_kernel<<<blocks, threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::edge
