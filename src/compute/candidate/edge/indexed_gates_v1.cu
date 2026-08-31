#include <Cellerator/compute/operation/edge/indexed_gates_v1.cuh>

#include <limits>

namespace cellerator::compute::operation::edge {
namespace {

__global__ void indexed_gate_kernel(indexed_gate_request_v1 request) {
    const std::uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t stride = gridDim.x * blockDim.x;
    for (std::uint32_t edge = first; edge < request.edges.local_edge_count;
        edge += stride) {
        const edge_coordinate_v1 coordinate = request.coordinates[edge];
        float gate = 1.0f;
        switch (request.kind) {
            case indexed_gate_kind_v1::per_source:
                gate = request.primary_gate[coordinate.source_local];
                break;
            case indexed_gate_kind_v1::per_destination:
                gate = request.primary_gate[coordinate.destination_local];
                break;
            case indexed_gate_kind_v1::per_component:
                gate = request.primary_gate[coordinate.component_local];
                break;
            case indexed_gate_kind_v1::factorized_source_destination:
                gate = request.primary_gate[coordinate.source_local]
                    * request.secondary_gate[coordinate.destination_local];
                break;
        }
        request.output[edge] = request.input[edge] * gate;
    }
}

} // namespace

status_v1 validate_indexed_gate_request_v1(
    const indexed_gate_request_v1 &request) noexcept {
    if (request.edges.local_edge_count == 0u
        || request.coordinates == nullptr || request.input == nullptr
        || request.output == nullptr || request.primary_gate == nullptr
        || request.source_count == 0u || request.destination_count == 0u
        || request.component_count == 0u || request.structure_id == 0u
        || request.structure_epoch == 0u || request.value_generation == 0u
        || request.edges.global_edge_begin
            > std::numeric_limits<std::uint64_t>::max()
                - request.edges.local_edge_count)
        return status_v1::invalid_argument;
    if (request.kind > indexed_gate_kind_v1::factorized_source_destination)
        return status_v1::unsupported;
    const bool needs_secondary = request.kind
        == indexed_gate_kind_v1::factorized_source_destination;
    if (needs_secondary != (request.secondary_gate != nullptr))
        return status_v1::invalid_argument;
    return status_v1::success;
}

status_v1 enqueue_indexed_gate_v1(
    const indexed_gate_request_v1 &request) noexcept {
    const status_v1 validation = validate_indexed_gate_request_v1(request);
    if (validation != status_v1::success) return validation;
    constexpr std::uint32_t threads = 256u;
    constexpr std::uint32_t maximum_blocks = 65535u;
    const std::uint32_t required =
        (request.edges.local_edge_count + threads - 1u) / threads;
    indexed_gate_kernel<<<required < maximum_blocks ? required : maximum_blocks,
        threads, 0u, request.stream>>>(request);
    return cudaGetLastError() == cudaSuccess
        ? status_v1::success : status_v1::cuda_failure;
}

} // namespace cellerator::compute::operation::edge
