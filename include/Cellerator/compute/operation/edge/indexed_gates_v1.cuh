#pragma once

#include <Cellerator/compute/operation/edge/edge_operations_v1.cuh>

namespace cellerator::compute::operation::edge {

enum class indexed_gate_kind_v1 : std::uint8_t {
    per_source = 0u,
    per_destination = 1u,
    per_component = 2u,
    factorized_source_destination = 3u
};

struct edge_coordinate_v1 {
    std::uint32_t source_local = 0u;
    std::uint32_t destination_local = 0u;
    std::uint32_t component_local = 0u;
};

struct indexed_gate_request_v1 {
    local_edge_slice_v1 edges{};
    const edge_coordinate_v1 *coordinates = nullptr;
    const float *input = nullptr;
    float *output = nullptr;
    indexed_gate_kind_v1 kind = indexed_gate_kind_v1::per_source;
    const float *primary_gate = nullptr;
    const float *secondary_gate = nullptr;
    std::uint32_t source_count = 0u;
    std::uint32_t destination_count = 0u;
    std::uint32_t component_count = 0u;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_indexed_gate_request_v1(
    const indexed_gate_request_v1 &request) noexcept;
status_v1 enqueue_indexed_gate_v1(
    const indexed_gate_request_v1 &request) noexcept;

static_assert(std::is_trivially_copyable<edge_coordinate_v1>::value,
    "edge coordinates are compact physical descriptors");

} // namespace cellerator::compute::operation::edge
