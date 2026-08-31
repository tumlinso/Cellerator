#pragma once

#include <cuda_runtime.h>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::operation::edge {

enum class status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    unsupported = 2u,
    cuda_failure = 3u
};

enum class map_kind_v1 : std::uint8_t {
    identity = 0u,
    affine = 1u,
    clamp = 2u,
    absolute = 3u,
    exponential = 4u,
    logarithm = 5u,
    reciprocal = 6u
};

enum class gate_kind_v1 : std::uint8_t {
    none = 0u,
    per_edge_multiplicative = 1u,
    per_edge_predicate = 2u
};

struct local_edge_slice_v1 {
    std::uint64_t global_edge_begin = 0u;
    std::uint32_t local_edge_count = 0u;
};

struct map_parameters_v1 {
    float first = 0.0f;
    float second = 0.0f;
};

struct edge_map_request_v1 {
    local_edge_slice_v1 edges{};
    const float *input = nullptr;
    float *output = nullptr;
    map_kind_v1 map = map_kind_v1::identity;
    gate_kind_v1 gate = gate_kind_v1::none;
    const void *per_edge_gate = nullptr;
    map_parameters_v1 parameters{};
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t value_generation = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_edge_map_request_v1(
    const edge_map_request_v1 &request) noexcept;
status_v1 enqueue_edge_map_v1(const edge_map_request_v1 &request) noexcept;

static_assert(std::is_trivially_copyable<local_edge_slice_v1>::value,
    "local edge slices are ABI values");
static_assert(std::is_trivially_copyable<edge_map_request_v1>::value,
    "edge map launches are non-owning bindings");

} // namespace cellerator::compute::operation::edge
