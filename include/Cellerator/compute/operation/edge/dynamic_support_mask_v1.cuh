#pragma once

#include <Cellerator/compute/operation/edge/edge_operations_v1.cuh>

namespace cellerator::compute::operation::edge {

enum class mask_encoding_v1 : std::uint8_t {
    byte_per_edge = 0u,
    packed_lsb_bits = 1u
};

struct dynamic_support_mask_request_v1 {
    local_edge_slice_v1 stable_support_superset{};
    const float *input = nullptr;
    float *output = nullptr;
    const std::uint8_t *active_mask = nullptr;
    mask_encoding_v1 encoding = mask_encoding_v1::byte_per_edge;
    std::uint64_t structure_id = 0u;
    std::uint64_t structure_epoch = 0u;
    std::uint64_t input_value_generation = 0u;
    std::uint64_t mask_value_generation = 0u;
    std::uint64_t output_value_generation = 0u;
    std::uint64_t profiler_correlation_id = 0u;
    cudaStream_t stream = nullptr;
};

status_v1 validate_dynamic_support_mask_request_v1(
    const dynamic_support_mask_request_v1 &request) noexcept;
status_v1 enqueue_dynamic_support_mask_v1(
    const dynamic_support_mask_request_v1 &request) noexcept;

} // namespace cellerator::compute::operation::edge
