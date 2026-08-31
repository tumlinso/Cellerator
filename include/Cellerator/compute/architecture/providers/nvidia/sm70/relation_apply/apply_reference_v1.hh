#pragma once

#include <cstdint>

namespace cellerator::compute::architecture::nvidia::sm70::relation_apply {

enum class apply_reference_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_offsets,
    insufficient_capacity,
    arithmetic_overflow,
};

struct apply_reference_request_v1 {
    const float *relation_tiles = nullptr;
    const std::uint32_t *destination_tile_offsets = nullptr;
    const std::uint32_t *tile_source_bases = nullptr;
    const float *dense_rhs = nullptr;
    float *output = nullptr;
    std::uint64_t output_capacity = 0u;
    std::uint32_t tile_count = 0u;
    std::uint32_t destination_group_count = 0u;
    std::uint32_t local_source_count = 0u;
    std::uint32_t rows_per_group = 0u;
    std::uint32_t dense_width = 0u;
};

// Independent host referee for dense relation tiles. It is intentionally
// simple, linear in useful multiply-adds, and shares no CUDA implementation.
apply_reference_status_v1 apply_dense_tile_reference_v1(
    const apply_reference_request_v1 &request) noexcept;

}  // namespace cellerator::compute::architecture::nvidia::sm70::relation_apply
