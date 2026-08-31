#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

inline constexpr std::uint32_t transpose_cover_schema_v1 = 1u;
inline constexpr std::uint32_t invalid_local_index_v1 = ~std::uint32_t{0u};

enum class transpose_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    insufficient_capacity,
    invalid_order,
    duplicate_identity,
    arithmetic_overflow,
    invalid_cover,
};

struct global_relation_edge_v1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t global_source_id = 0u;
    std::uint64_t global_destination_id = 0u;
};

// The caller supplies both deterministic orders. source_order is ordered by
// (global_source_id, global_destination_id, logical_edge_id); identity_order
// is ordered by logical_edge_id. This permits exact linear validation without
// hidden allocation, hashing, or quadratic duplicate scans.
struct transpose_cover_input_v1 {
    const global_relation_edge_v1 *edges = nullptr;
    const std::uint64_t *source_order = nullptr;
    const std::uint64_t *identity_order = nullptr;
    std::uint64_t edge_count = 0u;
    std::uint64_t forward_cover_id = 0u;
    std::uint64_t transpose_cover_id = 0u;
};

struct transpose_cover_requirements_v1 {
    std::uint64_t placement_count = 0u;
    std::uint64_t owner_count = 0u;
};

struct transpose_edge_placement_v1 {
    std::uint64_t logical_edge_id = 0u;
    std::uint64_t global_source_id = 0u;
    std::uint64_t global_destination_id = 0u;
    std::uint32_t local_source_index = invalid_local_index_v1;
    std::uint32_t local_destination_index = invalid_local_index_v1;
    std::uint64_t projection_value_position = 0u;
};

struct source_owner_schedule_v1 {
    std::uint64_t global_source_id = 0u;
    std::uint64_t placement_begin = 0u;
    std::uint64_t placement_count = 0u;
    std::uint32_t local_source_index = invalid_local_index_v1;
    std::uint32_t reserved = 0u;
};

struct transpose_cover_storage_v1 {
    transpose_edge_placement_v1 *placements = nullptr;
    std::uint64_t placement_capacity = 0u;
    source_owner_schedule_v1 *owners = nullptr;
    std::uint64_t owner_capacity = 0u;
};

struct transpose_cover_view_v1 {
    std::uint32_t schema_version = transpose_cover_schema_v1;
    std::uint32_t reserved = 0u;
    std::uint64_t forward_cover_id = 0u;
    std::uint64_t transpose_cover_id = 0u;
    const transpose_edge_placement_v1 *placements = nullptr;
    std::uint64_t placement_count = 0u;
    const source_owner_schedule_v1 *owners = nullptr;
    std::uint64_t owner_count = 0u;
};

transpose_status_v1 query_transpose_cover_requirements_v1(
    const transpose_cover_input_v1 &input,
    transpose_cover_requirements_v1 *requirements) noexcept;

transpose_status_v1 build_transpose_cover_v1(
    const transpose_cover_input_v1 &input,
    const transpose_cover_storage_v1 &storage,
    transpose_cover_view_v1 *cover) noexcept;

transpose_status_v1 validate_transpose_cover_v1(
    const transpose_cover_view_v1 &cover) noexcept;

static_assert(std::is_trivially_copyable<transpose_edge_placement_v1>::value,
    "transpose placements must remain pointer-free");
static_assert(std::is_trivially_copyable<source_owner_schedule_v1>::value,
    "transpose schedules must remain pointer-free");
static_assert(std::is_trivially_copyable<transpose_cover_view_v1>::value,
    "transpose covers must remain non-owning views");

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
