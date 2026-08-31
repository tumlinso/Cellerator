#pragma once

#include <cstdint>
#include <type_traits>

#include <Cellerator/execution/index_space/hierarchical_index_space_v1.hh>
#include <Cellerator/geometry/index/scalable_views_v2.hh>

namespace cellerator::geometry {

enum class scale_validation_code_v2 : std::uint32_t {
    valid = 0u,
    null_pointer,
    invalid_width,
    width_too_small,
    arithmetic_overflow,
    component_order,
    aggregate_discontinuity,
    aggregate_extent_mismatch,
    global_index_out_of_range,
    offset_not_monotonic,
    offset_extent_mismatch,
    local_index_out_of_range,
    duplicate_edge,
    missing_edge,
    workspace_too_small,
};

struct scale_validation_result_v2 {
    scale_validation_code_v2 code = scale_validation_code_v2::valid;
    std::uint32_t reserved = 0u;
    std::uint64_t component = 0u;
    std::uint64_t item = 0u;
    std::uint64_t operations = 0u;
};

// Caller-owned generation marks permit exact cover validation using memory
// bounded by the largest local component rather than the aggregate relation.
struct cover_validation_workspace_v2 {
    std::uint64_t *marks = nullptr;
    std::uint64_t capacity = 0u;
    std::uint64_t generation = 1u;
};

bool checked_add_u64_v2(std::uint64_t left, std::uint64_t right,
                        std::uint64_t *out) noexcept;
bool checked_multiply_u64_v2(std::uint64_t left, std::uint64_t right,
                             std::uint64_t *out) noexcept;
bool local_width_can_represent_v2(execution::local_index_width_v1 width,
                                  std::uint64_t extent) noexcept;
bool load_compact_index_v2(const void *data,
                           execution::local_index_width_v1 width,
                           std::uint64_t position,
                           std::uint64_t *out) noexcept;

scale_validation_result_v2 validate_hierarchical_index_space_v1(
    const execution::hierarchical_index_space_view_v1 &view) noexcept;

scale_validation_result_v2 validate_scalable_support_v2(
    const scalable_support_view_v2 &view) noexcept;

scale_validation_result_v2 validate_exact_cover_v2(
    const scalable_cover_view_v2 &cover,
    const scalable_support_view_v2 &support,
    cover_validation_workspace_v2 workspace) noexcept;

// Linear count/scan construction primitive.  output has count + 1 entries;
// operations, when non-null, is incremented exactly once per input element.
scale_validation_code_v2 exclusive_scan_counts_v2(
    const std::uint64_t *counts, std::uint64_t count,
    std::uint64_t *output, std::uint64_t output_capacity,
    std::uint64_t *operations = nullptr) noexcept;

static_assert(std::is_trivially_copyable_v<scale_validation_result_v2>);
static_assert(std::is_trivially_copyable_v<cover_validation_workspace_v2>);

}  // namespace cellerator::geometry
