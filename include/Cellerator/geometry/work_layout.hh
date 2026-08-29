#pragma once

#include <Cellerator/geometry/work_window.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

inline constexpr u32 work_layout_schema_version = 1u;
inline constexpr u32 invalid_work_item = ~u32{0u};

// A portable work layout is an exact permutation of the caller-selected work
// window. Indices address `work_window_view_v1::members`, not the full axis.
// Consequently every representable entry names a real work item; physical
// padding and invalid sentinels belong only to provider projections.
struct work_layout_view_v1 {
    u32 schema_version = work_layout_schema_version;
    u32 reserved = 0u;
    work_window_id work_window{};
    execution::axis_identity axis{};
    u32 work_count = 0u;
    const u32 *execution_to_window = nullptr;
    const u32 *window_to_execution = nullptr;
};

enum class work_layout_validation_code : u8 {
    ok = 0u,
    unsupported_version = 1u,
    nonzero_reserved = 2u,
    invalid_work_window = 3u,
    invalid_work_window_identity = 4u,
    axis_mismatch = 5u,
    work_count_mismatch = 6u,
    missing_permutation = 7u,
    missing_inverse = 8u,
    work_item_out_of_bounds = 9u,
    duplicate_work_item = 10u,
    inverse_out_of_bounds = 11u,
    inverse_mismatch = 12u
};

struct work_layout_validation_result {
    work_layout_validation_code code = work_layout_validation_code::ok;
    u32 index = invalid_work_item;

    constexpr explicit operator bool() const noexcept {
        return code == work_layout_validation_code::ok;
    }
};

enum class work_layout_build_code : u8 {
    ok = 0u,
    invalid_argument = 1u,
    invalid_work_window = 2u,
    insufficient_inverse_capacity = 3u,
    work_item_out_of_bounds = 4u,
    duplicate_work_item = 5u
};

struct work_layout_build_result {
    work_layout_build_code code = work_layout_build_code::ok;
    u32 index = invalid_work_item;

    constexpr explicit operator bool() const noexcept {
        return code == work_layout_build_code::ok;
    }
};

// Build the inverse into caller-owned cold workspace and bind `output` to both
// arrays. The arrays must not overlap. No allocation or canonicalization
// occurs. `execution_to_window` must remain alive for as long as the resulting
// view is used.
work_layout_build_result build_work_layout(
    const work_window_view_v1 &window,
    const u32 *execution_to_window,
    u32 work_count,
    u32 *window_to_execution,
    u32 inverse_capacity,
    work_layout_view_v1 *output) noexcept;

// Independent validation does not trust build_work_layout or either direction
// of the permutation. It proves both arrays are bounded, bijective, and exact
// inverses over the validated work window.
work_layout_validation_result validate_work_layout(
    const work_window_view_v1 &window,
    const work_layout_view_v1 &layout) noexcept;

constexpr u32 work_layout_axis_position(
    const work_window_view_v1 &window,
    const work_layout_view_v1 &layout,
    u32 execution_position) noexcept {
    if (execution_position >= layout.work_count
        || layout.execution_to_window == nullptr || window.members == nullptr)
        return invalid_work_item;
    const u32 window_index = layout.execution_to_window[execution_position];
    return window_index < window.member_count
        ? window.members[window_index]
        : invalid_work_item;
}

static_assert(std::is_trivially_copyable<work_layout_view_v1>::value,
    "work-layout views must remain pointer-copyable");
static_assert(std::is_trivially_copyable<work_layout_validation_result>::value,
    "work-layout validation results must remain trivially copyable");
static_assert(std::is_trivially_copyable<work_layout_build_result>::value,
    "work-layout build results must remain trivially copyable");

} // namespace cellerator::geometry
