#pragma once

#include <Cellerator/execution/identity.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

using u8 = std::uint8_t;
using u32 = std::uint32_t;

inline constexpr u32 work_window_schema_version = 1u;

struct work_window_tag;
using work_window_id = execution::persistent_identity<work_window_tag>;

// A work window binds one caller-selected set of real work items to exactly
// one biological axis. The kind states how the bound axis participates in the
// operation; it does not weaken or replace the axis identity.
enum class work_window_kind : u8 {
    relation_rows = 1u,
    dense_columns = 2u,
    grouped_operation_instances = 3u
};

// Membership is explicit and caller-owned. Members are positions in `axis`,
// not canonical identities and not padding slots. Cellerator may reorder or
// regroup these members after preparation, but may not infer or enlarge the
// selected set. This is a cold source view and owns no memory.
struct work_window_view_v1 {
    u32 schema_version = work_window_schema_version;
    work_window_kind kind = work_window_kind::relation_rows;
    u8 reserved[3]{};
    work_window_id identity{};
    execution::axis_identity axis{};
    u32 axis_extent = 0u;
    u32 member_count = 0u;
    const u32 *members = nullptr;
};

enum class work_window_validation_code : u8 {
    ok = 0u,
    unsupported_version = 1u,
    invalid_kind = 2u,
    invalid_identity = 3u,
    invalid_axis = 4u,
    invalid_extent = 5u,
    invalid_member_count = 6u,
    missing_members = 7u,
    member_out_of_bounds = 8u,
    duplicate_member = 9u,
    nonzero_reserved = 10u
};

struct work_window_validation_result {
    work_window_validation_code code = work_window_validation_code::ok;
    u32 member_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == work_window_validation_code::ok;
    }
};

constexpr bool valid_work_window_kind(work_window_kind kind) noexcept {
    return kind == work_window_kind::relation_rows
        || kind == work_window_kind::dense_columns
        || kind == work_window_kind::grouped_operation_instances;
}

// Validation is intentionally allocation-free. Duplicate checking is a cold
// O(n^2) reference path; later preparation may use caller-provided workspace
// when it compiles larger windows into portable work layouts.
constexpr work_window_validation_result validate_work_window(
    const work_window_view_v1 &window) noexcept {
    if (window.schema_version != work_window_schema_version)
        return {work_window_validation_code::unsupported_version, 0u};
    if (!valid_work_window_kind(window.kind))
        return {work_window_validation_code::invalid_kind, 0u};
    if (window.reserved[0] != 0u || window.reserved[1] != 0u
        || window.reserved[2] != 0u)
        return {work_window_validation_code::nonzero_reserved, 0u};
    if (!execution::valid_identity(window.identity))
        return {work_window_validation_code::invalid_identity, 0u};
    if (!execution::valid_axis_identity(window.axis))
        return {work_window_validation_code::invalid_axis, 0u};
    if (window.axis_extent == 0u)
        return {work_window_validation_code::invalid_extent, 0u};
    if (window.member_count == 0u || window.member_count > window.axis_extent)
        return {work_window_validation_code::invalid_member_count, 0u};
    if (window.members == nullptr)
        return {work_window_validation_code::missing_members, 0u};

    for (u32 index = 0u; index < window.member_count; ++index) {
        if (window.members[index] >= window.axis_extent)
            return {work_window_validation_code::member_out_of_bounds, index};
        for (u32 previous = 0u; previous < index; ++previous)
            if (window.members[previous] == window.members[index])
                return {work_window_validation_code::duplicate_member, index};
    }
    return {};
}

constexpr bool work_window_contains(
    const work_window_view_v1 &window,
    u32 axis_position) noexcept {
    if (axis_position >= window.axis_extent || window.members == nullptr)
        return false;
    for (u32 index = 0u; index < window.member_count; ++index)
        if (window.members[index] == axis_position)
            return true;
    return false;
}

static_assert(std::is_trivially_copyable<work_window_view_v1>::value,
    "work-window source view must remain pointer-copyable");
static_assert(std::is_trivially_copyable<work_window_validation_result>::value,
    "work-window validation result must remain trivially copyable");

} // namespace cellerator::geometry
