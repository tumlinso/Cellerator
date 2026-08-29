#pragma once

#include <Cellerator/geometry/work_window.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry {

inline constexpr u32 admissibility_schema_version = 1u;
inline constexpr u32 invalid_admissibility_record_index = ~u32{0u};

// Admissibility is cold compiler input. It constrains regrouping of the real
// work items selected by a work window; it never describes physical padding or
// a provider schedule.
enum class admissibility_constraint_kind : u8 {
    fixed_position = 1u,
    fixed_original_group_membership = 2u,
    must_link = 3u,
    cannot_share_group = 4u,
    precedence = 5u,
    partition_barrier = 6u,
    bounded_exchange_window = 7u
};

// All axis positions name real members of the associated work window unless a
// kind explicitly states otherwise. `related` has kind-specific meaning:
//
// - fixed_position: destination position in the portable work permutation;
// - fixed_original_group_membership: required original group;
// - must_link, cannot_share_group, precedence: another axis position.
//
// partition_barrier uses lower_bound and upper_bound for two adjacent original
// groups and leaves subject/related zero. bounded_exchange_window constrains
// subject to the inclusive original-group range [lower_bound, upper_bound].
// Unused fields must be zero so future versions can extend the record safely.
struct admissibility_record_v1 {
    admissibility_constraint_kind kind =
        admissibility_constraint_kind::fixed_position;
    u8 reserved[3]{};
    execution::axis_identity axis{};
    u32 subject = 0u;
    u32 related = 0u;
    u32 lower_bound = 0u;
    u32 upper_bound = 0u;
};

// `original_group_count` is required only by group-qualified constraints. The
// caller owns the records and their lifetime. A default-constructed view is the
// cheap permissive contract: it has no constraints and requires no pointer,
// group metadata, graph compilation, or traversal.
struct admissibility_view_v1 {
    u32 schema_version = admissibility_schema_version;
    u32 reserved = 0u;
    u32 original_group_count = 0u;
    u32 record_count = 0u;
    const admissibility_record_v1 *records = nullptr;
};

enum class admissibility_validation_code : u8 {
    ok = 0u,
    unsupported_version = 1u,
    nonzero_reserved = 2u,
    missing_records = 3u,
    invalid_kind = 4u,
    invalid_axis = 5u,
    axis_mismatch = 6u,
    subject_not_in_window = 7u,
    related_not_in_window = 8u,
    self_relation = 9u,
    invalid_position = 10u,
    missing_original_groups = 11u,
    group_out_of_bounds = 12u,
    invalid_partition_barrier = 13u,
    invalid_exchange_window = 14u,
    nonzero_unused_field = 15u,
    conflicting_constraint = 16u
};

struct admissibility_validation_result {
    admissibility_validation_code code = admissibility_validation_code::ok;
    u32 record_index = invalid_admissibility_record_index;
    u32 conflicting_record_index = invalid_admissibility_record_index;

    constexpr explicit operator bool() const noexcept {
        return code == admissibility_validation_code::ok;
    }
};

constexpr bool valid_admissibility_constraint_kind(
    admissibility_constraint_kind kind) noexcept {
    return kind == admissibility_constraint_kind::fixed_position
        || kind
            == admissibility_constraint_kind::fixed_original_group_membership
        || kind == admissibility_constraint_kind::must_link
        || kind == admissibility_constraint_kind::cannot_share_group
        || kind == admissibility_constraint_kind::precedence
        || kind == admissibility_constraint_kind::partition_barrier
        || kind == admissibility_constraint_kind::bounded_exchange_window;
}

constexpr bool admissibility_is_permissive(
    const admissibility_view_v1 &admissibility) noexcept {
    return admissibility.record_count == 0u;
}

// This validates only the admissibility layer. The work window remains
// independently validated by validate_work_window. In particular, the
// zero-record route returns after checking only the fixed-size view header.
admissibility_validation_result validate_admissibility(
    const work_window_view_v1 &window,
    const admissibility_view_v1 &admissibility) noexcept;

static_assert(std::is_trivially_copyable<admissibility_record_v1>::value,
    "admissibility records must remain pointer-copyable");
static_assert(std::is_trivially_copyable<admissibility_view_v1>::value,
    "admissibility source views must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<admissibility_validation_result>::value,
    "admissibility validation results must remain trivially copyable");

} // namespace cellerator::geometry
