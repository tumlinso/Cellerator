#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t destination_disjoint_schema_version_v1 = 1u;

struct destination_interval_v1 {
    std::uint64_t begin = 0u;
    std::uint64_t count = 0u;
};

// A cold, caller-owned decomposition view.  The intervals must be an exact,
// ordered partition of [0, destination_extent).  Therefore every destination
// is produced once and no cross-fragment result reduction is required.
struct destination_disjoint_relation_apply_v1 {
    std::uint32_t schema_version = destination_disjoint_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    std::uint64_t destination_extent = 0u;
    const destination_interval_v1 *intervals = nullptr;
    std::uint64_t interval_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::destination;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    std::uint8_t reserved2[5]{};
};

enum class destination_disjoint_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_destination_extent,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct destination_disjoint_validation_result_v1 {
    destination_disjoint_validation_code_v1 code =
        destination_disjoint_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == destination_disjoint_validation_code_v1::ok;
    }
};

destination_disjoint_validation_result_v1
validate_destination_disjoint_relation_apply_v1(
    const destination_disjoint_relation_apply_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<destination_interval_v1>);
static_assert(std::is_standard_layout_v<destination_interval_v1>);
static_assert(
    std::is_trivially_copyable_v<destination_disjoint_relation_apply_v1>);
static_assert(
    std::is_standard_layout_v<destination_disjoint_relation_apply_v1>);
static_assert(
    std::is_trivially_copyable_v<destination_disjoint_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
