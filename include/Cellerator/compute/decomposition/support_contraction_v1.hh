#pragma once

#include <Cellerator/compute/decomposition/destination_disjoint_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t support_contraction_schema_version_v1 = 1u;

enum class support_contraction_split_v1 : std::uint8_t {
    destination_disjoint = 1u,
    source_partial = 2u
};

struct support_contraction_decomposition_v1 {
    std::uint32_t schema_version = support_contraction_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    std::uint64_t split_extent = 0u;
    const destination_interval_v1 *intervals = nullptr;
    std::uint64_t interval_count = 0u;
    support_contraction_split_v1 split =
        support_contraction_split_v1::destination_disjoint;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::destination;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = false;
    bool requires_partial_algebra = false;
    std::uint8_t reserved2[2]{};
};

enum class support_contraction_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_split,
    invalid_split_extent,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct support_contraction_validation_result_v1 {
    support_contraction_validation_code_v1 code =
        support_contraction_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == support_contraction_validation_code_v1::ok;
    }
};

support_contraction_validation_result_v1
validate_support_contraction_decomposition_v1(
    const support_contraction_decomposition_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<support_contraction_split_v1>);
static_assert(
    std::is_trivially_copyable_v<support_contraction_decomposition_v1>);
static_assert(std::is_standard_layout_v<support_contraction_decomposition_v1>);
static_assert(
    std::is_trivially_copyable_v<support_contraction_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
