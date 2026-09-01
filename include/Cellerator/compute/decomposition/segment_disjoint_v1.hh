#pragma once

#include <Cellerator/compute/decomposition/destination_disjoint_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t segment_disjoint_schema_version_v1 = 1u;

// Whole segments are independent semantic units.  Intervals index segments,
// not their member elements, so no fragment can split a reduction or
// normalization segment.
struct segment_disjoint_decomposition_v1 {
    std::uint32_t schema_version = segment_disjoint_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::relation_algebra_problem *problem = nullptr;
    std::uint64_t segment_count = 0u;
    const destination_interval_v1 *segment_intervals = nullptr;
    std::uint64_t segment_interval_count = 0u;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = false;
    bool requires_partial_algebra = false;
    std::uint8_t reserved2[4]{};
};

enum class segment_disjoint_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_segment_count,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct segment_disjoint_validation_result_v1 {
    segment_disjoint_validation_code_v1 code =
        segment_disjoint_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == segment_disjoint_validation_code_v1::ok;
    }
};

segment_disjoint_validation_result_v1 validate_segment_disjoint_v1(
    const segment_disjoint_decomposition_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<segment_disjoint_decomposition_v1>);
static_assert(std::is_standard_layout_v<segment_disjoint_decomposition_v1>);
static_assert(std::is_trivially_copyable_v<segment_disjoint_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
