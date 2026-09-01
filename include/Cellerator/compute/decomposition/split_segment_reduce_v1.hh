#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/relation_algebra_v2/relation_algebra.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t split_segment_reduce_schema_version_v1 = 1u;

struct split_segment_fragment_v1 {
    std::uint64_t segment_index = 0u;
    std::uint64_t member_begin = 0u;
    std::uint64_t member_count = 0u;
};

struct split_segment_reduce_decomposition_v1 {
    std::uint32_t schema_version = split_segment_reduce_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::relation_algebra_problem *problem = nullptr;
    const std::uint64_t *segment_member_counts = nullptr;
    std::uint64_t segment_count = 0u;
    const split_segment_fragment_v1 *fragments = nullptr;
    std::uint64_t fragment_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::logical_edge;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = true;
    bool requires_partial_algebra = true;
    std::uint8_t reserved2[3]{};
};

enum class split_segment_reduce_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    missing_segment_counts,
    invalid_segment_count,
    missing_fragments,
    invalid_vocabulary,
    invalid_partial_result_contract,
    segment_index_mismatch,
    empty_fragment,
    member_offset_mismatch,
    member_range_overflow,
    incomplete_segment,
    extra_fragment
};

struct split_segment_reduce_validation_result_v1 {
    split_segment_reduce_validation_code_v1 code =
        split_segment_reduce_validation_code_v1::ok;
    std::uint64_t segment_index = 0u;
    std::uint64_t fragment_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == split_segment_reduce_validation_code_v1::ok;
    }
};

split_segment_reduce_validation_result_v1 validate_split_segment_reduce_v1(
    const split_segment_reduce_decomposition_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<split_segment_fragment_v1>);
static_assert(std::is_standard_layout_v<split_segment_fragment_v1>);
static_assert(
    std::is_trivially_copyable_v<split_segment_reduce_decomposition_v1>);
static_assert(std::is_standard_layout_v<split_segment_reduce_decomposition_v1>);
static_assert(std::is_trivially_copyable_v<
    split_segment_reduce_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
