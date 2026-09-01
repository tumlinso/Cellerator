#pragma once

#include <Cellerator/compute/decomposition/destination_disjoint_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t source_k_schema_version_v1 = 1u;

// Splitting the source/K reduction axis produces destination-shaped partials.
// This contract records only exact structural coverage; it deliberately does
// not claim that combining those partials is numerically legal.
struct source_k_relation_apply_v1 {
    std::uint32_t schema_version = source_k_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    std::uint64_t source_extent = 0u;
    const destination_interval_v1 *source_intervals = nullptr;
    std::uint64_t source_interval_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::source;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = true;
    bool requires_partial_algebra = true;
    std::uint8_t reserved2[3]{};
};

enum class source_k_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_source_extent,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct source_k_validation_result_v1 {
    source_k_validation_code_v1 code = source_k_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == source_k_validation_code_v1::ok;
    }
};

source_k_validation_result_v1 validate_source_k_relation_apply_v1(
    const source_k_relation_apply_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<source_k_relation_apply_v1>);
static_assert(std::is_standard_layout_v<source_k_relation_apply_v1>);
static_assert(std::is_trivially_copyable_v<source_k_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
