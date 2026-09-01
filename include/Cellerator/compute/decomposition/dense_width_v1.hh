#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t dense_width_schema_version_v1 = 1u;

struct dense_width_interval_v1 {
    std::uint32_t begin = 0u;
    std::uint32_t count = 0u;
};

// Dense channels are independent relation-apply outputs.  Exact disjoint
// coverage permits fragment outputs to remain in persistent order without a
// numerical partial-result merge.
struct dense_width_relation_apply_v1 {
    std::uint32_t schema_version = dense_width_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    const dense_width_interval_v1 *intervals = nullptr;
    std::uint64_t interval_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::dense_channel;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = false;
    std::uint8_t reserved2[4]{};
};

enum class dense_width_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_dense_width,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct dense_width_validation_result_v1 {
    dense_width_validation_code_v1 code = dense_width_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == dense_width_validation_code_v1::ok;
    }
};

dense_width_validation_result_v1 validate_dense_width_relation_apply_v1(
    const dense_width_relation_apply_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<dense_width_interval_v1>);
static_assert(std::is_standard_layout_v<dense_width_interval_v1>);
static_assert(std::is_trivially_copyable_v<dense_width_relation_apply_v1>);
static_assert(std::is_standard_layout_v<dense_width_relation_apply_v1>);
static_assert(std::is_trivially_copyable_v<dense_width_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
