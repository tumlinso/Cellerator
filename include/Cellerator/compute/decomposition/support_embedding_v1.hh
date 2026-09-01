#pragma once

#include <Cellerator/compute/decomposition/dense_width_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t support_embedding_schema_version_v1 = 1u;

// Embedding coordinates are independent outputs of support contraction.
// Exact channel slices therefore need concatenation/view assembly, not a
// numerical partial-result algebra.
struct support_embedding_decomposition_v1 {
    std::uint32_t schema_version = support_embedding_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    const dense_width_interval_v1 *embedding_intervals = nullptr;
    std::uint64_t embedding_interval_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::dense_channel;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = false;
    bool requires_partial_algebra = false;
    std::uint8_t reserved2[3]{};
};

enum class support_embedding_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_embedding_width,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct support_embedding_validation_result_v1 {
    support_embedding_validation_code_v1 code =
        support_embedding_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == support_embedding_validation_code_v1::ok;
    }
};

support_embedding_validation_result_v1
validate_support_embedding_decomposition_v1(
    const support_embedding_decomposition_v1 &decomposition) noexcept;

static_assert(
    std::is_trivially_copyable_v<support_embedding_decomposition_v1>);
static_assert(std::is_standard_layout_v<support_embedding_decomposition_v1>);
static_assert(
    std::is_trivially_copyable_v<support_embedding_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
