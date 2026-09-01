#pragma once

#include <Cellerator/compute/decomposition/destination_disjoint_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t transpose_source_partials_schema_version_v1 =
    1u;

// A transpose relation apply reduces over forward destinations and produces
// forward-source-shaped partials.  Exact destination/K coverage is structural
// evidence only; a separate partial algebra must authorize their combination.
struct transpose_source_partials_v1 {
    std::uint32_t schema_version =
        transpose_source_partials_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    std::uint64_t destination_k_extent = 0u;
    const destination_interval_v1 *destination_k_intervals = nullptr;
    std::uint64_t destination_k_interval_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::destination;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_source_partials = true;
    bool requires_partial_algebra = true;
    std::uint8_t reserved2[3]{};
};

enum class transpose_source_partials_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_destination_k_extent,
    missing_intervals,
    invalid_interval_count,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_interval,
    interval_offset_mismatch,
    interval_range_overflow,
    incomplete_partition
};

struct transpose_source_partials_validation_result_v1 {
    transpose_source_partials_validation_code_v1 code =
        transpose_source_partials_validation_code_v1::ok;
    std::uint64_t interval_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == transpose_source_partials_validation_code_v1::ok;
    }
};

transpose_source_partials_validation_result_v1
validate_transpose_source_partials_v1(
    const transpose_source_partials_v1 &decomposition) noexcept;

static_assert(std::is_trivially_copyable_v<transpose_source_partials_v1>);
static_assert(std::is_standard_layout_v<transpose_source_partials_v1>);
static_assert(std::is_trivially_copyable_v<
    transpose_source_partials_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
