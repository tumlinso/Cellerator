#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <Cellerator/geometry/relation_cover.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t support_edge_rectangle_schema_version_v1 = 1u;

enum class support_edge_rectangle_mode_v1 : std::uint8_t {
    logical_edge = 1u,
    semantic_rectangle = 2u
};

struct support_edge_rectangle_fragment_v1 {
    std::uint64_t logical_edge_begin = 0u;
    std::uint64_t logical_edge_count = 0u;
    std::uint32_t first_component = 0u;
    std::uint32_t component_count = 0u;
};

struct support_edge_rectangle_decomposition_v1 {
    std::uint32_t schema_version = support_edge_rectangle_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    std::uint64_t relation_index = 0u;
    const geometry::relation_cover_view_v1 *cover = nullptr;
    const support_edge_rectangle_fragment_v1 *fragments = nullptr;
    std::uint64_t fragment_count = 0u;
    support_edge_rectangle_mode_v1 mode =
        support_edge_rectangle_mode_v1::logical_edge;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = true;
    bool requires_partial_algebra = true;
    std::uint8_t reserved2[3]{};
};

enum class support_edge_rectangle_validation_code_v1 : std::uint8_t {
    ok = 0u,
    unsupported_schema,
    nonzero_reserved,
    invalid_identity,
    missing_problem,
    invalid_problem,
    unsupported_operation,
    invalid_relation_index,
    missing_cover,
    invalid_cover,
    relation_edge_count_mismatch,
    missing_fragments,
    invalid_fragment_count,
    invalid_mode,
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_fragment,
    edge_offset_mismatch,
    edge_range_overflow,
    component_offset_mismatch,
    component_range_overflow,
    nonrectangular_component,
    rectangle_edge_mismatch,
    incomplete_partition
};

struct support_edge_rectangle_validation_result_v1 {
    support_edge_rectangle_validation_code_v1 code =
        support_edge_rectangle_validation_code_v1::ok;
    std::uint64_t fragment_index = 0u;
    std::uint32_t component_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == support_edge_rectangle_validation_code_v1::ok;
    }
};

support_edge_rectangle_validation_result_v1
validate_support_edge_rectangle_decomposition_v1(
    const support_edge_rectangle_decomposition_v1 &decomposition,
    geometry::relation_cover_validation_workspace workspace) noexcept;

static_assert(
    std::is_trivially_copyable_v<support_edge_rectangle_fragment_v1>);
static_assert(std::is_standard_layout_v<support_edge_rectangle_fragment_v1>);
static_assert(
    std::is_trivially_copyable_v<support_edge_rectangle_decomposition_v1>);
static_assert(
    std::is_standard_layout_v<support_edge_rectangle_decomposition_v1>);
static_assert(std::is_trivially_copyable_v<
    support_edge_rectangle_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
