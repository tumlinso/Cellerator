#pragma once

#include <Cellerator/compute/decomposition/vocabulary_v1.hh>
#include <Cellerator/compute/operation/operation_core_v2/schema.hh>
#include <Cellerator/geometry/relation_cover.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::decomposition {

inline constexpr std::uint32_t edge_component_schema_version_v1 = 1u;

struct edge_component_fragment_v1 {
    std::uint32_t first_component = 0u;
    std::uint32_t component_count = 0u;
    std::uint64_t logical_edge_begin = 0u;
    std::uint64_t logical_edge_count = 0u;
};

// Each fragment owns whole semantic components and their exact logical-edge
// slices.  Components may share destinations, so fragment outputs are partial
// until a separately validated algebra combines them.
struct edge_component_relation_apply_v1 {
    std::uint32_t schema_version = edge_component_schema_version_v1;
    std::uint32_t reserved = 0u;
    operation::v2::stable_id decomposition_identity{};
    const operation::v2::operation_problem *problem = nullptr;
    std::uint64_t relation_index = 0u;
    const geometry::relation_cover_view_v1 *cover = nullptr;
    const edge_component_fragment_v1 *fragments = nullptr;
    std::uint64_t fragment_count = 0u;
    split_axis_kind_v1 split_axis = split_axis_kind_v1::semantic_component;
    decomposition_kind_v1 kind = decomposition_kind_v1::disjoint;
    fragment_role_v1 fragment_role = fragment_role_v1::owned;
    bool produces_partial_results = true;
    bool requires_partial_algebra = true;
    std::uint8_t reserved2[3]{};
};

enum class edge_component_validation_code_v1 : std::uint8_t {
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
    invalid_vocabulary,
    invalid_partial_result_contract,
    empty_fragment,
    component_offset_mismatch,
    component_range_overflow,
    logical_edge_offset_mismatch,
    logical_edge_count_mismatch,
    incomplete_component_partition
};

struct edge_component_validation_result_v1 {
    edge_component_validation_code_v1 code =
        edge_component_validation_code_v1::ok;
    std::uint64_t fragment_index = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == edge_component_validation_code_v1::ok;
    }
};

edge_component_validation_result_v1 validate_edge_component_relation_apply_v1(
    const edge_component_relation_apply_v1 &decomposition,
    geometry::relation_cover_validation_workspace workspace) noexcept;

static_assert(std::is_trivially_copyable_v<edge_component_fragment_v1>);
static_assert(std::is_standard_layout_v<edge_component_fragment_v1>);
static_assert(std::is_trivially_copyable_v<edge_component_relation_apply_v1>);
static_assert(std::is_standard_layout_v<edge_component_relation_apply_v1>);
static_assert(std::is_trivially_copyable_v<edge_component_validation_result_v1>);

}  // namespace cellerator::compute::decomposition
