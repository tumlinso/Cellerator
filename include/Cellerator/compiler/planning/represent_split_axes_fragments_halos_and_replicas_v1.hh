#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::planning {

enum class planning_split_axis_v1 : std::uint8_t {
    none = 0u,
    source,
    destination,
    relation_edges,
    semantic_components,
    segments,
    modules,
    extents,
};

enum planning_fragment_role_v1 : std::uint32_t {
    exact_input_read_v1 = 1u << 0u,
    exact_output_owner_v1 = 1u << 1u,
    exact_contribution_owner_v1 = 1u << 2u,
    read_only_halo_v1 = 1u << 3u,
    physical_replica_v1 = 1u << 4u,
};

struct planning_fragment_v1 {
    std::uint64_t fragment_identity = 0u;
    std::uint64_t contributor_identity = 0u;
    std::uint64_t logical_begin = 0u;
    std::uint64_t logical_count = 0u;
    std::uint64_t extent_lower_bound = 0u;
    std::uint64_t extent_upper_bound = 0u;
    std::uint64_t input_order_identity = 0u;
    std::uint64_t output_order_identity = 0u;
    std::uint64_t replica_group_identity = 0u;
    std::uint32_t roles = 0u;
};

struct planning_decomposition_v1 {
    std::uint64_t decomposition_identity = 0u;
    planning_split_axis_v1 split_axis = planning_split_axis_v1::none;
    std::uint64_t exact_logical_extent = 0u;
    std::vector<planning_fragment_v1> fragments;
};

enum class planning_decomposition_validation_code_v1 : std::uint8_t {
    ok = 0u,
    invalid_decomposition,
    invalid_fragment,
    invalid_extent_bounds,
    invalid_role,
    duplicate_contributor,
    overlapping_exact_coverage,
    incomplete_exact_coverage,
    invalid_halo,
    invalid_replica,
};

[[nodiscard]] planning_decomposition_validation_code_v1
validate_planning_decomposition_v1(
    const planning_decomposition_v1& decomposition) noexcept;

}  // namespace Cellerator::compiler::planning
