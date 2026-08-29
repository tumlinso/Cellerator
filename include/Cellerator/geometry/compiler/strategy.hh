#pragma once

#include <Cellerator/geometry/compiler/problem.hh>
#include <Cellerator/geometry/relation_cover.hh>
#include <Cellerator/geometry/work_layout.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler {

inline constexpr u32 geometry_search_policy_schema_version = 1u;
inline constexpr u32 geometry_strategy_schema_version = 1u;

enum class geometry_search_tier : u8 {
    instant = 1u,
    bounded = 2u,
    offline = 3u,
    external = 4u
};

enum geometry_search_tier_mask : u32 {
    geometry_search_tier_none = 0u,
    geometry_search_tier_instant = 1u << 0u,
    geometry_search_tier_bounded = 1u << 1u,
    geometry_search_tier_offline = 1u << 2u,
    geometry_search_tier_external = 1u << 3u
};

struct geometry_search_policy_v1 {
    u32 schema_version = geometry_search_policy_schema_version;
    geometry_search_tier tier = geometry_search_tier::instant;
    u8 reserved[3]{};
    u64 strategy_id = 0u;
    u64 maximum_iterations = 0u;
    u64 maximum_work_units = 0u;
};

struct geometry_strategy_requirements_v1 {
    u64 workspace_bytes = 0u;
    u64 workspace_alignment = 1u;
    u64 work_item_capacity = 0u;
    u64 component_capacity = 0u;
    u64 logical_edge_capacity = 0u;
};

struct geometry_strategy_workspace_v1 {
    void *data = nullptr;
    u64 bytes = 0u;
};

struct geometry_solution_buffers_v1 {
    u32 *execution_to_window = nullptr;
    u32 *window_to_execution = nullptr;
    u64 work_item_capacity = 0u;
    semantic_component_v1 *components = nullptr;
    u64 component_capacity = 0u;
    u64 *logical_edge_ids = nullptr;
    u64 logical_edge_capacity = 0u;
};

struct geometry_solution_v1 {
    work_layout_view_v1 work_layout{};
    relation_cover_view_v1 relation_cover{};
    u64 strategy_id = 0u;
    geometry_search_tier tier = geometry_search_tier::instant;
    u8 reserved[7]{};
};

enum class geometry_strategy_status : u8 {
    ok = 0u,
    invalid_argument = 1u,
    unsupported_policy = 2u,
    strategy_not_found = 3u,
    invalid_registry = 4u,
    requirements_failed = 5u,
    insufficient_workspace = 6u,
    misaligned_workspace = 7u,
    insufficient_output_capacity = 8u,
    strategy_failed = 9u
};

using geometry_strategy_requirements_fn = geometry_strategy_status (*)(
    const geometry_problem_v1 &,
    const geometry_search_policy_v1 &,
    geometry_strategy_requirements_v1 *) noexcept;

using geometry_strategy_execute_fn = geometry_strategy_status (*)(
    const geometry_problem_v1 &,
    const geometry_search_policy_v1 &,
    geometry_strategy_workspace_v1,
    geometry_solution_buffers_v1,
    geometry_solution_v1 *) noexcept;

struct geometry_strategy_descriptor_v1 {
    u32 schema_version = geometry_strategy_schema_version;
    u32 supported_tiers = geometry_search_tier_none;
    u64 strategy_id = 0u;
    geometry_strategy_requirements_fn query_requirements = nullptr;
    geometry_strategy_execute_fn execute = nullptr;
};

constexpr u32 geometry_search_tier_bit(geometry_search_tier tier) noexcept {
    switch (tier) {
    case geometry_search_tier::instant:
        return geometry_search_tier_instant;
    case geometry_search_tier::bounded:
        return geometry_search_tier_bounded;
    case geometry_search_tier::offline:
        return geometry_search_tier_offline;
    case geometry_search_tier::external:
        return geometry_search_tier_external;
    }
    return geometry_search_tier_none;
}

constexpr bool strategy_supports_tier(
    const geometry_strategy_descriptor_v1 &strategy,
    geometry_search_tier tier) noexcept {
    const u32 bit = geometry_search_tier_bit(tier);
    return bit != 0u && (strategy.supported_tiers & bit) != 0u;
}

static_assert(std::is_trivially_copyable<geometry_search_policy_v1>::value,
    "geometry search policies must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<geometry_strategy_requirements_v1>::value,
    "geometry strategy requirements must remain pointer-copyable");
static_assert(std::is_trivially_copyable<geometry_solution_buffers_v1>::value,
    "geometry solution buffers must remain pointer-copyable");
static_assert(std::is_trivially_copyable<geometry_solution_v1>::value,
    "geometry solutions must remain pointer-copyable");
static_assert(
    std::is_trivially_copyable<geometry_strategy_descriptor_v1>::value,
    "geometry strategy descriptors must remain pointer-copyable");

} // namespace cellerator::geometry::compiler
