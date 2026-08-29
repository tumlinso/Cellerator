#include <Cellerator/geometry/compiler/strategy_registry.hh>

#include <cstdint>

namespace cellerator::geometry::compiler {
namespace {

constexpr u32 all_search_tier_bits = geometry_search_tier_instant
    | geometry_search_tier_bounded | geometry_search_tier_offline
    | geometry_search_tier_external;

bool workspace_is_aligned(
    const geometry_strategy_workspace_v1 &workspace,
    u64 alignment) noexcept {
    if (alignment <= 1u)
        return true;
    if ((alignment & (alignment - 1u)) != 0u || workspace.data == nullptr)
        return false;
    const auto address = reinterpret_cast<std::uintptr_t>(workspace.data);
    return address % alignment == 0u;
}

bool buffers_satisfy(
    const geometry_solution_buffers_v1 &buffers,
    const geometry_strategy_requirements_v1 &requirements) noexcept {
    if (buffers.work_item_capacity < requirements.work_item_capacity
        || buffers.component_capacity < requirements.component_capacity
        || buffers.logical_edge_capacity < requirements.logical_edge_capacity)
        return false;
    if (requirements.work_item_capacity != 0u
        && (buffers.execution_to_window == nullptr
            || buffers.window_to_execution == nullptr))
        return false;
    if (requirements.component_capacity != 0u && buffers.components == nullptr)
        return false;
    return requirements.logical_edge_capacity == 0u
        || buffers.logical_edge_ids != nullptr;
}

} // namespace

geometry_strategy_status validate_geometry_strategy_registry(
    geometry_strategy_registry_v1 registry) noexcept {
    if (registry.strategy_count == 0u || registry.strategies == nullptr)
        return geometry_strategy_status::invalid_registry;
    for (u32 index = 0u; index < registry.strategy_count; ++index) {
        const geometry_strategy_descriptor_v1 &strategy =
            registry.strategies[index];
        if (strategy.schema_version != geometry_strategy_schema_version
            || strategy.strategy_id == 0u
            || strategy.supported_tiers == geometry_search_tier_none
            || (strategy.supported_tiers & ~all_search_tier_bits) != 0u
            || strategy.query_requirements == nullptr
            || strategy.execute == nullptr)
            return geometry_strategy_status::invalid_registry;
        for (u32 previous = 0u; previous < index; ++previous)
            if (registry.strategies[previous].strategy_id
                == strategy.strategy_id)
                return geometry_strategy_status::invalid_registry;
    }
    return geometry_strategy_status::ok;
}

const geometry_strategy_descriptor_v1 *find_geometry_strategy(
    geometry_strategy_registry_v1 registry,
    u64 strategy_id,
    geometry_search_tier tier) noexcept {
    if (validate_geometry_strategy_registry(registry)
            != geometry_strategy_status::ok
        || strategy_id == 0u || geometry_search_tier_bit(tier) == 0u)
        return nullptr;
    for (u32 index = 0u; index < registry.strategy_count; ++index)
        if (registry.strategies[index].strategy_id == strategy_id
            && strategy_supports_tier(registry.strategies[index], tier))
            return &registry.strategies[index];
    return nullptr;
}

geometry_strategy_status query_geometry_strategy_requirements(
    geometry_strategy_registry_v1 registry,
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_requirements_v1 *requirements) noexcept {
    if (policy.schema_version != geometry_search_policy_schema_version
        || requirements == nullptr)
        return geometry_strategy_status::invalid_argument;
    const geometry_strategy_descriptor_v1 *strategy = find_geometry_strategy(
        registry, policy.strategy_id, policy.tier);
    if (strategy == nullptr)
        return geometry_strategy_status::strategy_not_found;
    geometry_strategy_requirements_v1 result{};
    const geometry_strategy_status status =
        strategy->query_requirements(problem, policy, &result);
    if (status != geometry_strategy_status::ok)
        return geometry_strategy_status::requirements_failed;
    if (result.workspace_alignment == 0u
        || (result.workspace_alignment & (result.workspace_alignment - 1u))
            != 0u)
        return geometry_strategy_status::requirements_failed;
    *requirements = result;
    return geometry_strategy_status::ok;
}

geometry_strategy_status run_geometry_strategy(
    geometry_strategy_registry_v1 registry,
    const geometry_problem_v1 &problem,
    const geometry_search_policy_v1 &policy,
    geometry_strategy_workspace_v1 workspace,
    geometry_solution_buffers_v1 buffers,
    geometry_solution_v1 *solution) noexcept {
    if (solution == nullptr)
        return geometry_strategy_status::invalid_argument;
    const geometry_strategy_descriptor_v1 *strategy = find_geometry_strategy(
        registry, policy.strategy_id, policy.tier);
    if (strategy == nullptr)
        return geometry_strategy_status::strategy_not_found;

    geometry_strategy_requirements_v1 requirements{};
    const geometry_strategy_status query_status =
        query_geometry_strategy_requirements(
            registry, problem, policy, &requirements);
    if (query_status != geometry_strategy_status::ok)
        return query_status;
    if (workspace.bytes < requirements.workspace_bytes
        || (requirements.workspace_bytes != 0u && workspace.data == nullptr))
        return geometry_strategy_status::insufficient_workspace;
    if (!workspace_is_aligned(workspace, requirements.workspace_alignment))
        return geometry_strategy_status::misaligned_workspace;
    if (!buffers_satisfy(buffers, requirements))
        return geometry_strategy_status::insufficient_output_capacity;

    geometry_solution_v1 result{};
    const geometry_strategy_status status = strategy->execute(
        problem, policy, workspace, buffers, &result);
    if (status != geometry_strategy_status::ok)
        return geometry_strategy_status::strategy_failed;
    result.strategy_id = strategy->strategy_id;
    result.tier = policy.tier;
    *solution = result;
    return geometry_strategy_status::ok;
}

} // namespace cellerator::geometry::compiler
