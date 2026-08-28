#include <Cellerator/planner/distributed/hierarchy.hh>

#include <cassert>
#include <cmath>
#include <cstdint>

namespace distributed = cellerator::distributed;
namespace execution = cellerator::execution;
namespace planner = cellerator::planner;

namespace {

struct fixture {
    execution::partition_id root{1u, 90u};
    execution::partition_id left{2u, 90u};
    execution::partition_id right{3u, 90u};
    distributed::nested_partition partitions[3]{
        {root, {}, 0u, 0},
        {left, root, 1u, 0},
        {right, root, 1u, 1}};
    distributed::partition_hierarchy_view hierarchy{
        distributed::hierarchy_schema_version, {10u, 90u}, partitions, 3u};
    distributed::shared_value_module modules[3]{
        {100u, root, distributed::invalid_hierarchy_index, 0u, 2u},
        {101u, left, 0u, 2u, 2u},
        {102u, right, 0u, 4u, 3u}};
    std::uint32_t indices[7]{0u, 1u, 1u, 2u, 0u, 2u, 3u};
    distributed::shared_value_hierarchy_view values{
        distributed::hierarchy_schema_version, hierarchy.identity,
        {20u, 90u}, {4u}, modules, indices, 3u, 7u, 4u};
    std::uint8_t active[3]{1u, 1u, 1u};
    distributed::module_activity_view activity{
        hierarchy.identity, values.structure, values.epoch, {7u}, active, 3u};
};

void test_nested_identity_and_shared_values() {
    fixture value;
    assert(distributed::validate_partition_hierarchy(value.hierarchy)
        == distributed::hierarchy_status::ok);
    assert(distributed::validate_shared_value_hierarchy(
        value.hierarchy, value.values) == distributed::hierarchy_status::ok);

    distributed::active_module storage[3]{};
    distributed::active_module_plan plan{
        value.hierarchy.identity, value.activity.generation, storage, 3u, 0u};
    value.active[1] = 0u;
    assert(distributed::build_active_module_plan(
        value.values, value.activity, &plan) == distributed::hierarchy_status::ok);
    assert(plan.count == 2u && plan.modules[0].module_index == 0u
        && plan.modules[1].module_index == 2u);
    assert(plan.modules[1].value_count == 3u);

    distributed::active_module_plan short_plan{
        value.hierarchy.identity, value.activity.generation, storage, 1u, 0u};
    assert(distributed::build_active_module_plan(
        value.values, value.activity, &short_plan)
        == distributed::hierarchy_status::insufficient_capacity);
    assert(short_plan.count == 2u);

    value.activity.generation.value = 8u;
    plan.generation = value.activity.generation;
    assert(distributed::build_active_module_plan(
        value.values, value.activity, &plan) == distributed::hierarchy_status::ok);
    assert(plan.count == 2u);

    plan.generation.value += 1u;
    assert(distributed::build_active_module_plan(
        value.values, value.activity, &plan)
        == distributed::hierarchy_status::stale_values);
    value.values.epoch.value += 1u;
    assert(distributed::build_active_module_plan(
        value.values, value.activity, &plan)
        == distributed::hierarchy_status::stale_structure);

    fixture malformed;
    malformed.partitions[2].parent = malformed.left;
    assert(distributed::validate_partition_hierarchy(malformed.hierarchy)
        == distributed::hierarchy_status::invalid_partition);
}

void test_ordered_boundary_plan_and_cost() {
    fixture value;
    const execution::order_id packed{30u, 90u};
    const execution::order_id canonical{31u, 90u};
    distributed::boundary_edge boundaries[3]{
        {0u, 1u, value.root, value.left, packed, packed,
            2u, 8u, true, {}},
        {1u, 2u, value.left, value.right, packed, canonical,
            4u, 16u, true, {}},
        {0u, 2u, value.root, value.right, packed, packed,
            8u, 32u, false, {}}};
    distributed::communication_step storage[3]{};
    distributed::communication_plan plan{
        value.hierarchy.identity, value.activity.generation, storage, 3u, 0u, {}};
    const distributed::boundary_cost_model cost{
        10.0, 20.0, 4.0, 2.0, 0.5};
    assert(distributed::plan_boundary_communication(value.hierarchy,
        value.values, value.activity, boundaries, 3u, cost, &plan)
        == distributed::hierarchy_status::ok);
    assert(plan.count == 2u);
    assert(plan.steps[0].boundary_index == 1u
        && plan.steps[0].transfer
            == distributed::boundary_transfer_kind::peer_copy
        && plan.steps[0].order == execution::order_transition_kind::transform);
    assert(std::fabs(plan.steps[0].phases.order_transform_ns - 2.0) < 1e-9);
    assert(std::fabs(plan.steps[0].phases.communication_ns - 14.0) < 1e-9);
    assert(plan.steps[1].boundary_index == 2u
        && plan.steps[1].transfer
            == distributed::boundary_transfer_kind::staged_copy);
    assert(std::fabs(plan.total.communication_ns - 50.0) < 1e-9
        && plan.total.communication_bytes == 48u);

    boundaries[0].destination_order = canonical;
    assert(distributed::plan_boundary_communication(value.hierarchy,
        value.values, value.activity, boundaries, 1u, cost, &plan)
        == distributed::hierarchy_status::ok);
    assert(plan.count == 1u
        && plan.steps[0].transfer
            == distributed::boundary_transfer_kind::local_reorder
        && plan.steps[0].phases.communication_bytes == 0u);
    boundaries[0].destination_order = packed;

    value.active[2] = 0u;
    assert(distributed::plan_boundary_communication(value.hierarchy,
        value.values, value.activity, boundaries, 3u, cost, &plan)
        == distributed::hierarchy_status::ok);
    assert(plan.count == 0u && plan.total.communication_bytes == 0u);
}

void test_connected_transition_and_hierarchy_cache_identity() {
    fixture value;
    distributed::communication_step step{};
    step.order = execution::order_transition_kind::transform;
    step.transfer = distributed::boundary_transfer_kind::peer_copy;
    step.phases.communication_ns = 14.0;
    step.phases.communication_bytes = 16u;
    planner::connected_transition_cost transition{};
    assert(distributed::make_connected_transition(step, 0u, {1u, 90u},
        {2u, 90u}, {3u, 90u}, &transition)
        == distributed::hierarchy_status::ok);
    assert(transition.order == execution::order_transition_kind::transform
        && transition.phases.communication_bytes == 16u);

    planner::connected_planning_keys left{};
    left.graph_identity = {10u, 90u};
    left.hierarchy = value.hierarchy.identity;
    left.stage_count = 1u;
    planner::connected_planning_keys right = left;
    assert(planner::same_connected_planning_keys(left, right));
    right.hierarchy.low += 1u;
    assert(!planner::same_connected_planning_keys(left, right));
}

} // namespace

int main() {
    test_nested_identity_and_shared_values();
    test_ordered_boundary_plan_and_cost();
    test_connected_transition_and_hierarchy_cache_identity();
    return 0;
}
