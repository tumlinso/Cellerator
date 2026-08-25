#include <Cellerator/distributed/hierarchy.hh>

#include <cmath>

namespace cellerator::distributed {
namespace {

bool same_partition(execution::partition_id lhs,
    execution::partition_id rhs) noexcept {
    return execution::same_identity(lhs, rhs);
}

bool same_order(execution::order_id lhs, execution::order_id rhs) noexcept {
    return execution::same_identity(lhs, rhs);
}

bool find_partition(const partition_hierarchy_view &hierarchy,
    execution::partition_id identity, std::uint32_t *index) noexcept {
    for (std::uint32_t candidate = 0u;
         candidate < hierarchy.partition_count; ++candidate)
        if (same_partition(hierarchy.partitions[candidate].identity, identity)) {
            if (index != nullptr) *index = candidate;
            return true;
        }
    return false;
}

bool active_module_at(const module_activity_view &activity,
    std::uint32_t index) noexcept {
    return activity.active[index] != 0u;
}

bool valid_cost_model(const boundary_cost_model &cost) noexcept {
    return std::isfinite(cost.peer_launch_ns) && cost.peer_launch_ns >= 0.0
        && std::isfinite(cost.staged_launch_ns)
        && cost.staged_launch_ns >= 0.0
        && std::isfinite(cost.peer_bytes_per_ns)
        && cost.peer_bytes_per_ns > 0.0
        && std::isfinite(cost.staged_bytes_per_ns)
        && cost.staged_bytes_per_ns > 0.0
        && std::isfinite(cost.order_transform_ns_per_value)
        && cost.order_transform_ns_per_value >= 0.0;
}

bool present(planner::operation_core::stable_id identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

} // namespace

hierarchy_status validate_partition_hierarchy(
    const partition_hierarchy_view &hierarchy) noexcept {
    if (hierarchy.schema_version != hierarchy_schema_version
        || hierarchy.partitions == nullptr || hierarchy.partition_count == 0u)
        return hierarchy_status::invalid_argument;
    if (!execution::valid_identity(hierarchy.identity))
        return hierarchy_status::invalid_identity;
    std::uint32_t root_count = 0u;
    for (std::uint32_t index = 0u; index < hierarchy.partition_count; ++index) {
        const nested_partition &partition = hierarchy.partitions[index];
        if (!execution::valid_identity(partition.identity)
            || partition.device_ordinal < 0)
            return hierarchy_status::invalid_partition;
        for (std::uint32_t prior = 0u; prior < index; ++prior)
            if (same_partition(
                    hierarchy.partitions[prior].identity, partition.identity))
                return hierarchy_status::invalid_partition;
        if (!execution::valid_identity(partition.parent)) {
            if (partition.level != 0u) return hierarchy_status::invalid_partition;
            ++root_count;
            continue;
        }
        std::uint32_t parent = 0u;
        if (!find_partition(hierarchy, partition.parent, &parent)
            || parent >= index
            || partition.level != hierarchy.partitions[parent].level + 1u)
            return hierarchy_status::invalid_partition;
    }
    return root_count == 1u ? hierarchy_status::ok
                            : hierarchy_status::invalid_partition;
}

hierarchy_status validate_shared_value_hierarchy(
    const partition_hierarchy_view &hierarchy,
    const shared_value_hierarchy_view &values) noexcept {
    const hierarchy_status hierarchy_valid =
        validate_partition_hierarchy(hierarchy);
    if (hierarchy_valid != hierarchy_status::ok) return hierarchy_valid;
    if (values.schema_version != hierarchy_schema_version
        || !execution::same_identity(hierarchy.identity, values.hierarchy)
        || values.modules == nullptr || values.module_count == 0u
        || (values.value_index_count != 0u && values.value_indices == nullptr)
        || values.shared_value_count == 0u)
        return hierarchy_status::invalid_argument;
    if (!execution::valid_identity(values.structure))
        return hierarchy_status::invalid_identity;
    if (values.epoch.value == 0u) return hierarchy_status::stale_structure;
    std::uint64_t expected_offset = 0u;
    for (std::uint32_t index = 0u; index < values.module_count; ++index) {
        const shared_value_module &module = values.modules[index];
        if (module.identity == 0u
            || !find_partition(hierarchy, module.partition, nullptr)
            || module.value_offset != expected_offset
            || static_cast<std::uint64_t>(module.value_offset)
                    + module.value_count > values.value_index_count
            || (module.parent_module != invalid_hierarchy_index
                && module.parent_module >= index))
            return hierarchy_status::invalid_module;
        for (std::uint32_t prior = 0u; prior < index; ++prior)
            if (values.modules[prior].identity == module.identity)
                return hierarchy_status::invalid_module;
        expected_offset += module.value_count;
    }
    if (expected_offset != values.value_index_count)
        return hierarchy_status::invalid_module;
    for (std::uint32_t index = 0u; index < values.value_index_count; ++index)
        if (values.value_indices[index] >= values.shared_value_count)
            return hierarchy_status::invalid_module;
    return hierarchy_status::ok;
}

hierarchy_status build_active_module_plan(
    const shared_value_hierarchy_view &values,
    const module_activity_view &activity,
    active_module_plan *plan) noexcept {
    if (plan == nullptr) return hierarchy_status::invalid_argument;
    plan->count = 0u;
    if (values.schema_version != hierarchy_schema_version
        || values.modules == nullptr || values.module_count == 0u)
        return hierarchy_status::invalid_argument;
    if (!execution::same_identity(values.hierarchy, activity.hierarchy)
        || !execution::same_identity(values.structure, activity.structure)
        || values.epoch.value != activity.epoch.value)
        return hierarchy_status::stale_structure;
    if (activity.generation.value == 0u || activity.active == nullptr
        || activity.module_count != values.module_count)
        return hierarchy_status::stale_values;
    if (!execution::same_identity(plan->hierarchy, values.hierarchy)
        || plan->generation.value != activity.generation.value
        || (plan->capacity != 0u && plan->modules == nullptr))
        return hierarchy_status::stale_values;
    std::uint32_t required = 0u;
    for (std::uint32_t index = 0u; index < values.module_count; ++index)
        if (active_module_at(activity, index)) ++required;
    if (required > plan->capacity) {
        plan->count = required;
        return hierarchy_status::insufficient_capacity;
    }
    for (std::uint32_t index = 0u; index < values.module_count; ++index) {
        if (!active_module_at(activity, index)) continue;
        const shared_value_module &module = values.modules[index];
        plan->modules[plan->count++] = {index, module.partition,
            module.value_offset, module.value_count};
    }
    return hierarchy_status::ok;
}

hierarchy_status plan_boundary_communication(
    const partition_hierarchy_view &hierarchy,
    const shared_value_hierarchy_view &values,
    const module_activity_view &activity,
    const boundary_edge *boundaries,
    std::uint32_t boundary_count,
    const boundary_cost_model &cost,
    communication_plan *plan) noexcept {
    if (plan == nullptr || boundaries == nullptr || boundary_count == 0u
        || !valid_cost_model(cost))
        return hierarchy_status::invalid_argument;
    plan->count = 0u;
    plan->total = {};
    if (validate_shared_value_hierarchy(hierarchy, values)
            != hierarchy_status::ok)
        return hierarchy_status::invalid_module;
    if (!execution::same_identity(values.hierarchy, activity.hierarchy)
        || !execution::same_identity(values.structure, activity.structure)
        || values.epoch.value != activity.epoch.value)
        return hierarchy_status::stale_structure;
    if (activity.generation.value == 0u || activity.active == nullptr
        || activity.module_count != values.module_count
        || !execution::same_identity(plan->hierarchy, hierarchy.identity)
        || plan->activity_generation.value != activity.generation.value
        || (plan->capacity != 0u && plan->steps == nullptr))
        return hierarchy_status::stale_values;

    for (std::uint32_t index = 0u; index < boundary_count; ++index) {
        const boundary_edge &edge = boundaries[index];
        if (edge.source_module >= values.module_count
            || edge.destination_module >= values.module_count
            || edge.value_count == 0u || edge.byte_count == 0u
            || !execution::valid_identity(edge.source_order)
            || !execution::valid_identity(edge.destination_order)
            || !same_partition(values.modules[edge.source_module].partition,
                edge.source_partition)
            || !same_partition(values.modules[edge.destination_module].partition,
                edge.destination_partition))
            return hierarchy_status::invalid_order;
        std::uint32_t source_partition = 0u;
        std::uint32_t destination_partition = 0u;
        if (!find_partition(hierarchy, edge.source_partition,
                &source_partition)
            || !find_partition(hierarchy, edge.destination_partition,
                &destination_partition))
            return hierarchy_status::invalid_partition;
        if (!active_module_at(activity, edge.source_module)
            || !active_module_at(activity, edge.destination_module))
            continue;
        const std::int32_t source_device =
            hierarchy.partitions[source_partition].device_ordinal;
        const std::int32_t destination_device =
            hierarchy.partitions[destination_partition].device_ordinal;
        const bool reorder = !same_order(
            edge.source_order, edge.destination_order);
        if (source_device == destination_device && !reorder) continue;
        if (plan->count == plan->capacity)
            return hierarchy_status::insufficient_capacity;

        communication_step &step = plan->steps[plan->count++];
        step = {};
        step.boundary_index = index;
        step.source_partition = edge.source_partition;
        step.destination_partition = edge.destination_partition;
        step.source_device = source_device;
        step.destination_device = destination_device;
        step.order = reorder ? execution::order_transition_kind::transform
                             : execution::order_transition_kind::preserve;
        if (reorder)
            step.phases.order_transform_ns = edge.value_count
                * cost.order_transform_ns_per_value;
        if (source_device == destination_device) {
            step.transfer = boundary_transfer_kind::local_reorder;
        } else if (edge.peer_access) {
            step.transfer = boundary_transfer_kind::peer_copy;
            step.phases.communication_ns = cost.peer_launch_ns
                + edge.byte_count / cost.peer_bytes_per_ns;
            step.phases.communication_bytes = edge.byte_count;
        } else {
            step.transfer = boundary_transfer_kind::staged_copy;
            step.phases.communication_ns = cost.staged_launch_ns
                + edge.byte_count / cost.staged_bytes_per_ns;
            step.phases.communication_bytes = edge.byte_count;
        }
        plan->total.order_transform_ns += step.phases.order_transform_ns;
        plan->total.communication_ns += step.phases.communication_ns;
        plan->total.communication_bytes += step.phases.communication_bytes;
    }
    return hierarchy_status::ok;
}

hierarchy_status make_connected_transition(
    const communication_step &step,
    std::uint32_t connected_boundary,
    planner::operation_core::stable_id producer,
    planner::operation_core::stable_id consumer,
    planner::operation_core::stable_id order_conversion,
    planner::connected_transition_cost *transition) noexcept {
    if (transition == nullptr || !present(producer) || !present(consumer)
        || (step.order != execution::order_transition_kind::preserve
            && step.order != execution::order_transition_kind::transform)
        || (step.order == execution::order_transition_kind::transform
            && !present(order_conversion))
        || (step.order == execution::order_transition_kind::preserve
            && present(order_conversion)))
        return hierarchy_status::invalid_argument;
    *transition = {};
    transition->boundary = connected_boundary;
    transition->producer = producer;
    transition->consumer = consumer;
    transition->order = step.order;
    transition->legal = true;
    transition->conversion = order_conversion;
    transition->phases = step.phases;
    return hierarchy_status::ok;
}

} // namespace cellerator::distributed
