#include <Cellerator/compute/decomposition/gate_input_v1.hh>

namespace cellerator::compute::decomposition {
namespace {

gate_input_validation_result_v1 failure(gate_input_validation_code_v1 code,
    std::uint64_t input_index = 0u) noexcept {
    return {code, input_index};
}

bool dependency_matches_axis(operation::v2::gate_indexing dependency,
    split_axis_kind_v1 axis) noexcept {
    using dependency_kind = operation::v2::gate_indexing;
    return (dependency == dependency_kind::per_edge
            && axis == split_axis_kind_v1::logical_edge)
        || (dependency == dependency_kind::per_source
            && axis == split_axis_kind_v1::source)
        || (dependency == dependency_kind::per_destination
            && axis == split_axis_kind_v1::destination)
        || (dependency == dependency_kind::per_component
            && axis == split_axis_kind_v1::semantic_component);
}

bool valid_replication(gate_input_replication_v1 replication) noexcept {
    return replication >= gate_input_replication_v1::partition_local
        && replication <= gate_input_replication_v1::producer_routed;
}

}  // namespace

gate_input_validation_result_v1 validate_gate_dependent_input_set_v1(
    const gate_dependent_input_set_v1 &set) noexcept {
    using code = gate_input_validation_code_v1;
    using dependency_kind = operation::v2::gate_indexing;

    if (set.schema_version != gate_input_schema_version_v1)
        return failure(code::unsupported_schema);
    if (set.reserved != 0u)
        return failure(code::nonzero_reserved);
    if (!operation::v2::valid_stable_id(set.identity))
        return failure(code::invalid_identity);
    if (set.problem == nullptr)
        return failure(code::missing_problem);
    if (!operation::v2::validate_relation_algebra_problem(*set.problem))
        return failure(code::invalid_problem);
    if (set.problem->core.kind
        != operation::v2::operation_kind::edge_map_or_gate)
        return failure(code::unsupported_operation);
    if (set.input_count == 0u)
        return failure(code::invalid_input_count);
    if (set.inputs == nullptr)
        return failure(code::missing_inputs);

    bool saw_factorized_source = false;
    bool saw_factorized_destination = false;
    std::uint64_t previous_operand = 0u;
    for (std::uint64_t index = 0u; index < set.input_count; ++index) {
        const auto &input = set.inputs[index];
        if (input.operand_index == operation::v2::invalid_binding_index)
            return failure(code::invalid_operand, index);
        if (index != 0u && input.operand_index <= previous_operand)
            return failure(code::operand_order_mismatch, index);
        previous_operand = input.operand_index;
        if (input.dependency <= dependency_kind::none
            || input.dependency > dependency_kind::predicate)
            return failure(code::invalid_dependency, index);

        if (set.problem->gate == dependency_kind::factorized_source_destination) {
            if (input.dependency == dependency_kind::per_source)
                saw_factorized_source = true;
            else if (input.dependency == dependency_kind::per_destination)
                saw_factorized_destination = true;
            else
                return failure(code::dependency_mismatch, index);
        } else if (input.dependency != set.problem->gate) {
            return failure(code::dependency_mismatch, index);
        }
        if (!valid_split_axis_kind_v1(input.split_axis)
            || input.split_axis == split_axis_kind_v1::none)
            return failure(code::invalid_split_axis, index);
        if (!valid_replication(input.replication))
            return failure(code::invalid_replication, index);

        if (input.replication == gate_input_replication_v1::partition_local) {
            if (!dependency_matches_axis(input.dependency, input.split_axis))
                return failure(code::invalid_replication, index);
            if (input.replica_or_halo_count != 1u)
                return failure(code::invalid_replica_count, index);
        } else {
            if (!input.read_only)
                return failure(code::mutable_replica, index);
            const std::uint32_t minimum = input.replication
                    == gate_input_replication_v1::replicated_read_only
                ? 2u : 1u;
            if (input.replica_or_halo_count < minimum)
                return failure(code::invalid_replica_count, index);
        }
    }
    if (set.problem->gate == dependency_kind::factorized_source_destination
        && (!saw_factorized_source || !saw_factorized_destination))
        return failure(code::missing_factorized_dependency, set.input_count);
    return {};
}

}  // namespace cellerator::compute::decomposition
