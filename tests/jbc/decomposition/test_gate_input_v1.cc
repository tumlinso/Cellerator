#include <Cellerator/execution/decomposition/gate_input.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::execution::decomposition;
namespace compute_decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

template<typename Identity>
Identity identity(std::uint64_t value) { return {value, value + 1u}; }

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        identity<::cellerator::structure::domain_id>(seed),
        identity<::cellerator::structure::order_id>(seed + 2u),
        identity<::cellerator::structure::geometry_id>(seed + 4u),
        identity<::cellerator::structure::partition_id>(seed + 6u)};
}

operation::relation_algebra_problem problem(operation::typed_relation &relation,
    operation::relation_binding_contract &binding,
    operation::relation_value_binding_contract &values) {
    const auto source = axis(10u);
    const auto destination = axis(20u);
    relation = {identity<::cellerator::structure::structure_id>(30u), {1u}, source,
        destination, identity<::cellerator::structure::order_id>(40u), 8u};
    binding.relation_index = 0u;
    binding.source_state_operand = 0u;
    binding.destination_state_operand = 1u;
    binding.relation_values = 0u;
    values.structure = relation.structure;
    values.epoch = relation.epoch;
    values.generation = {1u};

    operation::relation_algebra_problem result{};
    result.core.kind = operation::operation_kind::edge_map_or_gate;
    result.core.persistent_problem_identity = {50u, 51u};
    result.core.operation_identity = {52u, 53u};
    result.core.relations = {&relation, 1u};
    result.core.values_axis = source;
    result.core.result_axis = destination;
    result.core.logical_edge_order = relation.logical_edge_order;
    result.core.expected_value_generation = {1u};
    result.core.numeric.relation_storage = execution::numeric_type::f32;
    result.core.numeric.state_storage = execution::numeric_type::f32;
    result.core.numeric.multiply = execution::numeric_type::f32;
    result.core.numeric.accumulation = execution::numeric_type::f32;
    result.core.numeric.output_storage = execution::numeric_type::f32;
    result.core.numeric.scalar = execution::numeric_type::f32;
    result.core.output.produced_axis = destination;
    result.core.output.canonical_axis = destination;
    result.core.logical_work_items = 8u;
    result.core.dense_width = 1u;
    result.bindings = {&binding, 1u};
    result.value_bindings = &values;
    result.value_binding_count = 1u;
    result.edge = operation::edge_operation::multiplicative_gate;
    result.gate = operation::gate_indexing::per_source;
    result.semantic_flags = operation::projection_aware_edge_values;
    return result;
}

}  // namespace

int main() {
    operation::typed_relation relation{};
    operation::relation_binding_contract binding{};
    operation::relation_value_binding_contract values{};
    auto algebra = problem(relation, binding, values);
    assert(operation::validate_relation_algebra_problem(algebra));

    decomposition::GateInputV1 input{};
    input.operand_index = 2u;
    input.dependency = operation::gate_indexing::per_source;
    input.split_axis = compute_decomposition::split_axis_kind_v1::source;
    decomposition::GateInputSetV1 set{};
    set.identity = {60u, 61u};
    set.problem = &algebra;
    set.inputs = &input;
    set.input_count = 1u;
    assert(decomposition::validate_gate_input_set_v1(set));

    input.split_axis = compute_decomposition::split_axis_kind_v1::destination;
    auto status = decomposition::validate_gate_input_set_v1(set);
    assert(status.code == decomposition::
        GateInputValidationCodeV1::invalid_replication);

    input.replication =
        decomposition::GateInputReplicationV1::replicated_read_only;
    input.replica_or_halo_count = 2u;
    assert(decomposition::validate_gate_input_set_v1(set));

    input.read_only = false;
    status = decomposition::validate_gate_input_set_v1(set);
    assert(status.code
        == decomposition::GateInputValidationCodeV1::mutable_replica);
    return 0;
}
