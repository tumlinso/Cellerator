#include <Cellerator/compiler/sema/field/field_semantics_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            {seed, 1}, {seed, 2}, {seed, 3}, {seed, 4}};
}

field::profile_required_semantic_field_request_v1 request() {
    field::profile_required_semantic_field_request_v1 value;
    value.field.stable_source_name = "first_profile_field.cell";
    value.field.explicit_field_name = "propagate";
    value.field.source = {{7, 100}, {7, 900}};
    value.field.profile_environment = {"pbmc3k", 11, 12};
    value.operation.source = {{7, 300}, {7, 500}};
    value.operation.declaration = reinterpret_cast<const void*>(0x1);
    value.operation.qualified_name = "cell::apply";
    value.operation.operation_identity = 71;
    value.operation.relation.structure = {21, 22};
    value.operation.relation.epoch = {3};
    value.operation.relation.source_axis = axis(31);
    value.operation.relation.destination_axis = axis(41);
    value.operation.relation.logical_edge_order = {51, 52};
    value.operation.relation.logical_edge_count = 1024;
    value.operation.values_axis = value.operation.relation.source_axis;
    value.operation.result_axis = value.operation.relation.destination_axis;
    value.operation.numeric.relation_storage = execution::numeric_type::f32;
    value.operation.numeric.state_storage = execution::numeric_type::f32;
    value.operation.numeric.multiply = execution::numeric_type::f32;
    value.operation.numeric.accumulation = execution::numeric_type::f32;
    value.operation.numeric.output_storage = execution::numeric_type::f32;
    value.operation.numeric.scalar = execution::numeric_type::f32;
    value.operation.output.produced_axis = value.operation.result_axis;
    value.operation.output.canonical_axis = value.operation.result_axis;
    value.operation.logical_work_items = 1024;
    value.operation.dense_width = 16;
    value.operation.requirement_flags = operation::require_forward |
        operation::dynamic_values;
    value.profiles = {{"baseline", 61, 62, 63, true, false}};
    value.selected_profile = "baseline";
    value.input_state = {21, 3, 8, 2, 4, field::semantic_lifetime_v1::alive};
    value.generation_transfer = {
        71, field::state_component_value_v1,
        field::generation_transition_kind_v1::known_operation, false};
    return value;
}

}  // namespace

int main() {
    auto input = request();
    field::profile_required_semantic_field_receipt_v1 receipt;
    if (field::deliver_the_first_profile_required_semantic_field_slice_v1(
            input, &receipt) != field::profile_required_semantic_field_status_v1::success) {
        std::cerr << "profile-required semantic field did not compile\n";
        return 1;
    }
    const auto problem = field::materialize_operation_problem_v1(receipt.operation_problem);
    if (!operation::validate_operation_problem(problem) ||
        problem.kind != operation::operation_kind::relation_apply ||
        problem.relations.relation_count != 1 ||
        problem.relations.relations != &receipt.operation_problem.relation ||
        problem.expected_value_generation.value != 9 ||
        receipt.selected_profile.state_identity != 61 ||
        receipt.output_state.value_generation != 9 ||
        receipt.resolved_declaration_name != "cell::apply" ||
        receipt.physical_execution_selected) {
        std::cerr << "semantic receipt does not match operation_core_v2\n";
        return 1;
    }

    auto missing_profile = request();
    missing_profile.field.profile_environment = {};
    if (field::deliver_the_first_profile_required_semantic_field_slice_v1(
            missing_profile, &receipt) !=
        field::profile_required_semantic_field_status_v1::missing_profile_environment) {
        std::cerr << "field compiled without its required profile environment\n";
        return 1;
    }
    auto wrong_axis = request();
    wrong_axis.operation.values_axis = axis(99);
    if (field::deliver_the_first_profile_required_semantic_field_slice_v1(
            wrong_axis, &receipt) != field::profile_required_semantic_field_status_v1::axis_mismatch) {
        std::cerr << "typed relation accepted the wrong biological axis\n";
        return 1;
    }
    auto outside = request();
    outside.operation.source = {{7, 901}, {7, 902}};
    if (field::deliver_the_first_profile_required_semantic_field_slice_v1(
            outside, &receipt) !=
        field::profile_required_semantic_field_status_v1::operation_outside_field) {
        std::cerr << "field accepted an operation outside its source ownership\n";
        return 1;
    }
    return 0;
}
