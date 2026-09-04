#include <Cellerator/compiler/sema/field/deliver_the_first_profile_required_semantic_field_slice_v1.hh>

#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

namespace operation = cellerator::compute::operation::v2;

bool same_axis(const cellerator::execution::persistent_axis_identity& lhs,
               const cellerator::execution::persistent_axis_identity& rhs) noexcept {
    return cellerator::execution::same_identity(lhs.domain, rhs.domain) &&
        cellerator::execution::same_identity(lhs.order, rhs.order) &&
        cellerator::execution::same_identity(lhs.geometry, rhs.geometry) &&
        cellerator::execution::same_identity(lhs.partition, rhs.partition);
}

operation::stable_id field_problem_identity(execution_field_identity_v1 field,
                                            std::uint64_t operation_identity) noexcept {
    operation::stable_id identity{
        field.low ^ (operation_identity + 0x9e3779b97f4a7c15ull),
        field.high ^ (operation_identity * 0xbf58476d1ce4e5b9ull),
    };
    if (!operation::valid_stable_id(identity)) identity.high = 1;
    return identity;
}

}  // namespace

operation::operation_problem materialize_operation_problem_v1(
    const semantic_operation_problem_v1& semantics) noexcept {
    operation::operation_problem problem;
    problem.kind = semantics.kind;
    problem.orientation = semantics.orientation;
    problem.value_ownership = operation::value_ownership_mode::logical_primary;
    problem.persistent_problem_identity = semantics.persistent_problem_identity;
    problem.operation_identity = semantics.operation_identity;
    problem.relations = {&semantics.relation, 1};
    problem.values_axis = semantics.values_axis;
    problem.result_axis = semantics.result_axis;
    problem.logical_edge_order = semantics.relation.logical_edge_order;
    problem.expected_value_generation = semantics.expected_value_generation;
    problem.numeric = semantics.numeric;
    problem.output = semantics.output;
    problem.determinism = semantics.determinism;
    problem.logical_work_items = semantics.logical_work_items;
    problem.dense_width = semantics.dense_width;
    problem.requirement_flags = semantics.requirement_flags;
    return problem;
}

profile_required_semantic_field_status_v1
deliver_the_first_profile_required_semantic_field_slice_v1(
    const profile_required_semantic_field_request_v1& request,
    profile_required_semantic_field_receipt_v1* receipt) noexcept {
    if (receipt == nullptr) {
        return profile_required_semantic_field_status_v1::invalid_output;
    }

    execution_field_semantics_v1 field;
    if (define_execution_field_semantic_ownership_v1(request.field, &field) !=
        execution_field_definition_status_v1::success) {
        return profile_required_semantic_field_status_v1::invalid_field;
    }
    const auto& resolved = request.operation;
    if (resolved.declaration == nullptr || resolved.qualified_name.empty() ||
        resolved.operation_identity == 0 || !resolved.source.valid()) {
        return profile_required_semantic_field_status_v1::unresolved_cpp_operation;
    }
    if (!execution_field_owns_operation_v1(field, resolved.source)) {
        return profile_required_semantic_field_status_v1::operation_outside_field;
    }
    if (!operation::validate_typed_relation(resolved.relation)) {
        return profile_required_semantic_field_status_v1::invalid_relation;
    }

    const auto& expected_values = resolved.orientation == operation::relation_orientation::forward
        ? resolved.relation.source_axis : resolved.relation.destination_axis;
    const auto& expected_result = resolved.orientation == operation::relation_orientation::forward
        ? resolved.relation.destination_axis : resolved.relation.source_axis;
    if (!same_axis(resolved.values_axis, expected_values) ||
        !same_axis(resolved.result_axis, expected_result)) {
        return profile_required_semantic_field_status_v1::axis_mismatch;
    }
    if (!field.profile_environment.bound() || request.selected_profile.empty()) {
        return profile_required_semantic_field_status_v1::missing_profile_environment;
    }

    representative_profile_binding_v1 profile_binding;
    if (implement_named_representative_profile_binding_v1(
            field, request.profiles, request.profile_aliases,
            {{resolved.operation_identity, request.selected_profile}}, &profile_binding) !=
        profile_binding_status_v1::success) {
        return profile_required_semantic_field_status_v1::profile_binding_failed;
    }

    automatic_semantic_state_v1 output_state = request.input_state;
    std::vector<materialized_generation_transition_v1> transitions;
    automatic_generation_transfer_v1 transfer = request.generation_transfer;
    if (transfer.operation_identity != resolved.operation_identity ||
        implement_automatic_lifetime_and_generation_transfer_v1(
            &output_state, transfer, 1, &transitions) != generation_transfer_status_v1::success ||
        transitions.size() != 1) {
        return profile_required_semantic_field_status_v1::generation_transfer_failed;
    }

    semantic_operation_problem_v1 semantic_problem;
    semantic_problem.persistent_problem_identity =
        field_problem_identity(field.identity, resolved.operation_identity);
    semantic_problem.operation_identity = {resolved.operation_identity, field.identity.low};
    semantic_problem.relation = resolved.relation;
    semantic_problem.kind = resolved.kind;
    semantic_problem.orientation = resolved.orientation;
    semantic_problem.values_axis = resolved.values_axis;
    semantic_problem.result_axis = resolved.result_axis;
    semantic_problem.expected_value_generation = {output_state.value_generation};
    semantic_problem.numeric = resolved.numeric;
    semantic_problem.output = resolved.output;
    semantic_problem.determinism = resolved.determinism;
    semantic_problem.logical_work_items = resolved.logical_work_items;
    semantic_problem.dense_width = resolved.dense_width;
    semantic_problem.requirement_flags = resolved.requirement_flags;
    const auto problem = materialize_operation_problem_v1(semantic_problem);
    if (!operation::validate_operation_problem(problem)) {
        return profile_required_semantic_field_status_v1::operation_problem_invalid;
    }

    profile_required_semantic_field_receipt_v1 result;
    result.field = std::move(field);
    result.profile_binding = std::move(profile_binding);
    result.selected_profile = result.profile_binding.operations.front();
    result.output_state = output_state;
    result.generation_transition = transitions.front();
    result.operation_problem = semantic_problem;
    result.resolved_declaration_name = resolved.qualified_name;
    result.physical_execution_selected = false;
    *receipt = std::move(result);
    return profile_required_semantic_field_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
