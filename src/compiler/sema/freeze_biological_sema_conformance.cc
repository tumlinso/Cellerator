#include <Cellerator/compiler/sema/freeze_biological_sema_conformance_v1.hh>

namespace cellerator::compiler::sema::v1 {
namespace {
using namespace compute::operation::v2;

source_operation_kind source_kind(operation_kind kind) noexcept {
    switch (kind) {
    case operation_kind::relation_apply: return source_operation_kind::relation_apply;
    case operation_kind::relation_apply_transpose: return source_operation_kind::relation_transpose;
    case operation_kind::contract_on_support: return source_operation_kind::support_contraction;
    case operation_kind::segment_reduce: return source_operation_kind::segment_statistics;
    case operation_kind::segment_normalize: return source_operation_kind::normalization;
    case operation_kind::edge_map_or_gate: return source_operation_kind::edge_map_or_gate;
    case operation_kind::relation_bundle_apply: return source_operation_kind::relation_bundle;
    case operation_kind::sparse_axis_update: return source_operation_kind::sparse_update;
    }
    return source_operation_kind::relation_apply;
}

output_effect effect(destination_update update) noexcept {
    switch (update) {
    case destination_update::overwrite: return output_effect::assign;
    case destination_update::accumulate: return output_effect::shared_destination_accumulate;
    case destination_update::affine_accumulate: return output_effect::epilogue;
    case destination_update::partial_write: return output_effect::partial_output;
    }
    return output_effect::assign;
}

bool same_id(execution::persistent_axis_identity a,
             execution::persistent_axis_identity b) noexcept {
    return execution::same_identity(a.domain, b.domain)
        && execution::same_identity(a.order, b.order)
        && execution::same_identity(a.geometry, b.geometry)
        && execution::same_identity(a.partition, b.partition);
}

bool same_core(const operation_problem &a, const operation_problem &b) noexcept {
    if (a.schema_version != b.schema_version || a.kind != b.kind
        || a.orientation != b.orientation || a.value_ownership != b.value_ownership
        || !same_stable_id(a.persistent_problem_identity, b.persistent_problem_identity)
        || !same_stable_id(a.operation_identity, b.operation_identity)
        || a.relations.relations != b.relations.relations
        || a.relations.relation_count != b.relations.relation_count
        || !same_id(a.values_axis, b.values_axis) || !same_id(a.result_axis, b.result_axis)
        || !execution::same_identity(a.logical_edge_order, b.logical_edge_order)
        || a.expected_value_generation.value != b.expected_value_generation.value
        || a.logical_work_items != b.logical_work_items || a.dense_width != b.dense_width
        || a.requirement_flags != b.requirement_flags)
        return false;
    return a.numeric.relation_storage == b.numeric.relation_storage
        && a.numeric.state_storage == b.numeric.state_storage
        && a.numeric.multiply == b.numeric.multiply
        && a.numeric.accumulation == b.numeric.accumulation
        && a.numeric.output_storage == b.numeric.output_storage
        && a.numeric.scalar == b.numeric.scalar
        && a.numeric.rounding == b.numeric.rounding
        && a.numeric.saturation == b.numeric.saturation
        && a.numeric.nan == b.numeric.nan
        && a.numeric.infinity == b.numeric.infinity
        && a.output.update == b.output.update
        && a.output.order == b.output.order
        && a.output.explicit_order_transform == b.output.explicit_order_transform
        && a.output.input_output_aliasing_legal == b.output.input_output_aliasing_legal
        && a.output.alpha_binding == b.output.alpha_binding
        && a.output.beta_binding == b.output.beta_binding
        && same_id(a.output.produced_axis, b.output.produced_axis)
        && same_id(a.output.canonical_axis, b.output.canonical_axis)
        && a.determinism.deterministic_required == b.determinism.deterministic_required
        && a.determinism.stable_work_order == b.determinism.stable_work_order
        && a.determinism.fixed_reduction_tree == b.determinism.fixed_reduction_tree
        && a.determinism.nondeterministic_atomics_permitted == b.determinism.nondeterministic_atomics_permitted
        && a.determinism.deterministic_seed_binding == b.determinism.deterministic_seed_binding;
}
}  // namespace

biological_sema_problem lower_through_biological_sema(
    const operation_problem &problem) noexcept {
    biological_sema_problem result{};
    result.preserved.core = problem;
    result.operation = source_kind(problem.kind);
    result.numeric = {problem.numeric.relation_storage, problem.numeric.state_storage,
        problem.numeric.multiply, problem.numeric.accumulation,
        problem.numeric.output_storage,
        problem.numeric.nan == nan_policy::reject ? nonfinite_contract::reject
                                                   : nonfinite_contract::propagate,
        precision_contract::mixed, approximation_contract::forbidden};
    result.output = resolve_output_effect(effect(problem.output.update),
                                          problem.output.input_output_aliasing_legal);
    result.output.requires_order_transform = problem.output.explicit_order_transform;
    return result;
}

biological_sema_problem lower_through_biological_sema(
    const relation_algebra_problem &problem) noexcept {
    auto result = lower_through_biological_sema(problem.core);
    result.preserved = problem;
    result.relation_algebra_present = true;
    return result;
}

operation_problem recover_operation_problem(
    const biological_sema_problem &problem) noexcept {
    return problem.preserved.core;
}

relation_algebra_problem recover_relation_algebra_problem(
    const biological_sema_problem &problem) noexcept {
    return problem.preserved;
}

bool planning_information_preserved(
    const relation_algebra_problem &source,
    const biological_sema_problem &lowered) noexcept {
    const auto recovered = recover_relation_algebra_problem(lowered);
    return lowered.relation_algebra_present && same_core(source.core, recovered.core)
        && source.bindings.bindings == recovered.bindings.bindings
        && source.bindings.binding_count == recovered.bindings.binding_count
        && source.value_bindings == recovered.value_bindings
        && source.value_binding_count == recovered.value_binding_count
        && source.segment == recovered.segment && source.edge == recovered.edge
        && source.gate == recovered.gate
        && source.semantic_flags == recovered.semantic_flags
        && source.reserved_flags == recovered.reserved_flags;
}

}  // namespace cellerator::compiler::sema::v1
