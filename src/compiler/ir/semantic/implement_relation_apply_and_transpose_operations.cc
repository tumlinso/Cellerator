#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>

#include <utility>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool same(semantic_identity_v1 left, semantic_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

bool same_numeric(const numeric_tuple_ir_v1& left, const numeric_tuple_ir_v1& right) noexcept {
    return left.storage == right.storage && left.compute == right.compute &&
        left.accumulation == right.accumulation && left.output == right.output;
}

cellerator::compute::operation::v2::stable_id stable(semantic_identity_v1 value) noexcept {
    return {value.low, value.high};
}

}  // namespace

lowered_relation_apply_v1::lowered_relation_apply_v1() noexcept { refresh_views(); }

lowered_relation_apply_v1::lowered_relation_apply_v1(
    const lowered_relation_apply_v1& other) noexcept
    : relation(other.relation), binding(other.binding), value_binding(other.value_binding),
      operation(other.operation), algebra(other.algebra) {
    refresh_views();
}

lowered_relation_apply_v1& lowered_relation_apply_v1::operator=(
    const lowered_relation_apply_v1& other) noexcept {
    if (this != &other) {
        relation = other.relation;
        binding = other.binding;
        value_binding = other.value_binding;
        operation = other.operation;
        algebra = other.algebra;
        refresh_views();
    }
    return *this;
}

lowered_relation_apply_v1::lowered_relation_apply_v1(
    lowered_relation_apply_v1&& other) noexcept
    : lowered_relation_apply_v1(static_cast<const lowered_relation_apply_v1&>(other)) {}

lowered_relation_apply_v1& lowered_relation_apply_v1::operator=(
    lowered_relation_apply_v1&& other) noexcept {
    return *this = static_cast<const lowered_relation_apply_v1&>(other);
}

void lowered_relation_apply_v1::refresh_views() noexcept {
    operation.relations = {&relation, 1};
    algebra.core = operation;
    algebra.bindings = {&binding, 1};
    algebra.value_bindings = &value_binding;
    algebra.value_binding_count = 1;
}

relation_apply_ir_validation_code_v1 validate_relation_apply_operation_ir_v1(
    const relation_apply_operation_ir_v1& operation) noexcept {
    if (!operation.identity.valid()) return relation_apply_ir_validation_code_v1::invalid_identity;
    if (validate_relation_ir_type_v1(operation.relation) != relation_ir_validation_code_v1::success)
        return relation_apply_ir_validation_code_v1::invalid_relation;
    if (validate_state_ir_type_v1(operation.source) != state_value_ir_validation_code_v1::success)
        return relation_apply_ir_validation_code_v1::invalid_source;
    if (validate_state_ir_type_v1(operation.result) != state_value_ir_validation_code_v1::success)
        return relation_apply_ir_validation_code_v1::invalid_result;
    const auto source_axis = operation.source.axes.back();
    const auto result_axis = operation.result.axes.back();
    const bool forward = operation.relation.orientation == relation_orientation_ir_v1::forward;
    if ((forward && (!same(source_axis, operation.relation.source_axis.identity) ||
                     !same(result_axis, operation.relation.destination_axis.identity))) ||
        (!forward && (!same(source_axis, operation.relation.destination_axis.identity) ||
                      !same(result_axis, operation.relation.source_axis.identity))))
        return relation_apply_ir_validation_code_v1::axis_mismatch;
    if (operation.source.dense_width != operation.result.dense_width)
        return relation_apply_ir_validation_code_v1::width_mismatch;
    if (!same_numeric(operation.source.numeric, operation.result.numeric))
        return relation_apply_ir_validation_code_v1::numeric_mismatch;
    using cellerator::compute::operation::v2::destination_update;
    if (operation.update < destination_update::overwrite ||
        operation.update > destination_update::partial_write)
        return relation_apply_ir_validation_code_v1::invalid_update;
    constexpr std::uint32_t required = relation_apply_reads_source_v1 |
        relation_apply_reads_values_v1 | relation_apply_writes_result_v1 |
        relation_apply_advances_result_generation_v1;
    if ((operation.effects & required) != required)
        return relation_apply_ir_validation_code_v1::invalid_effects;
    return relation_apply_ir_validation_code_v1::success;
}

relation_apply_ir_validation_code_v1 lower_relation_apply_operation_v1(
    const relation_apply_operation_ir_v1& operation,
    lowered_relation_apply_v1* lowered) noexcept {
    const auto status = validate_relation_apply_operation_ir_v1(operation);
    if (status != relation_apply_ir_validation_code_v1::success || lowered == nullptr)
        return status == relation_apply_ir_validation_code_v1::success
            ? relation_apply_ir_validation_code_v1::invalid_identity : status;
    const auto relation = typed_relation_from_relation_ir_v1(operation.relation);
    if (!relation) return relation_apply_ir_validation_code_v1::invalid_relation;

    lowered_relation_apply_v1 result;
    result.relation = *relation;
    result.operation.schema_version = cellerator::compute::operation::v2::operation_core_schema_version;
    result.operation.kind = operation.relation.orientation == relation_orientation_ir_v1::forward
        ? cellerator::compute::operation::v2::operation_kind::relation_apply
        : cellerator::compute::operation::v2::operation_kind::relation_apply_transpose;
    result.operation.orientation = operation.relation.orientation == relation_orientation_ir_v1::forward
        ? cellerator::compute::operation::v2::relation_orientation::forward
        : cellerator::compute::operation::v2::relation_orientation::transpose;
    result.operation.persistent_problem_identity = stable(operation.identity);
    result.operation.operation_identity = stable(operation.identity);
    result.operation.values_axis = result.operation.orientation ==
            cellerator::compute::operation::v2::relation_orientation::forward
        ? result.relation.source_axis : result.relation.destination_axis;
    result.operation.result_axis = result.operation.orientation ==
            cellerator::compute::operation::v2::relation_orientation::forward
        ? result.relation.destination_axis : result.relation.source_axis;
    result.operation.logical_edge_order = result.relation.logical_edge_order;
    result.operation.expected_value_generation = {operation.relation.value_generation};
    result.operation.numeric = to_operation_numeric_policy_v1(operation.source.numeric);
    result.operation.output.produced_axis = result.operation.result_axis;
    result.operation.output.canonical_axis = result.operation.result_axis;
    result.operation.output.update = operation.update;
    result.operation.output.input_output_aliasing_legal = operation.result.alias.may_alias_input;
    result.operation.determinism.deterministic_required = operation.deterministic;
    result.operation.determinism.stable_work_order = operation.deterministic;
    result.operation.logical_work_items = operation.relation.logical_edge_count;
    result.operation.dense_width = operation.source.dense_width;
    result.operation.requirement_flags = result.operation.orientation ==
            cellerator::compute::operation::v2::relation_orientation::forward
        ? cellerator::compute::operation::v2::require_forward
        : cellerator::compute::operation::v2::require_backward;

    result.binding = {0, 0, 1, 2};
    result.value_binding.structure = result.relation.structure;
    result.value_binding.epoch = result.relation.epoch;
    result.value_binding.generation = {operation.relation.value_generation};
    result.value_binding.layout = cellerator::execution::value_layout_kind::logical_edge_order;
    result.value_binding.ownership = cellerator::compute::operation::v2::value_ownership_mode::logical_primary;
    result.algebra.segment = cellerator::compute::operation::v2::segment_operation::none;
    result.algebra.edge = cellerator::compute::operation::v2::edge_operation::none;
    result.algebra.gate = cellerator::compute::operation::v2::gate_indexing::none;
    result.algebra.semantic_flags =
        cellerator::compute::operation::v2::alpha_applied_once |
        cellerator::compute::operation::v2::beta_applied_once |
        cellerator::compute::operation::v2::support_superset_preserved;
    result.refresh_views();
    *lowered = std::move(result);
    return relation_apply_ir_validation_code_v1::success;
}

}  // namespace Cellerator::compiler::ir::semantic
