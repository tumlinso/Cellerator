#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>

#include <utility>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool same(semantic_identity_v1 left, semantic_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

cellerator::compute::operation::v2::stable_id stable(semantic_identity_v1 value) noexcept {
    return {value.low, value.high};
}

}  // namespace

lowered_relation_apply_v1::lowered_relation_apply_v1() noexcept { refresh_views(); }

lowered_relation_apply_v1::lowered_relation_apply_v1(
    const lowered_relation_apply_v1& other) noexcept
    : transport_status(other.transport_status), semantic(other.semantic), relation(other.relation), binding(other.binding), value_binding(other.value_binding),
      operation(other.operation), algebra(other.algebra) {
    refresh_views();
}

lowered_relation_apply_v1& lowered_relation_apply_v1::operator=(
    const lowered_relation_apply_v1& other) noexcept {
    if (this != &other) {
        transport_status = other.transport_status;
        semantic = other.semantic;
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
    if (transport_status != relation_transport_status_v1::available) {
        operation = {};
        operation.schema_version = 0;
        operation.kind = static_cast<cellerator::compute::operation::v2::operation_kind>(0);
        algebra = {};
        algebra.core = operation;
        return;
    }
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
    if (operation.source.axes.size() != 1 || operation.result.axes.size() != 1)
        return relation_apply_ir_validation_code_v1::axis_mismatch;
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
    const auto& input_endpoint = forward ? operation.relation.source_axis : operation.relation.destination_axis;
    const auto& output_endpoint = forward ? operation.relation.destination_axis : operation.relation.source_axis;
    if (!same(operation.source.order, input_endpoint.order.identity) ||
        !same(operation.result.order, output_endpoint.order.identity))
        return relation_apply_ir_validation_code_v1::axis_mismatch;
    if (operation.source.numeric.compute != operation.result.numeric.compute ||
        operation.source.numeric.accumulation != operation.result.numeric.accumulation ||
        operation.source.numeric.output != operation.result.numeric.storage ||
        operation.result.numeric.output != operation.result.numeric.storage)
        return relation_apply_ir_validation_code_v1::numeric_mismatch;
    using cellerator::compute::operation::v2::destination_update;
    if (operation.update < destination_update::overwrite ||
        operation.update > destination_update::partial_write)
        return relation_apply_ir_validation_code_v1::invalid_update;
    constexpr std::uint32_t required = relation_apply_reads_source_v1 |
        relation_apply_reads_values_v1 | relation_apply_writes_result_v1 |
        relation_apply_advances_result_generation_v1;
    if (operation.effects != required)
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
    namespace canonical = cellerator::compute::relation;
    const auto& source_axis = operation.relation.source_axis;
    const auto& destination_axis = operation.relation.destination_axis;
    if (source_axis.extent.kind != extent_knowledge_kind_v1::exact ||
        destination_axis.extent.kind != extent_knowledge_kind_v1::exact)
        return relation_apply_ir_validation_code_v1::axis_mismatch;
    result.semantic.topology = {
        {operation.relation.structure_identity.low, operation.relation.structure_identity.high},
        {operation.relation.structure_epoch},
        {relation->source_axis, source_axis.extent.upper_bound},
        {relation->destination_axis, destination_axis.extent.upper_bound},
        {operation.relation.logical_edge_order.low, operation.relation.logical_edge_order.high},
        operation.relation.logical_edge_count};
    result.semantic.direction = operation.relation.orientation == relation_orientation_ir_v1::forward
        ? canonical::orientation::forward : canonical::orientation::transpose;
    result.semantic.arithmetic = {operation.relation_storage, operation.source.numeric.storage,
        operation.source.numeric.compute, operation.source.numeric.accumulation,
        operation.result.numeric.output, operation.permit_fma, operation.permit_reassociation,
        operation.nonfinite};
    result.semantic.dense_width = operation.source.dense_width;
    if (operation.update == cellerator::compute::operation::v2::destination_update::overwrite)
        result.semantic.update = canonical::output_update::overwrite;
    else if (operation.update == cellerator::compute::operation::v2::destination_update::accumulate)
        result.semantic.update = canonical::output_update::accumulate;
    else return relation_apply_ir_validation_code_v1::invalid_update;
    result.semantic.input_output_aliasing_legal = operation.result.alias.may_alias_input;
    if (!canonical::validate(result.semantic))
        return relation_apply_ir_validation_code_v1::numeric_mismatch;
    result.relation = *relation;
    if (!result.semantic.arithmetic.permit_fma || !result.semantic.arithmetic.permit_reassociation) {
        // v2 has no FMA/reassociation permissions. Preserve the mathematical
        // descriptor without publishing a permissive substitute transport.
        result.transport_status = relation_transport_status_v1::unsupported_arithmetic_policy;
        result.refresh_views();
        *lowered = std::move(result);
        return relation_apply_ir_validation_code_v1::success;
    }
    result.transport_status = relation_transport_status_v1::available;
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
    const auto& arithmetic = result.semantic.arithmetic;
    auto& numeric = result.operation.numeric;
    numeric.relation_storage = arithmetic.relation_storage;
    numeric.state_storage = arithmetic.input_storage;
    numeric.multiply = arithmetic.multiply;
    numeric.accumulation = arithmetic.accumulation;
    numeric.output_storage = arithmetic.output_storage;
    numeric.scalar = arithmetic.multiply;
    using namespace cellerator::compute::operation::v2;
    numeric.rounding = rounding_policy::nearest_even;
    numeric.saturation = saturation_policy::none;
    numeric.nan = arithmetic.nonfinite == canonical::nonfinite_policy::reject
        ? nan_policy::reject : nan_policy::propagate;
    numeric.infinity = arithmetic.nonfinite == canonical::nonfinite_policy::reject
        ? infinity_policy::reject : infinity_policy::propagate;
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
