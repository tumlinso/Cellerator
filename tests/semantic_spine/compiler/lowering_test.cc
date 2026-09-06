#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;
using cellerator::execution::numeric_type;

namespace {

axis_ir_type_v1 axis(semantic_identity_v1 identity, semantic_identity_v1 domain,
                     semantic_identity_v1 order, std::uint64_t seed, const char* tag) {
    axis_ir_type_v1 result;
    result.identity = identity;
    result.domain = {domain, tag};
    result.order = {order, domain, false};
    result.geometry = {{seed, seed + 1}, domain};
    result.partition = {{seed + 2, seed + 3}, domain, {seed + 4, seed + 5}};
    result.extent = {extent_knowledge_kind_v1::exact, 64, 64};
    return result;
}

state_ir_type_v1 state(std::uint64_t seed, semantic_identity_v1 axis_identity) {
    state_ir_type_v1 result;
    result.identity = {seed, seed + 1};
    result.axes = {axis_identity};
    result.dense_width = 16;
    result.numeric = {numeric_type::f32, numeric_type::f32,
                      numeric_type::f32, numeric_type::f32};
    result.order = {seed + 2, seed + 3};
    result.generation = {1, true};
    return result;
}

relation_ir_type_v1 relation() {
    relation_ir_type_v1 result;
    result.source_axis = axis({10, 11}, {1, 2}, {3, 4}, 20, "gene");
    result.destination_axis = axis({12, 13}, {5, 6}, {7, 8}, 30, "cell");
    result.structure_identity = {40, 41};
    result.structure_epoch = 2;
    result.logical_edge_identity = {42, 43};
    result.logical_edge_order = {44, 45};
    result.logical_edge_count = 1024;
    result.support_identity = {46, 47};
    result.value_plane_identity = {48, 49};
    result.value_generation = 3;
    result.active_support_generation = 4;
    return result;
}

}  // namespace

int main() {
    relation_apply_operation_ir_v1 apply;
    apply.identity = {100, 101};
    apply.relation = relation();
    apply.source = state(110, apply.relation.source_axis.identity);
    apply.result = state(120, apply.relation.destination_axis.identity);

    apply.source.order = apply.relation.source_axis.order.identity;
    apply.result.order = apply.relation.destination_axis.order.identity;
    lowered_relation_apply_v1 lowered;
    assert(lower_relation_apply_operation_v1(apply, &lowered) ==
           relation_apply_ir_validation_code_v1::success);
    assert(lowered.operation.kind ==
           cellerator::compute::operation::v2::operation_kind::relation_apply);
    assert(lowered.algebra.core.operation_identity.low == apply.identity.low);
    assert(lowered.algebra.bindings.binding_count == 1);
    assert(lowered.algebra.bindings.bindings->source_state_operand == 0);
    assert(lowered.algebra.value_bindings->generation.value == 3);
    assert(lowered.algebra.core.dense_width == 16);

    namespace canonical = cellerator::compute::relation;
    canonical::operation_descriptor cpp;
    cpp.topology.identity = {40, 41};
    cpp.topology.epoch = {2};
    const cellerator::execution::serialized_record_header header{
        cellerator::execution::biological_abi_version,
        cellerator::execution::serialized_record_kind::persistent_axis_identity,
        sizeof(cellerator::execution::persistent_axis_identity)};
    cpp.topology.source = {{header, {1, 2}, {3, 4}, {20, 21}, {22, 23}}, 64};
    cpp.topology.destination = {{header, {5, 6}, {7, 8}, {30, 31}, {32, 33}}, 64};
    cpp.topology.logical_edge_order = {44, 45};
    cpp.topology.edge_count = 1024;
    cpp.dense_width = 16;
    assert(canonical::equivalent(cpp, lowered.semantic));
    auto independent_identity = apply;
    independent_identity.identity = {500, 501};
    independent_identity.relation.value_generation = 99;
    lowered_relation_apply_v1 equivalent;
    assert(lower_relation_apply_operation_v1(independent_identity, &equivalent) ==
           relation_apply_ir_validation_code_v1::success);
    assert(canonical::equivalent(lowered.semantic, equivalent.semantic));
    auto changed = apply;
    changed.relation_storage = numeric_type::f32;
    assert(lower_relation_apply_operation_v1(changed, &equivalent) ==
           relation_apply_ir_validation_code_v1::success);
    assert(!canonical::equivalent(lowered.semantic, equivalent.semantic));
    changed = apply;
    changed.permit_fma = false;
    assert(lower_relation_apply_operation_v1(changed, &equivalent) ==
           relation_apply_ir_validation_code_v1::success);
    assert(!canonical::equivalent(lowered.semantic, equivalent.semantic));
    auto transpose = apply;
    transpose.identity = {102, 103};
    transpose.relation.orientation = relation_orientation_ir_v1::transpose;
    transpose.source.axes.back() = transpose.relation.destination_axis.identity;
    transpose.result.axes.back() = transpose.relation.source_axis.identity;
    transpose.source.order = transpose.relation.destination_axis.order.identity;
    transpose.result.order = transpose.relation.source_axis.order.identity;
    assert(lower_relation_apply_operation_v1(transpose, &lowered) ==
           relation_apply_ir_validation_code_v1::success);
    assert(lowered.operation.kind ==
           cellerator::compute::operation::v2::operation_kind::relation_apply_transpose);
    assert(lowered.operation.orientation ==
           cellerator::compute::operation::v2::relation_orientation::transpose);

    cpp.direction = canonical::orientation::transpose;
    assert(canonical::equivalent(cpp, lowered.semantic));
    lowered_relation_apply_v1 copied = lowered;
    assert(canonical::equivalent(cpp, copied.semantic));
    assert(copied.operation.relations.relations == &copied.relation);
    transpose.result.dense_width = 8;
    assert(validate_relation_apply_operation_ir_v1(transpose) ==
           relation_apply_ir_validation_code_v1::width_mismatch);

    std::cout << "apply_and_transpose=typed format_independent=true\n";
}
