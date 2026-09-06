#include <Cellerator/compiler/sema/relation_spine_bridge.hh>
#include <cassert>
#include <iostream>
using namespace Cellerator::compiler::ir::semantic;
using namespace Cellerator::compiler::sema;
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

}
int main() {
    relation_apply_operation_ir_v1 op;
    op.identity = {100,101};
    op.relation = relation();
    op.source = state(110, op.relation.source_axis.identity);
    op.result = state(120, op.relation.destination_axis.identity);
    op.source.order = op.relation.source_axis.order.identity;
    op.result.order = op.relation.destination_axis.order.identity;
    namespace canonical = cellerator::compute::relation;
    using code = relation_apply_ir_validation_code_v1;
    lowered_relation_apply_v1 initial, changed;
    assert(lower_relation_apply_operation_v1(op, &initial) == code::success);
    auto bad = op;
    bad.source.order.high++;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::axis_mismatch);
    bad = op;
    bad.source.axes.push_back({800,801});
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::axis_mismatch);
    bad = op;
    bad.effects |= 1u << 31;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::invalid_effects);
    bad = op;
    bad.effects &= ~relation_apply_advances_result_generation_v1;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::invalid_effects);
    assert(canonical::equivalent(changed.semantic, canonical::operation_descriptor{}));
    bad = op;
    bad.relation.structure_identity.high++;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::success);
    assert(!canonical::equivalent(initial.semantic, changed.semantic));
    bad = op;
    bad.relation.logical_edge_order.high++;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::success);
    assert(!canonical::equivalent(initial.semantic, changed.semantic));
    bad = op;
    bad.relation.value_generation++;
    bad.source.generation.value++;
    bad.result.generation.value++;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::success);
    assert(canonical::equivalent(initial.semantic, changed.semantic));
    assert(changed.value_binding.generation.value == bad.relation.value_generation);
    bad = op;
    bad.source.numeric.output = numeric_type::f16;
    bad.result.numeric.storage = bad.result.numeric.output = numeric_type::f16;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::success);
    assert(changed.semantic.arithmetic.input_storage == numeric_type::f32);
    assert(changed.semantic.arithmetic.output_storage == numeric_type::f16);
    bad = op;
    bad.result.alias.may_alias_input = true;
    bad.result.alias.alias_class = 1;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::success);
    assert(changed.semantic.input_output_aliasing_legal);
    assert(!canonical::equivalent(initial.semantic, changed.semantic));
    bad = op;
    bad.update = cellerator::compute::operation::v2::destination_update::affine_accumulate;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::invalid_update);
    auto cpp_bad = initial.semantic;
    cpp_bad.update = canonical::output_update::affine_accumulate;
    assert(!canonical::validate(cpp_bad));
    bad = op;
    bad.relation_storage = numeric_type::invalid;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::numeric_mismatch);
    cpp_bad = initial.semantic;
    cpp_bad.arithmetic.relation_storage = numeric_type::invalid;
    assert(!canonical::validate(cpp_bad));
    bad = op;
    bad.relation.source_axis.extent = {extent_knowledge_kind_v1::exact,0,0};
    bad.relation.destination_axis.extent = {extent_knowledge_kind_v1::exact,0,0};
    bad.relation.logical_edge_count = 0;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::success);
    assert(canonical::validate(changed.semantic));
    bad.relation.logical_edge_count = 1;
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::numeric_mismatch);
    cpp_bad = changed.semantic;
    cpp_bad.topology.edge_count = 1;
    assert(!canonical::validate(cpp_bad));
    bad = op;
    bad.relation.source_axis.extent = {extent_knowledge_kind_v1::bounded,1,64};
    assert(lower_relation_apply_operation_v1(bad, &changed) == code::axis_mismatch);

    relation_spine_environment env;
    auto rel = op.relation;
    rel.source_axis.domain.nominal_tag = "gene";
    rel.destination_axis.domain.nominal_tag = "module";
    env.axes = {{"genes",rel.source_axis},{"modules",rel.destination_axis}};
    env.relations = {{"regulation",rel,numeric_type::f16}};
    env.states = {{"x",op.source},{"y",op.result}};
    const std::string declarations = "domain gene; domain module; axis<gene> genes; axis<module> modules; relation<f16,gene,module> regulation; state<f32,gene> x; state<f32,module> y;";
    env.states[1].state.dense_width++;
    auto source_error = lower_relation_source_slice_v1(declarations,
        "y = x -[regulation]-> modules;", env, {100,101});
    assert(!source_error.accepted());
    assert(source_error.diagnostics[0].message == "dense width mismatch");
    assert(source_error.diagnostics[0].range.begin == 0 && source_error.diagnostics[0].range.end == 29);
    env.states[1].state.dense_width--;
    env.axes[0].axis.geometry.identity.high++;
    source_error = lower_relation_source_slice_v1(declarations,
        "y = x -[regulation]-> modules;", env, {100,101});
    assert(!source_error.accepted());
    assert(source_error.diagnostics[0].message == "input axis metadata does not match relation endpoint");
    std::cout << "diagnostics: identity high bits, order, effects, empty shapes, independent arithmetic, alias representation and source ranges passed\n";
}
