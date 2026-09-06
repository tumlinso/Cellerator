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
    relation_spine_environment env;
    auto rel = relation();
    rel.source_axis.domain.nominal_tag = "gene";
    rel.destination_axis.domain.nominal_tag = "module";
    env.axes = {{"genes", rel.source_axis}, {"modules", rel.destination_axis}};
    env.relations = {{"regulation", rel, numeric_type::f16}};
    env.states = {{"x", state(110, rel.source_axis.identity)}, {"y", state(120, rel.destination_axis.identity)}};
    env.states[0].state.order = rel.source_axis.order.identity;
    env.states[1].state.order = rel.destination_axis.order.identity;
    const std::string declarations =
        "domain gene; domain module; axis<gene> genes; axis<module> modules; "
        "relation<f16,gene,module> regulation; state<f32,gene> x; state<f32,module> y;";
    const auto parse = [&](std::string_view expression) {
        return lower_relation_source_slice_v1(declarations, expression, env, {100,101});
    };
    auto forward = parse("y = x -[regulation]-> modules;");
    for (const auto& diagnostic : forward.diagnostics) std::cerr << diagnostic.message << '\n';
    assert(forward.accepted());
    assert(forward.lowered.transport_status == relation_transport_status_v1::available);
    assert(forward.lowered.operation.numeric.relation_storage == numeric_type::f16);
    assert(forward.lowered.algebra.core.numeric.relation_storage == numeric_type::f16);
    namespace canonical = cellerator::compute::relation;
    canonical::operation_descriptor cpp;
    const cellerator::execution::serialized_record_header header{
        cellerator::execution::biological_abi_version,
        cellerator::execution::serialized_record_kind::persistent_axis_identity,
        sizeof(cellerator::execution::persistent_axis_identity)};
    cpp.topology = {{40,41}, {2}, {{header,{1,2},{3,4},{20,21},{22,23}},64},
        {{header,{5,6},{7,8},{30,31},{32,33}},64}, {44,45},1024};
    cpp.dense_width = 16;
    assert(canonical::equivalent(cpp, forward.lowered.semantic));
    assert(forward.provenance.relation_symbol == "regulation");
    assert(forward.provenance.expression_range.end == 29);
    auto transpose = parse("x = y -[transpose(regulation)]-> genes;");
    assert(transpose.accepted());
    cpp.direction = canonical::orientation::transpose;
    assert(canonical::equivalent(cpp, transpose.lowered.semantic));
    assert(!canonical::equivalent(forward.lowered.semantic, transpose.lowered.semantic));
    assert(!parse("y = x -[transpose(regulation)]-> modules;").accepted());
    assert(!parse("y = x -[missing]-> modules;").accepted());
    assert(!parse("y = missing -[regulation]-> modules;").accepted());
    assert(!parse("y = x -[regulation]-> genes;").accepted());
    assert(!parse("y = x -[regulation where active]-> modules;").accepted());
    assert(!parse("y = x -[regulation]-> modules; ignored();").accepted());
    assert(!parse("y = x -[transpose(regulation) + ignored]-> modules;").accepted());
    auto changed_declarations = declarations;
    changed_declarations.replace(changed_declarations.find("f16"), 3, "f32");
    assert(!lower_relation_source_slice_v1(changed_declarations,
        "y = x -[regulation]-> modules;", env, {100,101}).accepted());
    changed_declarations = declarations;
    changed_declarations.replace(changed_declarations.find("state<f32,gene> x"), 16, "state<f16,gene> x");
    assert(!lower_relation_source_slice_v1(changed_declarations,
        "y = x -[regulation]-> modules;", env, {100,101}).accepted());
    auto wrong_env = env;
    wrong_env.axes.push_back(wrong_env.axes.front());
    assert(!lower_relation_source_slice_v1(declarations,
        "y = x -[regulation]-> modules;", wrong_env, {100,101}).accepted());
    wrong_env = env;
    wrong_env.states[0].state.order.low++;
    assert(!lower_relation_source_slice_v1(declarations,
        "y = x -[regulation]-> modules;", wrong_env, {100,101}).accepted());
    wrong_env = env;
    wrong_env.relations[0].relation.structure_identity.low++;
    auto changed_binding = lower_relation_source_slice_v1(declarations,
        "y = x -[regulation]-> modules;", wrong_env, {100,101});
    assert(changed_binding.accepted());
    assert(!canonical::equivalent(forward.lowered.semantic, changed_binding.lowered.semantic));
    assert(!lower_relation_source_slice_v1(declarations + " state<f32,gene> ignored = foo();",
        "y = x -[regulation]-> modules;", env, {100,101}).accepted());
    auto accumulate = parse("y += x -[regulation on genes]-> modules;");
    assert(accumulate.accepted());
    assert(accumulate.lowered.semantic.update == canonical::output_update::accumulate);
    std::cout << "source_slice: parser/Sema/IR canonical equivalence and negative inputs passed; full .cell execution deferred\n";
}
