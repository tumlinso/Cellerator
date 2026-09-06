#include <Cellerator/compiler/ir/semantic/implement_gradient_and_publication_operations_v1.hh>
#include <cassert>
#include <iostream>
namespace ir = Cellerator::compiler::ir::semantic;
namespace ce = cellerator::compute::relation;
namespace ex = cellerator::execution;
ce::axis_descriptor axis(std::uint64_t id, std::uint64_t extent) {
    ce::axis_descriptor a{};
    a.extent = extent;
    a.identity.header = {ex::biological_abi_version,
        ex::serialized_record_kind::persistent_axis_identity, sizeof(ex::persistent_axis_identity)};
    a.identity.domain = {id, 1}; a.identity.order = {id, 2};
    a.identity.geometry = {id, 3}; a.identity.partition = {id, 4};
    return a;
}
int main() {
    using kind = ir::gradient_publication_operation_ir_v1;
    using code = ir::gradient_publication_status_ir_v1;
    ir::gradient_publication_program_ir_v1 p{};
    p.program_identity = 10; p.prepared_generation = 3;
    auto& c = p.calculus;
    c.forward.topology = {{11, 12}, {2}, axis(1, 20), axis(3, 19), {5, 6}, 262};
    c.forward.dense_width = 16;
    c.transpose = c.forward; c.transpose.direction = ce::orientation::transpose;
    c.update = ce::value_update_kind::gradient_step;
    for (unsigned i = 0; i < 5; ++i) {
        ir::gradient_publication_stage_ir_v1 s{};
        s.identity = 20 + i; s.dependencies = (1u << i) - 1;
        s.kind = i == 0 ? kind::forward : i == 1 ? kind::transpose :
            i == 2 ? kind::value_gradient : i == 3 ? kind::gradient_step : kind::publish_generation;
        s.input_axis = i == 1 ? c.forward.topology.destination : c.forward.topology.source;
        s.output_axis = i == 1 ? c.forward.topology.source : c.forward.topology.destination;
        s.gradient_order = c.forward.topology.logical_edge_order;
        s.consumed_generation = i == 4 ? 4 : 3;
        s.published_generation = i == 3 ? 4 : 0;
        p.stages.push_back(s);
    }
    ce::relation_effect_sequence effects{};
    assert(ir::lower_gradient_publication_program_ir_v1(p, &effects) == code::success);
    assert(effects.initial_generation.value == 3 && effects.count == 5);
    const ce::relation_effect_kind kinds[] = {ce::relation_effect_kind::forward,
        ce::relation_effect_kind::transpose, ce::relation_effect_kind::edge_gradient,
        ce::relation_effect_kind::value_update, ce::relation_effect_kind::publication};
    for (unsigned i = 0; i < 5; ++i) {
        assert(effects.stages[i].identity == 20 + i && effects.stages[i].kind == kinds[i]);
        assert(effects.stages[i].dependencies == (1u << i) - 1);
        assert(effects.stages[i].reads.value == (i == 4 ? 4 : 3));
        assert(effects.stages[i].writes.value == (i == 3 ? 4 : 0));
    }
    auto reject = [](const auto& bad) {
        assert(ir::validate_gradient_publication_program_ir_v1(bad) != code::success);
    };
    auto bad = p; bad.program_identity = 0; reject(bad);
    bad = p; bad.stages[1].identity = bad.stages[0].identity; reject(bad);
    bad = p; bad.stages[0].input_axis.identity.domain.high++; reject(bad);
    bad = p; bad.stages[1].output_axis.identity.order.high++; reject(bad);
    bad = p; bad.stages[2].gradient_order.high++; reject(bad);
    bad = p; bad.calculus.gradient = static_cast<ce::gradient_arithmetic>(255); reject(bad);
    bad = p; bad.stages.erase(bad.stages.begin() + 2); reject(bad);
    bad = p; bad.stages.erase(bad.stages.begin() + 3); reject(bad);
    bad = p; bad.stages[4].dependencies = 0; reject(bad);
    bad = p; bad.stages[4].consumed_generation = 3; reject(bad);
    bad = p; bad.stages[4].published_generation = 5; reject(bad);
    // Canonicalization and training-owned update policy are no longer IR stages.
    // Exact axes/orders and the core update/publication dependency replace them.
    std::cout << "gradient_closure=core publication=3_to_4 identity_order_numeric_rejections=PASS\n";
}
