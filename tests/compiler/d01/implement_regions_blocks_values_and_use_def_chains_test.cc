#include <Cellerator/compiler/ir/common/implement_regions_blocks_values_and_use_def_chains_v1.hh>

#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    ir_graph graph;
    const auto body = graph.add_region();
    const auto entry = graph.add_block(body);
    const auto exit = graph.add_block(body);
    assert(graph.add_control_edge(entry, exit));
    const auto producer = graph.add_operation(entry);
    const auto old_value = graph.add_value(producer, "!state<f32,gene>");
    const auto replacement = graph.add_value(producer, "!state<f32,gene>");
    const auto consumer = graph.add_operation(exit, {old_value});
    const auto nested = graph.add_region(consumer);
    assert(graph.operation(consumer)->regions.front().slot == nested.slot);
    assert(graph.value(old_value)->uses.size() == 1u);
    assert(graph.replace_all_uses(old_value, replacement));
    assert(graph.value(old_value)->uses.empty());
    assert(graph.value(replacement)->uses.size() == 1u);
    assert(graph.operation(consumer)->operands.front().slot == replacement.slot);
    assert(graph.block(entry)->successors.front().slot == exit.slot);
    assert(graph.region(body)->blocks.size() == 2u);
    assert(graph.block({99u, 1u}) == nullptr);
}
