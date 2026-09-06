#include <Cellerator/compiler/sema/implement_operation_kind_resolution_v1.hh>
#include <Cellerator/compiler/ir/semantic/implement_bundle_chain_moments_hierarchy_and_exchange_op_v1.hh>
#include <cassert>

using namespace cellerator::compiler::sema::v1;
using primitive_kind = cellerator::compute::operation::v2::operation_kind;

int main() {
    const primitive_kind primitives[] = {
        primitive_kind::relation_apply, primitive_kind::relation_apply_transpose,
        primitive_kind::contract_on_support, primitive_kind::segment_reduce,
        primitive_kind::segment_normalize, primitive_kind::edge_map_or_gate,
        primitive_kind::sparse_axis_update, primitive_kind::relation_bundle_apply};
    const composition_kind compositions[] = {composition_kind::relation_chain,
        composition_kind::moments, composition_kind::hierarchy,
        composition_kind::exchange, composition_kind::gradient};
    assert(operation_kind_coverage_count() == 14);
    for (unsigned i = 1; i <= 14; ++i) {
        auto *entry = resolve_operation_kind(static_cast<source_operation_kind>(i));
        assert(entry && entry->syntax && entry->syntax[0]);
        if (i <= 8) {
            assert(entry->primitive() && *entry->primitive() == primitives[i-1]);
        } else {
            assert(!entry->primitive());
            if (i <= 13)
                assert(std::get<composition_kind>(entry->meaning) == compositions[i-9]);
            else
                assert(std::get<effect_kind>(entry->meaning) == effect_kind::publication);
            // A direct variant extraction cannot accidentally yield a substitute.
            assert(std::get_if<primitive_kind>(&entry->meaning) == nullptr);
        }
    }
    namespace ir = Cellerator::compiler::ir::semantic;
    using graph_kind = ir::semantic_graph_operation_kind_v1;
    using composition = cellerator::compute::operation::v2::composition_kind;
    assert(ir::lower_semantic_graph_kind_v1(graph_kind::relation_bundle)
        == composition::bundle_to_shared_destination);
    assert(ir::lower_semantic_graph_kind_v1(graph_kind::paired_moments)
        == composition::relation_moments_pair);
    assert(ir::lower_semantic_graph_kind_v1(graph_kind::typed_exchange)
        == composition::sparse_exchange);
    for (auto kind : {graph_kind::relation_chain, graph_kind::incidence_pool,
            graph_kind::incidence_broadcast, static_cast<graph_kind>(255)})
        assert(!ir::lower_semantic_graph_kind_v1(kind));
    ir::semantic_operation_graph_v1 graph;
    graph.identity = {1, 2};
    ir::semantic_graph_node_v1 node;
    node.identity = {3, 4};
    node.kind = graph_kind::relation_chain;
    node.input_axes = {{5, 6}};
    node.output_axes = {{7, 8}};
    node.intermediate_axes = {{9, 10}};
    graph.nodes = {node};
    auto preserved = ir::round_trip_operation_portfolio_graph_v1(graph);
    assert(preserved && preserved->nodes.front().kind == graph_kind::relation_chain);
    assert(!ir::lower_semantic_graph_kind_v1(preserved->nodes.front().kind));
    graph.nodes.front().kind = static_cast<graph_kind>(255);
    assert(ir::validate_semantic_operation_graph_v1(graph)
        == ir::semantic_graph_validation_code_v1::invalid_node);
    assert(!resolve_operation_kind(static_cast<source_operation_kind>(0)));
    assert(!resolve_operation_kind(static_cast<source_operation_kind>(255)));
}
