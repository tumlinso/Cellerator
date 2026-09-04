#include <Cellerator/compiler/ir/semantic/implement_bundle_chain_moments_hierarchy_and_exchange_op_v1.hh>

#include <array>
#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    const semantic_identity_v1 genes{1, 2};
    const semantic_identity_v1 modules{3, 4};
    const semantic_identity_v1 cells{5, 6};
    const std::array<semantic_graph_operation_kind_v1, 6> portfolio{{
        semantic_graph_operation_kind_v1::relation_bundle,
        semantic_graph_operation_kind_v1::relation_chain,
        semantic_graph_operation_kind_v1::paired_moments,
        semantic_graph_operation_kind_v1::incidence_pool,
        semantic_graph_operation_kind_v1::incidence_broadcast,
        semantic_graph_operation_kind_v1::typed_exchange,
    }};
    for (std::size_t index = 0; index < portfolio.size(); ++index) {
        semantic_operation_graph_v1 graph;
        graph.identity = {100 + index, 200 + index};
        semantic_graph_node_v1 node;
        node.identity = {300 + index, 400 + index};
        node.kind = portfolio[index];
        node.input_axes = {genes};
        node.output_axes = node.kind == semantic_graph_operation_kind_v1::paired_moments
            ? std::vector<semantic_identity_v1>{modules, cells}
            : std::vector<semantic_identity_v1>{modules};
        if (node.kind == semantic_graph_operation_kind_v1::relation_chain)
            node.intermediate_axes = {cells};
        if (node.kind == semantic_graph_operation_kind_v1::typed_exchange)
            node.effects |= graph_communicates_v1;
        graph.nodes = {node};
        const auto round_trip = round_trip_operation_portfolio_graph_v1(graph);
        assert(round_trip && round_trip->nodes.front().kind == node.kind);
        assert(round_trip->nodes.front().output_axes.size() == node.output_axes.size());
        (void)lower_semantic_graph_kind_v1(node.kind);
    }

    semantic_operation_graph_v1 chain;
    chain.identity = {500, 501};
    chain.nodes = {
        {{510, 511}, semantic_graph_operation_kind_v1::incidence_pool,
         {genes}, {modules}, {}, graph_reads_inputs_v1 | graph_writes_outputs_v1},
        {{520, 521}, semantic_graph_operation_kind_v1::incidence_broadcast,
         {modules}, {cells}, {}, graph_reads_inputs_v1 | graph_writes_outputs_v1},
    };
    chain.dependencies = {{510, 520, modules}};
    assert(validate_semantic_operation_graph_v1(chain) ==
           semantic_graph_validation_code_v1::success);
    chain.dependencies.front().exchanged_axis = genes;
    assert(validate_semantic_operation_graph_v1(chain) ==
           semantic_graph_validation_code_v1::axis_mismatch);

    std::cout << "portfolio_kinds=6 explicit_intermediate_axes=true\n";
}
