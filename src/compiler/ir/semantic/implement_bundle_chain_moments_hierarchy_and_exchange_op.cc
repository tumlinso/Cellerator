#include <Cellerator/compiler/ir/semantic/implement_bundle_chain_moments_hierarchy_and_exchange_op_v1.hh>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool same(semantic_identity_v1 left, semantic_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

}  // namespace

semantic_graph_validation_code_v1 validate_semantic_operation_graph_v1(
    const semantic_operation_graph_v1& graph) noexcept {
    if (!graph.identity.valid()) return semantic_graph_validation_code_v1::invalid_identity;
    std::unordered_map<std::uint64_t, std::size_t> node_index;
    for (std::size_t index = 0; index < graph.nodes.size(); ++index) {
        const auto& node = graph.nodes[index];
        if (!node.identity.valid() || !node_index.emplace(node.identity.low, index).second)
            return semantic_graph_validation_code_v1::invalid_node;
        if (node.input_axes.empty() || node.output_axes.empty() ||
            std::any_of(node.input_axes.begin(), node.input_axes.end(),
                        [](semantic_identity_v1 axis) { return !axis.valid(); }) ||
            std::any_of(node.output_axes.begin(), node.output_axes.end(),
                        [](semantic_identity_v1 axis) { return !axis.valid(); }) ||
            std::any_of(node.intermediate_axes.begin(), node.intermediate_axes.end(),
                        [](semantic_identity_v1 axis) { return !axis.valid(); }))
            return semantic_graph_validation_code_v1::invalid_axis;
        constexpr std::uint32_t effects = graph_reads_inputs_v1 | graph_writes_outputs_v1;
        if ((node.effects & effects) != effects ||
            (node.kind == semantic_graph_operation_kind_v1::typed_exchange &&
             (node.effects & graph_communicates_v1) == 0))
            return semantic_graph_validation_code_v1::invalid_effects;
        if (node.kind == semantic_graph_operation_kind_v1::relation_chain &&
            node.intermediate_axes.empty())
            return semantic_graph_validation_code_v1::invalid_axis;
        if (node.kind == semantic_graph_operation_kind_v1::paired_moments &&
            node.output_axes.size() != 2)
            return semantic_graph_validation_code_v1::invalid_axis;
    }
    std::vector<std::size_t> indegree(graph.nodes.size());
    std::vector<std::vector<std::size_t>> consumers(graph.nodes.size());
    for (const auto& dependency : graph.dependencies) {
        const auto producer = node_index.find(dependency.producer);
        const auto consumer = node_index.find(dependency.consumer);
        if (producer == node_index.end() || consumer == node_index.end() ||
            producer == consumer || !dependency.exchanged_axis.valid())
            return semantic_graph_validation_code_v1::invalid_dependency;
        const auto& produced = graph.nodes[producer->second].output_axes;
        const auto& consumed = graph.nodes[consumer->second].input_axes;
        if (std::none_of(produced.begin(), produced.end(), [&](auto axis) {
                return same(axis, dependency.exchanged_axis);
            }) || std::none_of(consumed.begin(), consumed.end(), [&](auto axis) {
                return same(axis, dependency.exchanged_axis);
            })) return semantic_graph_validation_code_v1::axis_mismatch;
        ++indegree[consumer->second];
        consumers[producer->second].push_back(consumer->second);
    }
    std::vector<std::size_t> ready;
    for (std::size_t index = 0; index < indegree.size(); ++index)
        if (indegree[index] == 0) ready.push_back(index);
    std::size_t visited = 0;
    while (!ready.empty()) {
        const auto node = ready.back();
        ready.pop_back();
        ++visited;
        for (const auto consumer : consumers[node])
            if (--indegree[consumer] == 0) ready.push_back(consumer);
    }
    return visited == graph.nodes.size()
        ? semantic_graph_validation_code_v1::success
        : semantic_graph_validation_code_v1::cycle;
}

std::optional<semantic_operation_graph_v1>
round_trip_operation_portfolio_graph_v1(const semantic_operation_graph_v1& graph) noexcept {
    if (validate_semantic_operation_graph_v1(graph) != semantic_graph_validation_code_v1::success)
        return std::nullopt;
    // The public portfolio representation preserves semantic axes and effects;
    // target composition records are a later lowering and never replace them.
    return graph;
}

cellerator::compute::operation::v2::composition_kind
lower_semantic_graph_kind_v1(semantic_graph_operation_kind_v1 kind) noexcept {
    using result = cellerator::compute::operation::v2::composition_kind;
    switch (kind) {
    case semantic_graph_operation_kind_v1::relation_bundle:
        return result::bundle_to_shared_destination;
    case semantic_graph_operation_kind_v1::relation_chain:
        return result::normalization_to_relation_apply;
    case semantic_graph_operation_kind_v1::paired_moments:
        return result::relation_moments_pair;
    case semantic_graph_operation_kind_v1::incidence_pool:
        return result::contraction_to_segment;
    case semantic_graph_operation_kind_v1::incidence_broadcast:
        return result::normalization_to_relation_apply;
    case semantic_graph_operation_kind_v1::typed_exchange:
        return result::sparse_exchange;
    }
    return result::relation_apply_to_epilogue;
}

}  // namespace Cellerator::compiler::ir::semantic
