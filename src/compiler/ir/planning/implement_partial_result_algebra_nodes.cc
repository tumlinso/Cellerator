#include <Cellerator/compiler/ir/planning/implement_partial_result_algebra_nodes_v1.hh>

namespace cellerator::compiler::ir::planning::v1 {

partial_algebra_node_status_v1 validate_partial_result_algebra_node_v1(
    const partial_result_algebra_node_v1 &node) noexcept {
    if (node.node.low == 0u && node.node.high == 0u) {
        return partial_algebra_node_status_v1::invalid_identity;
    }
    if (!decomposition::validate_partial_result_algebra_v1(node.algebra)) {
        return partial_algebra_node_status_v1::invalid_algebra;
    }
    const bool deterministic =
        (node.algebra.flags & decomposition::deterministic_tree_required_v1) != 0u;
    if (deterministic && (node.leaf_count < 2u || node.reference_tree == nullptr ||
                          node.reference_tree_edge_count != node.leaf_count - 1u)) {
        return partial_algebra_node_status_v1::missing_tree;
    }
    for (std::uint32_t index = 0u; index != node.reference_tree_edge_count; ++index) {
        const auto &edge = node.reference_tree[index];
        if (edge.reserved != 0u) {
            return partial_algebra_node_status_v1::nonzero_reserved;
        }
        const std::uint32_t maximum_input = node.leaf_count + index;
        if (edge.left >= maximum_input || edge.right >= maximum_input ||
            edge.left == edge.right || edge.result != maximum_input) {
            return partial_algebra_node_status_v1::invalid_tree_edge;
        }
    }
    return partial_algebra_node_status_v1::ok;
}

}  // namespace cellerator::compiler::ir::planning::v1
