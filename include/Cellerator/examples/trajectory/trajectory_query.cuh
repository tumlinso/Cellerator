#pragma once

#include <Cellerator/compute/operators/graph/supernode_reduce.cuh>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cellerator::trajectory {

namespace cg = ::cellerator::compute::graph;

struct trajectory_id_span {
    const std::uint32_t *data = nullptr;
    std::uint32_t count = 0u;
};

inline trajectory_id_span subtree_supernode_span(const cg::TreeOverlay &tree, std::uint32_t root) {
    if (root >= tree.nodes) throw std::out_of_range("subtree root out of range");
    if (tree.euler_to_node.size() != tree.nodes || tree.tout[root] < tree.tin[root]) {
        throw std::invalid_argument("tree does not expose a valid Euler projection");
    }
    return {
        tree.euler_to_node.data() + tree.tin[root],
        tree.tout[root] - tree.tin[root]
    };
}

inline std::vector<std::uint32_t> subtree_supernodes(const cg::TreeOverlay &tree, std::uint32_t root) {
    const trajectory_id_span span = subtree_supernode_span(tree, root);
    return std::vector<std::uint32_t>(span.data, span.data + span.count);
}

inline std::vector<std::uint32_t> path_to_root(const cg::TreeOverlay &tree, std::uint32_t node) {
    if (node >= tree.nodes) throw std::out_of_range("path_to_root node out of range");
    // O(depth) host helper.
    std::vector<std::uint32_t> path;
    std::uint32_t current = node;
    while (true) {
        path.push_back(current);
        if (tree.parent[current] < 0 || static_cast<std::uint32_t>(tree.parent[current]) == current) break;
        current = static_cast<std::uint32_t>(tree.parent[current]);
    }
    std::reverse(path.begin(), path.end());
    return path;
}

inline std::vector<std::uint32_t> supernode_cells(const cg::SupernodeTable &table, std::uint32_t supernode_id) {
    if (supernode_id >= table.count) throw std::out_of_range("supernode id out of range");
    const std::uint32_t begin = table.member_row_ptr[supernode_id];
    const std::uint32_t end = table.member_row_ptr[supernode_id + 1u];
    return std::vector<std::uint32_t>(
        table.member_cell_ids.data() + begin,
        table.member_cell_ids.data() + end);
}

inline trajectory_id_span supernode_cell_span(
    const cg::SupernodeTable &table,
    std::uint32_t supernode_id) {
    if (supernode_id >= table.count) throw std::out_of_range("supernode id out of range");
    const std::uint32_t begin = table.member_row_ptr[supernode_id];
    return {
        table.member_cell_ids.data() + begin,
        table.member_row_ptr[supernode_id + 1u] - begin
    };
}

} // namespace cellerator::trajectory
