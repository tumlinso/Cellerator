#pragma once

#include "../compute/graph/supernode_reduce.cuh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace cellerator::trajectory {

namespace cg = ::cellerator::compute::graph;

inline std::vector<std::uint32_t> subtree_supernodes(const cg::TreeOverlay &tree, std::uint32_t root) {
    if (root >= tree.nodes) throw std::out_of_range("subtree root out of range");
    std::vector<std::uint32_t> nodes;
    nodes.reserve(tree.nodes);
    for (std::uint32_t node = 0; node < tree.nodes; ++node) {
        if (tree.tin[node] >= tree.tin[root] && tree.tout[node] <= tree.tout[root]) {
            nodes.push_back(node);
        }
    }
    return nodes;
}

inline std::vector<std::uint32_t> path_to_root(const cg::TreeOverlay &tree, std::uint32_t node) {
    if (node >= tree.nodes) throw std::out_of_range("path_to_root node out of range");
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
    std::vector<std::uint32_t> cells;
    const std::uint32_t begin = table.member_row_ptr[supernode_id];
    const std::uint32_t end = table.member_row_ptr[supernode_id + 1u];
    cells.reserve(end - begin);
    for (std::uint32_t off = begin; off < end; ++off) {
        cells.push_back(table.member_cell_ids[off]);
    }
    return cells;
}

} // namespace cellerator::trajectory
