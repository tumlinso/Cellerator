#pragma once

#include "forward_prune.cuh"
#include "record_table.cuh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cellerator::compute::graph {

struct TreeOverlay {
    std::uint32_t nodes = 0;
    std::vector<std::int32_t> parent;
    std::vector<float> parent_score;
    std::vector<std::uint32_t> child_row_ptr;
    std::vector<std::uint32_t> child_ids;
    std::vector<std::uint32_t> depth;
    std::vector<std::uint32_t> tin;
    std::vector<std::uint32_t> tout;
    std::vector<std::uint32_t> up;
    std::uint32_t up_levels = 0;
};

struct SupernodeTable {
    std::uint32_t count = 0;
    std::int32_t latent_dim = 0;
    std::vector<std::uint32_t> super_of_cell;
    std::vector<std::uint32_t> member_row_ptr;
    std::vector<std::uint32_t> member_cell_ids;
    std::vector<std::uint32_t> row_begin;
    std::vector<std::uint32_t> row_end;
    std::vector<std::int32_t> embryo_id;
    std::vector<float> time_begin;
    std::vector<float> time_end;
    std::vector<float> support_mass;
    std::vector<__half> centroid;
    std::vector<std::uint8_t> is_branch;
};

struct SupernodeDag {
    std::uint32_t rows = 0;
    std::vector<std::uint32_t> row_ptr;
    std::vector<std::uint32_t> dst;
    std::vector<float> mass;
    std::vector<float> score;
};

inline TreeOverlay build_principal_tree(const ForwardCsrGraph &graph) {
    const InboundCsrGraph inbound = build_inbound_graph(graph);
    TreeOverlay tree;
    tree.nodes = graph.rows;
    tree.parent.assign(tree.nodes, -1);
    tree.parent_score.assign(tree.nodes, -std::numeric_limits<float>::infinity());
    tree.depth.assign(tree.nodes, 0u);
    tree.tin.assign(tree.nodes, 0u);
    tree.tout.assign(tree.nodes, 0u);

    for (std::uint32_t node = 0; node < tree.nodes; ++node) {
        for (std::uint32_t edge = inbound.row_ptr[node]; edge < inbound.row_ptr[node + 1u]; ++edge) {
            if (inbound.score[edge] > tree.parent_score[node]) {
                tree.parent[node] = static_cast<std::int32_t>(inbound.src[edge]);
                tree.parent_score[node] = inbound.score[edge];
            }
        }
    }

    tree.child_row_ptr.assign(static_cast<std::size_t>(tree.nodes) + 1u, 0u);
    for (std::int32_t parent : tree.parent) {
        if (parent >= 0) ++tree.child_row_ptr[static_cast<std::size_t>(parent + 1)];
    }
    for (std::size_t i = 1; i < tree.child_row_ptr.size(); ++i) tree.child_row_ptr[i] += tree.child_row_ptr[i - 1u];
    tree.child_ids.assign(tree.child_row_ptr.back(), 0u);
    std::vector<std::uint32_t> cursor = tree.child_row_ptr;
    for (std::uint32_t node = 0; node < tree.nodes; ++node) {
        const std::int32_t parent = tree.parent[node];
        if (parent < 0) continue;
        tree.child_ids[cursor[static_cast<std::size_t>(parent)]++] = node;
    }

    tree.up_levels = 1u;
    while ((1u << tree.up_levels) <= std::max<std::uint32_t>(1u, tree.nodes)) ++tree.up_levels;
    tree.up.assign(static_cast<std::size_t>(tree.up_levels) * static_cast<std::size_t>(tree.nodes), std::numeric_limits<std::uint32_t>::max());

    std::uint32_t timer = 0;
    std::vector<std::pair<std::uint32_t, bool>> stack;
    for (std::uint32_t root = 0; root < tree.nodes; ++root) {
        if (tree.parent[root] >= 0) continue;
        stack.emplace_back(root, false);
        tree.depth[root] = 0;
        while (!stack.empty()) {
            const auto [node, exiting] = stack.back();
            stack.pop_back();
            if (exiting) {
                tree.tout[node] = timer++;
                continue;
            }

            tree.tin[node] = timer++;
            tree.up[node] = tree.parent[node] >= 0 ? static_cast<std::uint32_t>(tree.parent[node]) : node;
            for (std::uint32_t level = 1; level < tree.up_levels; ++level) {
                const std::uint32_t prev = tree.up[static_cast<std::size_t>(level - 1u) * tree.nodes + node];
                tree.up[static_cast<std::size_t>(level) * tree.nodes + node] =
                    tree.up[static_cast<std::size_t>(level - 1u) * tree.nodes + prev];
            }
            stack.emplace_back(node, true);

            const std::uint32_t child_begin = tree.child_row_ptr[node];
            const std::uint32_t child_end = tree.child_row_ptr[node + 1u];
            for (std::uint32_t off = child_end; off > child_begin; --off) {
                const std::uint32_t child = tree.child_ids[off - 1u];
                tree.depth[child] = tree.depth[node] + 1u;
                stack.emplace_back(child, false);
            }
        }
    }

    return tree;
}

inline SupernodeTable build_supernodes(
    const TrajectoryRecordTable &table,
    const TreeOverlay &tree,
    float max_chain_merge_dt,
    float min_parent_score) {
    table.validate();
    if (tree.nodes != table.rows) throw std::invalid_argument("tree.nodes must match table.rows");

    SupernodeTable supernodes;
    supernodes.latent_dim = table.latent_dim;
    supernodes.super_of_cell.assign(table.rows, std::numeric_limits<std::uint32_t>::max());
    supernodes.member_row_ptr.push_back(0u);

    const auto child_count = [&](std::uint32_t node) {
        return tree.child_row_ptr[node + 1u] - tree.child_row_ptr[node];
    };

    for (std::uint32_t row = 0; row < table.rows; ++row) {
        if (supernodes.super_of_cell[row] != std::numeric_limits<std::uint32_t>::max()) continue;

        const std::int32_t parent = tree.parent[row];
        const bool starts_chain =
            parent < 0
            || child_count(static_cast<std::uint32_t>(parent)) != 1u
            || tree.parent_score[row] < min_parent_score
            || table.developmental_time[row] - table.developmental_time[static_cast<std::uint32_t>(parent)] > max_chain_merge_dt;
        if (!starts_chain) continue;

        const std::uint32_t super_id = supernodes.count++;
        const std::uint32_t begin = row;
        std::vector<std::uint32_t> members{ row };
        supernodes.super_of_cell[row] = super_id;

        std::uint32_t current = row;
        while (child_count(current) == 1u) {
            const std::uint32_t child = tree.child_ids[tree.child_row_ptr[current]];
            if (tree.parent_score[child] < min_parent_score) break;
            if (table.developmental_time[child] - table.developmental_time[current] > max_chain_merge_dt) break;
            supernodes.super_of_cell[child] = super_id;
            members.push_back(child);
            current = child;
        }

        const auto max_it = std::max_element(members.begin(), members.end());
        supernodes.member_cell_ids.insert(supernodes.member_cell_ids.end(), members.begin(), members.end());
        supernodes.member_row_ptr.push_back(static_cast<std::uint32_t>(supernodes.member_cell_ids.size()));
        supernodes.row_begin.push_back(begin);
        supernodes.row_end.push_back(*max_it + 1u);
        supernodes.embryo_id.push_back(table.embryo_id[begin]);
        supernodes.time_begin.push_back(table.developmental_time[begin]);
        supernodes.time_end.push_back(table.developmental_time[current]);
        supernodes.support_mass.push_back(static_cast<float>(members.size()));
        supernodes.is_branch.push_back(0u);

        std::vector<float> centroid_acc(static_cast<std::size_t>(table.latent_dim), 0.0f);
        for (std::uint32_t cell : members) {
            const __half *row_ptr = table.latent_row_ptr(cell);
            for (std::int32_t d = 0; d < table.latent_dim; ++d) centroid_acc[static_cast<std::size_t>(d)] += __half2float(row_ptr[d]);
        }
        const float inv = 1.0f / static_cast<float>(members.size());
        for (float &value : centroid_acc) value *= inv;
        for (float value : centroid_acc) supernodes.centroid.push_back(__float2half_rn(value));
    }

    for (std::uint32_t row = 0; row < table.rows; ++row) {
        if (supernodes.super_of_cell[row] != std::numeric_limits<std::uint32_t>::max()) continue;
        const std::uint32_t super_id = supernodes.count++;
        supernodes.super_of_cell[row] = super_id;
        supernodes.member_cell_ids.push_back(row);
        supernodes.member_row_ptr.push_back(static_cast<std::uint32_t>(supernodes.member_cell_ids.size()));
        supernodes.row_begin.push_back(row);
        supernodes.row_end.push_back(row + 1u);
        supernodes.embryo_id.push_back(table.embryo_id[row]);
        supernodes.time_begin.push_back(table.developmental_time[row]);
        supernodes.time_end.push_back(table.developmental_time[row]);
        supernodes.support_mass.push_back(1.0f);
        supernodes.is_branch.push_back(0u);
        const __half *row_ptr = table.latent_row_ptr(row);
        for (std::int32_t d = 0; d < table.latent_dim; ++d) supernodes.centroid.push_back(row_ptr[d]);
    }

    return supernodes;
}

inline SupernodeDag build_supernode_dag(
    const ForwardCsrGraph &graph,
    const SupernodeTable &supernodes) {
    if (supernodes.super_of_cell.size() != graph.rows) {
        throw std::invalid_argument("supernodes.super_of_cell must align with graph rows");
    }

    std::vector<std::unordered_map<std::uint32_t, std::pair<float, float>>> rows(supernodes.count);
    for (std::uint32_t src = 0; src < graph.rows; ++src) {
        const std::uint32_t src_super = supernodes.super_of_cell[src];
        for (std::uint32_t edge = graph.row_ptr[src]; edge < graph.row_ptr[src + 1u]; ++edge) {
            const std::uint32_t dst_super = supernodes.super_of_cell[graph.dst[edge]];
            if (src_super == dst_super) continue;
            auto &entry = rows[src_super][dst_super];
            entry.first += 1.0f;
            entry.second = std::max(entry.second, graph.score[edge]);
        }
    }

    SupernodeDag dag;
    dag.rows = supernodes.count;
    dag.row_ptr.assign(static_cast<std::size_t>(dag.rows) + 1u, 0u);
    for (std::uint32_t row = 0; row < dag.rows; ++row) {
        dag.row_ptr[row + 1u] = dag.row_ptr[row] + static_cast<std::uint32_t>(rows[row].size());
        std::vector<std::pair<std::uint32_t, std::pair<float, float>>> ordered(rows[row].begin(), rows[row].end());
        std::sort(ordered.begin(), ordered.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.second.first > rhs.second.first) return true;
            if (lhs.second.first < rhs.second.first) return false;
            return lhs.first < rhs.first;
        });
        for (const auto &entry : ordered) {
            dag.dst.push_back(entry.first);
            dag.mass.push_back(entry.second.first);
            dag.score.push_back(entry.second.second);
        }
    }

    return dag;
}

} // namespace cellerator::compute::graph
