#pragma once

#include "forward_candidates.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cellerator::compute::graph {

struct ForwardCsrGraph {
    std::uint32_t rows = 0;
    std::vector<std::uint32_t> row_ptr;
    std::vector<std::uint32_t> dst;
    std::vector<float> score;
    std::vector<float> delta_t;
    std::vector<std::uint8_t> flags;
};

struct InboundCsrGraph {
    std::uint32_t rows = 0;
    std::vector<std::uint32_t> row_ptr;
    std::vector<std::uint32_t> src;
    std::vector<float> score;
};

inline ForwardCsrGraph prune_candidate_edges(
    const CandidateEdgeTable &candidates,
    const ForwardNeighborConfig &config = ForwardNeighborConfig()) {
    if (candidates.dst.size() != candidates.similarity.size() || candidates.dst.size() != candidates.delta_t.size()) {
        throw std::invalid_argument("CandidateEdgeTable arrays must align");
    }
    if (candidates.rows == 0) {
        return ForwardCsrGraph{
            0,
            std::vector<std::uint32_t>(1, 0),
            {},
            {},
            {},
            {}
        };
    }
    if (candidates.k == 0u || candidates.k > static_cast<std::uint32_t>(detail::max_candidate_k)
        || candidates.dst.size() != static_cast<std::size_t>(candidates.rows) * candidates.k) {
        throw std::invalid_argument("CandidateEdgeTable must be a bounded fixed-K table with K in [1, 8]");
    }

    ForwardCsrGraph graph;
    graph.rows = candidates.rows;
    graph.row_ptr.resize(static_cast<std::size_t>(graph.rows) + 1u, 0u);
    const std::size_t maximum_edges = static_cast<std::size_t>(graph.rows)
        * std::min(config.max_out_degree, candidates.k);
    graph.dst.reserve(maximum_edges);
    graph.score.reserve(maximum_edges);
    graph.delta_t.reserve(maximum_edges);
    graph.flags.reserve(maximum_edges);

    for (std::uint32_t row = 0; row < graph.rows; ++row) {
        struct alignas(16) edge {
            std::uint32_t dst;
            float score;
            float dt;
            std::uint32_t flags;
        };
        edge row_edges[detail::max_candidate_k]{};
        std::uint32_t row_edge_count = 0u;

        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(candidates.k);
        for (std::uint32_t slot = 0; slot < candidates.k; ++slot) {
            const std::size_t off = base + static_cast<std::size_t>(slot);
            if (candidates.dst[off] == std::numeric_limits<std::uint32_t>::max()) continue;
            if (!std::isfinite(candidates.similarity[off])) continue;
            if (candidates.similarity[off] < config.min_similarity) continue;
            row_edges[row_edge_count++] = edge{
                candidates.dst[off], candidates.similarity[off], candidates.delta_t[off], 0u};
        }

        std::sort(row_edges, row_edges + row_edge_count, [](const edge &lhs, const edge &rhs) {
            if (lhs.score > rhs.score) return true;
            if (lhs.score < rhs.score) return false;
            if (lhs.dt < rhs.dt) return true;
            if (lhs.dt > rhs.dt) return false;
            return lhs.dst < rhs.dst;
        });

        edge kept[detail::max_candidate_k]{};
        std::uint32_t kept_count = 0u;
        for (std::uint32_t index = 0u; index < row_edge_count; ++index) {
            const edge &candidate = row_edges[index];
            bool dominated = false;
            for (std::uint32_t prior = 0u; prior < kept_count; ++prior) {
                const edge &existing = kept[prior];
                if (candidate.dt > existing.dt * config.shortcut_ratio
                    && candidate.score + config.shortcut_similarity_gap < existing.score) {
                    dominated = true;
                    break;
                }
            }
            if (dominated) continue;
            kept[kept_count++] = candidate;
            if (kept_count >= config.max_out_degree) break;
        }

        graph.row_ptr[static_cast<std::size_t>(row + 1u)] =
            graph.row_ptr[static_cast<std::size_t>(row)] + kept_count;
        for (std::uint32_t index = 0u; index < kept_count; ++index) {
            const edge &entry = kept[index];
            graph.dst.push_back(entry.dst);
            graph.score.push_back(entry.score);
            graph.delta_t.push_back(entry.dt);
            graph.flags.push_back(static_cast<std::uint8_t>(entry.flags));
        }
    }

    return graph;
}

inline InboundCsrGraph build_inbound_graph(const ForwardCsrGraph &graph) {
    if (graph.row_ptr.size() != static_cast<std::size_t>(graph.rows) + 1u) {
        throw std::invalid_argument("ForwardCsrGraph row_ptr must equal rows + 1");
    }

    InboundCsrGraph inbound;
    inbound.rows = graph.rows;
    inbound.row_ptr.assign(static_cast<std::size_t>(graph.rows) + 1u, 0u);

    for (std::uint32_t dst : graph.dst) {
        if (dst >= graph.rows) throw std::out_of_range("graph dst is out of range");
        ++inbound.row_ptr[static_cast<std::size_t>(dst + 1u)];
    }
    for (std::size_t i = 1; i < inbound.row_ptr.size(); ++i) {
        inbound.row_ptr[i] += inbound.row_ptr[i - 1u];
    }

    inbound.src.assign(graph.dst.size(), 0u);
    inbound.score.assign(graph.score.size(), 0.0f);
    std::vector<std::uint32_t> cursor = inbound.row_ptr;
    for (std::uint32_t row = 0; row < graph.rows; ++row) {
        for (std::uint32_t edge = graph.row_ptr[row]; edge < graph.row_ptr[row + 1u]; ++edge) {
            const std::uint32_t dst = graph.dst[edge];
            const std::uint32_t off = cursor[dst]++;
            inbound.src[off] = row;
            inbound.score[off] = graph.score[edge];
        }
    }
    return inbound;
}

} // namespace cellerator::compute::graph
