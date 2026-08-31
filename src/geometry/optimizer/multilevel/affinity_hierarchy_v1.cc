#include "Cellerator/geometry/optimizer/multilevel/affinity_hierarchy_v1.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::geometry::optimizer::multilevel {
namespace {

multilevel_status_v1 failure(
    multilevel_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

bool edge_less(const affinity_edge_v1 &lhs, const affinity_edge_v1 &rhs) noexcept {
    if (lhs.lhs != rhs.lhs) return lhs.lhs < rhs.lhs;
    if (lhs.rhs != rhs.rhs) return lhs.rhs < rhs.rhs;
    return lhs.stable_identity < rhs.stable_identity;
}

std::uint32_t canonicalize_edges(
    affinity_edge_v1 *edges,
    std::uint32_t count,
    double *retained_affinity) noexcept {
    for (std::uint32_t index = 0u; index < count; ++index) {
        if (edges[index].lhs > edges[index].rhs) {
            std::swap(edges[index].lhs, edges[index].rhs);
        }
    }
    std::sort(edges, edges + count, edge_less);
    std::uint32_t write = 0u;
    double retained = 0.0;
    for (std::uint32_t read = 0u; read < count; ++read) {
        affinity_edge_v1 edge = edges[read];
        if (edge.lhs == edge.rhs) continue;
        if (write != 0u && edges[write - 1u].lhs == edge.lhs
            && edges[write - 1u].rhs == edge.rhs) {
            edges[write - 1u].affinity += edge.affinity;
            edges[write - 1u].stable_identity = std::min(
                edges[write - 1u].stable_identity, edge.stable_identity);
        } else {
            edges[write++] = edge;
        }
    }
    for (std::uint32_t index = 0u; index < write; ++index) {
        retained += edges[index].affinity;
    }
    *retained_affinity = retained;
    return write;
}

bool better_partner(
    double candidate_affinity,
    std::uint64_t candidate_identity,
    std::uint32_t candidate_local,
    double current_affinity,
    std::uint64_t current_identity,
    std::uint32_t current_local) noexcept {
    return candidate_affinity > current_affinity
        || (candidate_affinity == current_affinity
            && (candidate_identity < current_identity
                || (candidate_identity == current_identity
                    && candidate_local < current_local)));
}

}  // namespace

multilevel_status_v1 validate_affinity_problem_v1(
    const affinity_problem_v1 &problem) noexcept {
    if (problem.schema_version != affinity_hierarchy_schema_v1
        || problem.structure_identity == 0u || problem.structure_epoch == 0u
        || problem.local_node_count == 0u
        || problem.node_global_identities == nullptr
        || (problem.local_edge_count != 0u && problem.edges == nullptr)
        || problem.aggregate_node_count < problem.local_node_count
        || problem.aggregate_edge_count < problem.local_edge_count) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
        const std::uint64_t identity = problem.node_global_identities[node];
        if (identity == 0u) {
            return failure(multilevel_status_code_v1::invalid_identity, node);
        }
        if (node != 0u && problem.node_global_identities[node - 1u] >= identity) {
            return failure(multilevel_status_code_v1::invalid_order, node);
        }
    }
    for (std::uint32_t index = 0u; index < problem.local_edge_count; ++index) {
        const affinity_edge_v1 &edge = problem.edges[index];
        if (edge.lhs >= problem.local_node_count || edge.rhs >= problem.local_node_count
            || edge.lhs == edge.rhs || edge.stable_identity == 0u
            || !std::isfinite(edge.affinity) || edge.affinity <= 0.0) {
            return failure(multilevel_status_code_v1::invalid_edge, index);
        }
    }
    return {};
}

multilevel_status_v1 build_affinity_hierarchy_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_config_v1 &config,
    affinity_hierarchy_workspace_v1 workspace,
    affinity_hierarchy_v1 *out) noexcept {
    if (out == nullptr || config.target_coarse_node_count == 0u
        || config.max_levels == 0u || !std::isfinite(config.minimum_affinity)
        || config.minimum_affinity < 0.0) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    multilevel_status_v1 status = validate_affinity_problem_v1(problem);
    if (!status) return status;
    if (workspace.node_capacity < problem.local_node_count
        || workspace.edge_capacity < problem.local_edge_count
        || workspace.level_capacity < config.max_levels
        || workspace.node_identities_a == nullptr || workspace.node_identities_b == nullptr
        || workspace.edges_a == nullptr || workspace.edges_b == nullptr
        || workspace.best_partner == nullptr || workspace.best_affinity == nullptr
        || workspace.fine_to_coarse == nullptr || workspace.levels == nullptr) {
        return failure(multilevel_status_code_v1::insufficient_capacity, 0u);
    }

    std::copy_n(problem.node_global_identities, problem.local_node_count,
        workspace.node_identities_a);
    std::copy_n(problem.edges, problem.local_edge_count, workspace.edges_a);

    std::uint64_t *current_ids = workspace.node_identities_a;
    std::uint64_t *next_ids = workspace.node_identities_b;
    affinity_edge_v1 *current_edges = workspace.edges_a;
    affinity_edge_v1 *next_edges = workspace.edges_b;
    std::uint32_t current_node_count = problem.local_node_count;
    std::uint32_t current_edge_count = problem.local_edge_count;
    std::uint64_t map_offset = 0u;
    std::uint32_t level_count = 0u;
    affinity_hierarchy_counters_v1 counters{};

    while (level_count < config.max_levels
        && current_node_count > config.target_coarse_node_count) {
        if (map_offset > workspace.fine_to_coarse_capacity
            || current_node_count > workspace.fine_to_coarse_capacity - map_offset) {
            return failure(multilevel_status_code_v1::insufficient_capacity, map_offset);
        }

        double retained_affinity = 0.0;
        current_edge_count = canonicalize_edges(
            current_edges, current_edge_count, &retained_affinity);
        counters.edges_sorted += current_edge_count;
        for (std::uint32_t node = 0u; node < current_node_count; ++node) {
            workspace.best_partner[node] = no_local_node_v1;
            workspace.best_affinity[node] = -std::numeric_limits<double>::infinity();
        }
        for (std::uint32_t index = 0u; index < current_edge_count; ++index) {
            const affinity_edge_v1 &edge = current_edges[index];
            if (edge.affinity < config.minimum_affinity) continue;
            const std::uint32_t endpoints[2] = {edge.lhs, edge.rhs};
            for (std::uint32_t side = 0u; side < 2u; ++side) {
                const std::uint32_t node = endpoints[side];
                const std::uint32_t candidate = endpoints[1u - side];
                const std::uint32_t current = workspace.best_partner[node];
                const std::uint64_t current_identity = current == no_local_node_v1
                    ? UINT64_MAX : current_ids[current];
                if (better_partner(edge.affinity, current_ids[candidate], candidate,
                        workspace.best_affinity[node], current_identity, current)) {
                    workspace.best_partner[node] = candidate;
                    workspace.best_affinity[node] = edge.affinity;
                }
            }
        }
        counters.edges_visited += current_edge_count;

        std::uint32_t *mapping = workspace.fine_to_coarse + map_offset;
        std::fill_n(mapping, current_node_count, no_local_node_v1);
        std::uint32_t coarse_count = 0u;
        for (std::uint32_t node = 0u; node < current_node_count; ++node) {
            if (mapping[node] != no_local_node_v1) continue;
            const std::uint32_t partner = workspace.best_partner[node];
            mapping[node] = coarse_count;
            std::uint64_t coarse_identity = current_ids[node];
            if (partner != no_local_node_v1 && mapping[partner] == no_local_node_v1) {
                mapping[partner] = coarse_count;
                coarse_identity = std::min(coarse_identity, current_ids[partner]);
            }
            next_ids[coarse_count++] = coarse_identity;
        }
        counters.nodes_visited += current_node_count;
        if (coarse_count == current_node_count) break;

        std::uint32_t next_edge_count = 0u;
        for (std::uint32_t index = 0u; index < current_edge_count; ++index) {
            const affinity_edge_v1 &edge = current_edges[index];
            std::uint32_t lhs = mapping[edge.lhs];
            std::uint32_t rhs = mapping[edge.rhs];
            if (lhs == rhs) continue;
            if (lhs > rhs) std::swap(lhs, rhs);
            next_edges[next_edge_count++] = {
                lhs, rhs, edge.affinity, edge.stable_identity};
        }
        counters.edges_contracted += current_edge_count;
        double next_retained = 0.0;
        next_edge_count = canonicalize_edges(
            next_edges, next_edge_count, &next_retained);

        workspace.levels[level_count] = {
            current_node_count,
            coarse_count,
            next_edge_count,
            map_offset,
            next_retained,
        };
        ++level_count;
        map_offset += current_node_count;
        current_node_count = coarse_count;
        current_edge_count = next_edge_count;
        std::swap(current_ids, next_ids);
        std::swap(current_edges, next_edges);
    }

    *out = {
        affinity_hierarchy_schema_v1,
        problem.structure_identity,
        problem.structure_epoch,
        problem.aggregate_node_count,
        problem.aggregate_edge_count,
        problem.local_node_count,
        level_count,
        workspace.levels,
        workspace.fine_to_coarse,
        map_offset,
        current_ids,
        current_node_count,
        counters,
    };
    return {};
}

}  // namespace cellerator::geometry::optimizer::multilevel
