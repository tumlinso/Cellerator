#include "Cellerator/geometry/optimizer/multilevel/coarse_refinement_v1.hh"

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

bool valid_hierarchy(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy) noexcept {
    if (hierarchy.schema_version != affinity_hierarchy_schema_v1
        || hierarchy.structure_identity != problem.structure_identity
        || hierarchy.structure_epoch != problem.structure_epoch
        || hierarchy.initial_local_node_count != problem.local_node_count
        || hierarchy.coarsest_node_count == 0u
        || hierarchy.coarsest_global_identities == nullptr
        || (hierarchy.level_count != 0u
            && (hierarchy.levels == nullptr || hierarchy.fine_to_coarse == nullptr))) {
        return false;
    }
    std::uint32_t expected_fine = problem.local_node_count;
    std::uint64_t expected_offset = 0u;
    for (std::uint32_t level = 0u; level < hierarchy.level_count; ++level) {
        const affinity_hierarchy_level_v1 &record = hierarchy.levels[level];
        if (record.fine_node_count != expected_fine
            || record.coarse_node_count == 0u
            || record.coarse_node_count >= record.fine_node_count
            || record.fine_to_coarse_offset != expected_offset
            || expected_offset > hierarchy.fine_to_coarse_count
            || record.fine_node_count > hierarchy.fine_to_coarse_count - expected_offset) {
            return false;
        }
        for (std::uint32_t node = 0u; node < record.fine_node_count; ++node) {
            if (hierarchy.fine_to_coarse[expected_offset + node]
                >= record.coarse_node_count) return false;
        }
        expected_offset += record.fine_node_count;
        expected_fine = record.coarse_node_count;
    }
    return expected_fine == hierarchy.coarsest_node_count
        && expected_offset == hierarchy.fine_to_coarse_count;
}

}  // namespace

multilevel_status_v1 solve_and_refine_multilevel_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const coarse_refinement_config_v1 &config,
    coarse_refinement_workspace_v1 workspace,
    multilevel_grouping_solution_v1 *out) noexcept {
    if (out == nullptr || config.schema_version != coarse_refinement_schema_v1
        || config.fine_nodes_per_group_capacity == 0u
        || config.group_capacity == 0u
        || !std::isfinite(config.minimum_move_gain)
        || config.minimum_move_gain < 0.0) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    multilevel_status_v1 status = validate_affinity_problem_v1(problem);
    if (!status) return status;
    if (!valid_hierarchy(problem, hierarchy)) {
        return failure(multilevel_status_code_v1::invalid_argument, 1u);
    }
    if (workspace.node_capacity < problem.local_node_count
        || workspace.edge_cover_capacity < problem.local_edge_count
        || workspace.group_capacity < config.group_capacity
        || workspace.assignment_a == nullptr || workspace.assignment_b == nullptr
        || workspace.node_weight_a == nullptr || workspace.node_weight_b == nullptr
        || workspace.proposed_group == nullptr || workspace.proposed_affinity == nullptr
        || workspace.move_gain == nullptr || workspace.move_allowed == nullptr
        || workspace.group_sizes == nullptr
        || (problem.local_edge_count != 0u && workspace.edge_cover == nullptr)) {
        return failure(multilevel_status_code_v1::insufficient_capacity, 0u);
    }

    coarse_refinement_counters_v1 counters{};
    std::uint64_t *current_weight = workspace.node_weight_a;
    std::uint64_t *next_weight = workspace.node_weight_b;
    std::fill_n(current_weight, problem.local_node_count, 1u);
    std::uint32_t current_count = problem.local_node_count;
    for (std::uint32_t level = 0u; level < hierarchy.level_count; ++level) {
        const affinity_hierarchy_level_v1 &record = hierarchy.levels[level];
        std::fill_n(next_weight, record.coarse_node_count, 0u);
        const std::uint32_t *mapping = hierarchy.fine_to_coarse
            + record.fine_to_coarse_offset;
        for (std::uint32_t node = 0u; node < current_count; ++node) {
            const std::uint32_t coarse = mapping[node];
            if (next_weight[coarse] > UINT64_MAX - current_weight[node]) {
                return failure(multilevel_status_code_v1::arithmetic_overflow, node);
            }
            next_weight[coarse] += current_weight[node];
        }
        counters.hierarchy_entries_visited += current_count;
        current_count = record.coarse_node_count;
        std::swap(current_weight, next_weight);
    }

    std::uint32_t group_count = 0u;
    std::uint64_t current_group_size = 0u;
    for (std::uint32_t node = 0u; node < hierarchy.coarsest_node_count; ++node) {
        const std::uint64_t weight = current_weight[node];
        if (weight > config.fine_nodes_per_group_capacity) {
            return failure(multilevel_status_code_v1::insufficient_capacity, node);
        }
        if (group_count == 0u
            || current_group_size > config.fine_nodes_per_group_capacity - weight) {
            if (group_count == config.group_capacity) {
                return failure(multilevel_status_code_v1::insufficient_capacity, group_count);
            }
            current_group_size = 0u;
            ++group_count;
        }
        workspace.assignment_a[node] = group_count - 1u;
        current_group_size += weight;
    }

    std::uint32_t *current_assignment = workspace.assignment_a;
    std::uint32_t *next_assignment = workspace.assignment_b;
    current_count = hierarchy.coarsest_node_count;
    for (std::uint32_t reverse = hierarchy.level_count; reverse != 0u; --reverse) {
        const affinity_hierarchy_level_v1 &record = hierarchy.levels[reverse - 1u];
        const std::uint32_t *mapping = hierarchy.fine_to_coarse
            + record.fine_to_coarse_offset;
        for (std::uint32_t node = 0u; node < record.fine_node_count; ++node) {
            next_assignment[node] = current_assignment[mapping[node]];
        }
        counters.hierarchy_entries_visited += record.fine_node_count;
        current_count = record.fine_node_count;
        std::swap(current_assignment, next_assignment);
    }
    if (current_count != problem.local_node_count) {
        return failure(multilevel_status_code_v1::invalid_argument, 2u);
    }

    std::fill_n(workspace.group_sizes, group_count, 0u);
    for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
        ++workspace.group_sizes[current_assignment[node]];
    }

    std::uint32_t passes = 0u;
    for (; passes < config.max_refinement_passes; ++passes) {
        for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
            workspace.proposed_group[node] = current_assignment[node];
            workspace.proposed_affinity[node] = -std::numeric_limits<double>::infinity();
            workspace.move_gain[node] = 0.0;
        }
        for (std::uint32_t index = 0u; index < problem.local_edge_count; ++index) {
            const affinity_edge_v1 &edge = problem.edges[index];
            const std::uint32_t endpoints[2] = {edge.lhs, edge.rhs};
            for (std::uint32_t side = 0u; side < 2u; ++side) {
                const std::uint32_t node = endpoints[side];
                const std::uint32_t candidate_group = current_assignment[endpoints[1u - side]];
                if (candidate_group == current_assignment[node]) continue;
                if (edge.affinity > workspace.proposed_affinity[node]
                    || (edge.affinity == workspace.proposed_affinity[node]
                        && candidate_group < workspace.proposed_group[node])) {
                    workspace.proposed_affinity[node] = edge.affinity;
                    workspace.proposed_group[node] = candidate_group;
                }
            }
        }
        counters.edges_visited += problem.local_edge_count;
        for (std::uint32_t index = 0u; index < problem.local_edge_count; ++index) {
            const affinity_edge_v1 &edge = problem.edges[index];
            const std::uint32_t endpoints[2] = {edge.lhs, edge.rhs};
            for (std::uint32_t side = 0u; side < 2u; ++side) {
                const std::uint32_t node = endpoints[side];
                const std::uint32_t other = endpoints[1u - side];
                const bool old_cut = current_assignment[node] != current_assignment[other];
                const bool new_cut = workspace.proposed_group[node] != current_assignment[other];
                workspace.move_gain[node] += (old_cut ? edge.affinity : 0.0)
                    - (new_cut ? edge.affinity : 0.0);
            }
        }
        counters.edges_visited += problem.local_edge_count;

        for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
            const std::uint32_t source = current_assignment[node];
            const std::uint32_t destination = workspace.proposed_group[node];
            workspace.move_allowed[node] = destination != source
                && workspace.move_gain[node] > config.minimum_move_gain ? 1u : 0u;
        }
        for (std::uint32_t index = 0u; index < problem.local_edge_count; ++index) {
            const affinity_edge_v1 &edge = problem.edges[index];
            if (workspace.move_allowed[edge.lhs] != 0u
                && workspace.move_allowed[edge.rhs] != 0u) {
                workspace.move_allowed[std::max(edge.lhs, edge.rhs)] = 0u;
            }
        }
        counters.edges_visited += problem.local_edge_count;

        std::uint32_t moved = 0u;
        for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
            ++counters.nodes_considered;
            if (workspace.move_allowed[node] == 0u) continue;
            const std::uint32_t source = current_assignment[node];
            const std::uint32_t destination = workspace.proposed_group[node];
            if (workspace.group_sizes[source] <= 1u
                || workspace.group_sizes[destination]
                    >= config.fine_nodes_per_group_capacity) continue;
            --workspace.group_sizes[source];
            ++workspace.group_sizes[destination];
            current_assignment[node] = destination;
            ++moved;
        }
        counters.moves_applied += moved;
        if (moved == 0u) break;
    }

    double grouped_affinity = 0.0;
    double residual_affinity = 0.0;
    for (std::uint32_t index = 0u; index < problem.local_edge_count; ++index) {
        const affinity_edge_v1 &edge = problem.edges[index];
        const std::uint32_t lhs_group = current_assignment[edge.lhs];
        const std::uint32_t rhs_group = current_assignment[edge.rhs];
        const bool grouped = lhs_group == rhs_group;
        workspace.edge_cover[index] = {
            edge.stable_identity,
            lhs_group,
            rhs_group,
            grouped ? cover_class_v1::grouped : cover_class_v1::residual,
        };
        grouped_affinity += grouped ? edge.affinity : 0.0;
        residual_affinity += grouped ? 0.0 : edge.affinity;
    }
    counters.edges_visited += problem.local_edge_count;

    *out = {
        coarse_refinement_schema_v1,
        problem.structure_identity,
        problem.structure_epoch,
        current_assignment,
        problem.local_node_count,
        group_count,
        workspace.edge_cover,
        problem.local_edge_count,
        grouped_affinity,
        residual_affinity,
        passes,
        counters,
    };
    return {};
}

}  // namespace cellerator::geometry::optimizer::multilevel
