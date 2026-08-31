#include "Cellerator/geometry/optimizer/multilevel/validation_v1.hh"

#include <algorithm>
#include <cmath>

namespace cellerator::geometry::optimizer::multilevel {
namespace {

multilevel_status_v1 failure(
    multilevel_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

bool near(double lhs, double rhs, double tolerance) noexcept {
    return std::abs(lhs - rhs) <= tolerance;
}

bool add_counter(std::uint64_t value, std::uint64_t *sum) noexcept {
    if (*sum > UINT64_MAX - value) return false;
    *sum += value;
    return true;
}

void hash_u64(std::uint64_t *hash, std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        *hash ^= static_cast<std::uint8_t>(value >> (byte * 8u));
        *hash *= 1099511628211ull;
    }
}

std::uint64_t provenance_identity(
    const affinity_problem_v1 &problem,
    const multilevel_provenance_v1 &provenance,
    const multilevel_grouping_solution_v1 &solution) noexcept {
    std::uint64_t hash = 1469598103934665603ull;
    hash_u64(&hash, problem.structure_identity);
    hash_u64(&hash, problem.structure_epoch);
    hash_u64(&hash, problem.aggregate_node_count);
    hash_u64(&hash, problem.aggregate_edge_count);
    for (std::uint64_t index = 0u; index < provenance.link_count; ++index) {
        hash_u64(&hash, provenance.links[index].level);
        hash_u64(&hash, provenance.links[index].fine_local);
        hash_u64(&hash, provenance.links[index].coarse_local);
    }
    for (std::uint32_t node = 0u; node < provenance.node_count; ++node) {
        hash_u64(&hash, provenance.nodes[node].global_identity);
        hash_u64(&hash, provenance.nodes[node].final_group);
    }
    hash_u64(&hash, solution.edge_cover_count);
    for (std::uint32_t index = 0u; index < solution.edge_cover_count; ++index) {
        const logical_edge_cover_v1 &cover = solution.edge_cover[index];
        hash_u64(&hash, cover.logical_edge_identity);
        hash_u64(&hash, cover.lhs_group);
        hash_u64(&hash, cover.rhs_group);
        hash_u64(&hash, static_cast<std::uint64_t>(cover.cover_class));
    }
    return hash == 0u ? 1u : hash;
}

}  // namespace

multilevel_status_v1 validate_multilevel_result_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const multilevel_grouping_solution_v1 &solution,
    const multilevel_validation_limits_v1 &limits,
    multilevel_validation_workspace_v1 workspace,
    multilevel_validation_report_v1 *out) noexcept {
    if (out == nullptr || !std::isfinite(limits.affinity_tolerance)
        || limits.affinity_tolerance < 0.0
        || limits.fine_nodes_per_group_capacity == 0u) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    multilevel_status_v1 status = validate_affinity_problem_v1(problem);
    if (!status) return status;
    if (hierarchy.structure_identity != problem.structure_identity
        || hierarchy.structure_epoch != problem.structure_epoch
        || hierarchy.initial_local_node_count != problem.local_node_count
        || solution.structure_identity != problem.structure_identity
        || solution.structure_epoch != problem.structure_epoch
        || solution.fine_node_count != problem.local_node_count
        || solution.fine_node_to_group == nullptr || solution.group_count == 0u
        || workspace.group_sizes == nullptr
        || workspace.group_capacity < solution.group_count
        || solution.edge_cover_count != problem.local_edge_count
        || (solution.edge_cover_count != 0u && solution.edge_cover == nullptr)) {
        return failure(multilevel_status_code_v1::invalid_argument, 1u);
    }
    if (hierarchy.level_count != 0u
        && (hierarchy.levels == nullptr || hierarchy.fine_to_coarse == nullptr)) {
        return failure(multilevel_status_code_v1::invalid_argument, 2u);
    }

    std::uint64_t link_count = 0u;
    std::uint32_t expected_fine = problem.local_node_count;
    for (std::uint32_t level = 0u; level < hierarchy.level_count; ++level) {
        const affinity_hierarchy_level_v1 &record = hierarchy.levels[level];
        if (record.fine_node_count != expected_fine || record.coarse_node_count == 0u
            || record.coarse_node_count >= record.fine_node_count
            || record.fine_to_coarse_offset != link_count
            || link_count > hierarchy.fine_to_coarse_count
            || record.fine_node_count > hierarchy.fine_to_coarse_count - link_count) {
            return failure(multilevel_status_code_v1::invalid_argument, level);
        }
        for (std::uint32_t fine = 0u; fine < record.fine_node_count; ++fine) {
            if (hierarchy.fine_to_coarse[link_count + fine]
                >= record.coarse_node_count) {
                return failure(multilevel_status_code_v1::invalid_argument,
                    link_count + fine);
            }
        }
        link_count += record.fine_node_count;
        expected_fine = record.coarse_node_count;
    }
    if (link_count != hierarchy.fine_to_coarse_count
        || expected_fine != hierarchy.coarsest_node_count) {
        return failure(multilevel_status_code_v1::invalid_argument, link_count);
    }

    std::uint64_t hierarchy_operations = 0u;
    std::uint64_t refinement_operations = 0u;
    if (!add_counter(hierarchy.counters.nodes_visited, &hierarchy_operations)
        || !add_counter(hierarchy.counters.edges_visited, &hierarchy_operations)
        || !add_counter(hierarchy.counters.edges_sorted, &hierarchy_operations)
        || !add_counter(hierarchy.counters.edges_contracted, &hierarchy_operations)
        || !add_counter(solution.counters.hierarchy_entries_visited,
            &refinement_operations)
        || !add_counter(solution.counters.edges_visited, &refinement_operations)
        || !add_counter(solution.counters.nodes_considered, &refinement_operations)) {
        return failure(multilevel_status_code_v1::arithmetic_overflow, 0u);
    }
    if (hierarchy_operations > limits.maximum_hierarchy_operations
        || refinement_operations > limits.maximum_refinement_operations) {
        return failure(multilevel_status_code_v1::insufficient_capacity,
            hierarchy_operations > limits.maximum_hierarchy_operations
                ? hierarchy_operations : refinement_operations);
    }

    double grouped_affinity = 0.0;
    double residual_affinity = 0.0;
    std::fill_n(workspace.group_sizes, solution.group_count, 0u);
    for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
        if (solution.fine_node_to_group[node] >= solution.group_count) {
            return failure(multilevel_status_code_v1::invalid_argument, node);
        }
        ++workspace.group_sizes[solution.fine_node_to_group[node]];
    }
    for (std::uint32_t group = 0u; group < solution.group_count; ++group) {
        const std::uint32_t size = workspace.group_sizes[group];
        if (size == 0u || size > limits.fine_nodes_per_group_capacity) {
            return failure(multilevel_status_code_v1::insufficient_capacity, group);
        }
    }
    for (std::uint32_t index = 0u; index < problem.local_edge_count; ++index) {
        const affinity_edge_v1 &edge = problem.edges[index];
        const logical_edge_cover_v1 &cover = solution.edge_cover[index];
        const std::uint32_t lhs_group = solution.fine_node_to_group[edge.lhs];
        const std::uint32_t rhs_group = solution.fine_node_to_group[edge.rhs];
        const cover_class_v1 expected_class = lhs_group == rhs_group
            ? cover_class_v1::grouped : cover_class_v1::residual;
        if (cover.logical_edge_identity != edge.stable_identity
            || cover.lhs_group != lhs_group || cover.rhs_group != rhs_group
            || cover.cover_class != expected_class) {
            return failure(multilevel_status_code_v1::invalid_edge, index);
        }
        grouped_affinity += expected_class == cover_class_v1::grouped
            ? edge.affinity : 0.0;
        residual_affinity += expected_class == cover_class_v1::residual
            ? edge.affinity : 0.0;
    }
    if (!near(grouped_affinity, solution.grouped_affinity,
            limits.affinity_tolerance)
        || !near(residual_affinity, solution.residual_affinity,
            limits.affinity_tolerance)) {
        return failure(multilevel_status_code_v1::invalid_edge, UINT64_MAX);
    }
    *out = {multilevel_validation_schema_v1, problem.local_node_count,
        link_count, problem.local_edge_count, hierarchy_operations,
        refinement_operations, grouped_affinity, residual_affinity};
    return {};
}

multilevel_status_v1 validate_multilevel_provenance_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const multilevel_grouping_solution_v1 &solution,
    const multilevel_provenance_v1 &provenance) noexcept {
    if (provenance.schema_version != streaming_provenance_schema_v1
        || provenance.structure_identity != problem.structure_identity
        || provenance.structure_epoch != problem.structure_epoch
        || provenance.link_count != hierarchy.fine_to_coarse_count
        || provenance.node_count != problem.local_node_count
        || (provenance.link_count != 0u && provenance.links == nullptr)
        || provenance.nodes == nullptr) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    std::uint64_t write = 0u;
    for (std::uint32_t level = 0u; level < hierarchy.level_count; ++level) {
        const affinity_hierarchy_level_v1 &record = hierarchy.levels[level];
        for (std::uint32_t fine = 0u; fine < record.fine_node_count; ++fine) {
            const multilevel_provenance_link_v1 &link = provenance.links[write];
            if (link.level != level || link.fine_local != fine
                || link.coarse_local
                    != hierarchy.fine_to_coarse[record.fine_to_coarse_offset + fine]) {
                return failure(multilevel_status_code_v1::invalid_argument, write);
            }
            ++write;
        }
    }
    for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
        if (provenance.nodes[node].global_identity
                != problem.node_global_identities[node]
            || provenance.nodes[node].final_group
                != solution.fine_node_to_group[node]) {
            return failure(multilevel_status_code_v1::invalid_identity, node);
        }
    }
    if (provenance.deterministic_identity
        != provenance_identity(problem, provenance, solution)) {
        return failure(multilevel_status_code_v1::invalid_identity, UINT64_MAX);
    }
    return {};
}

multilevel_status_v1 compare_multilevel_replay_v1(
    const affinity_hierarchy_v1 &lhs_hierarchy,
    const multilevel_grouping_solution_v1 &lhs_solution,
    const affinity_hierarchy_v1 &rhs_hierarchy,
    const multilevel_grouping_solution_v1 &rhs_solution) noexcept {
    if (lhs_hierarchy.structure_identity != rhs_hierarchy.structure_identity
        || lhs_hierarchy.structure_epoch != rhs_hierarchy.structure_epoch
        || lhs_hierarchy.level_count != rhs_hierarchy.level_count
        || lhs_hierarchy.fine_to_coarse_count != rhs_hierarchy.fine_to_coarse_count
        || lhs_hierarchy.coarsest_node_count != rhs_hierarchy.coarsest_node_count
        || lhs_solution.fine_node_count != rhs_solution.fine_node_count
        || lhs_solution.group_count != rhs_solution.group_count
        || lhs_solution.edge_cover_count != rhs_solution.edge_cover_count) {
        return failure(multilevel_status_code_v1::invalid_identity, 0u);
    }
    for (std::uint32_t level = 0u; level < lhs_hierarchy.level_count; ++level) {
        const affinity_hierarchy_level_v1 &lhs = lhs_hierarchy.levels[level];
        const affinity_hierarchy_level_v1 &rhs = rhs_hierarchy.levels[level];
        if (lhs.fine_node_count != rhs.fine_node_count
            || lhs.coarse_node_count != rhs.coarse_node_count
            || lhs.coarse_edge_count != rhs.coarse_edge_count
            || lhs.fine_to_coarse_offset != rhs.fine_to_coarse_offset
            || lhs.retained_affinity != rhs.retained_affinity) {
            return failure(multilevel_status_code_v1::invalid_identity, level);
        }
    }
    for (std::uint64_t index = 0u; index < lhs_hierarchy.fine_to_coarse_count; ++index) {
        if (lhs_hierarchy.fine_to_coarse[index]
            != rhs_hierarchy.fine_to_coarse[index]) {
            return failure(multilevel_status_code_v1::invalid_identity, index);
        }
    }
    for (std::uint32_t node = 0u; node < lhs_solution.fine_node_count; ++node) {
        if (lhs_solution.fine_node_to_group[node]
            != rhs_solution.fine_node_to_group[node]) {
            return failure(multilevel_status_code_v1::invalid_identity, node);
        }
    }
    for (std::uint32_t edge = 0u; edge < lhs_solution.edge_cover_count; ++edge) {
        const logical_edge_cover_v1 &lhs = lhs_solution.edge_cover[edge];
        const logical_edge_cover_v1 &rhs = rhs_solution.edge_cover[edge];
        if (lhs.logical_edge_identity != rhs.logical_edge_identity
            || lhs.lhs_group != rhs.lhs_group || lhs.rhs_group != rhs.rhs_group
            || lhs.cover_class != rhs.cover_class) {
            return failure(multilevel_status_code_v1::invalid_identity, edge);
        }
    }
    return {};
}

multilevel_status_v1 validate_workload_objective_v1(
    const workload_affinity_profile_v1 &profile,
    const multilevel_grouping_solution_v1 &solution,
    double grouped_unit_cost,
    double residual_unit_cost,
    const workload_solution_objective_v1 &expected,
    double tolerance) noexcept {
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    workload_solution_objective_v1 actual{};
    const multilevel_status_v1 status = evaluate_workload_solution_v1(
        profile, solution, grouped_unit_cost, residual_unit_cost, &actual);
    if (!status) return status;
    if (actual.mixture_identity != expected.mixture_identity
        || !near(actual.grouped_cost, expected.grouped_cost, tolerance)
        || !near(actual.residual_cost, expected.residual_cost, tolerance)
        || !near(actual.total_cost, expected.total_cost, tolerance)) {
        return failure(multilevel_status_code_v1::invalid_identity,
            actual.mixture_identity);
    }
    return {};
}

}  // namespace cellerator::geometry::optimizer::multilevel
