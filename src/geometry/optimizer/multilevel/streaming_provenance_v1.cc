#include "Cellerator/geometry/optimizer/multilevel/streaming_provenance_v1.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cellerator::geometry::optimizer::multilevel {
namespace {

multilevel_status_v1 failure(
    multilevel_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

void hash_u64(std::uint64_t *hash, std::uint64_t value) noexcept {
    for (std::uint32_t byte = 0u; byte < 8u; ++byte) {
        *hash ^= static_cast<std::uint8_t>(value >> (byte * 8u));
        *hash *= 1099511628211ull;
    }
}

bool add_bytes(std::uint64_t count, std::uint64_t width, std::uint64_t *total) noexcept {
    if (count != 0u && width > UINT64_MAX / count) return false;
    const std::uint64_t bytes = count * width;
    if (*total > UINT64_MAX - bytes) return false;
    *total += bytes;
    return true;
}

}  // namespace

multilevel_status_v1 append_affinity_batch_v1(
    streaming_affinity_builder_v1 *builder,
    const affinity_edge_v1 *edges,
    std::uint32_t edge_count) noexcept {
    if (builder == nullptr
        || builder->schema_version != streaming_provenance_schema_v1
        || builder->structure_identity == 0u || builder->structure_epoch == 0u
        || builder->local_node_count == 0u
        || builder->node_global_identities == nullptr
        || (edge_count != 0u && edges == nullptr)
        || builder->edge_count > builder->edge_capacity
        || (builder->edge_capacity != 0u && builder->edge_storage == nullptr)) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    if (edge_count > builder->edge_capacity - builder->edge_count) {
        return failure(multilevel_status_code_v1::insufficient_capacity, edge_count);
    }
    std::uint64_t previous = builder->last_edge_identity;
    for (std::uint32_t index = 0u; index < edge_count; ++index) {
        const affinity_edge_v1 &edge = edges[index];
        if (edge.lhs >= builder->local_node_count || edge.rhs >= builder->local_node_count
            || edge.lhs == edge.rhs || edge.stable_identity == 0u
            || edge.stable_identity <= previous || !std::isfinite(edge.affinity)
            || edge.affinity <= 0.0) {
            return failure(multilevel_status_code_v1::invalid_edge, index);
        }
        previous = edge.stable_identity;
    }
    if (edge_count != 0u) {
        std::copy_n(edges, edge_count, builder->edge_storage + builder->edge_count);
    }
    builder->edge_count += edge_count;
    builder->last_edge_identity = previous;
    ++builder->appended_batch_count;
    return {};
}

multilevel_status_v1 finalize_affinity_stream_v1(
    const streaming_affinity_builder_v1 &builder,
    affinity_problem_v1 *out) noexcept {
    if (out == nullptr || builder.aggregate_node_count < builder.local_node_count
        || builder.aggregate_edge_count < builder.edge_count
        || (builder.edge_count != 0u && builder.edge_storage == nullptr)) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    affinity_problem_v1 problem{
        affinity_hierarchy_schema_v1,
        builder.structure_identity,
        builder.structure_epoch,
        builder.aggregate_node_count,
        builder.aggregate_edge_count,
        builder.node_global_identities,
        builder.local_node_count,
        builder.edge_storage,
        builder.edge_count,
    };
    const multilevel_status_v1 status = validate_affinity_problem_v1(problem);
    if (!status) return status;
    *out = problem;
    return {};
}

multilevel_status_v1 capture_multilevel_provenance_v1(
    const affinity_problem_v1 &problem,
    const affinity_hierarchy_v1 &hierarchy,
    const multilevel_grouping_solution_v1 &solution,
    multilevel_provenance_workspace_v1 workspace,
    multilevel_provenance_v1 *out) noexcept {
    if (out == nullptr || hierarchy.structure_identity != problem.structure_identity
        || hierarchy.structure_epoch != problem.structure_epoch
        || solution.structure_identity != problem.structure_identity
        || solution.structure_epoch != problem.structure_epoch
        || solution.fine_node_count != problem.local_node_count
        || solution.fine_node_to_group == nullptr
        || (solution.edge_cover_count != 0u && solution.edge_cover == nullptr)
        || workspace.link_capacity < hierarchy.fine_to_coarse_count
        || workspace.node_capacity < problem.local_node_count
        || (hierarchy.fine_to_coarse_count != 0u && workspace.links == nullptr)
        || workspace.nodes == nullptr) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }

    std::uint64_t write = 0u;
    for (std::uint32_t level = 0u; level < hierarchy.level_count; ++level) {
        const affinity_hierarchy_level_v1 &record = hierarchy.levels[level];
        for (std::uint32_t fine = 0u; fine < record.fine_node_count; ++fine) {
            workspace.links[write++] = {
                level,
                fine,
                hierarchy.fine_to_coarse[record.fine_to_coarse_offset + fine],
            };
        }
    }
    if (write != hierarchy.fine_to_coarse_count) {
        return failure(multilevel_status_code_v1::invalid_argument, write);
    }
    for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
        const std::uint32_t group = solution.fine_node_to_group[node];
        if (group >= solution.group_count) {
            return failure(multilevel_status_code_v1::invalid_argument, node);
        }
        workspace.nodes[node] = {problem.node_global_identities[node], group};
    }

    std::uint64_t required_bytes = 0u;
    if (!add_bytes(write, sizeof(multilevel_provenance_link_v1), &required_bytes)
        || !add_bytes(problem.local_node_count,
            sizeof(multilevel_node_provenance_v1), &required_bytes)) {
        return failure(multilevel_status_code_v1::arithmetic_overflow, 0u);
    }
    std::uint64_t hash = 1469598103934665603ull;
    hash_u64(&hash, problem.structure_identity);
    hash_u64(&hash, problem.structure_epoch);
    hash_u64(&hash, problem.aggregate_node_count);
    hash_u64(&hash, problem.aggregate_edge_count);
    for (std::uint64_t index = 0u; index < write; ++index) {
        hash_u64(&hash, workspace.links[index].level);
        hash_u64(&hash, workspace.links[index].fine_local);
        hash_u64(&hash, workspace.links[index].coarse_local);
    }
    for (std::uint32_t node = 0u; node < problem.local_node_count; ++node) {
        hash_u64(&hash, workspace.nodes[node].global_identity);
        hash_u64(&hash, workspace.nodes[node].final_group);
    }
    hash_u64(&hash, solution.edge_cover_count);
    for (std::uint32_t index = 0u; index < solution.edge_cover_count; ++index) {
        const logical_edge_cover_v1 &cover = solution.edge_cover[index];
        hash_u64(&hash, cover.logical_edge_identity);
        hash_u64(&hash, cover.lhs_group);
        hash_u64(&hash, cover.rhs_group);
        hash_u64(&hash, static_cast<std::uint64_t>(cover.cover_class));
    }
    if (hash == 0u) hash = 1u;

    const std::uint64_t hierarchy_operations = hierarchy.counters.nodes_visited
        + hierarchy.counters.edges_visited + hierarchy.counters.edges_sorted
        + hierarchy.counters.edges_contracted;
    const std::uint64_t refinement_operations = solution.counters.hierarchy_entries_visited
        + solution.counters.edges_visited + solution.counters.nodes_considered;
    *out = {
        streaming_provenance_schema_v1,
        problem.structure_identity,
        problem.structure_epoch,
        hash,
        workspace.links,
        write,
        workspace.nodes,
        problem.local_node_count,
        required_bytes,
        hierarchy_operations,
        refinement_operations,
    };
    return {};
}

}  // namespace cellerator::geometry::optimizer::multilevel
