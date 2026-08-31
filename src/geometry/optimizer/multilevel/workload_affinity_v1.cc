#include "Cellerator/geometry/optimizer/multilevel/workload_affinity_v1.hh"

#include <cmath>
#include <cstring>

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

void hash_double(std::uint64_t *hash, double value) noexcept {
    std::uint64_t bits = 0u;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    hash_u64(hash, bits);
}

bool lookup_weight(
    const workload_weight_v1 *weights,
    std::uint32_t count,
    std::uint64_t identity,
    double *weight,
    std::uint64_t *steps) noexcept {
    std::uint32_t begin = 0u;
    std::uint32_t end = count;
    while (begin < end) {
        ++*steps;
        const std::uint32_t middle = begin + (end - begin) / 2u;
        if (weights[middle].identity < identity) begin = middle + 1u;
        else end = middle;
    }
    if (begin == count || weights[begin].identity != identity) return false;
    *weight = weights[begin].weight;
    return true;
}

multilevel_status_v1 validate_profile(
    const workload_affinity_profile_v1 &profile) noexcept {
    if (profile.schema_version != workload_affinity_schema_v1
        || profile.structure_identity == 0u || profile.structure_epoch == 0u
        || profile.source_skeleton_identity == 0u || profile.local_node_count == 0u
        || profile.aggregate_node_count < profile.local_node_count
        || profile.node_global_identities == nullptr
        || profile.contribution_count == 0u || profile.contributions == nullptr
        || profile.operation_weight_count == 0u || profile.operation_weights == nullptr
        || profile.work_window_weight_count == 0u || profile.work_window_weights == nullptr) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    for (std::uint32_t index = 0u; index < profile.local_node_count; ++index) {
        if (profile.node_global_identities[index] == 0u
            || (index != 0u && profile.node_global_identities[index - 1u]
                >= profile.node_global_identities[index])) {
            return failure(multilevel_status_code_v1::invalid_order, index);
        }
    }
    const workload_weight_v1 *tables[2] = {
        profile.operation_weights, profile.work_window_weights};
    const std::uint32_t counts[2] = {
        profile.operation_weight_count, profile.work_window_weight_count};
    for (std::uint32_t table = 0u; table < 2u; ++table) {
        for (std::uint32_t index = 0u; index < counts[table]; ++index) {
            if (tables[table][index].identity == 0u
                || !std::isfinite(tables[table][index].weight)
                || tables[table][index].weight <= 0.0
                || (index != 0u && tables[table][index - 1u].identity
                    >= tables[table][index].identity)) {
                return failure(multilevel_status_code_v1::invalid_order, index);
            }
        }
    }
    for (std::uint32_t index = 0u; index < profile.contribution_count; ++index) {
        const workload_affinity_contribution_v1 &item = profile.contributions[index];
        if (item.logical_edge_identity == 0u || item.operation_identity == 0u
            || item.work_window_identity == 0u || item.lhs >= profile.local_node_count
            || item.rhs >= profile.local_node_count || item.lhs == item.rhs
            || !std::isfinite(item.frequency) || item.frequency <= 0.0
            || !std::isfinite(item.affinity) || item.affinity <= 0.0) {
            return failure(multilevel_status_code_v1::invalid_argument, index);
        }
        if (index != 0u) {
            const workload_affinity_contribution_v1 &previous = profile.contributions[index - 1u];
            const bool ordered = previous.logical_edge_identity < item.logical_edge_identity
                || (previous.logical_edge_identity == item.logical_edge_identity
                    && (previous.operation_identity < item.operation_identity
                        || (previous.operation_identity == item.operation_identity
                            && previous.work_window_identity < item.work_window_identity)));
            if (!ordered) return failure(multilevel_status_code_v1::invalid_order, index);
            if (previous.logical_edge_identity == item.logical_edge_identity
                && (previous.lhs != item.lhs || previous.rhs != item.rhs)) {
                return failure(multilevel_status_code_v1::invalid_edge, index);
            }
        }
    }
    return {};
}

multilevel_status_v1 weighted_contribution(
    const workload_affinity_profile_v1 &profile,
    const workload_affinity_contribution_v1 &item,
    double *weighted,
    std::uint64_t *steps) noexcept {
    double operation_weight = 0.0;
    double window_weight = 0.0;
    if (!lookup_weight(profile.operation_weights, profile.operation_weight_count,
            item.operation_identity, &operation_weight, steps)
        || !lookup_weight(profile.work_window_weights, profile.work_window_weight_count,
            item.work_window_identity, &window_weight, steps)) {
        return failure(multilevel_status_code_v1::invalid_identity,
            item.logical_edge_identity);
    }
    const double value = item.affinity * item.frequency
        * operation_weight * window_weight;
    if (!std::isfinite(value) || value <= 0.0) {
        return failure(multilevel_status_code_v1::arithmetic_overflow,
            item.logical_edge_identity);
    }
    *weighted = value;
    return {};
}

}  // namespace

multilevel_status_v1 build_workload_affinity_v1(
    const workload_affinity_profile_v1 &profile,
    affinity_edge_v1 *edge_storage,
    std::uint32_t edge_capacity,
    workload_affinity_result_v1 *out) noexcept {
    if (out == nullptr || edge_storage == nullptr) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    multilevel_status_v1 status = validate_profile(profile);
    if (!status) return status;
    workload_affinity_counters_v1 counters{};
    std::uint32_t write = 0u;
    std::uint64_t hash = 1469598103934665603ull;
    hash_u64(&hash, profile.source_skeleton_identity);
    for (std::uint32_t index = 0u; index < profile.contribution_count; ++index) {
        const workload_affinity_contribution_v1 &item = profile.contributions[index];
        double weighted = 0.0;
        status = weighted_contribution(profile, item, &weighted,
            &counters.weight_search_steps);
        if (!status) return status;
        if (index == 0u || profile.contributions[index - 1u].logical_edge_identity
            != item.logical_edge_identity) {
            if (write == edge_capacity) {
                return failure(multilevel_status_code_v1::insufficient_capacity, write);
            }
            edge_storage[write++] = {
                item.lhs, item.rhs, weighted, item.logical_edge_identity};
        } else {
            edge_storage[write - 1u].affinity += weighted;
            if (!std::isfinite(edge_storage[write - 1u].affinity)) {
                return failure(multilevel_status_code_v1::arithmetic_overflow,
                    item.logical_edge_identity);
            }
        }
        hash_u64(&hash, item.logical_edge_identity);
        hash_u64(&hash, item.operation_identity);
        hash_u64(&hash, item.work_window_identity);
        hash_double(&hash, weighted);
        ++counters.contributions_visited;
    }
    counters.logical_edges_emitted = write;
    if (profile.aggregate_edge_count < write) {
        return failure(multilevel_status_code_v1::invalid_argument, write);
    }
    if (hash == 0u) hash = 1u;
    affinity_problem_v1 problem{
        affinity_hierarchy_schema_v1,
        profile.structure_identity,
        profile.structure_epoch,
        profile.aggregate_node_count,
        profile.aggregate_edge_count,
        profile.node_global_identities,
        profile.local_node_count,
        edge_storage,
        write,
    };
    status = validate_affinity_problem_v1(problem);
    if (!status) return status;
    *out = {workload_affinity_schema_v1, profile.source_skeleton_identity,
        hash, problem, counters};
    return {};
}

multilevel_status_v1 evaluate_workload_solution_v1(
    const workload_affinity_profile_v1 &profile,
    const multilevel_grouping_solution_v1 &solution,
    double grouped_unit_cost,
    double residual_unit_cost,
    workload_solution_objective_v1 *out) noexcept {
    if (out == nullptr || solution.fine_node_to_group == nullptr
        || solution.structure_identity != profile.structure_identity
        || solution.structure_epoch != profile.structure_epoch
        || solution.fine_node_count != profile.local_node_count
        || !std::isfinite(grouped_unit_cost) || grouped_unit_cost < 0.0
        || !std::isfinite(residual_unit_cost) || residual_unit_cost < 0.0) {
        return failure(multilevel_status_code_v1::invalid_argument, 0u);
    }
    multilevel_status_v1 status = validate_profile(profile);
    if (!status) return status;
    double grouped = 0.0;
    double residual = 0.0;
    std::uint64_t steps = 0u;
    std::uint64_t hash = 1469598103934665603ull;
    hash_u64(&hash, profile.source_skeleton_identity);
    for (std::uint32_t index = 0u; index < profile.contribution_count; ++index) {
        const workload_affinity_contribution_v1 &item = profile.contributions[index];
        double weighted = 0.0;
        status = weighted_contribution(profile, item, &weighted, &steps);
        if (!status) return status;
        const bool same_group = solution.fine_node_to_group[item.lhs]
            == solution.fine_node_to_group[item.rhs];
        grouped += same_group ? weighted * grouped_unit_cost : 0.0;
        residual += same_group ? 0.0 : weighted * residual_unit_cost;
        hash_u64(&hash, item.logical_edge_identity);
        hash_u64(&hash, item.operation_identity);
        hash_u64(&hash, item.work_window_identity);
        hash_double(&hash, weighted);
    }
    if (!std::isfinite(grouped) || !std::isfinite(residual)) {
        return failure(multilevel_status_code_v1::arithmetic_overflow, 0u);
    }
    if (hash == 0u) hash = 1u;
    const double total = grouped + residual;
    if (!std::isfinite(total)) {
        return failure(multilevel_status_code_v1::arithmetic_overflow, 1u);
    }
    *out = {workload_affinity_schema_v1, hash, grouped, residual,
        total, profile.contribution_count, steps};
    return {};
}

}  // namespace cellerator::geometry::optimizer::multilevel
