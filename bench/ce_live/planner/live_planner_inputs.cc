#include <bench/ce_live/planner/live_planner_inputs.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::ce_live::planner_inputs {
namespace {

bool valid_identities(const relation_identity_input &identities) noexcept {
    return execution::valid_identity(identities.source_domain)
        && execution::valid_identity(identities.destination_domain)
        && execution::valid_identity(identities.source_order)
        && execution::valid_identity(identities.destination_order)
        && execution::valid_identity(identities.geometry)
        && execution::valid_identity(identities.partition)
        && execution::valid_identity(identities.structure)
        && identities.structure_epoch.value != 0u;
}

live_input_status derive_structure(
    const quantitative_relation_input &relation,
    structural_statistics *statistics) noexcept {
    if (relation.destination_offsets == nullptr
        || relation.source_count == 0u || relation.destination_count == 0u
        || (relation.logical_edge_count != 0u
            && relation.source_indices == nullptr)
        || relation.destination_offsets[0] != 0u
        || relation.destination_offsets[relation.destination_count]
            != relation.logical_edge_count)
        return live_input_status::invalid_support;
    std::uint64_t minimum_degree = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t maximum_degree = 0u;
    for (std::uint32_t row = 0u; row < relation.destination_count; ++row) {
        const std::uint64_t begin = relation.destination_offsets[row];
        const std::uint64_t end = relation.destination_offsets[row + 1u];
        if (begin > end || end > relation.logical_edge_count)
            return live_input_status::invalid_support;
        const std::uint64_t degree = end - begin;
        minimum_degree = std::min(minimum_degree, degree);
        maximum_degree = std::max(maximum_degree, degree);
        for (std::uint64_t edge = begin; edge < end; ++edge)
            if (relation.source_indices[edge] >= relation.source_count)
                return live_input_status::invalid_support;
    }
    statistics->source_count = relation.source_count;
    statistics->destination_count = relation.destination_count;
    statistics->logical_edge_count = relation.logical_edge_count;
    statistics->minimum_destination_degree = minimum_degree;
    statistics->maximum_destination_degree = maximum_degree;
    statistics->mean_destination_degree =
        static_cast<double>(relation.logical_edge_count)
        / static_cast<double>(relation.destination_count);
    statistics->density = static_cast<double>(relation.logical_edge_count)
        / (static_cast<double>(relation.source_count)
            * static_cast<double>(relation.destination_count));
    return live_input_status::ok;
}

live_input_status derive_values(const quantitative_relation_input &relation,
    quantitative_statistics *statistics) noexcept {
    if (relation.observed_generation.value == 0u
        || (relation.logical_edge_count != 0u && relation.values == nullptr))
        return live_input_status::invalid_values;
    statistics->observed_generation = relation.observed_generation;
    if (relation.logical_edge_count == 0u) return live_input_status::ok;
    statistics->minimum = relation.values[0];
    statistics->maximum = relation.values[0];
    for (std::uint64_t edge = 0u; edge < relation.logical_edge_count; ++edge) {
        const double value = relation.values[edge];
        if (!std::isfinite(value)) return live_input_status::invalid_values;
        if (value != 0.0) ++statistics->nonzero_count;
        statistics->minimum = std::min(statistics->minimum, value);
        statistics->maximum = std::max(statistics->maximum, value);
        statistics->l1_norm += std::fabs(value);
    }
    return live_input_status::ok;
}

} // namespace

live_input_status derive_live_planner_input(
    const quantitative_relation_input &relation,
    compute::math::core::stable_id problem,
    const cellerator::planner::device_performance_key &device,
    const cellerator::planner::runtime_build_key &build,
    reuse_horizons reuse,
    std::uint32_t numeric_policy,
    std::uint32_t determinism_policy,
    std::uint32_t output_order_policy,
    std::uint32_t graph_policy,
    live_planner_input *output) noexcept {
    if (output == nullptr || (problem.low == 0u && problem.high == 0u))
        return live_input_status::invalid_argument;
    *output = {};
    if (!valid_identities(relation.identities))
        return live_input_status::invalid_identity;
    if (reuse.structure == 0u || reuse.projection == 0u || reuse.value == 0u)
        return live_input_status::invalid_reuse;
    live_input_status status = derive_structure(relation, &output->structure);
    if (status != live_input_status::ok) return status;
    status = derive_values(relation, &output->values);
    if (status != live_input_status::ok) return status;

    output->keys.problem.identity = problem;
    output->keys.structures.count = 1u;
    output->keys.structures.structures[0] = {
        relation.identities.structure, relation.identities.structure_epoch};
    output->keys.geometry = {relation.identities.source_domain,
        relation.identities.destination_domain, relation.identities.geometry,
        relation.identities.source_order, relation.identities.destination_order,
        relation.identities.partition};
    output->keys.device = device;
    output->keys.build = build;
    output->keys.policy = {reuse.structure, reuse.projection, reuse.value,
        numeric_policy, determinism_policy, output_order_policy, graph_policy};
    return live_input_status::ok;
}

live_input_status account_candidate_phases(
    const candidate_phase_input &input,
    cellerator::planner::total_cost *output) noexcept {
    const cellerator::planner::planner_status status =
        cellerator::planner::compute_total_cost(input.phases,
            input.reuse.structure, input.reuse.projection,
            input.reuse.value, output);
    if (status) return live_input_status::ok;
    return status.code == cellerator::planner::planner_status_code::invalid_cost
        ? live_input_status::invalid_cost
        : live_input_status::invalid_reuse;
}

} // namespace cellerator::ce_live::planner_inputs
