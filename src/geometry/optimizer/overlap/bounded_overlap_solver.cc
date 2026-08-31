#include <Cellerator/geometry/optimizer/overlap/bounded_overlap_solver.hh>

#include <limits>

namespace cellerator::geometry::optimizer::overlap {
namespace {

bool add(std::uint64_t value, std::uint64_t *total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
        return false;
    }
    *total += value;
    return true;
}

bool add_cost(replication_unit_cost value, replication_unit_cost *total) noexcept {
    return add(value.source_state, &total->source_state)
        && add(value.dense_input_movement, &total->dense_input_movement)
        && add(value.value_maps, &total->value_maps)
        && add(value.gradient_reconciliation, &total->gradient_reconciliation)
        && add(value.persistent_bytes, &total->persistent_bytes)
        && add(value.construction, &total->construction)
        && add(value.canonical_recovery, &total->canonical_recovery);
}

bool cost_total(replication_unit_cost value, std::uint64_t *total) noexcept {
    *total = 0;
    return add(value.source_state, total)
        && add(value.dense_input_movement, total)
        && add(value.value_maps, total)
        && add(value.gradient_reconciliation, total)
        && add(value.persistent_bytes, total)
        && add(value.construction, total)
        && add(value.canonical_recovery, total);
}

bool baseline_contains(
    source_group_dictionary_view baseline,
    source_group_id group,
    source_id source) noexcept {
    const std::uint64_t begin = baseline.group_offsets[group];
    const std::uint64_t end = baseline.group_offsets[group + 1];
    for (std::uint64_t index = begin; index < end; ++index) {
        if (baseline.source_ids[index] == source) {
            return true;
        }
    }
    return false;
}

bool selected_duplicate(
    const overlap_proposal *proposals,
    const std::uint8_t *state,
    std::uint64_t proposal_count,
    std::uint64_t index) noexcept {
    for (std::uint64_t candidate = 0; candidate < proposal_count; ++candidate) {
        if (state[candidate] == 1
            && proposals[candidate].source == proposals[index].source
            && proposals[candidate].destination_group == proposals[index].destination_group) {
            return true;
        }
    }
    return false;
}

}  // namespace

contract_status solve_bounded_overlap(
    source_group_dictionary_view baseline,
    const overlap_proposal *proposals,
    std::uint64_t proposal_count,
    bounded_overlap_config config,
    bounded_overlap_workspace_view workspace,
    bounded_overlap_output_view output,
    bounded_overlap_result *result) noexcept {
    if (result == nullptr || (proposal_count != 0 && proposals == nullptr)) {
        return {contract_error::null_pointer, 0};
    }
    *result = {};
    const contract_status baseline_status = validate_source_group_dictionary(baseline);
    if (!baseline_status) {
        return baseline_status;
    }
    if (workspace.source_capacity < baseline.source_count
        || workspace.group_capacity < baseline.group_count
        || workspace.proposal_capacity < proposal_count
        || (baseline.source_count != 0 && workspace.source_use_counts == nullptr)
        || (baseline.group_count != 0 && workspace.group_sizes == nullptr)
        || (proposal_count != 0 && workspace.proposal_state == nullptr)
        || output.selected_capacity < config.maximum_replicated_memberships
        || (config.maximum_replicated_memberships != 0
            && output.selected_proposal_indices == nullptr)) {
        return {contract_error::insufficient_workspace, proposal_count};
    }

    for (std::uint64_t source = 0; source < baseline.source_count; ++source) {
        workspace.source_use_counts[source] = 0;
    }
    for (std::uint64_t group = 0; group < baseline.group_count; ++group) {
        workspace.group_sizes[group] = baseline.group_offsets[group + 1] - baseline.group_offsets[group];
    }
    for (std::uint64_t membership = 0; membership < baseline.membership_count; ++membership) {
        ++workspace.source_use_counts[baseline.source_ids[membership]];
    }
    for (std::uint64_t index = 0; index < proposal_count; ++index) {
        workspace.proposal_state[index] = 0;
        if (proposals[index].source >= baseline.source_count) {
            return {contract_error::source_out_of_range, index};
        }
        if (proposals[index].destination_group >= baseline.group_count) {
            return {contract_error::owner_group_out_of_range, index};
        }
    }

    while (result->selected_count < config.maximum_replicated_memberships) {
        std::uint64_t best = proposal_count;
        std::uint64_t best_net = 0;
        for (std::uint64_t index = 0; index < proposal_count; ++index) {
            if (workspace.proposal_state[index] != 0) {
                continue;
            }
            const overlap_proposal proposal = proposals[index];
            if (baseline_contains(baseline, proposal.destination_group, proposal.source)
                || selected_duplicate(
                    proposals, workspace.proposal_state, proposal_count, index)) {
                workspace.proposal_state[index] = 2;
                ++result->rejected_duplicate_count;
                continue;
            }
            if (workspace.source_use_counts[proposal.source]
                    >= config.maximum_memberships_per_source
                || workspace.group_sizes[proposal.destination_group]
                    >= config.maximum_sources_per_group) {
                workspace.proposal_state[index] = 3;
                ++result->rejected_bound_count;
                continue;
            }
            std::uint64_t duplication = 0;
            if (!cost_total(proposal.duplication_cost, &duplication)) {
                return {contract_error::integer_overflow, index};
            }
            const std::uint64_t net = proposal.predicted_benefit > duplication
                ? proposal.predicted_benefit - duplication
                : 0;
            if (net > best_net || (net == best_net && net != 0 && index < best)) {
                best = index;
                best_net = net;
            }
        }
        if (best == proposal_count) {
            break;
        }

        const overlap_proposal selected = proposals[best];
        std::uint64_t selected_cost = 0;
        if (!cost_total(selected.duplication_cost, &selected_cost)
            || !add(selected.predicted_benefit, &result->total_predicted_benefit)
            || !add(selected_cost, &result->total_duplication_cost)
            || !add_cost(selected.duplication_cost, &result->charged_duplication)) {
            return {contract_error::integer_overflow, best};
        }
        workspace.proposal_state[best] = 1;
        ++workspace.source_use_counts[selected.source];
        ++workspace.group_sizes[selected.destination_group];
        output.selected_proposal_indices[result->selected_count++] = best;
    }
    result->net_predicted_benefit =
        result->total_predicted_benefit - result->total_duplication_cost;
    return {};
}

}  // namespace cellerator::geometry::optimizer::overlap
