#include <Cellerator/geometry/optimizer/overlap/overlap_contract.hh>

#include <limits>

namespace cellerator::geometry::optimizer::overlap {
namespace {

bool add_checked(std::uint64_t value, std::uint64_t *total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
        return false;
    }
    *total += value;
    return true;
}

bool multiply_checked(std::uint64_t a, std::uint64_t b, std::uint64_t *result) noexcept {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

}  // namespace

contract_status validate_source_group_dictionary(source_group_dictionary_view dictionary) noexcept {
    if (dictionary.group_count != 0 && dictionary.group_offsets == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    if (dictionary.membership_count != 0 && dictionary.source_ids == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    if (dictionary.group_count == 0) {
        return dictionary.membership_count == 0
            ? contract_status{}
            : contract_status{contract_error::invalid_offset, 0};
    }
    if (dictionary.group_offsets[0] != 0
        || dictionary.group_offsets[dictionary.group_count] != dictionary.membership_count) {
        return {contract_error::invalid_offset, dictionary.group_count};
    }
    for (std::uint64_t group = 0; group < dictionary.group_count; ++group) {
        const std::uint64_t begin = dictionary.group_offsets[group];
        const std::uint64_t end = dictionary.group_offsets[group + 1];
        if (begin >= end || end > dictionary.membership_count) {
            return {begin == end ? contract_error::empty_group : contract_error::invalid_offset, group};
        }
        source_id previous = 0;
        for (std::uint64_t index = begin; index < end; ++index) {
            const source_id source = dictionary.source_ids[index];
            if (source >= dictionary.source_count) {
                return {contract_error::source_out_of_range, index};
            }
            if (index != begin && source <= previous) {
                return {source == previous ? contract_error::duplicate_source_in_group
                                           : contract_error::invalid_offset,
                        index};
            }
            previous = source;
        }
    }
    return {};
}

contract_status validate_logical_contribution_ownership(
    logical_contribution_ownership_view ownership,
    std::uint64_t group_count) noexcept {
    if (ownership.owner_count != 0 && ownership.owners == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    if (ownership.owner_count != ownership.contribution_count) {
        return {contract_error::missing_contribution_owner, ownership.owner_count};
    }
    logical_contribution_id previous = 0;
    for (std::uint64_t index = 0; index < ownership.owner_count; ++index) {
        const logical_contribution_owner owner = ownership.owners[index];
        if (owner.contribution >= ownership.contribution_count) {
            return {contract_error::contribution_out_of_range, index};
        }
        if (owner.owner_group >= group_count) {
            return {contract_error::owner_group_out_of_range, index};
        }
        if (index != 0 && owner.contribution <= previous) {
            return {owner.contribution == previous ? contract_error::duplicate_contribution_owner
                                                   : contract_error::missing_contribution_owner,
                    index};
        }
        if (owner.contribution != index) {
            return {contract_error::missing_contribution_owner, index};
        }
        previous = owner.contribution;
    }
    return {};
}

contract_status evaluate_replication_cost(
    source_group_dictionary_view dictionary,
    replication_unit_cost unit_cost,
    replication_workspace_view workspace,
    replication_cost *result) noexcept {
    if (result == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    *result = {};
    const contract_status status = validate_source_group_dictionary(dictionary);
    if (!status) {
        return status;
    }
    if (workspace.source_capacity < dictionary.source_count
        || (dictionary.source_count != 0 && workspace.source_use_counts == nullptr)) {
        return {contract_error::insufficient_workspace, dictionary.source_count};
    }
    for (std::uint64_t source = 0; source < dictionary.source_count; ++source) {
        workspace.source_use_counts[source] = 0;
    }
    for (std::uint64_t index = 0; index < dictionary.membership_count; ++index) {
        ++workspace.source_use_counts[dictionary.source_ids[index]];
    }
    for (std::uint64_t source = 0; source < dictionary.source_count; ++source) {
        const std::uint64_t count = workspace.source_use_counts[source];
        if (count > 1) {
            result->replicated_memberships += count - 1;
            ++result->replicated_sources;
        }
    }

    std::uint64_t unit_total = 0;
    if (!add_checked(unit_cost.source_state, &unit_total)
        || !add_checked(unit_cost.dense_input_movement, &unit_total)
        || !add_checked(unit_cost.value_maps, &unit_total)
        || !add_checked(unit_cost.gradient_reconciliation, &unit_total)
        || !add_checked(unit_cost.persistent_bytes, &unit_total)
        || !add_checked(unit_cost.construction, &unit_total)
        || !add_checked(unit_cost.canonical_recovery, &unit_total)
        || !multiply_checked(unit_total, result->replicated_memberships, &result->total)) {
        *result = {};
        return {contract_error::integer_overflow, 0};
    }
    result->repeated = unit_cost;
    return {};
}

contract_status query_is_disjoint(
    source_group_dictionary_view dictionary,
    replication_workspace_view workspace,
    bool *result) noexcept {
    if (result == nullptr) {
        return {contract_error::null_pointer, 0};
    }
    replication_cost cost;
    const contract_status status = evaluate_replication_cost(dictionary, {}, workspace, &cost);
    if (!status) {
        *result = false;
        return status;
    }
    *result = cost.replicated_memberships == 0;
    return {};
}

}  // namespace cellerator::geometry::optimizer::overlap
