#pragma once

#include <Cellerator/geometry/optimizer/overlap/overlap_contract.hh>

namespace cellerator::geometry::optimizer::overlap {

struct overlap_proposal {
    source_id source = 0;
    source_group_id destination_group = 0;
    std::uint64_t predicted_benefit = 0;
    replication_unit_cost duplication_cost{};
};

struct bounded_overlap_config {
    std::uint64_t maximum_replicated_memberships = 0;
    std::uint64_t maximum_memberships_per_source = 1;
    std::uint64_t maximum_sources_per_group = 0;
};

struct bounded_overlap_workspace_view {
    std::uint64_t *source_use_counts = nullptr;
    std::uint64_t source_capacity = 0;
    std::uint64_t *group_sizes = nullptr;
    std::uint64_t group_capacity = 0;
    std::uint8_t *proposal_state = nullptr;
    std::uint64_t proposal_capacity = 0;
};

struct bounded_overlap_output_view {
    std::uint64_t *selected_proposal_indices = nullptr;
    std::uint64_t selected_capacity = 0;
};

struct bounded_overlap_result {
    std::uint64_t selected_count = 0;
    std::uint64_t rejected_duplicate_count = 0;
    std::uint64_t rejected_bound_count = 0;
    std::uint64_t total_predicted_benefit = 0;
    replication_unit_cost charged_duplication{};
    std::uint64_t total_duplication_cost = 0;
    std::uint64_t net_predicted_benefit = 0;
};

contract_status solve_bounded_overlap(
    source_group_dictionary_view baseline,
    const overlap_proposal *proposals,
    std::uint64_t proposal_count,
    bounded_overlap_config config,
    bounded_overlap_workspace_view workspace,
    bounded_overlap_output_view output,
    bounded_overlap_result *result) noexcept;

}  // namespace cellerator::geometry::optimizer::overlap
