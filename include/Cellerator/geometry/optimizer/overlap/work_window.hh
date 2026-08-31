#pragma once

#include <Cellerator/geometry/optimizer/overlap/bounded_overlap_solver.hh>

namespace cellerator::geometry::optimizer::overlap {

struct windowed_overlap_proposal {
    overlap_proposal proposal{};
    std::uint64_t first_window = 0;
    std::uint64_t past_last_window = 0;
};

enum class overlap_solution_kind : std::uint8_t {
    disjoint_baseline = 0,
    bounded_overlap = 1
};

struct work_window_workspace_view {
    overlap_proposal *filtered_proposals = nullptr;
    std::uint64_t filtered_proposal_capacity = 0;
    std::uint64_t *filtered_to_original = nullptr;
    std::uint64_t filtered_index_capacity = 0;
    std::uint64_t *filtered_selected_indices = nullptr;
    std::uint64_t filtered_selected_capacity = 0;
    bounded_overlap_workspace_view solver{};
};

struct work_window_result {
    overlap_solution_kind kind = overlap_solution_kind::disjoint_baseline;
    std::uint64_t window = 0;
    std::uint64_t eligible_proposal_count = 0;
    bounded_overlap_result overlap{};
};

contract_status solve_work_window_overlap(
    source_group_dictionary_view dataset_source_skeleton,
    const windowed_overlap_proposal *proposals,
    std::uint64_t proposal_count,
    std::uint64_t window,
    bounded_overlap_config config,
    work_window_workspace_view workspace,
    bounded_overlap_output_view output,
    work_window_result *result) noexcept;

}  // namespace cellerator::geometry::optimizer::overlap
