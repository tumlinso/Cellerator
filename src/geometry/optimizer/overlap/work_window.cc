#include <Cellerator/geometry/optimizer/overlap/work_window.hh>

namespace cellerator::geometry::optimizer::overlap {

contract_status solve_work_window_overlap(
    source_group_dictionary_view dataset_source_skeleton,
    const windowed_overlap_proposal *proposals,
    std::uint64_t proposal_count,
    std::uint64_t window,
    bounded_overlap_config config,
    work_window_workspace_view workspace,
    bounded_overlap_output_view output,
    work_window_result *result) noexcept {
    if (result == nullptr || (proposal_count != 0 && proposals == nullptr)) {
        return {contract_error::null_pointer, 0};
    }
    *result = {};
    result->window = window;
    if (workspace.filtered_proposal_capacity < proposal_count
        || workspace.filtered_index_capacity < proposal_count
        || workspace.filtered_selected_capacity < config.maximum_replicated_memberships
        || (proposal_count != 0
            && (workspace.filtered_proposals == nullptr
                || workspace.filtered_to_original == nullptr))
        || (config.maximum_replicated_memberships != 0
            && workspace.filtered_selected_indices == nullptr)) {
        return {contract_error::insufficient_workspace, proposal_count};
    }

    for (std::uint64_t index = 0; index < proposal_count; ++index) {
        const windowed_overlap_proposal candidate = proposals[index];
        if (candidate.first_window >= candidate.past_last_window) {
            return {contract_error::invalid_offset, index};
        }
        if (window >= candidate.first_window && window < candidate.past_last_window) {
            const std::uint64_t filtered = result->eligible_proposal_count++;
            workspace.filtered_proposals[filtered] = candidate.proposal;
            workspace.filtered_to_original[filtered] = index;
        }
    }

    const contract_status status = solve_bounded_overlap(
        dataset_source_skeleton,
        workspace.filtered_proposals,
        result->eligible_proposal_count,
        config,
        workspace.solver,
        {workspace.filtered_selected_indices, workspace.filtered_selected_capacity},
        &result->overlap);
    if (!status) {
        return status;
    }
    if (output.selected_capacity < result->overlap.selected_count
        || (result->overlap.selected_count != 0
            && output.selected_proposal_indices == nullptr)) {
        return {contract_error::insufficient_workspace, result->overlap.selected_count};
    }
    for (std::uint64_t selected = 0; selected < result->overlap.selected_count; ++selected) {
        output.selected_proposal_indices[selected]
            = workspace.filtered_to_original[workspace.filtered_selected_indices[selected]];
    }
    result->kind = result->overlap.selected_count == 0
        ? overlap_solution_kind::disjoint_baseline
        : overlap_solution_kind::bounded_overlap;
    return {};
}

}  // namespace cellerator::geometry::optimizer::overlap
