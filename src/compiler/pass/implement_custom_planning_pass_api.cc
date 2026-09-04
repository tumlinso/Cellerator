#include <Cellerator/compiler/pass/implement_custom_planning_pass_api_v1.hh>

#include <algorithm>
#include <set>

namespace cellerator::compiler::pass::v1 {

planning_pass_status_v1 run_custom_planning_pass_v1(
    planning_pass_context_v1& context, planning_pass_run_v1 pass) noexcept {
    if (context.atoms == nullptr || context.evidence == nullptr
        || context.decompositions == nullptr || context.candidates == nullptr
        || context.selected_candidate == nullptr || context.diagnostics == nullptr
        || pass == nullptr)
        return planning_pass_status_v1::invalid_context;
    const auto old_decompositions = *context.decompositions;
    const auto old_candidates = *context.candidates;
    const auto old_selection = *context.selected_candidate;
    if (context.mode == planning_pass_mode_v1::replace) {
        context.decompositions->clear();
        context.candidates->clear();
        *context.selected_candidate = 0;
    }
    if (!pass(context)) {
        *context.decompositions = old_decompositions;
        *context.candidates = old_candidates;
        *context.selected_candidate = old_selection;
        return planning_pass_status_v1::pass_failed;
    }
    std::set<std::uint64_t> atom_ids;
    for (const auto atom : *context.atoms)
        if (atom.id == 0 || !atom_ids.insert(atom.id).second)
            return planning_pass_status_v1::invalid_result;
    std::set<std::uint64_t> decomposition_ids;
    for (const auto& decomposition : *context.decompositions) {
        if (decomposition.id == 0
            || !decomposition_ids.insert(decomposition.id).second)
            return planning_pass_status_v1::invalid_result;
        for (auto atom : decomposition.covered_atoms)
            if (atom_ids.count(atom) == 0)
                return planning_pass_status_v1::invalid_result;
    }
    const auto selected = std::find_if(context.candidates->begin(),
        context.candidates->end(), [&](const auto& candidate) {
            return candidate.id == *context.selected_candidate
                && decomposition_ids.count(candidate.decomposition) != 0
                && candidate.total_cost >= 0 && !candidate.provider.empty();
        });
    if (selected == context.candidates->end()) {
        context.diagnostics->push_back("custom planning pass produced invalid selection");
        return planning_pass_status_v1::invalid_result;
    }
    return planning_pass_status_v1::success;
}

}  // namespace cellerator::compiler::pass::v1
