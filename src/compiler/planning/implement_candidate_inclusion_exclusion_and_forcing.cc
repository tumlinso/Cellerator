#include <Cellerator/compiler/planning/implement_candidate_inclusion_exclusion_and_forcing_v1.hh>

#include <algorithm>
#include <limits>
#include <unordered_map>

namespace Cellerator::compiler::planning {

candidate_selection_v1 apply_candidate_edits_v1(
    const std::vector<candidate_choice_v1>& candidates,
    const std::vector<candidate_edit_v1>& edits) {
    candidate_selection_v1 result{};
    std::unordered_map<std::uint64_t, std::size_t> index;
    std::vector<bool> offered(candidates.size(), true);
    std::vector<candidate_edit_authority_v1> authorities(
        candidates.size(), candidate_edit_authority_v1::automatic);
    for (std::size_t i = 0u; i < candidates.size(); ++i) index[candidates[i].candidate_identity] = i;

    const candidate_edit_v1* forced = nullptr;
    for (const auto& edit : edits) {
        candidate_edit_receipt_v1 receipt{edit.candidate_identity, edit.authority, edit.mode};
        const auto found = index.find(edit.candidate_identity);
        if (found == index.end()) {
            receipt.diagnostic = candidate_edit_diagnostic_v1::unknown_candidate;
            result.receipts.push_back(receipt);
            continue;
        }
        const auto candidate_index = found->second;
        if (edit.authority < authorities[candidate_index]) {
            receipt.diagnostic = candidate_edit_diagnostic_v1::lower_authority_ignored;
            result.receipts.push_back(receipt);
            continue;
        }
        authorities[candidate_index] = edit.authority;
        receipt.applied = true;
        if (edit.mode == candidate_edit_mode_v1::exclude) offered[candidate_index] = false;
        else offered[candidate_index] = true;
        if (edit.mode == candidate_edit_mode_v1::force ||
            edit.mode == candidate_edit_mode_v1::unsafe_force) {
            if (forced == nullptr || edit.authority >= forced->authority) forced = &edit;
            if (!candidates[candidate_index].admissible)
                receipt.diagnostic = candidate_edit_diagnostic_v1::impossible_choice;
            else if (candidates[candidate_index].dominated)
                receipt.diagnostic = candidate_edit_diagnostic_v1::dominated_choice;
        }
        result.receipts.push_back(receipt);
    }

    for (std::size_t i = 0u; i < candidates.size(); ++i)
        if (offered[i]) result.offered_candidates.push_back(candidates[i].candidate_identity);

    if (forced != nullptr) {
        const auto& choice = candidates[index[forced->candidate_identity]];
        if (choice.admissible || forced->mode == candidate_edit_mode_v1::unsafe_force) {
            result.selected_candidate_identity = choice.candidate_identity;
            result.selection_mode = forced->mode;
            result.unsafe = !choice.admissible;
            return result;
        }
    }

    std::uint64_t best_cost = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t i = 0u; i < candidates.size(); ++i) {
        const auto& candidate = candidates[i];
        if (offered[i] && candidate.admissible && !candidate.dominated &&
            candidate.predicted_nanoseconds < best_cost) {
            best_cost = candidate.predicted_nanoseconds;
            result.selected_candidate_identity = candidate.candidate_identity;
        }
    }
    if (result.selected_candidate_identity == 0u)
        result.receipts.push_back({0u, candidate_edit_authority_v1::automatic,
            candidate_edit_mode_v1::offer,
            candidate_edit_diagnostic_v1::no_candidate_available, false});
    return result;
}

}  // namespace Cellerator::compiler::planning
