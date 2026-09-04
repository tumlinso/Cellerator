#include <Cellerator/compiler/sema/field/implement_custom_candidate_and_forced_realization_contro_v1.hh>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Cellerator::compiler::sema::field {

realization_control_status_v1 implement_custom_candidate_and_forced_realization_contro_v1(
    const std::vector<controllable_realization_candidate_v1>& candidates,
    const custom_candidate_or_forced_realization_v1& control,
    resolved_realization_control_v1* resolved) noexcept {
    if (resolved == nullptr) return realization_control_status_v1::invalid_output;
    for (const auto& candidate : candidates) {
        if (candidate.candidate_identity == 0 || candidate.decomposition_identity == 0 ||
            candidate.realization_identity == 0 ||
            !std::isfinite(candidate.estimated_total_cost) || candidate.estimated_total_cost < 0.0) {
            return realization_control_status_v1::invalid_candidate;
        }
    }
    if (control.selected_identity == 0) {
        return realization_control_status_v1::selected_object_unavailable;
    }

    resolved_realization_control_v1 result;
    if (control.kind == realization_control_kind_v1::offer_candidate) {
        const auto offered = std::find_if(candidates.begin(), candidates.end(), [&control](const auto& candidate) {
            return candidate.candidate_identity == control.selected_identity;
        });
        if (offered == candidates.end()) {
            return realization_control_status_v1::selected_object_unavailable;
        }
        double best_cost = std::numeric_limits<double>::infinity();
        for (const auto& candidate : candidates) {
            if (!candidate.legal) continue;
            result.considered_candidates.push_back(candidate.candidate_identity);
            if (candidate.estimated_total_cost < best_cost) {
                best_cost = candidate.estimated_total_cost;
                result.selected_candidate_identity = candidate.candidate_identity;
            }
        }
        if (result.selected_candidate_identity == 0) {
            return realization_control_status_v1::no_legal_candidate;
        }
        result.custom_candidate_won =
            result.selected_candidate_identity == control.selected_identity;
    } else {
        const auto selected = std::find_if(candidates.begin(), candidates.end(), [&control](const auto& candidate) {
            switch (control.kind) {
                case realization_control_kind_v1::force_candidate:
                    return candidate.candidate_identity == control.selected_identity;
                case realization_control_kind_v1::force_decomposition:
                    return candidate.decomposition_identity == control.selected_identity;
                case realization_control_kind_v1::force_realization:
                    return candidate.realization_identity == control.selected_identity;
                case realization_control_kind_v1::offer_candidate:
                    return false;
            }
            return false;
        });
        if (selected == candidates.end()) {
            return realization_control_status_v1::selected_object_unavailable;
        }
        if (!selected->legal && !control.explicitly_unsafe) {
            return realization_control_status_v1::selected_object_illegal;
        }
        result.considered_candidates.push_back(selected->candidate_identity);
        result.selected_candidate_identity = selected->candidate_identity;
        result.forced = true;
        result.unsafe = !selected->legal;
    }

    *resolved = std::move(result);
    return realization_control_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
