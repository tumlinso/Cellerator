#include <Cellerator/compiler/sema/field/implement_hard_semantic_and_execution_constraints_v1.hh>

#include <cmath>
#include <utility>

namespace Cellerator::compiler::sema::field {

hard_constraint_status_v1 implement_hard_semantic_and_execution_constraints_v1(
    const hard_execution_constraints_v1& constraints,
    const std::vector<constrained_plan_candidate_v1>& candidates,
    hard_constraint_result_v1* result) noexcept {
    if (result == nullptr) return hard_constraint_status_v1::invalid_output;
    if (!std::isfinite(constraints.maximum_numerical_error) ||
        constraints.maximum_numerical_error < 0.0) {
        return hard_constraint_status_v1::invalid_constraint;
    }

    hard_constraint_result_v1 filtered;
    for (const auto& candidate : candidates) {
        if (candidate.candidate_identity == 0 || !std::isfinite(candidate.numerical_error) ||
            candidate.numerical_error < 0.0 || candidate.candidate_family == 0) {
            return hard_constraint_status_v1::invalid_candidate;
        }
        std::string reason;
        if (constraints.require_determinism && !candidate.deterministic) {
            reason = "does not satisfy deterministic semantics";
        } else if (constraints.require_exactness && !candidate.exact) {
            reason = "does not satisfy exact execution semantics";
        } else if (candidate.numerical_error > constraints.maximum_numerical_error) {
            reason = "exceeds the numerical tolerance";
        } else if (constraints.maximum_memory_bytes != 0 &&
                   candidate.memory_bytes > constraints.maximum_memory_bytes) {
            reason = "exceeds the hard memory bound";
        } else if ((candidate.target_capabilities & constraints.required_target_capabilities) !=
                   constraints.required_target_capabilities) {
            reason = "lacks required target capabilities";
        } else if (constraints.allowed_candidate_families != 0 &&
                   (candidate.candidate_family & constraints.allowed_candidate_families) == 0) {
            reason = "candidate family is not allowed";
        } else if (constraints.required_order_identity != 0 &&
                   candidate.output_order_identity != constraints.required_order_identity) {
            reason = "does not preserve the required output order";
        } else if (constraints.forbid_synchronization && candidate.synchronizes) {
            reason = "introduces forbidden synchronization";
        }

        if (reason.empty()) {
            filtered.legal_candidates.push_back(candidate.candidate_identity);
        } else {
            filtered.rejections.push_back({candidate.candidate_identity, std::move(reason)});
        }
    }
    *result = std::move(filtered);
    return result->legal_candidates.empty() ? hard_constraint_status_v1::no_legal_continuation
                                            : hard_constraint_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
