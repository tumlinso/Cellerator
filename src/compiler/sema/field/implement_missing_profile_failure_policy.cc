#include <Cellerator/compiler/sema/field/implement_missing_profile_failure_policy_v1.hh>

#include <utility>

namespace Cellerator::compiler::sema::field {

missing_profile_policy_status_v1 implement_missing_profile_failure_policy_v1(
    const missing_profile_policy_request_v1& request,
    missing_profile_policy_result_v1* result) noexcept {
    if (result == nullptr) return missing_profile_policy_status_v1::invalid_output;
    missing_profile_policy_result_v1 decision;
    if (request.activation != compilation_activation_kind_v1::biological_compilation) {
        decision.compilation_allowed = true;
        *result = std::move(decision);
        return missing_profile_policy_status_v1::success;
    }
    if (request.representative_profile_bound || request.generic_reference_profile_selected) {
        decision.compilation_allowed = true;
        decision.uses_generic_reference_profile =
            !request.representative_profile_bound && request.generic_reference_profile_selected;
        *result = std::move(decision);
        return missing_profile_policy_status_v1::success;
    }
    decision.diagnostic =
        "activated biological compilation requires a representative semantic profile; "
        "bind a named build profile or explicitly select the generic reference profile";
    *result = std::move(decision);
    return missing_profile_policy_status_v1::representative_profile_required;
}

}  // namespace Cellerator::compiler::sema::field
