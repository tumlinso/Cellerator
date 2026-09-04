#include <Cellerator/compiler/pass/implement_complete_built_in_stage_replacement_v1.hh>

namespace cellerator::compiler::pass::v1 {

stage_replacement_receipt_v1 run_stage_replacement_v1(
    const stage_replacement_request_v1& request) noexcept {
    stage_replacement_receipt_v1 receipt;
    if (!valid_pipeline_stage_v1({request.phase, interception_side_v1::before})) {
        receipt.status = stage_replacement_status_v1::invalid_stage;
        receipt.diagnostic = "invalid replaceable pipeline phase";
        return receipt;
    }
    stage_replacement_context_v1 context{request.phase, request.stage_state, {}};
    if (request.replacement != nullptr) {
        receipt.replacement_attempted = true;
        if (request.replacement(context)) {
            receipt.replacement_selected = true;
            receipt.diagnostic = context.diagnostic;
            return receipt;
        }
        receipt.diagnostic = context.diagnostic;
        if (request.policy == stage_replacement_policy_v1::force_replacement) {
            receipt.status = stage_replacement_status_v1::replacement_failed;
            return receipt;
        }
    } else if (request.policy == stage_replacement_policy_v1::force_replacement) {
        receipt.status = stage_replacement_status_v1::missing_implementation;
        receipt.diagnostic = "forced replacement is unavailable";
        return receipt;
    }
    if (request.built_in == nullptr) {
        receipt.status = stage_replacement_status_v1::missing_implementation;
        receipt.diagnostic = "no built-in fallback is available";
        return receipt;
    }
    receipt.fallback_used = receipt.replacement_attempted;
    context.diagnostic.clear();
    if (!request.built_in(context)) {
        receipt.status = stage_replacement_status_v1::built_in_failed;
    }
    receipt.diagnostic = context.diagnostic;
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
