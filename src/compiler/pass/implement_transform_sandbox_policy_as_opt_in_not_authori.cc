#include <Cellerator/compiler/pass/implement_transform_sandbox_policy_as_opt_in_not_authori_v1.hh>

namespace cellerator::compiler::pass::v1 {

transform_sandbox_receipt_v1 execute_transform_with_policy_v1(
    const transform_sandbox_policy_v1& policy, transform_execute_v1 execute,
    transform_verify_v1 verify, void* user_data) noexcept {
    transform_sandbox_receipt_v1 receipt;
    receipt.executed_mode = policy.requested_mode;
    receipt.isolated = policy.requested_mode !=
        transform_execution_mode_v1::trusted_in_process;
    if (execute == nullptr) {
        receipt.observation = transform_observation_v1::rejected;
        receipt.diagnostic = "transform entry point is unavailable";
    } else {
        receipt.observation = execute(user_data);
        if (receipt.observation == transform_observation_v1::success
            && policy.requested_mode == transform_execution_mode_v1::isolated_verified) {
            receipt.verified = verify != nullptr && verify(user_data);
            if (!receipt.verified) {
                receipt.observation = transform_observation_v1::rejected;
                receipt.diagnostic = "transform result failed verification";
            }
        }
    }
    if (receipt.observation != transform_observation_v1::success) {
        receipt.continuation_allowed = policy.unsafe_continue_after_failure;
        if (receipt.diagnostic.empty()) {
            receipt.diagnostic = "transform failed under explicitly selected execution policy";
        }
    }
    // The policy is advisory execution configuration selected by the caller. It
    // never silently replaces trusted mode with isolation or makes itself an
    // authority over whether expert-controlled compilation may continue.
    return receipt;
}

}  // namespace cellerator::compiler::pass::v1
