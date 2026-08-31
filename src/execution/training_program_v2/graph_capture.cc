#include <Cellerator/execution/training_program_v2/graph_capture.hh>

#include <cstdint>

namespace cellerator::execution::training_v2 {
namespace {

training_result_v2 error(training_status_v2 code, const char *message) noexcept {
    return {code, message};
}

bool valid_launch(const training_graph_launch_binding_v2 &binding) noexcept {
    return binding.generation.value != 0u && binding.source != nullptr
        && binding.destination_gradient != nullptr
        && binding.destination != nullptr && binding.source_gradient != nullptr
        && binding.value_gradient != nullptr
        && (binding.transient_workspace_bytes == 0u
            || binding.transient_workspace != nullptr)
        && binding.stream_token != 0u;
}

} // namespace

training_result_v2 validate_training_graph_capture_v2(
    const training_graph_capture_v2 &capture,
    const training_program_v2 &program,
    graph_capture_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (capture.program_identity == 0u
        || capture.program_identity != program.program_identity
        || !same_handle(capture.structure, program.structure)
        || capture.epoch.value != program.epoch.value
        || capture.prepared_generation.value != program.prepared_generation.value
        || capture.stage_count != program.stage_count
        || capture.stage_count == 0u || capture.stage_identities == nullptr
        || capture.graph_identity == 0u || !capture.pointer_rebind_supported
        || !capture.stream_rebind_supported
        || !capture.update_policy_owned_by_caller
        || capture.production_promoted || !program.graph_capture_required)
        return error(training_status_v2::invalid_stage_graph,
            "training graph capture envelope is invalid or promoted");
    for (std::uint64_t index = 0u; index < capture.stage_count; ++index) {
        if (capture.stage_identities[index] == 0u
            || capture.stage_identities[index]
                != program.stages[index].stage_identity
            || !program.stages[index].graph_capture_compatible)
            return error(training_status_v2::invalid_stage_graph,
                "captured stage identity or compatibility is invalid");
    }
    receipt.validated_stage_count = capture.stage_count;
    return {};
}

training_result_v2 validate_training_graph_rebind_v2(
    const training_graph_capture_v2 &capture,
    const training_graph_launch_binding_v2 &previous,
    const training_graph_launch_binding_v2 &next,
    const caller_update_policy_binding_v2 &update_policy,
    graph_capture_receipt_v2 &receipt) noexcept {
    receipt = {};
    if (!valid_launch(previous) || !valid_launch(next)
        || previous.generation.value != next.generation.value
        || next.generation.value < capture.prepared_generation.value)
        return error(training_status_v2::stale_generation,
            "training graph launch generation or binding is invalid");
    if (update_policy.caller_policy_identity == 0u
        || update_policy.prepared_update_candidate_identity == 0u
        || update_policy.caller_policy_state == nullptr
        || update_policy.caller_policy_state_bytes == 0u
        || !capture.update_policy_owned_by_caller)
        return error(training_status_v2::invalid_argument,
            "caller update policy remains unbound or enters core ownership");
    receipt.validated_stage_count = capture.stage_count;
    receipt.pointers_rebound = previous.source != next.source
        || previous.destination_gradient != next.destination_gradient
        || previous.destination != next.destination
        || previous.source_gradient != next.source_gradient
        || previous.value_gradient != next.value_gradient
        || previous.transient_workspace != next.transient_workspace;
    receipt.stream_rebound = previous.stream_token != next.stream_token;
    receipt.reprepare_required = false;
    receipt.update_policy_separate = true;
    return {};
}

} // namespace cellerator::execution::training_v2
