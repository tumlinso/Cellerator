#include <Cellerator/compiler/ir/semantic/implement_generation_and_epoch_transition_operations_v1.hh>

namespace Cellerator::compiler::ir::semantic {
namespace {

bool same(semantic_identity_v1 left, semantic_identity_v1 right) noexcept {
    return left.low == right.low && left.high == right.high;
}

std::uint64_t& generation(semantic_lifetime_state_v1& state,
                          semantic_lifetime_layer_v1 layer) noexcept {
    if (layer == semantic_lifetime_layer_v1::structure) return state.structure_epoch;
    if (layer == semantic_lifetime_layer_v1::values) return state.value_generation;
    if (layer == semantic_lifetime_layer_v1::support) return state.support_generation;
    return state.order_generation;
}

bool& validity(semantic_lifetime_state_v1& state,
               semantic_lifetime_layer_v1 layer) noexcept {
    if (layer == semantic_lifetime_layer_v1::structure) return state.structure_valid;
    if (layer == semantic_lifetime_layer_v1::values) return state.values_valid;
    if (layer == semantic_lifetime_layer_v1::support) return state.support_valid;
    return state.order_valid;
}

void report(semantic_transition_diagnostic_v1* diagnostic,
            semantic_transition_status_v1 code,
            semantic_lifetime_layer_v1 layer,
            std::uint64_t expected,
            std::uint64_t observed,
            const char* message) {
    if (diagnostic != nullptr) *diagnostic = {code, layer, expected, observed, message};
}

}  // namespace

semantic_transition_status_v1 validate_semantic_lifetime_use_v1(
    const semantic_lifetime_state_v1& state,
    semantic_lifetime_layer_v1 layer,
    std::uint64_t expected,
    semantic_transition_diagnostic_v1* diagnostic) noexcept {
    auto copy = state;
    const auto observed = generation(copy, layer);
    if (!state.object.valid() || !validity(copy, layer) || observed == 0 || observed != expected) {
        report(diagnostic, semantic_transition_status_v1::stale_use, layer,
               expected, observed, "semantic lifetime use is stale or invalidated");
        return semantic_transition_status_v1::stale_use;
    }
    report(diagnostic, semantic_transition_status_v1::success, layer,
           expected, observed, "semantic lifetime use is current");
    return semantic_transition_status_v1::success;
}

semantic_transition_status_v1 apply_semantic_transition_v1(
    const semantic_transition_operation_v1& operation,
    semantic_lifetime_state_v1* state,
    semantic_lifetime_state_v1* cloned_state,
    semantic_transition_diagnostic_v1* diagnostic) noexcept {
    if (state == nullptr || !operation.identity.valid() || !operation.source_object.valid()) {
        report(diagnostic, semantic_transition_status_v1::invalid_operation,
               operation.layer, operation.expected, 0, "invalid transition operation");
        return semantic_transition_status_v1::invalid_operation;
    }
    if (!same(state->object, operation.source_object)) {
        report(diagnostic, semantic_transition_status_v1::object_mismatch,
               operation.layer, operation.expected, 0, "transition source object mismatch");
        return semantic_transition_status_v1::object_mismatch;
    }
    auto& current = generation(*state, operation.layer);
    auto& valid = validity(*state, operation.layer);
    if (operation.expected != 0 && (!valid || current != operation.expected))
        return validate_semantic_lifetime_use_v1(*state, operation.layer,
                                                 operation.expected, diagnostic);
    switch (operation.kind) {
    case semantic_transition_kind_v1::invalidate:
        valid = false;
        break;
    case semantic_transition_kind_v1::publish:
        if (operation.produced <= current) return semantic_transition_status_v1::invalid_transition;
        current = operation.produced;
        valid = true;
        break;
    case semantic_transition_kind_v1::clone:
        if (cloned_state == nullptr || !operation.target_object.valid())
            return semantic_transition_status_v1::invalid_transition;
        *cloned_state = *state;
        cloned_state->object = operation.target_object;
        break;
    case semantic_transition_kind_v1::trusted_assertion:
        if (!operation.trusted || operation.assertion_reason.empty()) {
            report(diagnostic, semantic_transition_status_v1::untrusted_assertion,
                   operation.layer, operation.expected, current,
                   "lifetime assertion requires explicit trust and reason");
            return semantic_transition_status_v1::untrusted_assertion;
        }
        valid = true;
        if (operation.produced != 0) current = operation.produced;
        break;
    case semantic_transition_kind_v1::epoch_boundary:
        if (operation.layer != semantic_lifetime_layer_v1::structure ||
            operation.produced <= state->structure_epoch)
            return semantic_transition_status_v1::invalid_transition;
        state->structure_epoch = operation.produced;
        state->structure_valid = true;
        state->values_valid = false;
        state->support_valid = false;
        state->order_valid = false;
        break;
    }
    report(diagnostic, semantic_transition_status_v1::success, operation.layer,
           operation.expected, current, "explicit semantic lifetime transition applied");
    return semantic_transition_status_v1::success;
}

}  // namespace Cellerator::compiler::ir::semantic
