#include <Cellerator/compiler/ir/semantic/implement_generation_and_epoch_transition_operations_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;

int main() {
    semantic_lifetime_state_v1 state{{1, 2}, 3, 4, 5, 6, true, true, true, true};
    semantic_transition_diagnostic_v1 diagnostic;
    semantic_transition_operation_v1 publish;
    publish.identity = {10, 11};
    publish.kind = semantic_transition_kind_v1::publish;
    publish.layer = semantic_lifetime_layer_v1::values;
    publish.source_object = state.object;
    publish.expected = 4;
    publish.produced = 5;
    assert(apply_semantic_transition_v1(publish, &state, nullptr, &diagnostic) ==
           semantic_transition_status_v1::success);
    assert(state.value_generation == 5);
    assert(validate_semantic_lifetime_use_v1(
        state, semantic_lifetime_layer_v1::values, 4, &diagnostic) ==
        semantic_transition_status_v1::stale_use);
    assert(diagnostic.observed == 5);

    semantic_transition_operation_v1 boundary;
    boundary.identity = {12, 13};
    boundary.kind = semantic_transition_kind_v1::epoch_boundary;
    boundary.layer = semantic_lifetime_layer_v1::structure;
    boundary.source_object = state.object;
    boundary.expected = 3;
    boundary.produced = 4;
    assert(apply_semantic_transition_v1(boundary, &state) ==
           semantic_transition_status_v1::success);
    assert(state.structure_epoch == 4 && !state.values_valid &&
           !state.support_valid && !state.order_valid);

    semantic_transition_operation_v1 assertion;
    assertion.identity = {14, 15};
    assertion.kind = semantic_transition_kind_v1::trusted_assertion;
    assertion.layer = semantic_lifetime_layer_v1::order;
    assertion.source_object = state.object;
    assertion.produced = 7;
    assert(apply_semantic_transition_v1(assertion, &state, nullptr, &diagnostic) ==
           semantic_transition_status_v1::untrusted_assertion);
    assertion.trusted = true;
    assertion.assertion_reason = "verified external order contract";
    assert(apply_semantic_transition_v1(assertion, &state, nullptr, &diagnostic) ==
           semantic_transition_status_v1::success);
    assert(state.order_valid && state.order_generation == 7);

    semantic_transition_operation_v1 clone = assertion;
    clone.identity = {16, 17};
    clone.kind = semantic_transition_kind_v1::clone;
    clone.target_object = {20, 21};
    clone.expected = 7;
    semantic_lifetime_state_v1 cloned;
    assert(apply_semantic_transition_v1(clone, &state, &cloned) ==
           semantic_transition_status_v1::success);
    assert(cloned.object.low == 20 && cloned.structure_epoch == state.structure_epoch);

    std::cout << "structure_epoch=4 stale_use=diagnosed trusted_assertion=accepted\n";
}
