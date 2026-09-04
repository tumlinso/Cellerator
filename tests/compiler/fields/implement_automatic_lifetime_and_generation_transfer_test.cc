#include <Cellerator/compiler/sema/field/implement_automatic_lifetime_and_generation_transfer_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    field::automatic_semantic_state_v1 initial;
    initial.object_identity = 9;
    auto loop_state = initial;
    std::vector<field::materialized_generation_transition_v1> transitions;
    field::automatic_generation_transfer_v1 update;
    update.operation_identity = 20;
    update.advance_components = field::state_component_value_v1 |
        field::state_component_support_v1;
    if (field::implement_automatic_lifetime_and_generation_transfer_v1(
            &loop_state, update, 3, &transitions) !=
            field::generation_transfer_status_v1::success ||
        loop_state.value_generation != 4 || loop_state.support_generation != 4 ||
        transitions.size() != 3 || transitions[1].kind !=
            field::generation_transition_kind_v1::loop_iteration) {
        std::cerr << "repeated generation updates were not materialized\n";
        return 1;
    }

    auto unchanged_branch = initial;
    field::automatic_semantic_state_v1 joined;
    field::materialized_generation_transition_v1 join;
    if (field::join_automatic_generation_branches_v1(
            loop_state, unchanged_branch, 30, &joined, &join) !=
            field::generation_transfer_status_v1::success ||
        joined.value_generation != 5 || joined.support_generation != 5 ||
        join.kind != field::generation_transition_kind_v1::branch_join) {
        std::cerr << "branch generation phi was not made explicit\n";
        return 1;
    }

    field::automatic_generation_transfer_v1 exit;
    exit.operation_identity = 40;
    exit.kind = field::generation_transition_kind_v1::field_exit;
    exit.advance_components = field::state_component_order_v1;
    exit.ends_lifetime = true;
    if (field::implement_automatic_lifetime_and_generation_transfer_v1(
            &joined, exit, 1, &transitions) != field::generation_transfer_status_v1::success ||
        joined.order_generation != 2 || joined.lifetime != field::semantic_lifetime_v1::ended ||
        field::implement_automatic_lifetime_and_generation_transfer_v1(
            &joined, update, 1, &transitions) !=
            field::generation_transfer_status_v1::use_after_lifetime) {
        std::cerr << "field-exit lifetime was not enforced\n";
        return 1;
    }
    return 0;
}
