#include <Cellerator/compiler/sema/field/implement_automatic_lifetime_and_generation_transfer_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::sema::field {
namespace {

void advance(automatic_semantic_state_v1& state, std::uint32_t components) noexcept {
    if ((components & state_component_structure_v1) != 0) ++state.structure_epoch;
    if ((components & state_component_value_v1) != 0) ++state.value_generation;
    if ((components & state_component_support_v1) != 0) ++state.support_generation;
    if ((components & state_component_order_v1) != 0) ++state.order_generation;
}

bool valid(const automatic_semantic_state_v1& state) noexcept {
    return state.object_identity != 0 && state.structure_epoch != 0 &&
        state.value_generation != 0 && state.support_generation != 0 &&
        state.order_generation != 0;
}

std::uint64_t join_generation(std::uint64_t lhs, std::uint64_t rhs) noexcept {
    return lhs == rhs ? lhs : std::max(lhs, rhs) + 1;
}

}  // namespace

generation_transfer_status_v1 implement_automatic_lifetime_and_generation_transfer_v1(
    automatic_semantic_state_v1* state,
    const automatic_generation_transfer_v1& transfer,
    std::uint32_t repeat_count,
    std::vector<materialized_generation_transition_v1>* transitions) noexcept {
    if (state == nullptr || transitions == nullptr || !valid(*state)) {
        return generation_transfer_status_v1::invalid_state;
    }
    if (transfer.operation_identity == 0 || repeat_count == 0) {
        return generation_transfer_status_v1::invalid_transfer;
    }
    if (state->lifetime == semantic_lifetime_v1::ended) {
        return generation_transfer_status_v1::use_after_lifetime;
    }
    for (std::uint32_t iteration = 0; iteration < repeat_count; ++iteration) {
        materialized_generation_transition_v1 record;
        record.operation_identity = transfer.operation_identity;
        record.kind = repeat_count == 1 ? transfer.kind :
            generation_transition_kind_v1::loop_iteration;
        record.before = *state;
        advance(*state, transfer.advance_components);
        if (transfer.ends_lifetime) state->lifetime = semantic_lifetime_v1::ended;
        record.after = *state;
        transitions->push_back(record);
        if (transfer.ends_lifetime && iteration + 1 != repeat_count) {
            return generation_transfer_status_v1::use_after_lifetime;
        }
    }
    return generation_transfer_status_v1::success;
}

generation_transfer_status_v1 join_automatic_generation_branches_v1(
    const automatic_semantic_state_v1& lhs,
    const automatic_semantic_state_v1& rhs,
    std::uint64_t join_identity,
    automatic_semantic_state_v1* joined,
    materialized_generation_transition_v1* transition) noexcept {
    if (joined == nullptr || transition == nullptr || join_identity == 0 ||
        !valid(lhs) || !valid(rhs)) {
        return generation_transfer_status_v1::invalid_state;
    }
    if (lhs.object_identity != rhs.object_identity) {
        return generation_transfer_status_v1::branch_identity_mismatch;
    }
    *joined = lhs;
    joined->structure_epoch = join_generation(lhs.structure_epoch, rhs.structure_epoch);
    joined->value_generation = join_generation(lhs.value_generation, rhs.value_generation);
    joined->support_generation = join_generation(lhs.support_generation, rhs.support_generation);
    joined->order_generation = join_generation(lhs.order_generation, rhs.order_generation);
    joined->lifetime = lhs.lifetime == rhs.lifetime ? lhs.lifetime :
        semantic_lifetime_v1::maybe_alive;
    transition->operation_identity = join_identity;
    transition->kind = generation_transition_kind_v1::branch_join;
    transition->before = lhs;
    transition->after = *joined;
    return generation_transfer_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
