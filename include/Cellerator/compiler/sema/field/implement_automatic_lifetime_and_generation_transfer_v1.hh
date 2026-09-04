#pragma once

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::sema::field {

enum semantic_state_component_v1 : std::uint32_t {
    state_component_none_v1 = 0,
    state_component_structure_v1 = 1u << 0,
    state_component_value_v1 = 1u << 1,
    state_component_support_v1 = 1u << 2,
    state_component_order_v1 = 1u << 3,
};

enum class semantic_lifetime_v1 : std::uint8_t {
    alive = 1,
    ended,
    maybe_alive,
};

struct automatic_semantic_state_v1 {
    std::uint64_t object_identity = 0;
    std::uint64_t structure_epoch = 1;
    std::uint64_t value_generation = 1;
    std::uint64_t support_generation = 1;
    std::uint64_t order_generation = 1;
    semantic_lifetime_v1 lifetime = semantic_lifetime_v1::alive;
};

enum class generation_transition_kind_v1 : std::uint8_t {
    known_operation = 1,
    loop_iteration,
    native_contract,
    branch_join,
    field_exit,
};

struct automatic_generation_transfer_v1 {
    std::uint64_t operation_identity = 0;
    std::uint32_t advance_components = state_component_none_v1;
    generation_transition_kind_v1 kind = generation_transition_kind_v1::known_operation;
    bool ends_lifetime = false;
};

struct materialized_generation_transition_v1 {
    std::uint64_t operation_identity = 0;
    generation_transition_kind_v1 kind = generation_transition_kind_v1::known_operation;
    automatic_semantic_state_v1 before;
    automatic_semantic_state_v1 after;
};

enum class generation_transfer_status_v1 : std::uint8_t {
    success = 0,
    invalid_state,
    invalid_transfer,
    use_after_lifetime,
    branch_identity_mismatch,
};

[[nodiscard]] generation_transfer_status_v1
implement_automatic_lifetime_and_generation_transfer_v1(
    automatic_semantic_state_v1* state,
    const automatic_generation_transfer_v1& transfer,
    std::uint32_t repeat_count,
    std::vector<materialized_generation_transition_v1>* transitions) noexcept;

[[nodiscard]] generation_transfer_status_v1 join_automatic_generation_branches_v1(
    const automatic_semantic_state_v1& lhs,
    const automatic_semantic_state_v1& rhs,
    std::uint64_t join_identity,
    automatic_semantic_state_v1* joined,
    materialized_generation_transition_v1* transition) noexcept;

}  // namespace Cellerator::compiler::sema::field
