#pragma once

#include <Cellerator/compiler/ir/semantic/implement_domain_and_axis_ir_types_v1.hh>

#include <cstdint>
#include <string>

namespace Cellerator::compiler::ir::semantic {

enum class semantic_lifetime_layer_v1 : std::uint8_t {
    structure = 1,
    values,
    support,
    order,
};

enum class semantic_transition_kind_v1 : std::uint8_t {
    invalidate = 1,
    publish,
    clone,
    trusted_assertion,
    epoch_boundary,
};

struct semantic_lifetime_state_v1 {
    semantic_identity_v1 object{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t support_generation = 0;
    std::uint64_t order_generation = 0;
    bool structure_valid = true;
    bool values_valid = true;
    bool support_valid = true;
    bool order_valid = true;
};

struct semantic_transition_operation_v1 {
    semantic_identity_v1 identity{};
    semantic_transition_kind_v1 kind = semantic_transition_kind_v1::invalidate;
    semantic_lifetime_layer_v1 layer = semantic_lifetime_layer_v1::values;
    semantic_identity_v1 source_object{};
    semantic_identity_v1 target_object{};
    std::uint64_t expected = 0;
    std::uint64_t produced = 0;
    bool trusted = false;
    std::string assertion_reason;
};

enum class semantic_transition_status_v1 : std::uint8_t {
    success = 0,
    invalid_operation,
    object_mismatch,
    stale_use,
    invalid_transition,
    untrusted_assertion,
};

struct semantic_transition_diagnostic_v1 {
    semantic_transition_status_v1 code = semantic_transition_status_v1::success;
    semantic_lifetime_layer_v1 layer = semantic_lifetime_layer_v1::values;
    std::uint64_t expected = 0;
    std::uint64_t observed = 0;
    std::string message;
};

[[nodiscard]] semantic_transition_status_v1
apply_semantic_transition_v1(
    const semantic_transition_operation_v1& operation,
    semantic_lifetime_state_v1* state,
    semantic_lifetime_state_v1* cloned_state = nullptr,
    semantic_transition_diagnostic_v1* diagnostic = nullptr) noexcept;

[[nodiscard]] semantic_transition_status_v1
validate_semantic_lifetime_use_v1(
    const semantic_lifetime_state_v1& state,
    semantic_lifetime_layer_v1 layer,
    std::uint64_t expected,
    semantic_transition_diagnostic_v1* diagnostic = nullptr) noexcept;

}  // namespace Cellerator::compiler::ir::semantic
