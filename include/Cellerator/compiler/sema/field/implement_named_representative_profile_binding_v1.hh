#pragma once

#include <Cellerator/compiler/sema/field/define_execution_field_semantic_ownership_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

struct representative_profile_state_v1 {
    std::string name;
    std::uint64_t state_identity = 0;
    std::uint64_t content_digest_low = 0;
    std::uint64_t content_digest_high = 0;
    bool baseline = false;
    bool activated = false;
};

struct representative_profile_alias_v1 {
    std::string alias;
    std::string state_name;
};

struct operation_profile_selection_v1 {
    std::uint64_t operation_identity = 0;
    std::string state_or_alias;
};

struct resolved_operation_profile_v1 {
    std::uint64_t operation_identity = 0;
    std::uint64_t state_identity = 0;
    std::string selected_name;
    bool activated = false;
};

struct representative_profile_binding_v1 {
    execution_field_identity_v1 field_identity{};
    std::vector<representative_profile_state_v1> states;
    std::vector<representative_profile_alias_v1> aliases;
    std::vector<resolved_operation_profile_v1> operations;
};

enum class profile_binding_status_v1 : std::uint8_t {
    success = 0,
    invalid_field,
    invalid_state,
    duplicate_state,
    missing_baseline,
    duplicate_alias,
    unresolved_alias,
    invalid_operation,
    unavailable_state,
};

[[nodiscard]] profile_binding_status_v1
implement_named_representative_profile_binding_v1(
    const execution_field_semantics_v1& field,
    const std::vector<representative_profile_state_v1>& compile_supplied_states,
    const std::vector<representative_profile_alias_v1>& aliases,
    const std::vector<operation_profile_selection_v1>& selections,
    representative_profile_binding_v1* binding) noexcept;

}  // namespace Cellerator::compiler::sema::field
