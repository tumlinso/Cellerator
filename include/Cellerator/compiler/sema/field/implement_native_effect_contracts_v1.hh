#pragma once

#include <Cellerator/compiler/frontend/cxx/bind_native_effect_contracts_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::sema::field {

enum native_publication_effect_v1 : std::uint32_t {
    publishes_nothing_v1 = 0,
    publishes_value_generation_v1 = 1u << 0,
    publishes_structure_epoch_v1 = 1u << 1,
    publishes_order_identity_v1 = 1u << 2,
    publishes_support_generation_v1 = 1u << 3,
};

enum class native_target_behavior_v1 : std::uint8_t {
    host_only = 1,
    device_only,
    host_and_device,
    target_dependent,
};

struct field_native_effect_contract_v1 {
    frontend::cxx::native_effect_contract_v1 effects;
    std::uint32_t publications = publishes_nothing_v1;
    native_target_behavior_v1 target = native_target_behavior_v1::host_only;
};

struct resolved_native_function_effects_v1 {
    const void* function_declaration = nullptr;
    std::string qualified_name;
    std::uint32_t effects = 0;
    std::uint32_t publications = publishes_nothing_v1;
    native_target_behavior_v1 target = native_target_behavior_v1::host_only;
    bool pure = false;
    bool deterministic = false;
    bool no_alias = false;
    bool verified = false;
    bool trusted_continuation_permitted = false;
    std::vector<std::string> diagnostics;
};

enum class field_native_effect_status_v1 : std::uint8_t {
    success = 0,
    invalid_input,
    schema_mismatch,
    missing_contract,
    duplicate_contract,
    function_mismatch,
    contract_violation,
};

[[nodiscard]] field_native_effect_status_v1 implement_native_effect_contracts_v1(
    std::uint32_t schema_version,
    const std::vector<frontend::cxx::overload_semantic_candidate_v1>& functions,
    const std::vector<field_native_effect_contract_v1>& contracts,
    const std::vector<frontend::cxx::observed_native_effects_v1>& observations,
    bool permit_trusted_continuation,
    std::vector<resolved_native_function_effects_v1>* resolved) noexcept;

}  // namespace Cellerator::compiler::sema::field
