#pragma once

#include <Cellerator/compiler/frontend/cxx/integrate_overload_resolution_and_cellerator_semantic_ca_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::frontend::cxx {

inline constexpr std::uint32_t native_effect_contract_schema_version_v1 = 1;

enum native_effect_claim_v1 : std::uint32_t {
    native_reads_value_v1 = 1u << 0,
    native_writes_value_v1 = 1u << 1,
    native_reads_topology_v1 = 1u << 2,
    native_writes_topology_v1 = 1u << 3,
    native_reads_order_v1 = 1u << 4,
    native_writes_order_v1 = 1u << 5,
    native_reads_support_v1 = 1u << 6,
    native_writes_support_v1 = 1u << 7,
};

enum class native_effect_contract_status_v1 : std::uint8_t {
    success = 0,
    schema_mismatch,
    missing_contract,
    duplicate_contract,
    function_mismatch,
    contract_violation,
};

struct native_effect_contract_v1 {
    std::string qualified_name;
    std::uint32_t permitted_effects = 0;
    bool pure = false;
    bool deterministic = false;
    bool no_alias = false;
};

struct observed_native_effects_v1 {
    std::uint32_t effects = 0;
    bool nondeterministic_result = false;
    bool alias_observed = false;
};

class native_effect_observer_v1 {
public:
    void observe(std::uint32_t effects) noexcept { observed_.effects |= effects; }
    void observe_nondeterminism() noexcept { observed_.nondeterministic_result = true; }
    void observe_alias() noexcept { observed_.alias_observed = true; }
    const observed_native_effects_v1& result() const noexcept { return observed_; }

private:
    observed_native_effects_v1 observed_{};
};

struct bound_native_effect_contract_v1 {
    const void* function_declaration = nullptr;
    native_effect_contract_v1 contract;
    observed_native_effects_v1 observed;
    bool verified = false;
    bool trusted_continuation_permitted = false;
    std::vector<std::string> diagnostics;
};

native_effect_contract_status_v1 bind_native_effect_contracts_v1(
    std::uint32_t schema_version,
    const std::vector<overload_semantic_candidate_v1>& functions,
    const std::vector<native_effect_contract_v1>& contracts,
    const std::vector<observed_native_effects_v1>& checked_observations,
    bool permit_trusted_continuation,
    std::vector<bound_native_effect_contract_v1>* bindings) noexcept;

}  // namespace Cellerator::compiler::frontend::cxx
