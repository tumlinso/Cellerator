#include <Cellerator/compiler/frontend/cxx/bind_native_effect_contracts_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::frontend::cxx {

native_effect_contract_status_v1 bind_native_effect_contracts_v1(
    std::uint32_t schema_version,
    const std::vector<overload_semantic_candidate_v1>& functions,
    const std::vector<native_effect_contract_v1>& contracts,
    const std::vector<observed_native_effects_v1>& checked_observations,
    bool permit_trusted_continuation,
    std::vector<bound_native_effect_contract_v1>* bindings) noexcept {
    if (bindings == nullptr || functions.empty() ||
        functions.size() != checked_observations.size()) {
        return native_effect_contract_status_v1::function_mismatch;
    }
    bindings->clear();
    if (schema_version != native_effect_contract_schema_version_v1) {
        return native_effect_contract_status_v1::schema_mismatch;
    }
    auto overall = native_effect_contract_status_v1::success;
    for (std::size_t index = 0; index < functions.size(); ++index) {
        const auto& function = functions[index];
        if (function.selected_declaration == nullptr || function.qualified_name.empty()) {
            return native_effect_contract_status_v1::function_mismatch;
        }
        const auto first = std::find_if(
            contracts.begin(), contracts.end(), [&function](const auto& contract) {
                return contract.qualified_name == function.qualified_name;
            });
        if (first == contracts.end()) {
            return native_effect_contract_status_v1::missing_contract;
        }
        if (std::find_if(std::next(first), contracts.end(), [&function](const auto& contract) {
                return contract.qualified_name == function.qualified_name;
            }) != contracts.end()) {
            return native_effect_contract_status_v1::duplicate_contract;
        }
        bound_native_effect_contract_v1 binding;
        binding.function_declaration = function.selected_declaration;
        binding.contract = *first;
        binding.observed = checked_observations[index];
        const auto undeclared = binding.observed.effects & ~binding.contract.permitted_effects;
        if (undeclared != 0) {
            binding.diagnostics.push_back("observed undeclared read/write topology/order/support/value effects");
        }
        if (binding.contract.pure && binding.observed.effects != 0) {
            binding.diagnostics.push_back("pure contract observed effects");
        }
        if (binding.contract.deterministic && binding.observed.nondeterministic_result) {
            binding.diagnostics.push_back("deterministic contract produced divergent checked results");
        }
        if (binding.contract.no_alias && binding.observed.alias_observed) {
            binding.diagnostics.push_back("no-alias contract observed overlapping storage");
        }
        binding.verified = binding.diagnostics.empty();
        binding.trusted_continuation_permitted =
            !binding.verified && permit_trusted_continuation;
        if (!binding.verified) {
            overall = native_effect_contract_status_v1::contract_violation;
        }
        bindings->push_back(std::move(binding));
    }
    return overall;
}

}  // namespace Cellerator::compiler::frontend::cxx
