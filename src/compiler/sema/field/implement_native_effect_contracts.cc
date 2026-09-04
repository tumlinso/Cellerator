#include <Cellerator/compiler/sema/field/implement_native_effect_contracts_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::sema::field {
namespace {

field_native_effect_status_v1 translate(
    frontend::cxx::native_effect_contract_status_v1 status) noexcept {
    using source = frontend::cxx::native_effect_contract_status_v1;
    switch (status) {
    case source::success: return field_native_effect_status_v1::success;
    case source::schema_mismatch: return field_native_effect_status_v1::schema_mismatch;
    case source::missing_contract: return field_native_effect_status_v1::missing_contract;
    case source::duplicate_contract: return field_native_effect_status_v1::duplicate_contract;
    case source::function_mismatch: return field_native_effect_status_v1::function_mismatch;
    case source::contract_violation: return field_native_effect_status_v1::contract_violation;
    }
    return field_native_effect_status_v1::invalid_input;
}

}  // namespace

field_native_effect_status_v1 implement_native_effect_contracts_v1(
    std::uint32_t schema_version,
    const std::vector<frontend::cxx::overload_semantic_candidate_v1>& functions,
    const std::vector<field_native_effect_contract_v1>& contracts,
    const std::vector<frontend::cxx::observed_native_effects_v1>& observations,
    bool permit_trusted_continuation,
    std::vector<resolved_native_function_effects_v1>* resolved) noexcept {
    if (resolved == nullptr) return field_native_effect_status_v1::invalid_input;
    resolved->clear();
    std::vector<frontend::cxx::native_effect_contract_v1> base_contracts;
    base_contracts.reserve(contracts.size());
    for (const auto& contract : contracts) base_contracts.push_back(contract.effects);

    std::vector<frontend::cxx::bound_native_effect_contract_v1> bindings;
    const auto base_status = frontend::cxx::bind_native_effect_contracts_v1(
        schema_version, functions, base_contracts, observations,
        permit_trusted_continuation, &bindings);
    if (base_status != frontend::cxx::native_effect_contract_status_v1::success &&
        base_status != frontend::cxx::native_effect_contract_status_v1::contract_violation) {
        return translate(base_status);
    }
    for (const auto& binding : bindings) {
        const auto contract = std::find_if(
            contracts.begin(), contracts.end(), [&binding](const auto& candidate) {
                return candidate.effects.qualified_name == binding.contract.qualified_name;
            });
        if (contract == contracts.end()) return field_native_effect_status_v1::missing_contract;
        resolved_native_function_effects_v1 result;
        result.function_declaration = binding.function_declaration;
        result.qualified_name = binding.contract.qualified_name;
        result.effects = binding.contract.permitted_effects;
        result.publications = contract->publications;
        result.target = contract->target;
        result.pure = binding.contract.pure;
        result.deterministic = binding.contract.deterministic;
        result.no_alias = binding.contract.no_alias;
        result.verified = binding.verified;
        result.trusted_continuation_permitted = binding.trusted_continuation_permitted;
        result.diagnostics = binding.diagnostics;
        resolved->push_back(std::move(result));
    }
    return translate(base_status);
}

}  // namespace Cellerator::compiler::sema::field
