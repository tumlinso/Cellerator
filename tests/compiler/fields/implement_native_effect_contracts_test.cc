#include <Cellerator/compiler/sema/field/implement_native_effect_contracts_v1.hh>

#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;
namespace field = Cellerator::compiler::sema::field;

int main() {
    int declaration = 0;
    cxx::overload_semantic_candidate_v1 function;
    function.selected_declaration = &declaration;
    function.qualified_name = "model::publish_values";

    field::field_native_effect_contract_v1 contract;
    contract.effects.qualified_name = function.qualified_name;
    contract.effects.permitted_effects = cxx::native_reads_value_v1 |
        cxx::native_writes_value_v1;
    contract.effects.deterministic = true;
    contract.effects.no_alias = true;
    contract.publications = field::publishes_value_generation_v1;
    contract.target = field::native_target_behavior_v1::host_and_device;

    cxx::observed_native_effects_v1 observed;
    observed.effects = contract.effects.permitted_effects;
    std::vector<field::resolved_native_function_effects_v1> resolved;
    if (field::implement_native_effect_contracts_v1(
            cxx::native_effect_contract_schema_version_v1, {function}, {contract},
            {observed}, false, &resolved) != field::field_native_effect_status_v1::success ||
        resolved.size() != 1 || !resolved[0].verified ||
        resolved[0].publications != field::publishes_value_generation_v1 ||
        resolved[0].target != field::native_target_behavior_v1::host_and_device) {
        std::cerr << "verified native effect contract was not attached\n";
        return 1;
    }

    observed.effects |= cxx::native_writes_topology_v1;
    if (field::implement_native_effect_contracts_v1(
            cxx::native_effect_contract_schema_version_v1, {function}, {contract},
            {observed}, true, &resolved) !=
            field::field_native_effect_status_v1::contract_violation ||
        resolved[0].verified || !resolved[0].trusted_continuation_permitted ||
        resolved[0].diagnostics.empty()) {
        std::cerr << "verified failure did not preserve trusted continuation\n";
        return 1;
    }
    return 0;
}
