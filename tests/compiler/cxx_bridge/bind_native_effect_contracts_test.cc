#include <Cellerator/compiler/frontend/cxx/bind_native_effect_contracts_v1.hh>

#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;

int main() {
    int declaration = 0;
    cxx::overload_semantic_candidate_v1 function;
    function.selected_declaration = &declaration;
    function.qualified_name = "model::reference_step";
    cxx::native_effect_contract_v1 claimed{
        "model::reference_step", cxx::native_reads_value_v1, true, true, true};

    cxx::native_effect_observer_v1 observer;
    // Checked-mode instrumentation around a deliberately dishonest reference function.
    observer.observe(cxx::native_reads_value_v1 | cxx::native_writes_value_v1 |
                     cxx::native_writes_topology_v1);
    observer.observe_nondeterminism();
    observer.observe_alias();
    std::vector<cxx::bound_native_effect_contract_v1> bindings;
    if (cxx::bind_native_effect_contracts_v1(
            cxx::native_effect_contract_schema_version_v1,
            {function}, {claimed}, {observer.result()}, true, &bindings) !=
            cxx::native_effect_contract_status_v1::contract_violation ||
        bindings.size() != 1 || bindings[0].verified ||
        !bindings[0].trusted_continuation_permitted || bindings[0].diagnostics.size() != 4) {
        std::cerr << "false effect contract was not detected with trusted continuation\n";
        return 1;
    }

    cxx::native_effect_observer_v1 honest_observer;
    honest_observer.observe(cxx::native_reads_value_v1);
    claimed.pure = false;
    if (cxx::bind_native_effect_contracts_v1(
            cxx::native_effect_contract_schema_version_v1,
            {function}, {claimed}, {honest_observer.result()}, false, &bindings) !=
            cxx::native_effect_contract_status_v1::success ||
        !bindings[0].verified || bindings[0].trusted_continuation_permitted) {
        return 1;
    }
    return 0;
}
