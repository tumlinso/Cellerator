#include <Cellerator/compiler/frontend/cxx/model_opaque_native_calls_v1.hh>

#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;

int main() {
    int unknown_declaration = 0;
    int pure_declaration = 0;
    std::vector<cxx::overload_semantic_candidate_v1> calls(2);
    calls[0].selected_declaration = &unknown_declaration;
    calls[0].qualified_name = "model::uncontracted_step";
    calls[0].mechanism = "ordinary-cxx-call";
    calls[1].selected_declaration = &pure_declaration;
    calls[1].qualified_name = "model::lookup";
    calls[1].mechanism = "ordinary-cxx-call";
    std::vector<cxx::native_call_contract_v1> contracts{
        {"model::lookup", cxx::native_effect_read_v1},
    };
    std::vector<cxx::opaque_native_call_v1> models;
    if (cxx::model_opaque_native_calls_v1(
            cxx::opaque_native_call_schema_version_v1,
            calls, contracts, &models) != cxx::opaque_native_call_status_v1::success ||
        models.size() != 2) return 1;
    const auto conservative = cxx::native_effect_read_v1 | cxx::native_effect_write_v1 |
        cxx::native_effect_escape_v1 | cxx::native_effect_synchronize_v1;
    if (!models[0].semantic_barrier || models[0].effects != conservative ||
        models[0].diagnostic.find("model::uncontracted_step") == std::string::npos ||
        cxx::may_reorder_across_native_call_v1(models[0], cxx::native_effect_read_v1) ||
        cxx::may_reorder_across_native_call_v1(models[0], cxx::native_effect_write_v1)) {
        std::cerr << "planner moved an effect across an opaque barrier\n";
        return 1;
    }
    if (models[1].semantic_barrier || !models[1].contract_applied ||
        !cxx::may_reorder_across_native_call_v1(models[1], cxx::native_effect_read_v1) ||
        cxx::may_reorder_across_native_call_v1(models[1], cxx::native_effect_write_v1)) {
        std::cerr << "contracted read-only call effect handling failed\n";
        return 1;
    }
    return 0;
}
