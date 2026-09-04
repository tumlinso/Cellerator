#include <Cellerator/compiler/sema/field/implement_opaque_native_call_barriers_v1.hh>

#include <algorithm>
#include <utility>

namespace Cellerator::compiler::sema::field {

opaque_barrier_status_v1 implement_opaque_native_call_barriers_v1(
    const frontend::cxx::opaque_native_call_v1& call,
    std::uint64_t statement_id,
    const std::vector<semantic_value_id_v1>& affected_values,
    opaque_native_call_barrier_v1* barrier) noexcept {
    if (barrier == nullptr || call.selected_declaration == nullptr ||
        call.qualified_name.empty()) {
        return opaque_barrier_status_v1::invalid_call;
    }
    if (!call.semantic_barrier || call.contract_applied) {
        return opaque_barrier_status_v1::contracted_call;
    }
    if (statement_id == 0) return opaque_barrier_status_v1::invalid_statement;
    if (affected_values.empty() ||
        std::find(affected_values.begin(), affected_values.end(), 0) !=
            affected_values.end()) {
        return opaque_barrier_status_v1::invalid_affected_value;
    }

    opaque_native_call_barrier_v1 result;
    result.statement_id = statement_id;
    result.qualified_name = call.qualified_name;
    result.invalidated_generation_values = affected_values;
    result.statement.statement_id = statement_id;
    result.statement.reads = affected_values;
    result.statement.writes = affected_values;
    result.statement.observable_effects = field_effect_opaque_v1 |
        field_effect_reads_memory_v1 | field_effect_writes_memory_v1 |
        field_effect_synchronizes_v1;
    result.diagnostic = call.diagnostic.empty()
        ? "uncontracted native call is an opaque execution-field barrier"
        : call.diagnostic;
    result.planning_report = "planning stopped across uncontracted call '" +
        call.qualified_name +
        "'; representative profile state and affected value generations invalidated";
    *barrier = std::move(result);
    return opaque_barrier_status_v1::success;
}

}  // namespace Cellerator::compiler::sema::field
