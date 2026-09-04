#include <Cellerator/compiler/sema/field/implement_opaque_native_call_barriers_v1.hh>

#include <iostream>

namespace cxx = Cellerator::compiler::frontend::cxx;
namespace field = Cellerator::compiler::sema::field;

int main() {
    int declaration = 0;
    cxx::opaque_native_call_v1 call;
    call.selected_declaration = &declaration;
    call.qualified_name = "legacy::mutate_state";
    call.semantic_barrier = true;
    call.effects = cxx::native_effect_read_v1 | cxx::native_effect_write_v1 |
        cxx::native_effect_escape_v1;
    call.diagnostic = "uncontracted native mutation";

    field::opaque_native_call_barrier_v1 barrier;
    if (field::implement_opaque_native_call_barriers_v1(
            call, 44, {5, 8}, &barrier) != field::opaque_barrier_status_v1::success ||
        !barrier.invalidates_profile_state || !barrier.stops_cross_call_planning ||
        barrier.invalidated_generation_values.size() != 2 ||
        barrier.diagnostic.empty() || barrier.planning_report.find("legacy::mutate_state") ==
            std::string::npos ||
        (barrier.statement.observable_effects & field::field_effect_opaque_v1) == 0) {
        std::cerr << "opaque call did not become an explained field barrier\n";
        return 1;
    }

    field::field_statement_semantics_v1 independent;
    independent.statement_id = 45;
    independent.reads = {20};
    independent.writes = {21};
    if (field::implement_statement_ordering_and_observable_effects_v1(
            barrier.statement, independent).reorder_permitted()) {
        std::cerr << "planning crossed an opaque call barrier\n";
        return 1;
    }

    call.semantic_barrier = false;
    call.contract_applied = true;
    if (field::implement_opaque_native_call_barriers_v1(
            call, 44, {5}, &barrier) != field::opaque_barrier_status_v1::contracted_call) {
        std::cerr << "contracted call was converted to an opaque barrier\n";
        return 1;
    }
    return 0;
}
