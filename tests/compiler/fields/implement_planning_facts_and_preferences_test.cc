#include <Cellerator/compiler/sema/field/implement_planning_facts_and_preferences_v1.hh>

#include <iostream>

namespace field = Cellerator::compiler::sema::field;

int main() {
    using kind = field::planning_fact_or_preference_kind_v1;
    const std::vector<field::planning_fact_or_preference_v1> hints{
        {1, kind::reuse, 64.0, 0, true, true},
        {2, kind::memory, 2.0, 0, false, true},
        {3, kind::latency, 1.0, 0, false, true},
        {4, kind::latency, 4.0, 0, false, true},
        {5, kind::throughput, 3.0, 0, false, true},
        {6, kind::compilation_budget, 10.0, 0, true, true},
        {7, kind::target_preference, 1.0, 70, false, true},
        {8, kind::graph_capture, 1.0, 0, false, false},
        {9, kind::canonical_output, 1.0, 0, false, true},
    };

    field::planning_facts_and_preferences_v1 resolved;
    if (field::implement_planning_facts_and_preferences_v1(hints, &resolved) !=
            field::planning_facts_and_preferences_status_v1::success ||
        resolved.hints.size() != hints.size() || resolved.applied_count != 7 ||
        resolved.ignored_count != 1 || resolved.dominated_count != 1 ||
        resolved.hints[2].disposition != field::planning_hint_disposition_v1::dominated ||
        resolved.hints[2].diagnostic.empty() ||
        resolved.hints[3].disposition != field::planning_hint_disposition_v1::applied ||
        resolved.hints[7].disposition != field::planning_hint_disposition_v1::ignored ||
        resolved.hints[7].diagnostic.empty()) {
        std::cerr << "planning facts and preferences were not resolved visibly\n";
        return 1;
    }

    // These records influence plan scoring only; none carries an operation result
    // or a replacement mathematical expression.
    return 0;
}
