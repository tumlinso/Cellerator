#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    resume::resumption_trace_v1 trace{};
    const auto compatible = resume::make_status_v1(
        resume::compatibility_code_v1::compatible,
        resume::lowering_stage_v1::local_realization);
    assert(resume::instrument_resumption_decision_v1(
        resume::lowering_stage_v1::local_realization, compatible, &trace));
    assert(!trace.fallback_used);
    assert(trace.selected_stage == resume::lowering_stage_v1::local_realization);
    assert(trace.bypassed_phase_count == 7u);
    assert(trace.replayed_phase_count == 0u);

    const auto stale = resume::make_status_v1(
        resume::compatibility_code_v1::value_generation_stale,
        resume::lowering_stage_v1::executable_recipe);
    assert(resume::instrument_resumption_decision_v1(
        resume::lowering_stage_v1::executable_recipe, stale, &trace));
    assert(trace.fallback_used);
    assert(trace.selected_stage == resume::lowering_stage_v1::packed_operand);
    assert(trace.bypassed_phase_count == 5u);
    assert(trace.replayed_phase_count == 1u);
    assert(trace.compatibility ==
        resume::compatibility_code_v1::value_generation_stale);
}
