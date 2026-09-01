#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const auto compatible = resume::make_status_v1(
        resume::compatibility_code_v1::compatible,
        resume::lowering_stage_v1::target_cover);
    assert(compatible);
    assert(compatible.earliest_compatible_stage ==
        resume::lowering_stage_v1::target_cover);

    const auto stale_values = resume::make_status_v1(
        resume::compatibility_code_v1::value_generation_stale,
        resume::lowering_stage_v1::executable_recipe, 9u);
    assert(!stale_values);
    assert(stale_values.earliest_compatible_stage ==
        resume::lowering_stage_v1::packed_operand);
    assert(stale_values.detail == 9u);

    const auto wrong_target = resume::make_status_v1(
        resume::compatibility_code_v1::target_mismatch,
        resume::lowering_stage_v1::local_realization);
    assert(wrong_target.earliest_compatible_stage ==
        resume::lowering_stage_v1::physical_projection);
}
