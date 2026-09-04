#include <Cellerator/compiler/pass/implement_complete_built_in_stage_replacement_v1.hh>

#include <array>
#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

namespace {
bool built_in(cp::stage_replacement_context_v1& context) noexcept {
    ++*static_cast<int*>(context.stage_state);
    context.diagnostic = "built-in";
    return true;
}
bool replacement(cp::stage_replacement_context_v1& context) noexcept {
    *static_cast<int*>(context.stage_state) += 10;
    context.diagnostic = "replacement";
    return true;
}
bool rejected(cp::stage_replacement_context_v1& context) noexcept {
    context.diagnostic = "not supported";
    return false;
}
}

int main() {
    constexpr std::array replaceable{
        cp::pipeline_phase_v1::profile_propagation,
        cp::pipeline_phase_v1::discovery,
        cp::pipeline_phase_v1::certification,
        cp::pipeline_phase_v1::decomposition,
        cp::pipeline_phase_v1::candidate_enumeration,
        cp::pipeline_phase_v1::cost_modeling,
        cp::pipeline_phase_v1::selection,
        cp::pipeline_phase_v1::realization,
        cp::pipeline_phase_v1::backend_emission,
    };
    for (const auto phase : replaceable) {
        int state = 0;
        const auto receipt = cp::run_stage_replacement_v1(
            {phase, built_in, replacement,
                cp::stage_replacement_policy_v1::force_replacement, &state});
        assert(receipt.status == cp::stage_replacement_status_v1::success);
        assert(receipt.replacement_selected && !receipt.fallback_used);
        assert(state == 10);
    }
    int fallback_state = 0;
    const auto fallback = cp::run_stage_replacement_v1(
        {cp::pipeline_phase_v1::discovery, built_in, rejected,
            cp::stage_replacement_policy_v1::prefer_replacement_with_fallback,
            &fallback_state});
    assert(fallback.status == cp::stage_replacement_status_v1::success);
    assert(fallback.fallback_used && fallback_state == 1);

    int forced_state = 0;
    const auto forced = cp::run_stage_replacement_v1(
        {cp::pipeline_phase_v1::discovery, built_in, rejected,
            cp::stage_replacement_policy_v1::force_replacement, &forced_state});
    assert(forced.status == cp::stage_replacement_status_v1::replacement_failed);
    assert(!forced.fallback_used && forced_state == 0);
}
