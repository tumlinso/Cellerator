#include <Cellerator/compiler/ir/common/implement_validation_mode_plumbing_v1.hh>

#include <array>
#include <cassert>

using namespace cellerator::compiler::ir;

int main() {
    constexpr std::array modes{trust_mode::verified, trust_mode::checked,
        trust_mode::trusted, trust_mode::unsafe, trust_mode::unchecked};
    for (const auto mode : modes) {
        validation_envelope envelope{mode, pipeline_stage::parser, true, false};
        for (const auto stage : {pipeline_stage::builder, pipeline_stage::pass,
                 pipeline_stage::serializer, pipeline_stage::backend}) {
            envelope = advance_validation(envelope, stage);
            assert(envelope.mode == mode && envelope.stage == stage);
        }
        const auto decision = evaluate_validation(envelope);
        const bool requires_checks = mode == trust_mode::verified || mode == trust_mode::checked;
        assert(decision.continue_pipeline != requires_checks);
        trust_mode decoded{};
        assert(deserialize_trust_mode(serialize_trust_mode(mode), decoded));
        assert(decoded == mode && !trust_mode_name(mode).empty());
        envelope.structurally_parseable = false;
        assert(!evaluate_validation(envelope).continue_pipeline);
    }
    trust_mode unused{};
    assert(!deserialize_trust_mode(255u, unused));
}
