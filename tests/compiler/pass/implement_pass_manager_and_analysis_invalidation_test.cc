#include <Cellerator/compiler/pass/implement_pass_manager_and_analysis_invalidation_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

namespace {
bool produce(cp::pass_context_v1& context, cp::pass_result_v1& result) noexcept {
    assert(context.scope_depth == 1);
    result.produced_analyses = 1;
    return true;
}
bool mutate(cp::pass_context_v1& context, cp::pass_result_v1& result) noexcept {
    assert(context.scope_depth == 2);
    result.changed = true;
    result.preserved_analyses = 0;
    result.produced_analyses = 2;
    result.diagnostic = "rewrote module";
    return true;
}
bool consume(cp::pass_context_v1&, cp::pass_result_v1&) noexcept { return true; }
bool fail(cp::pass_context_v1&, cp::pass_result_v1& result) noexcept {
    result.diagnostic = "deliberate failure";
    return false;
}
bool cancelled(void* value) noexcept { return *static_cast<bool*>(value); }
}  // namespace

int main() {
    const std::vector<cp::pass_descriptor_v1> pipeline{
        {"produce", {cp::pipeline_phase_v1::discovery,
            cp::interception_side_v1::before}, 1, 0, produce},
        {"mutate", {cp::pipeline_phase_v1::discovery,
            cp::interception_side_v1::after}, 2, 1, mutate},
        {"consume", {cp::pipeline_phase_v1::certification,
            cp::interception_side_v1::before}, 0, 2, consume}};
    const auto first = cp::run_pass_pipeline_v1(pipeline, {});
    const auto replay = cp::run_pass_pipeline_v1(pipeline, {});
    assert(first.status == cp::pass_status_v1::success);
    assert(first.ordered_passes.size() == 3 && first.final_analyses == 2);
    assert(first.deterministic_replay_hash == replay.deterministic_replay_hash);
    assert(first.diagnostics.size() == 1);

    auto missing = pipeline;
    missing[2].required_analyses = 1;
    assert(cp::run_pass_pipeline_v1(missing, {}).status
        == cp::pass_status_v1::missing_required_analysis);
    auto failure = pipeline;
    failure[2].run = fail;
    assert(cp::run_pass_pipeline_v1(failure, {}).status
        == cp::pass_status_v1::pass_failed);
    bool stop = true;
    assert(cp::run_pass_pipeline_v1(pipeline, {}, cancelled, &stop).status
        == cp::pass_status_v1::cancelled);
}
