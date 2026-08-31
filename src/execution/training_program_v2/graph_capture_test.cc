#include <Cellerator/execution/training_program_v2/graph_capture.hh>

#include <array>
#include <cassert>

using namespace cellerator::execution;
using namespace cellerator::execution::training_v2;

namespace {

axis_identity axis(std::uint32_t value) {
    return {{value, 1u}, {value + 1u, 1u}, {value + 2u, 1u},
        {value + 3u, 1u}};
}

} // namespace

int main() {
    std::array<training_stage_v2, 3> stages{};
    std::array<std::uint64_t, 3> stage_ids{{10u, 20u, 30u}};
    for (std::size_t index = 0u; index < stages.size(); ++index) {
        stages[index].stage_identity = stage_ids[index];
        stages[index].candidate_identity = stage_ids[index] + 1u;
        stages[index].input_axis = axis(1u);
        stages[index].output_axis = axis(10u);
    }
    training_program_v2 program{};
    program.stage_count = static_cast<std::uint32_t>(stages.size());
    program.stages = stages.data();
    program.program_identity = 40u;
    program.structure = {41u, 1u};
    program.epoch = {42u};
    program.prepared_generation = {43u};
    program.graph_capture_required = true;
    const training_graph_capture_v2 capture{40u, {41u, 1u}, {42u}, {43u},
        stage_ids.size(), stage_ids.data(), 44u, true, true, true, false, {}};
    graph_capture_receipt_v2 receipt{};
    assert(validate_training_graph_capture_v2(capture, program, receipt));

    std::array<float, 4> first{};
    std::array<float, 4> second{};
    const training_graph_launch_binding_v2 before{{43u}, first.data(),
        first.data(), first.data(), first.data(), first.data(), first.data(),
        sizeof(first), 50u};
    const training_graph_launch_binding_v2 after{{43u}, second.data(),
        second.data(), second.data(), second.data(), second.data(), second.data(),
        sizeof(second), 51u};
    const caller_update_policy_binding_v2 policy{
        60u, 61u, first.data(), sizeof(first)};
    assert(validate_training_graph_rebind_v2(
        capture, before, after, policy, receipt));
    assert(receipt.pointers_rebound && receipt.stream_rebound
        && !receipt.reprepare_required && receipt.update_policy_separate);

    auto stale = after;
    stale.generation = {44u};
    assert(!validate_training_graph_rebind_v2(
        capture, before, stale, policy, receipt));
}
