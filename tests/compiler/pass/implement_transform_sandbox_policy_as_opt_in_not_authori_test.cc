#include <Cellerator/compiler/pass/implement_transform_sandbox_policy_as_opt_in_not_authori_v1.hh>

#include <cassert>

namespace cp = cellerator::compiler::pass::v1;

namespace {
cp::transform_observation_v1 observe(void* data) noexcept {
    return *static_cast<cp::transform_observation_v1*>(data);
}
bool verify(void*) noexcept { return true; }
}

int main() {
    for (const auto mode : {cp::transform_execution_mode_v1::trusted_in_process,
             cp::transform_execution_mode_v1::isolated_subprocess}) {
        for (const auto failure : {cp::transform_observation_v1::crashed,
                 cp::transform_observation_v1::timed_out}) {
            auto observed = failure;
            const auto stopped = cp::execute_transform_with_policy_v1(
                {mode, 10, 4096, false}, observe, nullptr, &observed);
            assert(stopped.executed_mode == mode && stopped.observation == failure);
            assert(!stopped.continuation_allowed);
            const auto expert = cp::execute_transform_with_policy_v1(
                {mode, 10, 4096, true}, observe, nullptr, &observed);
            assert(expert.continuation_allowed);
        }
    }
    auto success = cp::transform_observation_v1::success;
    const auto checked = cp::execute_transform_with_policy_v1(
        {cp::transform_execution_mode_v1::isolated_verified, 10, 4096, false},
        observe, verify, &success);
    assert(checked.isolated && checked.verified
        && checked.observation == cp::transform_observation_v1::success);
}
