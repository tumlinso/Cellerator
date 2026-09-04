#include <Cellerator/compiler/backend/nvcc/integrate_asynchronous_readiness_and_streams_v1.hh>

#include <cassert>
#include <vector>

int main() {
    using namespace cellerator::compiler::backend::nvcc::v1;

    const std::vector<symbolic_stage_dependency> dependencies{
        {0, 1, 8, 3}, {0, 1, 8, 3}, {1, 1, 8, 4}};
    const auto plan = lower_stage_dependencies(dependencies, 1, 8, 5);
    assert(plan);
    assert(plan->graph_compatible);
    assert(plan->elided_same_stream_waits == 1);
    assert(plan->actions.size() == 3);
    assert(plan->actions[0].kind ==
           asynchronous_action_kind::wait_for_generation);
    assert(plan->actions[0].stream == 1);
    assert(plan->actions[1].kind == asynchronous_action_kind::launch_stage);
    assert(plan->actions[2].kind ==
           asynchronous_action_kind::publish_generation);
    assert(plan->actions[2].generation == 5);

    assert(!lower_stage_dependencies({{0, 2, 8, 3}}, 1, 8, 5));
    assert(!lower_stage_dependencies({}, 1, 0, 5));
}
