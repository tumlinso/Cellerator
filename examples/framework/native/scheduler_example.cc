#include <Cellerator/execution/conditional/scheduler.hh>
#include <iostream>

using namespace cellerator::execution::conditional;

int main() {
    // Two abstract execution devices and a storage-only node. These are cost
    // metadata, not hardware discovery. Numerical callbacks below run on CPU.
    HardwareTopology topology({{0, 8192, 1, false}, {1, 4096}, {2, 4096}},
        {{0, 1, 1000, 0, false}, {0, 2, 1, 0, false}, {1, 2, 1000, 0, true}});
    const ExpertKey expert{12, 7};
    auto parameter = std::make_shared<float>(3);
    ParameterBuffers buffers;
    buffers.parameters = {parameter, parameter.get(), sizeof(float), "float32", "host"};
    ParameterStore store;
    store.register_initial(expert, buffers);
    ResidencySnapshot residency{1, {{expert, 0, sizeof(float), ResidencyState::ready, false}}, {}};
    PlacementPlanner planner(topology, {{expert, sizeof(float), .001}}, residency);
    DispatchPlan selected({{expert, {1, 100, 2}, .5}, {expert, {2, 200, 4}, 2},
                           {expert, {1, 100, 2}, .5}});
    auto plan = planner.plan(selected, [](const Origin& origin) { return origin.rank; },
                             sizeof(float), sizeof(float), 9, {{{expert, 1}, 5}});
    std::vector<float> outputs(selected.size());
    PlannedExecution execution(plan, store, 9, 1);
    std::vector<bool> started(plan.decisions.size(), false);
    auto run = [&](std::size_t index, const DispatchDecision& decision,
                   const DispatchPlan& work, const ParameterLease& lease) {
        // A real backend retains physical residency pins and starts copies or a
        // CUDA kernel here. This example uses a synthetic event to show resume.
        if (!started[index]) { started[index] = true; return TaskPoll::yield(100 + index); }
        const auto scalar = *static_cast<const float*>(lease.buffers().parameters.data);
        for (auto occurrence : decision.demand.selection_indices)
            outputs[work.selection_to_dispatch()[occurrence]] =
                scalar * static_cast<float>(work.selections()[occurrence].origin.row);
        return TaskPoll::done();
    };
    while (!execution.done()) {
        execution.advance(run, 1);
        if (!execution.done() && execution.states()[execution.completed_decisions()] == TaskState::yielded)
            execution.signal(100 + execution.completed_decisions());
    }
    execution.close();
    const auto combined = combine<float>(selected, outputs, 1);
    if (combined != std::vector<float>{6, 24}) return 1;
    std::cout << "exact revision " << expert.revision << ": " << combined[0] << ", " << combined[1]
              << "; " << plan.decisions.size() << " placement decisions, "
              << plan.proposed_replicas.size() << " proposed replicas\n";
}
