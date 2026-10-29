#pragma once

#include <Cellerator/execution/conditional/parameter_store.hh>

namespace cellerator::execution::conditional {

// Pin every exact parameter revision before reading inputs or executing any
// module. Updates/retirement during callbacks cannot replace the versions used
// by this invocation. There is no implicit "latest" revision substitution.
// This helper is synchronous: an asynchronous runtime must retain its own
// leases until all device work/backward readers have completed.
template<class InputStorage, class OutputStorage = InputStorage, class Accumulator = OutputStorage,
         class CombinedOutput = Accumulator, class InputProvider, class ExpertCallback>
std::vector<CombinedOutput> execute_parameterized(
    const DispatchPlan& plan, const ParameterStore& store,
    std::size_t input_features, std::size_t output_features,
    InputProvider&& provider, ExpertCallback&& callback) {
    auto leases = store.snapshot(plan.experts());
    std::size_t next = 0;
    return execute<InputStorage, OutputStorage, Accumulator, CombinedOutput>(
        plan, input_features, output_features, std::forward<InputProvider>(provider),
        [&](const ExpertKey& key, const InputBatch<InputStorage>& inputs, OutputBatch<OutputStorage>& outputs) {
            const auto& lease = leases.at(next++);
            if (lease.key() != key) throw std::logic_error("parameter snapshot does not match dispatch grouping");
            callback(lease.revision(), inputs, outputs);
        });
}

} // namespace cellerator::execution::conditional
