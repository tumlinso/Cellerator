#include <Cellerator/compiler/backend/nvcc/integrate_asynchronous_readiness_and_streams_v1.hh>

#include <set>
#include <tuple>

namespace cellerator::compiler::backend::nvcc::v1 {

std::optional<asynchronous_stage_plan> lower_stage_dependencies(
    const std::vector<symbolic_stage_dependency>& dependencies,
    std::uint32_t launch_stream,
    std::uint64_t result_structure_epoch,
    std::uint64_t result_generation) {
    if (result_structure_epoch == 0 || result_generation == 0) {
        return std::nullopt;
    }

    asynchronous_stage_plan result;
    std::set<std::tuple<std::uint32_t, std::uint64_t, std::uint64_t>> waits;
    for (const auto& dependency : dependencies) {
        if (dependency.structure_epoch == 0 || dependency.generation == 0 ||
            dependency.consumer_stream != launch_stream) {
            return std::nullopt;
        }
        if (dependency.producer_stream == dependency.consumer_stream) {
            ++result.elided_same_stream_waits;
            continue;
        }
        const auto key = std::make_tuple(dependency.producer_stream,
                                         dependency.structure_epoch,
                                         dependency.generation);
        if (waits.insert(key).second) {
            result.actions.push_back({
                asynchronous_action_kind::wait_for_generation,
                launch_stream,
                dependency.structure_epoch,
                dependency.generation});
        }
    }
    result.actions.push_back({asynchronous_action_kind::launch_stage,
                              launch_stream, 0, 0});
    result.actions.push_back({asynchronous_action_kind::publish_generation,
                              launch_stream, result_structure_epoch,
                              result_generation});
    return result;
}

} // namespace cellerator::compiler::backend::nvcc::v1
