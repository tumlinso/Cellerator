#pragma once

#include <cstdint>
#include <optional>
#include <vector>

namespace cellerator::compiler::backend::nvcc::v1 {

struct symbolic_stage_dependency {
    std::uint32_t producer_stream = 0;
    std::uint32_t consumer_stream = 0;
    std::uint64_t structure_epoch = 0;
    std::uint64_t generation = 0;
};

enum class asynchronous_action_kind : std::uint8_t {
    wait_for_generation = 0,
    launch_stage,
    publish_generation,
};

struct asynchronous_action {
    asynchronous_action_kind kind = asynchronous_action_kind::launch_stage;
    std::uint32_t stream = 0;
    std::uint64_t structure_epoch = 0;
    std::uint64_t generation = 0;
};

struct asynchronous_stage_plan {
    std::vector<asynchronous_action> actions;
    std::uint32_t elided_same_stream_waits = 0;
    bool graph_compatible = true;
};

// Produces stream-ordered actions only: it never emits host or device
// synchronization, and the caller retains ownership of every stream/event.
[[nodiscard]] std::optional<asynchronous_stage_plan> lower_stage_dependencies(
    const std::vector<symbolic_stage_dependency>& dependencies,
    std::uint32_t launch_stream,
    std::uint64_t result_structure_epoch,
    std::uint64_t result_generation);

} // namespace cellerator::compiler::backend::nvcc::v1
