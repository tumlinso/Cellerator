#pragma once

#include <cstdint>

namespace cellerator::execution::program {

enum class program_status : std::uint32_t {
    success = 0, invalid_argument, invalid_stage_graph, insufficient_bindings,
    launch_failed
};

struct launch_binding_v2 {
    const void* input = nullptr;
    void* output = nullptr;
    const void* values = nullptr;
    void* workspace = nullptr;
    std::uint64_t workspace_bytes = 0;
};

using stage_launch_v2 = program_status (*)(
        const void* prepared_state,
        const launch_binding_v2& binding,
        void* caller_stream) noexcept;

struct prepared_stage_v2 {
    std::uint64_t stable_stage_id = 0;
    std::uint64_t candidate_id = 0;
    const void* prepared_state = nullptr;
    stage_launch_v2 launch = nullptr;
    std::uint64_t first_dependency = 0;
    std::uint32_t dependency_count = 0;
    std::uint32_t binding_index = 0;
    std::uint64_t required_workspace_bytes = 0;
};

struct prepared_program_v2 {
    std::uint32_t version = 2;
    std::uint32_t flags = 0;
    const prepared_stage_v2* stages = nullptr;
    std::uint64_t stage_count = 0;
    const std::uint64_t* dependencies = nullptr;
    std::uint64_t dependency_count = 0;
};

program_status validate_prepared_program_v2(
        const prepared_program_v2& program) noexcept;

program_status execute_prepared_program_v2(
        const prepared_program_v2& program,
        const launch_binding_v2* bindings,
        std::uint64_t binding_count,
        void* caller_stream) noexcept;

}  // namespace cellerator::execution::program
