#pragma once

#include "Cellerator/execution/program/program_v2.h"

#include <cstdint>

namespace cellerator::execution::program {

struct legacy_program_entry_v1 {
    std::uint64_t stable_stage_id = 0;
    std::uint64_t candidate_id = 0;
    const void* prepared_state = nullptr;
    stage_launch_v2 launch = nullptr;
    std::uint32_t binding_index = 0;
    std::uint32_t reserved = 0;
    std::uint64_t required_workspace_bytes = 0;
};

struct legacy_program_v1 {
    const legacy_program_entry_v1* entries = nullptr;
    std::uint32_t entry_count = 0;
};

// Deprecated source adapter only. It creates v2 state in caller storage;
// execution always flows through execute_prepared_program_v2.
program_status adapt_legacy_program_v1(
        const legacy_program_v1& legacy,
        prepared_stage_v2* stage_storage,
        std::uint64_t stage_capacity,
        std::uint64_t* dependency_storage,
        std::uint64_t dependency_capacity,
        prepared_program_v2* program) noexcept;

}  // namespace cellerator::execution::program
