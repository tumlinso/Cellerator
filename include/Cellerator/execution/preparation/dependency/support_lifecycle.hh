#pragma once

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::dependency {

enum class SupportLifecycleStatus : std::uint8_t {
    success = 0u,
    invalid_epoch,
    stale_epoch,
    live_forward_tapes,
    no_live_forward_tapes,
};

// Host publication gate for immutable support snapshots. A replacement epoch
// can publish only after every forward tape that pins the current epoch retires.
struct SupportSnapshotLifecycle {
    std::uint64_t current_structure_epoch = 0u;
    std::uint64_t live_forward_tapes = 0u;

    SupportLifecycleStatus pin_forward_tape(std::uint64_t expected_epoch) noexcept;
    SupportLifecycleStatus retire_forward_tape(std::uint64_t expected_epoch) noexcept;
    SupportLifecycleStatus publish_structure(std::uint64_t next_epoch) noexcept;
};

// Mutable execution generations are distinct from immutable topology and
// invalidate only the caches whose reads depend on each generation.
struct SupportRuntimeGenerations {
    std::uint64_t structure_epoch = 0u;
    std::uint64_t state_generation = 0u;
    std::uint64_t parameter_generation = 0u;
};

static_assert(std::is_trivially_copyable<SupportRuntimeGenerations>::value,
              "runtime generations are compact values");

} // namespace cellerator::execution::dependency
