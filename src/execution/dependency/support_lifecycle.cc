#include <Cellerator/execution/dependency/support_lifecycle.hh>

#include <limits>

namespace cellerator::execution::dependency {

SupportLifecycleStatus SupportSnapshotLifecycle::pin_forward_tape(
    std::uint64_t expected_epoch) noexcept {
    if (expected_epoch == 0u || current_structure_epoch == 0u)
        return SupportLifecycleStatus::invalid_epoch;
    if (expected_epoch != current_structure_epoch)
        return SupportLifecycleStatus::stale_epoch;
    if (live_forward_tapes == std::numeric_limits<std::uint64_t>::max())
        return SupportLifecycleStatus::invalid_epoch;
    ++live_forward_tapes;
    return SupportLifecycleStatus::success;
}

SupportLifecycleStatus SupportSnapshotLifecycle::retire_forward_tape(
    std::uint64_t expected_epoch) noexcept {
    if (expected_epoch == 0u || current_structure_epoch == 0u)
        return SupportLifecycleStatus::invalid_epoch;
    if (expected_epoch != current_structure_epoch)
        return SupportLifecycleStatus::stale_epoch;
    if (live_forward_tapes == 0u)
        return SupportLifecycleStatus::no_live_forward_tapes;
    --live_forward_tapes;
    return SupportLifecycleStatus::success;
}

SupportLifecycleStatus SupportSnapshotLifecycle::publish_structure(
    std::uint64_t next_epoch) noexcept {
    if (next_epoch == 0u || current_structure_epoch == 0u)
        return SupportLifecycleStatus::invalid_epoch;
    if (live_forward_tapes != 0u)
        return SupportLifecycleStatus::live_forward_tapes;
    if (next_epoch <= current_structure_epoch)
        return SupportLifecycleStatus::stale_epoch;
    current_structure_epoch = next_epoch;
    return SupportLifecycleStatus::success;
}

} // namespace cellerator::execution::dependency
