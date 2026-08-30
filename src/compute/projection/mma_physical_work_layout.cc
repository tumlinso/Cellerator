#include <Cellerator/compute/projection/physical_mma_hybrid.hh>
#include <Cellerator/geometry/work_layout.hh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::projection {

// Build a provider-local padded schedule over portable execution positions.
// The portable work layout is read only: sentinels exist solely in the
// physical_to_execution array and never enter either semantic permutation.
bool build_mma_physical_work_layout_v1(
    const geometry::work_layout_view_v1 &semantic,
    std::uint32_t group_width,
    std::uint32_t *physical_to_execution,
    std::uint32_t physical_capacity,
    std::uint32_t *execution_to_physical,
    std::uint32_t inverse_capacity,
    std::uint32_t *physical_count) noexcept {
    if (physical_count == nullptr || group_width == 0u
        || group_width > mma_group_extent_limit_v1
        || semantic.schema_version != geometry::work_layout_schema_version
        || semantic.reserved != 0u || semantic.work_count == 0u
        || semantic.execution_to_window == nullptr
        || semantic.window_to_execution == nullptr
        || execution_to_physical == nullptr
        || inverse_capacity < semantic.work_count)
        return false;

    for (std::uint32_t execution = 0u; execution < semantic.work_count;
        ++execution) {
        const std::uint32_t window = semantic.execution_to_window[execution];
        if (window >= semantic.work_count
            || semantic.window_to_execution[window] != execution)
            return false;
    }

    const std::uint32_t remainder = semantic.work_count % group_width;
    const std::uint32_t padding = remainder == 0u ? 0u : group_width - remainder;
    if (semantic.work_count
        > std::numeric_limits<std::uint32_t>::max() - padding)
        return false;
    const std::uint32_t count = semantic.work_count + padding;
    if (physical_to_execution == nullptr || physical_capacity < count)
        return false;

    for (std::uint32_t physical = 0u; physical < semantic.work_count;
        ++physical) {
        physical_to_execution[physical] = physical;
        execution_to_physical[physical] = physical;
    }
    for (std::uint32_t physical = semantic.work_count; physical < count;
        ++physical)
        physical_to_execution[physical] = geometry::invalid_work_item;
    *physical_count = count;
    return true;
}

} // namespace cellerator::compute::projection
