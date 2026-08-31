#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::geometry {

// Maps a work-window slice to a prepared component dispatch without imposing a
// fixed number of windows, candidates, or chunks.
struct work_layout_entry_v2 {
    std::uint64_t component_identity = 0u;
    std::uint64_t aggregate_work_begin = 0u;
    std::uint64_t window_component_position = 0u;
    std::uint64_t dispatch_component_position = 0u;
    std::uint32_t local_begin = 0u;
    std::uint32_t local_count = 0u;
};

struct work_layout_view_v2 {
    std::uint64_t layout_identity = 0u;
    std::uint64_t input_order_identity = 0u;
    std::uint64_t output_order_identity = 0u;
    std::uint64_t aggregate_work_count = 0u;
    const work_layout_entry_v2 *entries = nullptr;
    std::uint64_t entry_count = 0u;
};

enum class work_layout_status_v2 : std::uint32_t {
    valid = 0u,
    null_pointer,
    component_order,
    aggregate_discontinuity,
    arithmetic_overflow,
    aggregate_extent_mismatch,
};

struct work_layout_validation_v2 {
    work_layout_status_v2 status = work_layout_status_v2::valid;
    std::uint32_t reserved = 0u;
    std::uint64_t entry = 0u;
    std::uint64_t operations = 0u;
};

inline work_layout_validation_v2 validate_work_layout_v2(
    const work_layout_view_v2 &layout) noexcept {
    work_layout_validation_v2 result{};
    if (layout.entry_count != 0u && layout.entries == nullptr) {
        result.status = work_layout_status_v2::null_pointer;
        return result;
    }
    std::uint64_t aggregate = 0u;
    std::uint64_t previous_identity = 0u;
    for (std::uint64_t index = 0u; index < layout.entry_count; ++index) {
        const auto &entry = layout.entries[index];
        ++result.operations;
        if (index != 0u && entry.component_identity < previous_identity) {
            result.status = work_layout_status_v2::component_order;
            result.entry = index;
            return result;
        }
        if (entry.aggregate_work_begin != aggregate) {
            result.status = work_layout_status_v2::aggregate_discontinuity;
            result.entry = index;
            return result;
        }
        if (entry.local_count >
            std::numeric_limits<std::uint64_t>::max() - aggregate) {
            result.status = work_layout_status_v2::arithmetic_overflow;
            result.entry = index;
            return result;
        }
        aggregate += entry.local_count;
        previous_identity = entry.component_identity;
    }
    if (aggregate != layout.aggregate_work_count) {
        result.status = work_layout_status_v2::aggregate_extent_mismatch;
        result.entry = layout.entry_count;
    }
    return result;
}

static_assert(std::is_trivially_copyable_v<work_layout_entry_v2>);
static_assert(std::is_trivially_copyable_v<work_layout_view_v2>);

}  // namespace cellerator::geometry
