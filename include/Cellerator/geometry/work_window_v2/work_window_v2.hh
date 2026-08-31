#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::geometry {

// One bounded slice of a reusable dataset-level component.  Local arithmetic
// is u32, while aggregate work and repetition counts remain u64.
struct work_window_component_v2 {
    std::uint64_t component_identity = 0u;
    std::uint64_t source_skeleton_identity = 0u;
    std::uint64_t aggregate_work_begin = 0u;
    std::uint64_t repetitions = 0u;
    std::uint32_t local_begin = 0u;
    std::uint32_t local_count = 0u;
};

struct work_window_view_v2 {
    std::uint64_t window_identity = 0u;
    std::uint64_t dataset_identity = 0u;
    std::uint64_t aggregate_work_count = 0u;
    const work_window_component_v2 *components = nullptr;
    std::uint64_t component_count = 0u;
};

enum class work_window_status_v2 : std::uint32_t {
    valid = 0u,
    null_pointer,
    component_order,
    aggregate_discontinuity,
    arithmetic_overflow,
    aggregate_extent_mismatch,
    output_too_small,
};

struct work_window_validation_v2 {
    work_window_status_v2 status = work_window_status_v2::valid;
    std::uint32_t reserved = 0u;
    std::uint64_t component = 0u;
    std::uint64_t operations = 0u;
};

inline work_window_validation_v2 validate_work_window_v2(
    const work_window_view_v2 &window) noexcept {
    work_window_validation_v2 result{};
    if (window.component_count != 0u && window.components == nullptr) {
        result.status = work_window_status_v2::null_pointer;
        return result;
    }
    std::uint64_t aggregate = 0u;
    std::uint64_t previous_identity = 0u;
    for (std::uint64_t index = 0u; index < window.component_count; ++index) {
        const auto &component = window.components[index];
        ++result.operations;
        if (index != 0u && component.component_identity <= previous_identity) {
            result.status = work_window_status_v2::component_order;
            result.component = index;
            return result;
        }
        if (component.aggregate_work_begin != aggregate) {
            result.status = work_window_status_v2::aggregate_discontinuity;
            result.component = index;
            return result;
        }
        if (component.local_count != 0u
            && component.repetitions >
                (std::numeric_limits<std::uint64_t>::max() - aggregate)
                    / component.local_count) {
            result.status = work_window_status_v2::arithmetic_overflow;
            result.component = index;
            return result;
        }
        aggregate += static_cast<std::uint64_t>(component.local_count)
            * component.repetitions;
        previous_identity = component.component_identity;
    }
    if (aggregate != window.aggregate_work_count) {
        result.status = work_window_status_v2::aggregate_extent_mismatch;
        result.component = window.component_count;
    }
    return result;
}

// Explicit, allocation-free adapter for legacy u32 component arrays.  It is a
// source migration route only; v2 remains the scalable production view.
struct legacy_work_window_arrays_v1 {
    const std::uint64_t *component_identities = nullptr;
    const std::uint64_t *source_skeleton_identities = nullptr;
    const std::uint32_t *local_begins = nullptr;
    const std::uint32_t *local_counts = nullptr;
    std::uint64_t component_count = 0u;
};

inline work_window_status_v2 upgrade_legacy_work_window_v1(
    const legacy_work_window_arrays_v1 &legacy,
    work_window_component_v2 *output, std::uint64_t output_capacity,
    std::uint64_t *aggregate_work_count,
    std::uint64_t *operations = nullptr) noexcept {
    if (aggregate_work_count == nullptr
        || (legacy.component_count != 0u
            && (legacy.component_identities == nullptr
                || legacy.source_skeleton_identities == nullptr
                || legacy.local_begins == nullptr
                || legacy.local_counts == nullptr
                || output == nullptr))) {
        return work_window_status_v2::null_pointer;
    }
    if (output_capacity < legacy.component_count) {
        return work_window_status_v2::output_too_small;
    }
    std::uint64_t aggregate = 0u;
    for (std::uint64_t index = 0u; index < legacy.component_count; ++index) {
        if (legacy.local_counts[index] >
            std::numeric_limits<std::uint64_t>::max() - aggregate) {
            return work_window_status_v2::arithmetic_overflow;
        }
        output[index].component_identity = legacy.component_identities[index];
        output[index].source_skeleton_identity =
            legacy.source_skeleton_identities[index];
        output[index].aggregate_work_begin = aggregate;
        output[index].repetitions = 1u;
        output[index].local_begin = legacy.local_begins[index];
        output[index].local_count = legacy.local_counts[index];
        aggregate += legacy.local_counts[index];
        if (operations != nullptr) {
            ++*operations;
        }
    }
    *aggregate_work_count = aggregate;
    return work_window_status_v2::valid;
}

static_assert(std::is_trivially_copyable_v<work_window_component_v2>);
static_assert(std::is_trivially_copyable_v<work_window_view_v2>);

}  // namespace cellerator::geometry
