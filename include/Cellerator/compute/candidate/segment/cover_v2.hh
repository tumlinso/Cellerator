#pragma once

#include <Cellerator/compute/candidate/segment/segment_v2.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::compute::segment {

enum class segment_cover_owner_v2 : std::uint8_t {
    mma = 1u,
    residual = 2u
};

// Cold host validation view for a cover-native segmented value order. Only
// occupied biological slots appear here: padding holes have no logical edge ID
// and therefore can never enter a segment or mutable value plane.
struct segment_cover_native_host_view_v2 {
    const std::uint64_t *offsets = nullptr;
    std::uint64_t offset_count = 0u;
    const std::uint64_t *local_to_global_value = nullptr;
    const segment_cover_owner_v2 *owners = nullptr;
    std::uint64_t occupied_value_count = 0u;
    std::uint64_t physical_slot_count = 0u;
    std::uint64_t mma_occupied_count = 0u;
    std::uint64_t residual_occupied_count = 0u;
};

struct segment_cover_validation_workspace_v2 {
    std::uint8_t *ownership_marks = nullptr;
    std::uint64_t ownership_mark_bytes = 0u;
};

struct segment_cover_validation_receipt_v2 {
    std::uint64_t exact_logical_values = 0u;
    std::uint64_t occupied_physical_values = 0u;
    std::uint64_t physical_holes = 0u;
    std::uint64_t mma_values = 0u;
    std::uint64_t residual_values = 0u;
    bool exact_disjoint_ownership = false;
    bool holes_are_non_biological = false;
    std::uint8_t reserved[6]{};
};

segment_result_v2 validate_segment_cover_native_partition_v2_host(
    const segment_plan_v2 &plan,
    const segment_cover_native_host_view_v2 &cover,
    const segment_cover_validation_workspace_v2 &workspace,
    segment_cover_validation_receipt_v2 &receipt) noexcept;

static_assert(
    std::is_trivially_copyable<segment_cover_validation_receipt_v2>::value,
    "segment cover receipt must remain pointer-free");

} // namespace cellerator::compute::segment
