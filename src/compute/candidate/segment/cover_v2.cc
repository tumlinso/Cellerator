#include <Cellerator/compute/candidate/segment/cover_v2.hh>

#include <cstdint>
#include <limits>

namespace cellerator::compute::segment {
namespace {

segment_result_v2 error(segment_status_v2 code, const char *message) noexcept {
    return {code, message};
}

} // namespace

segment_result_v2 validate_segment_cover_native_partition_v2_host(
    const segment_plan_v2 &plan,
    const segment_cover_native_host_view_v2 &cover,
    const segment_cover_validation_workspace_v2 &workspace,
    segment_cover_validation_receipt_v2 &receipt) noexcept {
    receipt = {};
    const segment_result_v2 valid = validate_segment_plan_v2(plan);
    if (!valid) return valid;
    if (plan.storage_order != segment_storage_order_v2::cover_native)
        return error(segment_status_v2::invalid_argument,
            "cover-native validation requires cover-native storage order");
    if (cover.occupied_value_count != plan.local_value_count
        || cover.offset_count
            != static_cast<std::uint64_t>(plan.local_segment_count) + 1u
        || cover.offsets == nullptr
        || (cover.occupied_value_count != 0u
            && (cover.local_to_global_value == nullptr
                || cover.owners == nullptr))
        || cover.physical_slot_count < cover.occupied_value_count
        || cover.mma_occupied_count > cover.occupied_value_count
        || cover.residual_occupied_count > cover.occupied_value_count
        || cover.mma_occupied_count + cover.residual_occupied_count
            != cover.occupied_value_count)
        return error(segment_status_v2::invalid_partition,
            "cover-native occupied and physical counts are inconsistent");
    if (cover.occupied_value_count
            > std::numeric_limits<std::uint64_t>::max()
                - plan.component_value_begin)
        return error(segment_status_v2::invalid_shape,
            "cover-native logical interval overflows");
    if (workspace.ownership_mark_bytes < cover.occupied_value_count
        || (cover.occupied_value_count != 0u
            && workspace.ownership_marks == nullptr))
        return error(segment_status_v2::insufficient_workspace,
            "cover-native exact ownership marks are insufficient");
    const segment_result_v2 offsets_valid =
        validate_segment_partition_offsets_v2_host(
            plan, cover.offsets, cover.offset_count);
    if (!offsets_valid) return offsets_valid;

    for (std::uint64_t index = 0u;
         index < cover.occupied_value_count; ++index)
        workspace.ownership_marks[index] = 0u;
    const std::uint64_t logical_end =
        plan.component_value_begin + plan.local_value_count;
    std::uint64_t mma_count = 0u;
    std::uint64_t residual_count = 0u;
    for (std::uint64_t physical = 0u;
         physical < cover.occupied_value_count; ++physical) {
        const std::uint64_t logical = cover.local_to_global_value[physical];
        if (logical < plan.component_value_begin || logical >= logical_end)
            return error(segment_status_v2::invalid_partition,
                "cover-native logical edge is outside the local component");
        const std::uint64_t local = logical - plan.component_value_begin;
        if (workspace.ownership_marks[local] != 0u)
            return error(segment_status_v2::invalid_partition,
                "cover-native logical edge has duplicate contribution ownership");
        const segment_cover_owner_v2 owner = cover.owners[physical];
        if (owner != segment_cover_owner_v2::mma
            && owner != segment_cover_owner_v2::residual)
            return error(segment_status_v2::invalid_partition,
                "cover-native contribution owner is invalid");
        workspace.ownership_marks[local] = 1u;
        mma_count += static_cast<std::uint64_t>(
            owner == segment_cover_owner_v2::mma);
        residual_count += static_cast<std::uint64_t>(
            owner == segment_cover_owner_v2::residual);
    }
    for (std::uint64_t local = 0u;
         local < cover.occupied_value_count; ++local)
        if (workspace.ownership_marks[local] != 1u)
            return error(segment_status_v2::invalid_partition,
                "cover-native exact cover prunes a logical edge");
    if (mma_count != cover.mma_occupied_count
        || residual_count != cover.residual_occupied_count)
        return error(segment_status_v2::invalid_partition,
            "cover-native owner census mismatches declared counts");

    receipt.exact_logical_values = plan.local_value_count;
    receipt.occupied_physical_values = cover.occupied_value_count;
    receipt.physical_holes =
        cover.physical_slot_count - cover.occupied_value_count;
    receipt.mma_values = mma_count;
    receipt.residual_values = residual_count;
    receipt.exact_disjoint_ownership = true;
    receipt.holes_are_non_biological = true;
    return {};
}

} // namespace cellerator::compute::segment
