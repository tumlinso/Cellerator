#include <Cellerator/geometry/compiler/v2/work_window.hh>

namespace cellerator::geometry::compiler::v2 {

workload_status validate_original_group_skeleton(
    const original_group_skeleton &skeleton) noexcept {
    if (!valid_identity(skeleton.identity) || skeleton.group_offsets == nullptr
        || skeleton.group_count == 0 || skeleton.original_item_ids == nullptr
        || skeleton.item_count == 0 || skeleton.group_offsets[0] != 0
        || skeleton.group_offsets[skeleton.group_count] != skeleton.item_count) {
        return {workload_status_code::invalid_argument, 0};
    }
    for (std::uint64_t group = 0; group < skeleton.group_count; ++group) {
        if (skeleton.group_offsets[group] >= skeleton.group_offsets[group + 1]) {
            return {workload_status_code::invalid_argument, group};
        }
    }
    return {};
}

workload_status validate_incremental_work_window(
    const original_group_skeleton &skeleton,
    const incremental_work_window &window) noexcept {
    const workload_status skeleton_status = validate_original_group_skeleton(skeleton);
    if (!skeleton_status) return skeleton_status;
    if (!valid_identity(window.identity)
        || window.skeleton_identity.low != skeleton.identity.low
        || window.skeleton_identity.high != skeleton.identity.high
        || window.active_original_group_ids == nullptr || window.active_group_count == 0
        || (window.change_count != 0 && window.changes == nullptr)) {
        return {workload_status_code::invalid_identity, 0};
    }
    for (std::uint64_t index = 0; index < window.active_group_count; ++index) {
        const std::uint64_t group = window.active_original_group_ids[index];
        if (group >= skeleton.group_count
            || (index != 0 && window.active_original_group_ids[index - 1] >= group)) {
            return {workload_status_code::invalid_argument, index};
        }
    }
    for (std::uint64_t index = 0; index < window.change_count; ++index) {
        const work_window_change change = window.changes[index];
        if (change.original_group_id >= skeleton.group_count
            || (change.kind != window_change_kind::add_group
                && change.kind != window_change_kind::remove_group)) {
            return {workload_status_code::invalid_argument, index};
        }
    }
    return {};
}

}  // namespace cellerator::geometry::compiler::v2
