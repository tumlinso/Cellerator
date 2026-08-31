#pragma once

#include <Cellerator/geometry/compiler/v2/workload_profile.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler::v2 {

struct original_group_skeleton {
    stable_identity identity{};
    const std::uint64_t *group_offsets = nullptr;
    std::uint64_t group_count = 0;
    const std::uint64_t *original_item_ids = nullptr;
    std::uint64_t item_count = 0;
};

enum class window_change_kind : std::uint8_t { add_group = 1, remove_group = 2 };

struct work_window_change {
    std::uint64_t original_group_id = 0;
    window_change_kind kind = window_change_kind::add_group;
    std::uint8_t reserved[7]{};
};

struct incremental_work_window {
    stable_identity identity{};
    stable_identity skeleton_identity{};
    stable_identity previous_window_identity{};
    const std::uint64_t *active_original_group_ids = nullptr;
    std::uint64_t active_group_count = 0;
    const work_window_change *changes = nullptr;
    std::uint64_t change_count = 0;
};

workload_status validate_original_group_skeleton(
    const original_group_skeleton &skeleton) noexcept;
workload_status validate_incremental_work_window(
    const original_group_skeleton &skeleton,
    const incremental_work_window &window) noexcept;

static_assert(std::is_trivially_copyable_v<original_group_skeleton>);
static_assert(std::is_trivially_copyable_v<work_window_change>);
static_assert(std::is_trivially_copyable_v<incremental_work_window>);

}  // namespace cellerator::geometry::compiler::v2
