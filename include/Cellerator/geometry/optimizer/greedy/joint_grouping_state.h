#pragma once

#include <cstdint>

namespace cellerator::geometry::optimizer::greedy {

enum class joint_grouping_status : std::uint32_t {
    success = 0,
    invalid_argument,
    invalid_problem,
    insufficient_storage,
    rectangle_table_full,
    arithmetic_overflow,
    state_mismatch,
};

// One bounded local component. Aggregate biological edge identity remains the
// uint64 edge index; local source, destination, and group indices stay compact.
struct joint_grouping_problem_view {
    std::uint32_t source_count = 0;
    std::uint32_t destination_count = 0;
    std::uint64_t edge_count = 0;
    const std::uint32_t* edge_sources = nullptr;       // edge_count
    const std::uint32_t* edge_destinations = nullptr;  // edge_count
};

struct joint_rectangle_record {
    std::uint32_t source_group = 0;
    std::uint32_t destination_group = 0;
    std::uint64_t contribution_count = 0;
    std::uint8_t occupied = 0;
    std::uint8_t reserved[7]{};
};

struct joint_grouping_storage {
    std::uint32_t* source_groups = nullptr;       // source_count
    std::uint32_t source_group_capacity = 0;
    std::uint32_t* destination_groups = nullptr;  // destination_count
    std::uint32_t destination_group_capacity = 0;
    std::uint64_t* source_group_sizes = nullptr;
    std::uint32_t source_group_size_capacity = 0;
    std::uint64_t* destination_group_sizes = nullptr;
    std::uint32_t destination_group_size_capacity = 0;
    std::uint32_t* edge_rectangle_slots = nullptr;  // edge_count
    std::uint64_t edge_rectangle_capacity = 0;
    joint_rectangle_record* rectangles = nullptr;
    std::uint32_t rectangle_capacity = 0;
};

struct mutable_joint_grouping_state {
    joint_grouping_problem_view problem{};
    joint_grouping_storage storage{};
    std::uint32_t source_group_count = 0;
    std::uint32_t destination_group_count = 0;
    std::uint32_t occupied_rectangle_count = 0;
    std::uint64_t generation = 0;
};

struct joint_grouping_validation_workspace {
    std::uint64_t* source_group_sizes = nullptr;
    std::uint32_t source_group_capacity = 0;
    std::uint64_t* destination_group_sizes = nullptr;
    std::uint32_t destination_group_capacity = 0;
};

// Initializes from explicit group assignments. Group IDs must be dense in
// [0, group_count). Empty groups are legal and remain visible to later moves.
joint_grouping_status initialize_joint_grouping_state(
        const joint_grouping_problem_view& problem,
        const std::uint32_t* initial_source_groups,
        const std::uint32_t* initial_destination_groups,
        std::uint32_t source_group_count,
        std::uint32_t destination_group_count,
        const joint_grouping_storage& storage,
        mutable_joint_grouping_state* state) noexcept;

// Rebuilds exact sparse rectangle census and edge ownership from the current
// assignments. This is a cold validation/recovery operation, never a hot move.
joint_grouping_status rebuild_joint_rectangle_census(
        mutable_joint_grouping_state* state) noexcept;

// Independently checks group sizes, every edge's rectangle key, and the exact
// contribution census without allocating or mutating the state.
joint_grouping_status validate_joint_grouping_state(
        const mutable_joint_grouping_state& state,
        const joint_grouping_validation_workspace& workspace,
        std::uint64_t* validated_contributions) noexcept;

}  // namespace cellerator::geometry::optimizer::greedy
