#pragma once

#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::distributed {

inline constexpr std::uint32_t hierarchy_schema_version = 1u;
inline constexpr std::uint32_t invalid_hierarchy_index = 0xffffffffu;

enum class hierarchy_status : std::uint8_t {
    ok = 0u,
    invalid_argument = 1u,
    invalid_identity = 2u,
    invalid_partition = 3u,
    invalid_module = 4u,
    stale_structure = 5u,
    stale_values = 6u,
    invalid_order = 7u,
    insufficient_capacity = 8u
};

struct nested_partition {
    execution::partition_id identity{};
    execution::partition_id parent{};
    std::uint32_t level = 0u;
    std::int32_t device_ordinal = -1;
};

// Preparation-time view. The records are immutable, pointer identity is not
// meaningful, and the persistent hierarchy identity changes with membership,
// ancestry, or placement.
struct partition_hierarchy_view {
    std::uint32_t schema_version = hierarchy_schema_version;
    execution::partition_hierarchy_id identity{};
    const nested_partition *partitions = nullptr;
    std::uint32_t partition_count = 0u;
};

struct shared_value_module {
    std::uint64_t identity = 0u;
    execution::partition_id partition{};
    std::uint32_t parent_module = invalid_hierarchy_index;
    std::uint32_t value_offset = 0u;
    std::uint32_t value_count = 0u;
};

// Compact module-to-shared-value index over one immutable semantic structure.
// A value index may occur in several modules; values themselves remain in the
// existing mutable value plane.
struct shared_value_hierarchy_view {
    std::uint32_t schema_version = hierarchy_schema_version;
    execution::partition_hierarchy_id hierarchy{};
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    const shared_value_module *modules = nullptr;
    const std::uint32_t *value_indices = nullptr;
    std::uint32_t module_count = 0u;
    std::uint32_t value_index_count = 0u;
    std::uint32_t shared_value_count = 0u;
};

// Per-generation activity is deliberately separate from the immutable index.
struct module_activity_view {
    execution::partition_hierarchy_id hierarchy{};
    execution::structure_id structure{};
    execution::structure_epoch epoch{};
    execution::value_generation generation{};
    const std::uint8_t *active = nullptr;
    std::uint32_t module_count = 0u;
};

struct active_module {
    std::uint32_t module_index = 0u;
    execution::partition_id partition{};
    std::uint32_t value_offset = 0u;
    std::uint32_t value_count = 0u;
};

struct active_module_plan {
    execution::partition_hierarchy_id hierarchy{};
    execution::value_generation generation{};
    active_module *modules = nullptr;
    std::uint32_t capacity = 0u;
    std::uint32_t count = 0u;
};

struct boundary_edge {
    std::uint32_t source_module = 0u;
    std::uint32_t destination_module = 0u;
    execution::partition_id source_partition{};
    execution::partition_id destination_partition{};
    execution::order_id source_order{};
    execution::order_id destination_order{};
    std::uint64_t value_count = 0u;
    std::uint64_t byte_count = 0u;
    bool peer_access = false;
    std::uint8_t reserved[7]{};
};

enum class boundary_transfer_kind : std::uint8_t {
    local_reorder = 1u,
    peer_copy = 2u,
    staged_copy = 3u
};

struct boundary_cost_model {
    // Supplied from measured device/topology evidence; never a topology law.
    double peer_launch_ns = 0.0;
    double staged_launch_ns = 0.0;
    double peer_bytes_per_ns = 0.0;
    double staged_bytes_per_ns = 0.0;
    double order_transform_ns_per_value = 0.0;
};

struct communication_step {
    std::uint32_t boundary_index = 0u;
    boundary_transfer_kind transfer = boundary_transfer_kind::local_reorder;
    execution::order_transition_kind order =
        execution::order_transition_kind::preserve;
    std::uint16_t reserved = 0u;
    execution::partition_id source_partition{};
    execution::partition_id destination_partition{};
    std::int32_t source_device = -1;
    std::int32_t destination_device = -1;
    planner::phase_costs phases{};
};

struct communication_plan {
    execution::partition_hierarchy_id hierarchy{};
    execution::value_generation activity_generation{};
    communication_step *steps = nullptr;
    std::uint32_t capacity = 0u;
    std::uint32_t count = 0u;
    planner::phase_costs total{};
};

hierarchy_status validate_partition_hierarchy(
    const partition_hierarchy_view &hierarchy) noexcept;
hierarchy_status validate_shared_value_hierarchy(
    const partition_hierarchy_view &hierarchy,
    const shared_value_hierarchy_view &values) noexcept;
hierarchy_status build_active_module_plan(
    const shared_value_hierarchy_view &values,
    const module_activity_view &activity,
    active_module_plan *plan) noexcept;
hierarchy_status plan_boundary_communication(
    const partition_hierarchy_view &hierarchy,
    const shared_value_hierarchy_view &values,
    const module_activity_view &activity,
    const boundary_edge *boundaries,
    std::uint32_t boundary_count,
    const boundary_cost_model &cost,
    communication_plan *plan) noexcept;
hierarchy_status make_connected_transition(
    const communication_step &step,
    std::uint32_t connected_boundary,
    planner::operation_core::stable_id producer,
    planner::operation_core::stable_id consumer,
    planner::operation_core::stable_id order_conversion,
    planner::connected_transition_cost *transition) noexcept;

static_assert(std::is_trivially_copyable<nested_partition>::value,
    "nested partition records must remain persistence friendly");
static_assert(std::is_trivially_copyable<shared_value_module>::value,
    "shared-value modules must remain device-copyable");
static_assert(std::is_trivially_copyable<boundary_edge>::value,
    "boundary edges must remain device-copyable");

} // namespace cellerator::distributed
