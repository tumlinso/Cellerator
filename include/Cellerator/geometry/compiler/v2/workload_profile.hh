#pragma once

#include <Cellerator/execution/identity.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::geometry::compiler::v2 {

inline constexpr std::uint32_t workload_profile_schema_version = 2u;

struct stable_identity {
    std::uint64_t low = 0;
    std::uint64_t high = 0;
};

enum class operation_kind : std::uint8_t {
    relation_apply = 1,
    relation_apply_transpose,
    contract_on_support,
    segment_reduce,
    segment_normalize,
    edge_map_or_gate,
    relation_bundle_apply,
    sparse_axis_update
};

enum class orientation : std::uint8_t { forward = 1, transpose = 2 };
enum class value_mode : std::uint8_t { logical_primary = 1, projection_primary = 2 };
enum class value_dynamics : std::uint8_t { static_values = 1, dynamic_values = 2 };

enum component_requirement_flag : std::uint32_t {
    packed_output_permitted = 1u << 0u,
    canonical_output_required = 1u << 1u,
    graph_capture_required = 1u << 2u,
    segment_operation_present = 1u << 3u,
    fusion_opportunity_present = 1u << 4u
};

struct reuse_horizons {
    std::uint64_t structure = 1;
    std::uint64_t projection = 1;
    std::uint64_t values = 1;
    std::uint64_t dense_layout = 1;
    std::uint64_t work_window = 1;
};

struct workload_component {
    stable_identity identity{};
    operation_kind operation = operation_kind::relation_apply;
    orientation relation_orientation = orientation::forward;
    value_mode values = value_mode::logical_primary;
    value_dynamics dynamics = value_dynamics::static_values;
    std::uint32_t dense_width_min = 0;
    std::uint32_t dense_width_max = 0;
    std::uint32_t dense_width_bucket = 0;
    std::uint32_t requirement_flags = 0;
    std::uint64_t frequency = 1;
    std::uint64_t repetitions = 1;
    reuse_horizons reuse{};
    stable_identity segment_operation{};
    stable_identity fusion_group{};
    std::uint64_t persistent_budget_bytes = 0;
    std::uint64_t transient_budget_bytes = 0;
};

struct workload_profile {
    std::uint32_t schema_version = workload_profile_schema_version;
    std::uint32_t record_bytes = sizeof(workload_profile);
    const workload_component *components = nullptr;
    std::uint64_t component_count = 0;
};

enum class workload_status_code : std::uint8_t {
    success = 0,
    invalid_header,
    invalid_argument,
    invalid_identity,
    invalid_width,
    invalid_reuse,
    invalid_requirements
};

struct workload_status {
    workload_status_code code = workload_status_code::success;
    std::uint64_t index = 0;
    constexpr explicit operator bool() const noexcept {
        return code == workload_status_code::success;
    }
};

constexpr bool valid_identity(stable_identity id) noexcept {
    return id.low != 0 || id.high != 0;
}

workload_status validate_workload_profile(const workload_profile &profile) noexcept;

static_assert(std::is_trivially_copyable_v<reuse_horizons>);
static_assert(std::is_trivially_copyable_v<workload_component>);
static_assert(std::is_trivially_copyable_v<workload_profile>);

}  // namespace cellerator::geometry::compiler::v2
