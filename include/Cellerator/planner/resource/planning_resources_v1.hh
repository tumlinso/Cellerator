#pragma once

#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::planner::resource {

inline constexpr std::uint32_t planning_resource_schema_v1 = 1u;

enum class planning_stage_kind_v1 : std::uint8_t {
    preparation = 1u,
    value_pack = 2u,
    input_pack = 3u,
    kernel = 4u,
    epilogue = 5u,
    order_transform = 6u,
    synchronization = 7u,
    communication = 8u,
};

enum planning_stage_flag_v1 : std::uint8_t {
    planning_stage_graph_capture_v1 = 1u << 0u,
    planning_stage_experimental_v1 = 1u << 1u,
    planning_stage_requires_measurement_v1 = 1u << 2u,
};

enum class resource_evidence_kind_v1 : std::uint8_t {
    declared = 1u,
    compiled_attribute_query = 2u,
};

enum class resource_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_argument,
    invalid_stage,
    invalid_resource,
    invalid_statistics,
    invalid_cost,
    arithmetic_overflow,
};

struct resource_status_v1 {
    resource_status_code_v1 code = resource_status_code_v1::success;
    std::uint64_t subject = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == resource_status_code_v1::success;
    }
};

struct mechanism_statistics_v1 {
    std::uint64_t logical_work_items = 0u;
    std::uint64_t physical_work_items = 0u;
    std::uint64_t useful_work_items = 0u;
    std::uint64_t padded_work_items = 0u;
    std::uint64_t relation_bytes = 0u;
    std::uint64_t dense_input_bytes = 0u;
    std::uint64_t output_bytes = 0u;
    std::uint64_t value_pack_bytes = 0u;
    std::uint64_t residual_edges = 0u;
    std::uint64_t group_count = 0u;
    std::uint64_t tile_count = 0u;
    std::uint64_t owner_work_items = 0u;
};

struct planning_stage_v1 {
    operation_core::stable_id identity{};
    std::uint64_t correlation_id = 0u;
    const char *static_name = nullptr;
    planning_stage_kind_v1 kind = planning_stage_kind_v1::kernel;
    std::uint8_t flags = 0u;
    std::uint16_t reserved = 0u;
    std::uint32_t launch_count = 0u;
    double analytical_ns = 0.0;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
};

struct stage_resource_receipt_v1 {
    operation_core::stable_id stage{};
    resource_evidence_kind_v1 evidence = resource_evidence_kind_v1::declared;
    std::uint8_t reserved0[3]{};
    std::uint32_t threads_per_cta = 0u;
    std::uint32_t warps_per_cta = 0u;
    std::uint32_t registers_per_thread = 0u;
    std::uint32_t static_shared_bytes = 0u;
    std::uint32_t dynamic_shared_bytes = 0u;
    std::uint32_t reserved1 = 0u;
};

struct candidate_resource_manifest_v1 {
    std::uint32_t schema_version = planning_resource_schema_v1;
    std::uint32_t reserved = 0u;
    operation_core::stable_id candidate{};
    operation_core::stable_id provider{};
    operation_core::stable_id capability{};
    execution::projection_id projection{};
    execution::geometry_id geometry{};
    mechanism_statistics_v1 mechanism{};
    const planning_stage_v1 *stages = nullptr;
    const stage_resource_receipt_v1 *resources = nullptr;
    std::uint64_t stage_count = 0u;
    std::uint64_t resource_count = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    bool cold_resource_query_complete = false;
    bool requires_measurement = true;
    std::uint8_t reserved2[6]{};
};

resource_status_v1 validate_candidate_resource_manifest_v1(
    const candidate_resource_manifest_v1 &manifest) noexcept;

resource_status_v1 compute_manifest_phase_costs_v1(
    const candidate_resource_manifest_v1 &manifest,
    phase_costs *costs) noexcept;

static_assert(std::is_trivially_copyable<planning_stage_v1>::value,
    "planning stages must remain pointer-first cold metadata");
static_assert(std::is_trivially_copyable<candidate_resource_manifest_v1>::value,
    "resource manifests must remain non-owning pointer-plus-count views");

}  // namespace cellerator::planner::resource
