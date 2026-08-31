#pragma once

#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_candidates_v1.hh>
#include <Cellerator/compute/operation/candidate_catalog_v3/catalog.h>
#include <Cellerator/planner/resource/planning_resources_v1.hh>

#include <cstdint>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {

inline constexpr std::uint64_t sm70_transpose_provider_id_v1 =
    0x534d373054525001u;
inline constexpr std::uint64_t transpose_operation_id_v1 =
    0x5452414e53504f53u;

struct transpose_integration_profile_v1 {
    std::uint64_t device_class_id = 0u;
    std::uint64_t projection_type_id = 0u;
    std::uint64_t capability_id = 0u;
    execution::projection_id projection{};
    execution::geometry_id geometry{};
    std::uint64_t logical_edge_count = 0u;
    std::uint64_t physical_work_items = 0u;
    std::uint64_t padded_work_items = 0u;
    std::uint64_t residual_edge_count = 0u;
    std::uint64_t owner_count = 0u;
    std::uint64_t relation_bytes = 0u;
    std::uint64_t dense_input_bytes = 0u;
    std::uint64_t output_bytes = 0u;
    std::uint64_t value_pack_bytes = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    double sparse_kernel_ns = 0.0;
    double mma_kernel_ns = 0.0;
};

struct transpose_integration_storage_v1 {
    operation::catalog_v3::candidate_descriptor_v3 *catalog_candidates = nullptr;
    operation::catalog_v3::candidate_stage_v3 *catalog_stages = nullptr;
    planner::resource::candidate_resource_manifest_v1 *resource_manifests = nullptr;
    planner::resource::planning_stage_v1 *planning_stages = nullptr;
    planner::resource::stage_resource_receipt_v1 *resource_receipts = nullptr;
    std::uint64_t capacity = 0u;
};

struct transpose_integration_view_v1 {
    operation::catalog_v3::candidate_catalog_view_v3 catalog{};
    const planner::resource::candidate_resource_manifest_v1 *resources = nullptr;
    std::uint64_t resource_count = 0u;
};

transpose_status_v1 build_transpose_integration_v1(
    const transpose_integration_profile_v1 &profile,
    const transpose_integration_storage_v1 &storage,
    transpose_integration_view_v1 *view) noexcept;

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
