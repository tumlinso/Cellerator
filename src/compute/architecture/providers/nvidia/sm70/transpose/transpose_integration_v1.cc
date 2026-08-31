#include <Cellerator/compute/architecture/providers/nvidia/sm70/transpose/transpose_integration_v1.hh>

#include <cmath>

namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose {
namespace {

void copy_name(const char *source, char (&destination)[48]) noexcept {
    std::uint32_t index = 0u;
    for (; index + 1u < sizeof(destination) && source[index] != '\0'; ++index)
        destination[index] = source[index];
    destination[index] = '\0';
}

bool valid_profile(const transpose_integration_profile_v1 &profile) noexcept {
    return profile.device_class_id != 0u && profile.projection_type_id != 0u
        && profile.capability_id != 0u
        && execution::valid_identity(profile.projection)
        && execution::valid_identity(profile.geometry)
        && profile.logical_edge_count != 0u
        && profile.physical_work_items >= profile.logical_edge_count
        && profile.padded_work_items
            == profile.physical_work_items - profile.logical_edge_count
        && profile.residual_edge_count <= profile.logical_edge_count
        && profile.owner_count != 0u
        && profile.owner_count <= profile.logical_edge_count
        && std::isfinite(profile.sparse_kernel_ns)
        && profile.sparse_kernel_ns >= 0.0
        && std::isfinite(profile.mma_kernel_ns) && profile.mma_kernel_ns >= 0.0;
}

} // namespace

transpose_status_v1 build_transpose_integration_v1(
    const transpose_integration_profile_v1 &profile,
    const transpose_integration_storage_v1 &storage,
    transpose_integration_view_v1 *view) noexcept {
    constexpr std::uint64_t count = 2u;
    if (view == nullptr || !valid_profile(profile) || storage.capacity < count
        || storage.catalog_candidates == nullptr || storage.catalog_stages == nullptr
        || storage.resource_manifests == nullptr || storage.planning_stages == nullptr
        || storage.resource_receipts == nullptr)
        return transpose_status_v1::invalid_argument;

    const transpose_candidate_catalog_v1 native = query_transpose_candidates_v1();
    // Catalog v3 requires ascending candidate identity. The MMA identity sorts
    // before sparse; ordering is metadata-only and does not imply promotion.
    const std::uint32_t native_order[]{1u, 0u};
    for (std::uint32_t index = 0u; index < count; ++index) {
        const transpose_candidate_v1 &candidate =
            native.candidates[native_order[index]];
        if (validate_transpose_candidate_v1(candidate)
            != transpose_status_v1::success)
            return transpose_status_v1::invalid_argument;

        operation::catalog_v3::candidate_stage_v3 &catalog_stage =
            storage.catalog_stages[index];
        catalog_stage = {};
        catalog_stage.stage_id = candidate.stage_id;
        catalog_stage.kernel_id = candidate.kernel_id;
        catalog_stage.stage_kind = 4u;
        catalog_stage.launch_count = 1u;
        copy_name(candidate.stable_name, catalog_stage.stable_name);

        operation::catalog_v3::candidate_descriptor_v3 &catalog_candidate =
            storage.catalog_candidates[index];
        catalog_candidate = {};
        catalog_candidate.identity = {candidate.candidate_id,
            sm70_transpose_provider_id_v1, profile.device_class_id,
            profile.projection_type_id, profile.capability_id,
            transpose_operation_id_v1, candidate.width_min, candidate.width_max,
            operation::catalog_v3::numerical_mode::precise,
            candidate.experimental
                ? operation::catalog_v3::candidate_class::experimental
                : operation::catalog_v3::candidate_class::production,
            candidate.requires_measurement, {}};
        catalog_candidate.stages = &catalog_stage;
        catalog_candidate.stage_count = 1u;
        catalog_candidate.resources = {profile.persistent_bytes,
            profile.transient_bytes,
            candidate.kind == transpose_candidate_kind_v1::mma16_source_owner
                ? 32u : 128u,
            0u};

        planner::resource::planning_stage_v1 &planning_stage =
            storage.planning_stages[index];
        planning_stage = {};
        planning_stage.identity = {candidate.stage_id, candidate.kernel_id};
        planning_stage.correlation_id = candidate.stage_id;
        planning_stage.static_name = candidate.stable_name;
        planning_stage.kind = planner::resource::planning_stage_kind_v1::kernel;
        planning_stage.flags = candidate.experimental
            ? planner::resource::planning_stage_experimental_v1
                | planner::resource::planning_stage_requires_measurement_v1
            : planner::resource::planning_stage_requires_measurement_v1;
        planning_stage.launch_count = 1u;
        planning_stage.analytical_ns =
            candidate.kind == transpose_candidate_kind_v1::mma16_source_owner
                ? profile.mma_kernel_ns : profile.sparse_kernel_ns;
        planning_stage.persistent_bytes = profile.persistent_bytes;
        planning_stage.transient_bytes = profile.transient_bytes;

        planner::resource::stage_resource_receipt_v1 &receipt =
            storage.resource_receipts[index];
        receipt = {};
        receipt.stage = planning_stage.identity;
        receipt.evidence = planner::resource::resource_evidence_kind_v1::declared;
        receipt.threads_per_cta =
            candidate.kind == transpose_candidate_kind_v1::mma16_source_owner
                ? 32u : 128u;
        receipt.warps_per_cta = receipt.threads_per_cta / 32u;

        planner::resource::candidate_resource_manifest_v1 &manifest =
            storage.resource_manifests[index];
        manifest = {};
        manifest.candidate = {candidate.candidate_id, candidate.kernel_id};
        manifest.provider = {sm70_transpose_provider_id_v1, 1u};
        manifest.capability = {profile.capability_id, 1u};
        manifest.projection = profile.projection;
        manifest.geometry = profile.geometry;
        manifest.mechanism.logical_work_items = profile.logical_edge_count;
        manifest.mechanism.physical_work_items = profile.physical_work_items;
        manifest.mechanism.useful_work_items = profile.logical_edge_count;
        manifest.mechanism.padded_work_items = profile.padded_work_items;
        manifest.mechanism.relation_bytes = profile.relation_bytes;
        manifest.mechanism.dense_input_bytes = profile.dense_input_bytes;
        manifest.mechanism.output_bytes = profile.output_bytes;
        manifest.mechanism.value_pack_bytes = profile.value_pack_bytes;
        manifest.mechanism.residual_edges = profile.residual_edge_count;
        manifest.mechanism.owner_work_items = profile.owner_count;
        manifest.stages = &planning_stage;
        manifest.resources = &receipt;
        manifest.stage_count = 1u;
        manifest.resource_count = 1u;
        manifest.persistent_bytes = profile.persistent_bytes;
        manifest.transient_bytes = profile.transient_bytes;
        manifest.cold_resource_query_complete = false;
        manifest.requires_measurement = true;
    }
    view->catalog = {storage.catalog_candidates, count};
    view->resources = storage.resource_manifests;
    view->resource_count = count;
    return transpose_status_v1::success;
}

} // namespace cellerator::compute::architecture::providers::nvidia::sm70::transpose
