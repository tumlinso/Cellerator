#include <Cellerator/execution/geometry_acquisition_diagnostics.hh>

#include <cmath>

namespace cellerator::execution {
namespace {

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool valid_work(const geometry_acquisition_work_v1 &work) noexcept {
    if (work.schema_version
            != geometry_acquisition_diagnostics_schema_version_v1
        || work.record_bytes != sizeof(geometry_acquisition_work_v1))
        return false;
    for (std::uint32_t value : work.reserved)
        if (value != 0u)
            return false;
    return finite_nonnegative(work.semantic_search_ns)
        && finite_nonnegative(work.target_refinement_ns)
        && finite_nonnegative(work.projection_construction_ns)
        && finite_nonnegative(work.projection_upload_ns)
        && finite_nonnegative(work.cpe2_prebind_ns)
        && finite_nonnegative(work.candidate_preparation_ns)
        && finite_nonnegative(work.value_pack_ns)
        && finite_nonnegative(work.input_pack_ns)
        && finite_nonnegative(work.kernel_ns)
        && finite_nonnegative(work.epilogue_ns)
        && finite_nonnegative(work.order_ns);
}

bool valid_stable(compute::math::core::stable_id identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

bool valid_reuse(
    const geometry_acquisition_reuse_diagnostics_v1 &reuse) noexcept {
    if (reuse.schema_version
            != geometry_acquisition_diagnostics_schema_version_v1
        || reuse.record_bytes
            != sizeof(geometry_acquisition_reuse_diagnostics_v1)
        || !valid_identity(reuse.structure)
        || reuse.epoch.value == 0u
        || !valid_identity(reuse.semantic_geometry)
        || !valid_identity(reuse.projection) || reuse.values.value == 0u
        || !valid_stable(reuse.dense_layout)
        || !valid_identity(reuse.work_window)
        || !valid_stable(reuse.prepared_program)
        || reuse.structure_observed_uses == 0u
        || reuse.semantic_geometry_observed_uses == 0u
        || reuse.projection_observed_uses == 0u
        || reuse.value_generation_observed_uses == 0u
        || reuse.dense_layout_observed_uses == 0u
        || reuse.work_window_observed_uses == 0u
        || reuse.prepared_program_observed_uses == 0u)
        return false;
    const bool has_graph_replay = valid_stable(reuse.graph_replay);
    if (has_graph_replay != (reuse.graph_replay_observed_uses != 0u))
        return false;
    for (std::uint32_t value : reuse.reserved)
        if (value != 0u)
            return false;
    return true;
}

} // namespace

geometry_acquisition_diagnostics_status_v1
map_geometry_acquisition_diagnostics_v1(
    const geometry_acquisition_work_v1 &work,
    const geometry_acquisition_reuse_diagnostics_v1 &reuse,
    geometry_acquisition_diagnostics_v1 *out) noexcept {
    if (out == nullptr)
        return geometry_acquisition_diagnostics_status_v1::invalid_argument;
    *out = {};
    if (work.schema_version
            != geometry_acquisition_diagnostics_schema_version_v1
        || work.record_bytes != sizeof(geometry_acquisition_work_v1)
        || reuse.schema_version
            != geometry_acquisition_diagnostics_schema_version_v1
        || reuse.record_bytes
            != sizeof(geometry_acquisition_reuse_diagnostics_v1))
        return geometry_acquisition_diagnostics_status_v1::invalid_header;
    for (std::uint32_t value : work.reserved)
        if (value != 0u)
            return geometry_acquisition_diagnostics_status_v1::nonzero_reserved;
    for (std::uint32_t value : reuse.reserved)
        if (value != 0u)
            return geometry_acquisition_diagnostics_status_v1::nonzero_reserved;
    if (!valid_work(work))
        return geometry_acquisition_diagnostics_status_v1::invalid_cost;
    if (!valid_reuse(reuse)) {
        const bool identities = valid_identity(reuse.structure)
            && reuse.epoch.value != 0u
            && valid_identity(reuse.semantic_geometry)
            && valid_identity(reuse.projection) && reuse.values.value != 0u
            && valid_stable(reuse.dense_layout)
            && valid_identity(reuse.work_window)
            && valid_stable(reuse.prepared_program);
        return identities
            ? geometry_acquisition_diagnostics_status_v1::invalid_reuse
            : geometry_acquisition_diagnostics_status_v1::invalid_identity;
    }

    geometry_acquisition_diagnostics_v1 candidate{};
    candidate.planner_phases.semantic_packing_ns = work.semantic_search_ns;
    candidate.planner_phases.projection_construction_ns =
        work.target_refinement_ns + work.projection_construction_ns
        + work.projection_upload_ns + work.cpe2_prebind_ns;
    candidate.planner_phases.backend_prepare_ns =
        work.candidate_preparation_ns;
    candidate.planner_phases.static_value_pack_ns = work.value_pack_ns;
    candidate.planner_phases.dynamic_input_pack_ns = work.input_pack_ns;
    candidate.planner_phases.kernel_ns = work.kernel_ns;
    candidate.planner_phases.epilogue_ns = work.epilogue_ns;
    candidate.planner_phases.order_transform_ns = work.order_ns;
    candidate.planner_phases.persistent_bytes = work.persistent_bytes;
    candidate.planner_phases.transient_bytes = work.transient_bytes;
    if (!finite_nonnegative(candidate.planner_phases.projection_construction_ns))
        return geometry_acquisition_diagnostics_status_v1::invalid_cost;
    candidate.reuse = reuse;
    candidate.persistent_projection_upload_bytes = work.projection_upload_bytes;
    *out = candidate;
    return geometry_acquisition_diagnostics_status_v1::success;
}

} // namespace cellerator::execution
