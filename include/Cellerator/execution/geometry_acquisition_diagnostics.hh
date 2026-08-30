#pragma once

#include <Cellerator/geometry/persistence/semantic_geometry_image_v1.hh>
#include <Cellerator/planner/end_to_end_planner.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

inline constexpr std::uint32_t
    geometry_acquisition_diagnostics_schema_version_v1 = 1u;

enum class geometry_acquisition_diagnostics_status_v1 : std::uint8_t {
    success = 0u,
    invalid_argument = 1u,
    invalid_header = 2u,
    invalid_identity = 3u,
    invalid_reuse = 4u,
    invalid_cost = 5u,
    nonzero_reserved = 6u
};

// Raw cold/steady-state observations are retained for diagnosis, but are not a
// second cost model. map_geometry_acquisition_diagnostics_v1 is the only path
// from these named observations into planner-v2 phase_costs.
struct geometry_acquisition_work_v1 {
    std::uint32_t schema_version =
        geometry_acquisition_diagnostics_schema_version_v1;
    std::uint32_t record_bytes = sizeof(geometry_acquisition_work_v1);
    double semantic_search_ns = 0.0;
    double target_refinement_ns = 0.0;
    double projection_construction_ns = 0.0;
    double projection_upload_ns = 0.0;
    double cpe2_prebind_ns = 0.0;
    double candidate_preparation_ns = 0.0;
    double value_pack_ns = 0.0;
    double input_pack_ns = 0.0;
    double kernel_ns = 0.0;
    double epilogue_ns = 0.0;
    double order_ns = 0.0;
    std::uint64_t projection_upload_bytes = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    std::uint32_t reserved[4]{};
};

// These counts are observations keyed by exact identities. Planner policy's
// structure/projection/value reuse horizons remain the amortization authority;
// diagnostics never replace or silently derive those policy values.
struct geometry_acquisition_reuse_diagnostics_v1 {
    std::uint32_t schema_version =
        geometry_acquisition_diagnostics_schema_version_v1;
    std::uint32_t record_bytes =
        sizeof(geometry_acquisition_reuse_diagnostics_v1);
    structure_id structure{};
    structure_epoch epoch{};
    geometry_id semantic_geometry{};
    projection_id projection{};
    value_generation values{};
    compute::math::core::stable_id dense_layout{};
    geometry::work_window_id work_window{};
    compute::math::core::stable_id prepared_program{};
    compute::math::core::stable_id graph_replay{};
    std::uint64_t structure_observed_uses = 0u;
    std::uint64_t semantic_geometry_observed_uses = 0u;
    std::uint64_t projection_observed_uses = 0u;
    std::uint64_t value_generation_observed_uses = 0u;
    std::uint64_t dense_layout_observed_uses = 0u;
    std::uint64_t work_window_observed_uses = 0u;
    std::uint64_t prepared_program_observed_uses = 0u;
    std::uint64_t graph_replay_observed_uses = 0u;
    std::uint32_t reserved[4]{};
};

struct geometry_acquisition_diagnostics_v1 {
    std::uint32_t schema_version =
        geometry_acquisition_diagnostics_schema_version_v1;
    std::uint32_t record_bytes = sizeof(geometry_acquisition_diagnostics_v1);
    planner::phase_costs planner_phases{};
    geometry_acquisition_reuse_diagnostics_v1 reuse{};
    // Persistent projection upload is included in projection_construction_ns,
    // not planner h2d_ns. This byte count remains diagnostic for transfer and
    // break-even reporting without charging it again per execution.
    std::uint64_t persistent_projection_upload_bytes = 0u;
    std::uint32_t reserved[4]{};
};

geometry_acquisition_diagnostics_status_v1
map_geometry_acquisition_diagnostics_v1(
    const geometry_acquisition_work_v1 &work,
    const geometry_acquisition_reuse_diagnostics_v1 &reuse,
    geometry_acquisition_diagnostics_v1 *out) noexcept;

static_assert(std::is_trivially_copyable<geometry_acquisition_work_v1>::value,
    "acquisition work observations must remain pointer-copyable");
static_assert(std::is_trivially_copyable<
    geometry_acquisition_reuse_diagnostics_v1>::value,
    "acquisition reuse diagnostics must remain pointer-copyable");
static_assert(std::is_trivially_copyable<
    geometry_acquisition_diagnostics_v1>::value,
    "mapped acquisition diagnostics must remain pointer-copyable");

} // namespace cellerator::execution
