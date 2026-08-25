#pragma once

#include <Cellerator/planner/end_to_end_planner.hh>

#include <CellPack/alternating_refinement.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::planner {

inline constexpr std::uint32_t objective_v2_calibration_schema_version = 1u;
inline constexpr std::uint32_t objective_v2_refinement_guidance_schema_version = 1u;

enum class objective_v2_prediction_state : std::uint8_t {
    calibrated = 1u,
    novel_regime = 2u,
    stale_identity = 3u
};

// Candidate-neutral work counts. Projection families expose the work their
// native traversal performs; stable candidate ids and names are deliberately
// absent. Future candidates may populate any combination of these mechanisms.
struct objective_v2_mechanism_statistics {
    std::uint64_t active_rows = 0u;
    std::uint64_t active_features = 0u;
    std::uint64_t logical_edges = 0u;
    std::uint64_t useful_interactions = 0u;
    std::uint64_t masked_row_lane_slots = 0u;
    std::uint64_t linear_edge_visits = 0u;
    std::uint64_t masked_feature_lane_slots = 0u;
    std::uint64_t dense_rhs_vector_elements = 0u;
    std::uint64_t feature_value_loads = 0u;
    std::uint64_t dynamic_input_pack_bytes = 0u;
    std::uint64_t output_order_bytes = 0u;
    std::uint64_t projection_bytes = 0u;
    std::uint64_t value_bytes = 0u;
    std::uint64_t transient_bytes = 0u;
    std::uint32_t launch_count = 0u;
    std::uint32_t dense_width = 0u;
    double projection_construction_ns = 0.0;
    double backend_prepare_ns = 0.0;
    double static_value_pack_ns = 0.0;
};

struct objective_v2_coefficients {
    double intercept_ns = 0.0;
    double useful_interaction_ns = 0.0;
    double masked_row_lane_slot_ns = 0.0;
    double linear_edge_visit_ns = 0.0;
    double masked_feature_lane_slot_ns = 0.0;
    double dense_rhs_vector_element_ns = 0.0;
    double feature_value_load_ns = 0.0;
    double launch_ns = 0.0;
    double input_pack_byte_ns = 0.0;
    double output_order_byte_ns = 0.0;
};

// Replaceable measured model. The shipped instance is intentionally bounded
// to CE-ARCH-76's V100 build and shape support; novel domains request empirical
// measurement instead of silently extrapolating.
struct objective_v2_calibration {
    std::uint32_t schema_version = objective_v2_calibration_schema_version;
    std::uint64_t model_identity = 0u;
    std::uint64_t evidence_revision = 0u;
    device_performance_key device{};
    runtime_build_key build{};
    std::uint64_t trained_rows = 0u;
    std::uint64_t trained_features = 0u;
    std::uint64_t trained_edges = 0u;
    std::uint32_t supported_dense_width_mask = 0u;
    std::uint32_t sample_count = 0u;
    std::uint64_t maximum_masked_row_lane_slots = 0u;
    std::uint64_t maximum_linear_edge_visits = 0u;
    std::uint64_t maximum_masked_feature_lane_slots = 0u;
    std::uint64_t maximum_dense_rhs_vector_elements = 0u;
    std::uint64_t maximum_feature_value_loads = 0u;
    std::uint32_t maximum_launch_count = 0u;
    double median_relative_error_percent = 0.0;
    double maximum_relative_error_percent = 0.0;
    double maximum_training_spread_percent = 0.0;
    objective_v2_coefficients coefficients{};
};

struct objective_v2_calibration_query {
    planning_keys keys{};
    objective_v2_mechanism_statistics statistics{};
    double practical_tolerance_percent = 2.0;
};

struct objective_v2_prediction {
    std::uint32_t schema_version = objective_v2_calibration_schema_version;
    objective_v2_prediction_state state =
        objective_v2_prediction_state::novel_regime;
    std::uint16_t reserved = 0u;
    planning_keys keys{};
    std::uint64_t model_identity = 0u;
    std::uint64_t evidence_revision = 0u;
    total_cost predicted{};
    double confidence = 0.0;
    double relative_error_bound_percent = 0.0;
    bool empirical_measurement_required = true;
    const char *reason = nullptr;
};

objective_v2_calibration ce_arch_76_v100_objective_v2_calibration() noexcept;

planner_status evaluate_calibrated_objective_v2(
    const objective_v2_calibration &calibration,
    const objective_v2_calibration_query &query,
    objective_v2_prediction *out) noexcept;

// Copies a calibrated estimate into the existing planner candidate. The
// normal measurement hook remains enabled and is still final authority.
planner_status apply_objective_v2_prediction(
    const objective_v2_prediction &prediction,
    const planning_keys &current_keys,
    planner_candidate *candidate) noexcept;

// CP-BP refinement consumes held-out measured runtime and preparation. The
// calibrated reuse semantics map those existing measurements into nanoseconds;
// no predicted score or candidate identity is persisted in CPK1.
planner_status make_objective_v2_refinement_weights(
    const objective_v2_calibration &calibration,
    std::uint64_t expected_reuse,
    cellpack::alternating_refinement_objective_weights *out) noexcept;

struct objective_v2_refinement_workload {
    std::uint64_t expected_reuse = 1u;
    std::uint64_t workload_profile_identity = 0u;
    std::uint64_t workload_evidence_revision = 0u;
    std::uint32_t minimum_bootstrap_samples = 0u;
    std::uint32_t reserved = 0u;
    double forward_weight = 0.0;
    double transpose_weight = 0.0;
    double active_interaction_scale = 0.0;
    double measured_partition_cut_edge_ns = 0.0;
    double bootstrap_mad_weight = 0.0;
};

struct objective_v2_refinement_guidance {
    std::uint32_t schema_version =
        objective_v2_refinement_guidance_schema_version;
    std::uint32_t reserved = 0u;
    std::uint64_t model_identity = 0u;
    std::uint64_t calibration_evidence_revision = 0u;
    std::uint64_t workload_profile_identity = 0u;
    std::uint64_t workload_evidence_revision = 0u;
    std::uint32_t minimum_bootstrap_samples = 0u;
    std::uint32_t reserved_count = 0u;
    cellpack::alternating_refinement_objective_weights weights{};
};

// Builds measured workload guidance. Forward/transpose totals and bootstrap
// spread come from validation packets. Activity uses the CE-ARCH-76 measured
// useful-interaction coefficient. Partition-cut cost is accepted only as an
// explicit caller-measured coefficient with its own evidence identity.
planner_status make_objective_v2_refinement_guidance(
    const objective_v2_calibration &calibration,
    const objective_v2_refinement_workload &workload,
    objective_v2_refinement_guidance *out) noexcept;

planner_status apply_objective_v2_refinement_guidance(
    const objective_v2_refinement_guidance &guidance,
    cellpack::alternating_refinement_config *config) noexcept;

static_assert(std::is_trivially_copyable<objective_v2_coefficients>::value,
    "objective v2 coefficients must remain replaceable data");
static_assert(std::is_trivially_copyable<objective_v2_prediction>::value,
    "objective v2 predictions must remain diagnostic-record friendly");
static_assert(std::is_trivially_copyable<objective_v2_refinement_guidance>::value,
    "objective v2 refinement guidance must remain pointer-free evidence data");

} // namespace cellerator::planner
