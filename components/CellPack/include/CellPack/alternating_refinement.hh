#pragma once

#include "CellPack/record_statistical_validation.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 alternating_refinement_schema_version = 2u;

enum class alternating_refinement_phase : u32 {
    baseline = 0u,
    gene_blocks = 1u,
    cell_order_and_tiles = 2u
};

enum class alternating_refinement_outcome : u32 {
    accepted = 1u,
    rejected_no_improvement = 2u,
    rejected_evaluation_error = 3u
};

enum class alternating_refinement_stop_reason : u32 {
    candidate_sequence_exhausted = 1u,
    iteration_cap = 2u,
    evaluation_cap = 3u,
    convergence = 4u,
    preprocessing_cap = 5u
};

// V2 retains the measured representation/runtime terms and adds an optional
// evidence-bound workload profile. It does not persist hardware predictions
// or candidate identity in the packing plan.
struct alternating_refinement_objective_weights {
    double encoded_bytes = 1.0;
    double metadata_bytes = 0.0;
    double active_block_references = 0.0;
    double tile_block_union_references = 0.0;
    double padding_slots = 0.0;
    double runtime_mean_nanoseconds = 0.0;
    double preprocessing_mean_nanoseconds = 0.0;
    double forward_mean_nanoseconds = 0.0;
    double transpose_mean_nanoseconds = 0.0;
    double active_interaction_nanoseconds = 0.0;
    double partition_cut_edge_nanoseconds = 0.0;
    double bootstrap_mad_nanoseconds = 0.0;
};

struct alternating_refinement_config {
    u32 schema_version = alternating_refinement_schema_version;
    u32 maximum_iterations = 16u;
    u32 maximum_evaluations = 16u;
    u32 maximum_consecutive_rejections = 4u;
    u64 maximum_preprocessing_nanoseconds = 0u;
    u64 dataset_identity = 0u;
    u64 feature_axis_identity = 0u;
    u32 feature_axis_identity_version = 0u;
    u64 row_domain_identity = 0u;
    u64 split_identity = 0u;
    u64 workload_profile_identity = 0u;
    u64 workload_evidence_revision = 0u;
    u64 seed = 0u;
    u32 minimum_bootstrap_samples = 0u;
    u32 reserved = 0u;
    double absolute_improvement_tolerance = 0.0;
    double relative_improvement_tolerance = 0.0;
    alternating_refinement_objective_weights weights{};
};

// Candidates are caller-materialized through the public CP-BP-04/07/08/09/11
// pipeline. The controller owns acceptance and rollback, never hidden packing.
struct alternating_refinement_observation {
    u32 schema_version = alternating_refinement_schema_version;
    alternating_refinement_phase phase = alternating_refinement_phase::baseline;
    u32 iteration = 0u;
    u64 candidate_identity = 0u;
    u64 parent_plan_identity = 0u;
    const frozen_packing_plan *plan = nullptr;
    packing_validation_metrics training{};
    packing_validation_metrics held_out{};
    bool evaluation_succeeded = true;
};

struct alternating_refinement_event {
    u32 schema_version = alternating_refinement_schema_version;
    u32 iteration = 0u;
    alternating_refinement_phase phase = alternating_refinement_phase::baseline;
    alternating_refinement_outcome outcome =
        alternating_refinement_outcome::rejected_evaluation_error;
    u64 candidate_identity = 0u;
    u64 parent_plan_identity = 0u;
    u64 candidate_plan_identity = 0u;
    double training_objective = 0.0;
    double held_out_objective = 0.0;
    double previous_best_held_out_objective = 0.0;
    double held_out_improvement = 0.0;
};

struct alternating_refinement_buffers {
    std::size_t event_capacity = 0u;
    alternating_refinement_event *events = nullptr;
};

struct alternating_refinement_result {
    u32 schema_version = alternating_refinement_schema_version;
    alternating_refinement_stop_reason stop_reason =
        alternating_refinement_stop_reason::candidate_sequence_exhausted;
    u32 attempted_iterations = 0u;
    u32 evaluated_candidates = 0u;
    u32 accepted_candidates = 0u;
    u32 rejected_candidates = 0u;
    u32 evaluation_errors = 0u;
    u32 consecutive_rejections = 0u;
    u64 controller_identity = 0u;
    u64 best_plan_identity = 0u;
    u64 total_preprocessing_nanoseconds = 0u;
    const frozen_packing_plan *best_plan = nullptr;
    packing_validation_metrics best_training{};
    packing_validation_metrics best_held_out{};
    double best_training_objective = 0.0;
    double best_held_out_objective = 0.0;
    const alternating_refinement_event *events = nullptr;
    u32 event_count = 0u;
};

// Same semantic identity used by CP-BP-11 frozen-plan validation. It is stable
// under pointer relocation but intentionally changes with canonical membership.
u64 alternating_refinement_plan_identity(
    const frozen_packing_plan &plan) noexcept;

validation_result evaluate_alternating_refinement_objective(
    const packing_validation_metrics &metrics,
    const alternating_refinement_objective_weights &weights,
    double *out);

validation_result run_alternating_refinement(
    const alternating_refinement_observation &baseline,
    const alternating_refinement_observation *candidates,
    u32 candidate_count,
    const alternating_refinement_config &config,
    const alternating_refinement_buffers &buffers,
    alternating_refinement_result *out);

} // namespace cellpack
