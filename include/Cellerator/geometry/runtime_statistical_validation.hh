#pragma once

#include "Cellerator/geometry/alternating_refinement.hh"
#include "Cellerator/geometry/tile_statistical_validation.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 runtime_statistical_validation_schema_version = 1u;

enum class runtime_timing_scope : u32 {
    device_resident_kernel = 1u
};

// One raw, reproducible timing packet. elapsed_nanoseconds is the aggregate of
// repeat_count launches after warmup_count launches; setup, allocation,
// transfers, and synchronization outside event timing are excluded in v1.
struct relearned_plan_runtime_observation {
    u32 schema_version = runtime_statistical_validation_schema_version;
    runtime_timing_scope timing_scope = runtime_timing_scope::device_resident_kernel;
    u64 controller_identity = 0u;
    u64 plan_identity = 0u;
    u64 bootstrap_identity = 0u;
    u64 split_identity = 0u;
    u64 dataset_identity = 0u;
    u64 feature_axis_identity = 0u;
    u64 row_domain_identity = 0u;
    u64 ordering_identity = 0u;
    u64 tile_identity = 0u;
    u64 operation_identity = 0u;
    u64 feature_weight_identity = 0u;
    u64 hardware_identity = 0u;
    u64 toolchain_identity = 0u;
    u64 input_nnz = 0u;
    u64 input_bytes = 0u;
    u64 elapsed_nanoseconds = 0u;
    u64 correctness_items = 0u;
    u64 correctness_mismatches = 0u;
    u32 warmup_count = 0u;
    u32 repeat_count = 0u;
    u32 launches_per_repeat = 0u;
    bool observed = true;
};

// Each bootstrap replicate must be the output of a real CP-BP-10 controller
// run. The plan is obtained from refinement->best_plan, never supplied again.
struct relearned_plan_runtime_input {
    const validation_bootstrap_provenance *bootstrap_provenance = nullptr;
    const u32 *row_multiplicities = nullptr;
    const alternating_refinement_result *refinement = nullptr;
    relearned_plan_runtime_observation runtime{};
};

struct relearned_plan_runtime_replicate {
    u32 schema_version = runtime_statistical_validation_schema_version;
    u64 bootstrap_identity = 0u;
    u64 controller_identity = 0u;
    u64 plan_identity = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 co_membership_pair_count = 0u;
    u64 co_membership_agreements = 0u;
    u64 co_membership_disagreements = 0u;
    bool exact_label_invariant_mapping = false;
    relearned_plan_runtime_observation runtime{};
    packing_validation_metrics training{};
    packing_validation_metrics held_out{};
};

struct relearned_plan_runtime_buffers {
    std::size_t replicate_capacity = 0u;
    relearned_plan_runtime_replicate *replicates = nullptr;
};

struct relearned_plan_runtime_stability_summary {
    u32 schema_version = runtime_statistical_validation_schema_version;
    u32 repeat_count = 0u;
    u32 exact_mapping_count = 0u;
    u64 reference_plan_identity = 0u;
    u64 dataset_identity = 0u;
    u64 feature_axis_identity = 0u;
    u64 row_domain_identity = 0u;
    u64 split_identity = 0u;
    u64 operation_identity = 0u;
    u64 feature_weight_identity = 0u;
    u64 hardware_identity = 0u;
    u64 toolchain_identity = 0u;
    validation_unit_kind unit_kind = validation_unit_kind::row_identity;
    bool claims_group_generalization = false;
    bootstrap_scalar_summary encoded_bytes{};
    bootstrap_scalar_summary metadata_bytes{};
    bootstrap_scalar_summary preprocessing_mean_nanoseconds{};
    bootstrap_scalar_summary runtime_mean_nanoseconds{};
    bootstrap_scalar_summary runtime_nnz_per_second{};
    bootstrap_scalar_summary runtime_gigabytes_per_second{};
    bootstrap_scalar_summary co_membership_agreement_fraction{};
};

validation_result validate_relearned_plan_runtime_observation(
    const relearned_plan_runtime_observation &observation);

validation_result evaluate_relearned_plan_runtime_stability(
    const validation_identity_view &identities,
    const relearned_plan_runtime_input *inputs,
    u32 input_count,
    const relearned_plan_runtime_buffers &buffers,
    relearned_plan_runtime_stability_summary *out);

} // namespace cellpack
