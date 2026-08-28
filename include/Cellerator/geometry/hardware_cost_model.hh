#pragma once

#include "Cellerator/geometry/evaluator.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 hardware_cost_model_schema_version = 1u;
inline constexpr u32 hardware_cost_feature_count = 13u;

enum class hardware_execution_path : u32 {
    direct_warp_tiles = 1u,
    csr_fallback = 2u
};

enum class hardware_cost_partition : u32 {
    calibration = 1u,
    held_out = 2u
};

// Representation and work counters needed by the v1 execution model. These
// are raw denominators, not pre-normalized features or a physical-format ABI.
struct hardware_cost_shape {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 tile_row_width = 0u;
    u32 feature_block_width = 0u;
    u64 nnz_count = 0u;
    u64 tile_count = 0u;
    u64 tile_block_count = 0u;
    u64 row_block_entry_count = 0u;
    u64 metadata_bytes = 0u;
    u64 payload_bytes = 0u;
    u64 input_output_bytes = 0u;
    u32 index_width_bytes = 0u;
    u32 alignment_bytes = 0u;
    // Caller-defined estimate, not a hardware-counter claim. Campaign identity
    // binds the estimation rule so a later profiler-calibrated rule is safe.
    u64 estimated_memory_transactions = 0u;
};

struct hardware_cost_observation {
    u32 schema_version = hardware_cost_model_schema_version;
    hardware_execution_path path = hardware_execution_path::direct_warp_tiles;
    hardware_cost_partition partition = hardware_cost_partition::calibration;
    u64 campaign_identity = 0u;
    u64 configuration_identity = 0u;
    u64 hardware_identity = 0u;
    u64 toolchain_identity = 0u;
    u64 operation_identity = 0u;
    u64 cost_policy_identity = 0u;
    hardware_cost_shape shape{};
    u64 median_elapsed_nanoseconds = 0u;
    u64 correctness_items = 0u;
    u64 correctness_mismatches = 0u;
    u32 warmup_count = 0u;
    u32 repeat_count = 0u;
    u32 launches_per_repeat = 0u;
};

struct hardware_cost_fit_config {
    u32 schema_version = hardware_cost_model_schema_version;
    u64 campaign_identity = 0u;
    u64 hardware_identity = 0u;
    u64 toolchain_identity = 0u;
    u64 operation_identity = 0u;
    u32 supported_feature_block_width_mask = 0u;
    double ridge_regularization = 1.0e-6;
};

struct hardware_cost_path_model {
    bool available = false;
    u32 calibration_count = 0u;
    double feature_mean[hardware_cost_feature_count]{};
    double feature_scale[hardware_cost_feature_count]{};
    double coefficients[hardware_cost_feature_count]{};
};

struct hardware_cost_model {
    u32 schema_version = hardware_cost_model_schema_version;
    u64 model_identity = 0u;
    u64 campaign_identity = 0u;
    u64 hardware_identity = 0u;
    u64 toolchain_identity = 0u;
    u64 operation_identity = 0u;
    u32 supported_feature_block_width_mask = 0u;
    double ridge_regularization = 0.0;
    hardware_cost_path_model direct_warp_tiles{};
    hardware_cost_path_model csr_fallback{};
};

struct hardware_cost_prediction_error {
    u32 schema_version = hardware_cost_model_schema_version;
    hardware_execution_path path = hardware_execution_path::direct_warp_tiles;
    hardware_cost_partition partition = hardware_cost_partition::calibration;
    u64 configuration_identity = 0u;
    double observed_nanoseconds = 0.0;
    double predicted_nanoseconds = 0.0;
    double absolute_error_nanoseconds = 0.0;
    double absolute_percentage_error = 0.0;
};

struct hardware_cost_error_summary {
    u32 observation_count = 0u;
    double mean_absolute_error_nanoseconds = 0.0;
    double root_mean_squared_error_nanoseconds = 0.0;
    double mean_absolute_percentage_error = 0.0;
    double maximum_absolute_percentage_error = 0.0;
};

struct hardware_cost_validation_buffers {
    std::size_t prediction_capacity = 0u;
    hardware_cost_prediction_error *predictions = nullptr;
};

struct hardware_cost_validation_report {
    u32 schema_version = hardware_cost_model_schema_version;
    u64 model_identity = 0u;
    hardware_cost_error_summary direct_calibration{};
    hardware_cost_error_summary direct_held_out{};
    hardware_cost_error_summary csr_calibration{};
    hardware_cost_error_summary csr_held_out{};
};

struct hardware_cost_candidate {
    u64 candidate_identity = 0u;
    u64 cost_policy_identity = 0u;
    hardware_execution_path path = hardware_execution_path::direct_warp_tiles;
    hardware_cost_shape shape{};
    u64 storage_bytes = 0u;
};

struct hardware_autotune_config {
    u32 schema_version = hardware_cost_model_schema_version;
    u32 supported_feature_block_width_mask = 0u;
    double storage_byte_weight = 1.0;
    double runtime_nanosecond_weight = 0.0;
};

struct hardware_autotune_result {
    u32 schema_version = hardware_cost_model_schema_version;
    u64 model_identity = 0u;
    u64 candidate_identity = 0u;
    u64 cost_policy_identity = 0u;
    hardware_execution_path path = hardware_execution_path::direct_warp_tiles;
    u32 feature_block_width = 0u;
    u64 storage_bytes = 0u;
    double predicted_runtime_nanoseconds = 0.0;
    double objective = 0.0;
};

u32 hardware_cost_block_width_bit(u32 width) noexcept;

validation_result validate_hardware_cost_observation(
    const hardware_cost_observation &observation);

validation_result fit_hardware_cost_model(
    const hardware_cost_observation *observations,
    u32 observation_count,
    const hardware_cost_fit_config &config,
    hardware_cost_model *out);

validation_result predict_hardware_cost(
    const hardware_cost_model &model,
    hardware_execution_path path,
    const hardware_cost_shape &shape,
    double *predicted_nanoseconds);

validation_result evaluate_hardware_cost_model(
    const hardware_cost_model &model,
    const hardware_cost_observation *observations,
    u32 observation_count,
    const hardware_cost_validation_buffers &buffers,
    hardware_cost_validation_report *out);

validation_result evaluate_hardware_aware_objective(
    u64 storage_bytes,
    double predicted_runtime_nanoseconds,
    const hardware_autotune_config &config,
    double *out);

validation_result select_hardware_cost_candidate(
    const hardware_cost_model &model,
    const hardware_cost_candidate *candidates,
    u32 candidate_count,
    const hardware_autotune_config &config,
    hardware_autotune_result *out);

} // namespace cellpack
