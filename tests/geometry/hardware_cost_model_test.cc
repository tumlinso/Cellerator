#include "Cellerator/geometry/hardware_cost_model.hh"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {

using cellpack::u32;
using cellpack::u64;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackHardwareCostModelTest: " << message << '\n';
        std::exit(1);
    }
}

void require_status(cellpack::validation_result status, const char *message) {
    if (!status) {
        std::cerr << "cellPackHardwareCostModelTest: " << message << ": "
                  << status.message << " (index=" << status.index << ")\n";
        std::exit(1);
    }
}

cellpack::hardware_cost_shape make_shape(u32 index) {
    cellpack::hardware_cost_shape shape;
    shape.row_count = 4096u << (index % 4u);
    shape.feature_count = 65536u;
    shape.tile_row_width = 32u;
    shape.feature_block_width = index % 3u == 0u ? 8u
        : (index % 3u == 1u ? 16u : 32u);
    shape.tile_count = shape.row_count / shape.tile_row_width;
    const u64 groups = 1u << (index % 6u);
    shape.tile_block_count = shape.tile_count * groups;
    shape.row_block_entry_count = shape.tile_block_count * (1u + index % 11u);
    shape.nnz_count = shape.row_block_entry_count
        * (1u + (index * 7u) % shape.feature_block_width);
    shape.metadata_bytes = shape.tile_count * 16u
        + shape.tile_block_count * 20u + shape.row_block_entry_count * 8u;
    shape.payload_bytes = shape.nnz_count * 2u;
    shape.input_output_bytes = shape.feature_count * 2u + shape.row_count * 4u;
    shape.index_width_bytes = 4u;
    shape.alignment_bytes = 4u;
    shape.estimated_memory_transactions = (shape.metadata_bytes + shape.payload_bytes
        + shape.input_output_bytes + 31u) / 32u;
    return shape;
}

u64 synthetic_runtime(cellpack::hardware_execution_path path,
    const cellpack::hardware_cost_shape &shape) {
    const double bytes = static_cast<double>(shape.metadata_bytes
        + shape.payload_bytes + shape.input_output_bytes);
    const double log_runtime =
        (path == cellpack::hardware_execution_path::direct_warp_tiles ? 4.1 : 4.7)
        + 0.05 * std::log1p(static_cast<double>(shape.row_count))
        + 0.08 * std::log1p(static_cast<double>(shape.nnz_count))
        + 0.11 * std::log1p(bytes)
        + 0.03 * std::log1p(static_cast<double>(shape.tile_count))
        + 0.07 * std::log1p(static_cast<double>(shape.tile_block_count))
        + 0.06 * std::log1p(static_cast<double>(shape.row_block_entry_count))
        + 0.02 * std::log1p(static_cast<double>(shape.feature_block_width))
        + 0.004 * static_cast<double>(shape.tile_block_count) / shape.tile_count
        + 0.006 * static_cast<double>(shape.row_block_entry_count)
            / shape.tile_block_count
        + 0.02 * std::log1p(static_cast<double>(shape.estimated_memory_transactions));
    return static_cast<u64>(std::llround(std::exp(log_runtime)));
}

std::vector<cellpack::hardware_cost_observation> make_observations() {
    std::vector<cellpack::hardware_cost_observation> observations;
    for (u32 index = 0u; index < 72u; ++index) {
        for (u32 path_index = 0u; path_index < 2u; ++path_index) {
            cellpack::hardware_cost_observation observation;
            observation.path = path_index == 0u
                ? cellpack::hardware_execution_path::direct_warp_tiles
                : cellpack::hardware_execution_path::csr_fallback;
            observation.partition = index % 6u == 0u
                ? cellpack::hardware_cost_partition::held_out
                : cellpack::hardware_cost_partition::calibration;
            observation.campaign_identity = 0x1100u;
            observation.configuration_identity = 1u + index * 2u + path_index;
            observation.hardware_identity = 0x2200u;
            observation.toolchain_identity = 0x3300u;
            observation.operation_identity = 0x4400u;
            observation.cost_policy_identity = 0x5500u;
            observation.shape = make_shape(index);
            observation.median_elapsed_nanoseconds = synthetic_runtime(
                observation.path, observation.shape);
            observation.correctness_items = observation.shape.row_count;
            observation.warmup_count = 3u;
            observation.repeat_count = 11u;
            observation.launches_per_repeat = 1u;
            observations.push_back(observation);
        }
    }
    return observations;
}

cellpack::hardware_cost_fit_config fit_config() {
    cellpack::hardware_cost_fit_config config;
    config.campaign_identity = 0x1100u;
    config.hardware_identity = 0x2200u;
    config.toolchain_identity = 0x3300u;
    config.operation_identity = 0x4400u;
    config.supported_feature_block_width_mask =
        cellpack::hardware_cost_block_width_bit(8u)
        | cellpack::hardware_cost_block_width_bit(16u)
        | cellpack::hardware_cost_block_width_bit(32u);
    config.ridge_regularization = 1.0e-6;
    return config;
}

void test_fit_predict_and_validate() {
    auto observations = make_observations();
    cellpack::hardware_cost_model first, second;
    require_status(cellpack::fit_hardware_cost_model(observations.data(),
        static_cast<u32>(observations.size()), fit_config(), &first), "fit model");
    require_status(cellpack::fit_hardware_cost_model(observations.data(),
        static_cast<u32>(observations.size()), fit_config(), &second),
        "repeat deterministic fit");
    require(first.model_identity != 0u && first.model_identity == second.model_identity,
        "fit must produce a stable nonzero identity");
    require(first.direct_warp_tiles.calibration_count == 60u,
        "direct calibration denominator");
    require(first.csr_fallback.calibration_count == 60u,
        "CSR calibration denominator");

    std::vector<cellpack::hardware_cost_prediction_error> errors(observations.size());
    cellpack::hardware_cost_validation_buffers buffers;
    buffers.prediction_capacity = errors.size();
    buffers.predictions = errors.data();
    cellpack::hardware_cost_validation_report report;
    require_status(cellpack::evaluate_hardware_cost_model(first, observations.data(),
        static_cast<u32>(observations.size()), buffers, &report), "evaluate model");
    require(report.direct_held_out.observation_count == 12u,
        "direct held-out denominator");
    require(report.csr_held_out.observation_count == 12u,
        "CSR held-out denominator");
    require(report.direct_held_out.mean_absolute_percentage_error < 0.10,
        "direct held-out error should stay bounded");
    require(report.csr_held_out.mean_absolute_percentage_error < 0.10,
        "CSR held-out error should stay bounded");

    double prediction = 0.0;
    require_status(cellpack::predict_hardware_cost(first,
        cellpack::hardware_execution_path::direct_warp_tiles, make_shape(17u),
        &prediction), "predict direct path");
    require(std::isfinite(prediction) && prediction > 0.0,
        "prediction must be finite and positive");

    auto corrupted = first;
    corrupted.direct_warp_tiles.coefficients[0] += 1.0;
    require(!cellpack::predict_hardware_cost(corrupted,
        cellpack::hardware_execution_path::direct_warp_tiles, make_shape(17u),
        &prediction), "model identity must detect coefficient tampering");
}

void test_autotune_and_width_constraints() {
    auto observations = make_observations();
    cellpack::hardware_cost_model model;
    require_status(cellpack::fit_hardware_cost_model(observations.data(),
        static_cast<u32>(observations.size()), fit_config(), &model), "fit selector model");

    cellpack::hardware_cost_candidate candidates[2];
    candidates[0].candidate_identity = 20u;
    candidates[0].cost_policy_identity = 0x5500u;
    candidates[0].path = cellpack::hardware_execution_path::direct_warp_tiles;
    candidates[0].shape = make_shape(10u);
    candidates[0].storage_bytes = 2000u;
    candidates[1] = candidates[0];
    candidates[1].candidate_identity = 10u;
    candidates[1].path = cellpack::hardware_execution_path::csr_fallback;
    candidates[1].storage_bytes = 1000u;

    cellpack::hardware_autotune_config config;
    config.supported_feature_block_width_mask = fit_config().supported_feature_block_width_mask;
    config.storage_byte_weight = 1.0;
    config.runtime_nanosecond_weight = 0.0;
    cellpack::hardware_autotune_result result;
    require_status(cellpack::select_hardware_cost_candidate(model, candidates, 2u,
        config, &result), "storage-only selection");
    require(result.candidate_identity == 10u, "storage-only objective must choose fewer bytes");

    config.runtime_nanosecond_weight = 1000.0;
    require_status(cellpack::select_hardware_cost_candidate(model, candidates, 2u,
        config, &result), "runtime-weighted selection");
    require(result.candidate_identity == 20u,
        "runtime weight must be able to select the faster direct path");

    config.runtime_nanosecond_weight = 0.0;
    candidates[1] = candidates[0];
    candidates[1].candidate_identity = 10u;
    require_status(cellpack::select_hardware_cost_candidate(model, candidates, 2u,
        config, &result), "deterministic tie selection");
    require(result.candidate_identity == 10u, "candidate identity must break exact ties");

    candidates[0].shape.feature_block_width = 8u;
    candidates[1].shape.feature_block_width = 16u;
    config.supported_feature_block_width_mask =
        cellpack::hardware_cost_block_width_bit(16u);
    require_status(cellpack::select_hardware_cost_candidate(model, candidates, 2u,
        config, &result), "supported-width selection");
    require(result.candidate_identity == 10u && result.feature_block_width == 16u,
        "supported-width mask must constrain plan selection");

    candidates[0].shape.feature_block_width = 7u;
    candidates[1].shape.feature_block_width = 7u;
    require(!cellpack::select_hardware_cost_candidate(model, candidates, 2u,
        config, &result), "unsupported widths must not be selected");
    require(cellpack::hardware_cost_block_width_bit(0u) == 0u
        && cellpack::hardware_cost_block_width_bit(32u) == 0x80000000u
        && cellpack::hardware_cost_block_width_bit(33u) == 0u,
        "width-mask boundaries");
}

void test_adversarial_provenance() {
    auto observations = make_observations();
    cellpack::hardware_cost_model model;

    auto tampered = observations;
    tampered[1].configuration_identity = tampered[0].configuration_identity;
    require(!cellpack::fit_hardware_cost_model(tampered.data(),
        static_cast<u32>(tampered.size()), fit_config(), &model),
        "duplicate configurations must be rejected");

    tampered = observations;
    tampered[3].correctness_mismatches = 1u;
    require(!cellpack::fit_hardware_cost_model(tampered.data(),
        static_cast<u32>(tampered.size()), fit_config(), &model),
        "incorrect benchmark samples must be rejected");

    tampered = observations;
    tampered[5].toolchain_identity ^= 1u;
    require(!cellpack::fit_hardware_cost_model(tampered.data(),
        static_cast<u32>(tampered.size()), fit_config(), &model),
        "mixed toolchain provenance must be rejected");

    tampered = observations;
    for (auto &observation : tampered) {
        if (observation.path == cellpack::hardware_execution_path::csr_fallback) {
            observation.partition = cellpack::hardware_cost_partition::calibration;
        }
    }
    require(!cellpack::fit_hardware_cost_model(tampered.data(),
        static_cast<u32>(tampered.size()), fit_config(), &model),
        "both paths require held-out observations");
}

} // namespace

int main() {
    test_fit_predict_and_validate();
    test_autotune_and_width_constraints();
    test_adversarial_provenance();
    std::cout << "cellPackHardwareCostModelTest: passed\n";
    return 0;
}
