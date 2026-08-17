#include "CellPack/hardware_cost_model.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace cellpack {
namespace {

constexpr u64 fnv1a_offset = 1469598103934665603ull;
constexpr u64 fnv1a_prime = 1099511628211ull;
constexpr u32 minimum_calibration_observations = 4u;

void hash_byte(u64 *hash, unsigned char value) noexcept {
    *hash ^= value;
    *hash *= fnv1a_prime;
}

void hash_u64(u64 *hash, u64 value) noexcept {
    for (u32 byte = 0u; byte < 8u; ++byte) {
        hash_byte(hash, static_cast<unsigned char>(value >> (byte * 8u)));
    }
}

void hash_double(u64 *hash, double value) noexcept {
    u64 bits = 0u;
    std::memcpy(&bits, &value, sizeof(bits));
    hash_u64(hash, bits);
}

u64 nonzero_hash(u64 value) noexcept { return value == 0u ? 1u : value; }

bool valid_path(hardware_execution_path path) noexcept {
    return path == hardware_execution_path::direct_warp_tiles
        || path == hardware_execution_path::csr_fallback;
}

bool valid_partition(hardware_cost_partition partition) noexcept {
    return partition == hardware_cost_partition::calibration
        || partition == hardware_cost_partition::held_out;
}

validation_result validate_shape(const hardware_cost_shape &shape) {
    if (shape.row_count == 0u || shape.feature_count == 0u || shape.nnz_count == 0u
        || shape.tile_row_width == 0u || shape.tile_row_width > 32u
        || shape.feature_block_width == 0u || shape.feature_block_width > 32u
        || shape.tile_count == 0u || shape.tile_block_count == 0u
        || shape.row_block_entry_count < shape.tile_block_count
        || shape.metadata_bytes == 0u || shape.payload_bytes == 0u
        || shape.input_output_bytes == 0u || shape.index_width_bytes == 0u
        || shape.index_width_bytes > 8u || shape.alignment_bytes == 0u
        || (shape.alignment_bytes & (shape.alignment_bytes - 1u)) != 0u
        || shape.estimated_memory_transactions == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "hardware-cost shape lacks raw work/representation denominators");
    }
    return validation_ok();
}

void raw_features(const hardware_cost_shape &shape,
    double features[hardware_cost_feature_count]) noexcept {
    features[0] = 1.0;
    features[1] = std::log1p(static_cast<double>(shape.row_count));
    features[2] = std::log1p(static_cast<double>(shape.nnz_count));
    features[3] = std::log1p(static_cast<double>(shape.metadata_bytes
        + shape.payload_bytes + shape.input_output_bytes));
    features[4] = std::log1p(static_cast<double>(shape.tile_count));
    features[5] = std::log1p(static_cast<double>(shape.tile_block_count));
    features[6] = std::log1p(static_cast<double>(shape.row_block_entry_count));
    features[7] = std::log1p(static_cast<double>(shape.feature_block_width));
    features[8] = static_cast<double>(shape.tile_block_count) / shape.tile_count;
    features[9] = static_cast<double>(shape.row_block_entry_count)
        / shape.tile_block_count;
    features[10] = std::log1p(static_cast<double>(shape.index_width_bytes));
    features[11] = std::log1p(static_cast<double>(shape.alignment_bytes));
    features[12] = std::log1p(static_cast<double>(shape.estimated_memory_transactions));
}

hardware_cost_path_model *path_model(
    hardware_cost_model *model, hardware_execution_path path) noexcept {
    return path == hardware_execution_path::direct_warp_tiles
        ? &model->direct_warp_tiles : &model->csr_fallback;
}

const hardware_cost_path_model *path_model(
    const hardware_cost_model &model, hardware_execution_path path) noexcept {
    return path == hardware_execution_path::direct_warp_tiles
        ? &model.direct_warp_tiles : &model.csr_fallback;
}

validation_result solve_system(
    double matrix[hardware_cost_feature_count][hardware_cost_feature_count],
    double rhs[hardware_cost_feature_count],
    double solution[hardware_cost_feature_count]) {
    for (u32 column = 0u; column < hardware_cost_feature_count; ++column) {
        u32 pivot = column;
        for (u32 row = column + 1u; row < hardware_cost_feature_count; ++row) {
            if (std::fabs(matrix[row][column]) > std::fabs(matrix[pivot][column])) {
                pivot = row;
            }
        }
        if (!std::isfinite(matrix[pivot][column])
            || std::fabs(matrix[pivot][column]) < 1.0e-15) {
            return validation_error(validation_code::invalid_matrix_view, column,
                "hardware-cost regression system is singular");
        }
        if (pivot != column) {
            for (u32 entry = column; entry < hardware_cost_feature_count; ++entry) {
                std::swap(matrix[column][entry], matrix[pivot][entry]);
            }
            std::swap(rhs[column], rhs[pivot]);
        }
        const double scale = matrix[column][column];
        for (u32 entry = column; entry < hardware_cost_feature_count; ++entry) {
            matrix[column][entry] /= scale;
        }
        rhs[column] /= scale;
        for (u32 row = 0u; row < hardware_cost_feature_count; ++row) {
            if (row == column) continue;
            const double factor = matrix[row][column];
            for (u32 entry = column; entry < hardware_cost_feature_count; ++entry) {
                matrix[row][entry] -= factor * matrix[column][entry];
            }
            rhs[row] -= factor * rhs[column];
        }
    }
    for (u32 index = 0u; index < hardware_cost_feature_count; ++index) {
        if (!std::isfinite(rhs[index])) {
            return validation_error(validation_code::integer_overflow, index,
                "hardware-cost regression coefficient is not finite");
        }
        solution[index] = rhs[index];
    }
    return validation_ok();
}

validation_result fit_path(
    const hardware_cost_observation *observations,
    u32 observation_count,
    hardware_execution_path path,
    double ridge,
    hardware_cost_path_model *out) {
    hardware_cost_path_model result;
    for (u32 index = 0u; index < observation_count; ++index) {
        if (observations[index].path == path
            && observations[index].partition == hardware_cost_partition::calibration) {
            ++result.calibration_count;
            double features[hardware_cost_feature_count];
            raw_features(observations[index].shape, features);
            for (u32 feature = 1u; feature < hardware_cost_feature_count; ++feature) {
                result.feature_mean[feature] += features[feature];
            }
        }
    }
    if (result.calibration_count < minimum_calibration_observations) {
        return validation_error(validation_code::insufficient_capacity,
            result.calibration_count,
            "hardware-cost path has too few calibration observations");
    }
    result.feature_mean[0] = 0.0;
    result.feature_scale[0] = 1.0;
    for (u32 feature = 1u; feature < hardware_cost_feature_count; ++feature) {
        result.feature_mean[feature] /= result.calibration_count;
    }
    for (u32 index = 0u; index < observation_count; ++index) {
        if (observations[index].path != path
            || observations[index].partition != hardware_cost_partition::calibration) continue;
        double features[hardware_cost_feature_count];
        raw_features(observations[index].shape, features);
        for (u32 feature = 1u; feature < hardware_cost_feature_count; ++feature) {
            const double delta = features[feature] - result.feature_mean[feature];
            result.feature_scale[feature] += delta * delta;
        }
    }
    for (u32 feature = 1u; feature < hardware_cost_feature_count; ++feature) {
        result.feature_scale[feature] = std::sqrt(
            result.feature_scale[feature] / result.calibration_count);
        if (result.feature_scale[feature] < 1.0e-12) result.feature_scale[feature] = 1.0;
    }

    double matrix[hardware_cost_feature_count][hardware_cost_feature_count]{};
    double rhs[hardware_cost_feature_count]{};
    for (u32 index = 0u; index < observation_count; ++index) {
        const hardware_cost_observation &observation = observations[index];
        if (observation.path != path
            || observation.partition != hardware_cost_partition::calibration) continue;
        double features[hardware_cost_feature_count];
        raw_features(observation.shape, features);
        features[0] = 1.0;
        for (u32 feature = 1u; feature < hardware_cost_feature_count; ++feature) {
            features[feature] = (features[feature] - result.feature_mean[feature])
                / result.feature_scale[feature];
        }
        const double target = std::log(
            static_cast<double>(observation.median_elapsed_nanoseconds));
        for (u32 row = 0u; row < hardware_cost_feature_count; ++row) {
            rhs[row] += features[row] * target;
            for (u32 column = 0u; column < hardware_cost_feature_count; ++column) {
                matrix[row][column] += features[row] * features[column];
            }
        }
    }
    for (u32 diagonal = 1u; diagonal < hardware_cost_feature_count; ++diagonal) {
        matrix[diagonal][diagonal] += ridge;
    }
    validation_result status = solve_system(matrix, rhs, result.coefficients);
    if (!status) return status;
    result.available = true;
    *out = result;
    return validation_ok();
}

u64 model_identity(const hardware_cost_model &model) noexcept {
    u64 hash = fnv1a_offset;
    hash_u64(&hash, hardware_cost_model_schema_version);
    hash_u64(&hash, model.campaign_identity);
    hash_u64(&hash, model.hardware_identity);
    hash_u64(&hash, model.toolchain_identity);
    hash_u64(&hash, model.operation_identity);
    hash_u64(&hash, model.supported_feature_block_width_mask);
    hash_double(&hash, model.ridge_regularization);
    for (const hardware_cost_path_model *path :
        {&model.direct_warp_tiles, &model.csr_fallback}) {
        hash_u64(&hash, path->available ? 1u : 0u);
        hash_u64(&hash, path->calibration_count);
        for (u32 feature = 0u; feature < hardware_cost_feature_count; ++feature) {
            hash_double(&hash, path->feature_mean[feature]);
            hash_double(&hash, path->feature_scale[feature]);
            hash_double(&hash, path->coefficients[feature]);
        }
    }
    return nonzero_hash(hash);
}

struct error_accumulator {
    u32 count = 0u;
    double absolute_total = 0.0;
    double squared_total = 0.0;
    double percentage_total = 0.0;
    double maximum_percentage = 0.0;

    void add(double absolute_error, double percentage_error) noexcept {
        ++count;
        absolute_total += absolute_error;
        squared_total += absolute_error * absolute_error;
        percentage_total += percentage_error;
        maximum_percentage = std::max(maximum_percentage, percentage_error);
    }

    hardware_cost_error_summary finish() const noexcept {
        hardware_cost_error_summary result;
        result.observation_count = count;
        if (count == 0u) return result;
        result.mean_absolute_error_nanoseconds = absolute_total / count;
        result.root_mean_squared_error_nanoseconds = std::sqrt(squared_total / count);
        result.mean_absolute_percentage_error = percentage_total / count;
        result.maximum_absolute_percentage_error = maximum_percentage;
        return result;
    }
};

} // namespace

u32 hardware_cost_block_width_bit(u32 width) noexcept {
    return width == 0u || width > 32u ? 0u : (u32{1u} << (width - 1u));
}

validation_result validate_hardware_cost_observation(
    const hardware_cost_observation &observation) {
    if (observation.schema_version != hardware_cost_model_schema_version) {
        return validation_error(validation_code::unsupported_version,
            observation.schema_version, "unsupported hardware-cost observation schema");
    }
    if (!valid_path(observation.path) || !valid_partition(observation.partition)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "hardware-cost observation path or partition is invalid");
    }
    if (observation.campaign_identity == 0u || observation.configuration_identity == 0u
        || observation.hardware_identity == 0u || observation.toolchain_identity == 0u
        || observation.operation_identity == 0u || observation.cost_policy_identity == 0u) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "hardware-cost observation identities must be explicit");
    }
    validation_result status = validate_shape(observation.shape);
    if (!status) return status;
    if (observation.median_elapsed_nanoseconds == 0u
        || observation.correctness_items == 0u
        || observation.correctness_mismatches != 0u
        || observation.warmup_count == 0u || observation.repeat_count == 0u
        || observation.launches_per_repeat == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "hardware-cost observation lacks timing/correctness denominators");
    }
    return validation_ok();
}

validation_result fit_hardware_cost_model(
    const hardware_cost_observation *observations,
    u32 observation_count,
    const hardware_cost_fit_config &config,
    hardware_cost_model *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "hardware-cost model output is null");
    }
    if (observations == nullptr || observation_count == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "hardware-cost observations are empty");
    }
    if (config.schema_version != hardware_cost_model_schema_version) {
        return validation_error(validation_code::unsupported_version,
            config.schema_version, "unsupported hardware-cost fit schema");
    }
    if (config.campaign_identity == 0u || config.hardware_identity == 0u
        || config.toolchain_identity == 0u || config.operation_identity == 0u
        || config.supported_feature_block_width_mask == 0u
        || !std::isfinite(config.ridge_regularization)
        || config.ridge_regularization <= 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "hardware-cost fit configuration is incomplete");
    }
    u32 held_out_direct = 0u, held_out_csr = 0u;
    for (u32 index = 0u; index < observation_count; ++index) {
        validation_result status = validate_hardware_cost_observation(observations[index]);
        if (!status) return status;
        const hardware_cost_observation &observation = observations[index];
        if (observation.campaign_identity != config.campaign_identity
            || observation.hardware_identity != config.hardware_identity
            || observation.toolchain_identity != config.toolchain_identity
            || observation.operation_identity != config.operation_identity
            || (hardware_cost_block_width_bit(observation.shape.feature_block_width)
                & config.supported_feature_block_width_mask) == 0u) {
            return validation_error(validation_code::invalid_signature, index,
                "hardware-cost observation disagrees with fit provenance");
        }
        for (u32 prior = 0u; prior < index; ++prior) {
            if (observations[prior].configuration_identity
                == observation.configuration_identity) {
                return validation_error(validation_code::duplicate_id, index,
                    "hardware-cost configuration identities must be unique");
            }
        }
        if (observation.partition == hardware_cost_partition::held_out) {
            if (observation.path == hardware_execution_path::direct_warp_tiles) {
                ++held_out_direct;
            } else {
                ++held_out_csr;
            }
        }
    }
    if (held_out_direct == 0u || held_out_csr == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "hardware-cost fit requires held-out observations for both paths");
    }

    hardware_cost_model result;
    result.campaign_identity = config.campaign_identity;
    result.hardware_identity = config.hardware_identity;
    result.toolchain_identity = config.toolchain_identity;
    result.operation_identity = config.operation_identity;
    result.supported_feature_block_width_mask =
        config.supported_feature_block_width_mask;
    result.ridge_regularization = config.ridge_regularization;
    validation_result status = fit_path(observations, observation_count,
        hardware_execution_path::direct_warp_tiles, config.ridge_regularization,
        &result.direct_warp_tiles);
    if (!status) return status;
    status = fit_path(observations, observation_count,
        hardware_execution_path::csr_fallback, config.ridge_regularization,
        &result.csr_fallback);
    if (!status) return status;
    result.model_identity = model_identity(result);
    *out = result;
    return validation_ok();
}

validation_result predict_hardware_cost(
    const hardware_cost_model &model,
    hardware_execution_path path,
    const hardware_cost_shape &shape,
    double *predicted_nanoseconds) {
    if (predicted_nanoseconds == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "hardware-cost prediction output is null");
    }
    if (model.schema_version != hardware_cost_model_schema_version
        || model.model_identity == 0u || model.model_identity != model_identity(model)
        || !valid_path(path)) {
        return validation_error(validation_code::unsupported_version, invalid_id,
            "hardware-cost model identity or path is invalid");
    }
    validation_result status = validate_shape(shape);
    if (!status) return status;
    if ((hardware_cost_block_width_bit(shape.feature_block_width)
        & model.supported_feature_block_width_mask) == 0u) {
        return validation_error(validation_code::invalid_plan_geometry,
            shape.feature_block_width, "hardware-cost block width is unsupported");
    }
    const hardware_cost_path_model *selected = path_model(model, path);
    if (!selected->available) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "hardware-cost execution path is unavailable");
    }
    double features[hardware_cost_feature_count];
    raw_features(shape, features);
    features[0] = 1.0;
    for (u32 feature = 1u; feature < hardware_cost_feature_count; ++feature) {
        features[feature] = (features[feature] - selected->feature_mean[feature])
            / selected->feature_scale[feature];
    }
    double log_prediction = 0.0;
    for (u32 feature = 0u; feature < hardware_cost_feature_count; ++feature) {
        log_prediction += selected->coefficients[feature] * features[feature];
    }
    const double prediction = std::exp(log_prediction);
    if (!std::isfinite(prediction) || prediction <= 0.0) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "hardware-cost prediction is not finite and positive");
    }
    *predicted_nanoseconds = prediction;
    return validation_ok();
}

validation_result evaluate_hardware_cost_model(
    const hardware_cost_model &model,
    const hardware_cost_observation *observations,
    u32 observation_count,
    const hardware_cost_validation_buffers &buffers,
    hardware_cost_validation_report *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "hardware-cost validation report is null");
    }
    if (observations == nullptr || observation_count == 0u
        || buffers.predictions == nullptr
        || buffers.prediction_capacity < observation_count) {
        return validation_error(validation_code::insufficient_capacity,
            observation_count, "hardware-cost validation buffers are insufficient");
    }
    error_accumulator direct_calibration, direct_held_out, csr_calibration,
        csr_held_out;
    for (u32 index = 0u; index < observation_count; ++index) {
        validation_result status = validate_hardware_cost_observation(observations[index]);
        if (!status) return status;
        const hardware_cost_observation &observation = observations[index];
        if (observation.campaign_identity != model.campaign_identity
            || observation.hardware_identity != model.hardware_identity
            || observation.toolchain_identity != model.toolchain_identity
            || observation.operation_identity != model.operation_identity) {
            return validation_error(validation_code::invalid_signature, index,
                "hardware-cost validation provenance disagrees with model");
        }
        double prediction = 0.0;
        status = predict_hardware_cost(model, observation.path, observation.shape,
            &prediction);
        if (!status) return status;
        const double observed = static_cast<double>(
            observation.median_elapsed_nanoseconds);
        const double absolute_error = std::fabs(prediction - observed);
        const double percentage_error = absolute_error / observed;
        hardware_cost_prediction_error packet;
        packet.path = observation.path;
        packet.partition = observation.partition;
        packet.configuration_identity = observation.configuration_identity;
        packet.observed_nanoseconds = observed;
        packet.predicted_nanoseconds = prediction;
        packet.absolute_error_nanoseconds = absolute_error;
        packet.absolute_percentage_error = percentage_error;
        buffers.predictions[index] = packet;
        error_accumulator *accumulator = nullptr;
        if (observation.path == hardware_execution_path::direct_warp_tiles) {
            accumulator = observation.partition == hardware_cost_partition::calibration
                ? &direct_calibration : &direct_held_out;
        } else {
            accumulator = observation.partition == hardware_cost_partition::calibration
                ? &csr_calibration : &csr_held_out;
        }
        accumulator->add(absolute_error, percentage_error);
    }
    hardware_cost_validation_report result;
    result.model_identity = model.model_identity;
    result.direct_calibration = direct_calibration.finish();
    result.direct_held_out = direct_held_out.finish();
    result.csr_calibration = csr_calibration.finish();
    result.csr_held_out = csr_held_out.finish();
    *out = result;
    return validation_ok();
}

validation_result evaluate_hardware_aware_objective(
    u64 storage_bytes,
    double predicted_runtime_nanoseconds,
    const hardware_autotune_config &config,
    double *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "hardware-aware objective output is null");
    }
    if (config.schema_version != hardware_cost_model_schema_version
        || config.supported_feature_block_width_mask == 0u
        || !std::isfinite(config.storage_byte_weight)
        || !std::isfinite(config.runtime_nanosecond_weight)
        || config.storage_byte_weight < 0.0 || config.runtime_nanosecond_weight < 0.0
        || (config.storage_byte_weight == 0.0
            && config.runtime_nanosecond_weight == 0.0)
        || storage_bytes == 0u || !std::isfinite(predicted_runtime_nanoseconds)
        || predicted_runtime_nanoseconds <= 0.0) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "hardware-aware objective configuration or denominators are invalid");
    }
    const double objective = config.storage_byte_weight * storage_bytes
        + config.runtime_nanosecond_weight * predicted_runtime_nanoseconds;
    if (!std::isfinite(objective)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "hardware-aware objective overflowed");
    }
    *out = objective;
    return validation_ok();
}

validation_result select_hardware_cost_candidate(
    const hardware_cost_model &model,
    const hardware_cost_candidate *candidates,
    u32 candidate_count,
    const hardware_autotune_config &config,
    hardware_autotune_result *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
            "hardware-autotune result is null");
    }
    if (candidates == nullptr || candidate_count == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "hardware-autotune candidate set is empty");
    }
    if ((config.supported_feature_block_width_mask
        & model.supported_feature_block_width_mask) == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "hardware-autotune and model width sets do not overlap");
    }
    hardware_autotune_result best;
    bool found = false;
    for (u32 index = 0u; index < candidate_count; ++index) {
        const hardware_cost_candidate &candidate = candidates[index];
        if (candidate.candidate_identity == 0u || candidate.cost_policy_identity == 0u
            || candidate.storage_bytes == 0u || !valid_path(candidate.path)) {
            return validation_error(validation_code::invalid_plan_geometry, index,
                "hardware-autotune candidate is invalid");
        }
        const u32 width_bit = hardware_cost_block_width_bit(
            candidate.shape.feature_block_width);
        if ((width_bit & config.supported_feature_block_width_mask
            & model.supported_feature_block_width_mask) == 0u) continue;
        double runtime = 0.0, objective = 0.0;
        validation_result status = predict_hardware_cost(
            model, candidate.path, candidate.shape, &runtime);
        if (!status) return status;
        status = evaluate_hardware_aware_objective(
            candidate.storage_bytes, runtime, config, &objective);
        if (!status) return status;
        if (!found || objective < best.objective
            || (objective == best.objective
                && candidate.candidate_identity < best.candidate_identity)) {
            found = true;
            best.model_identity = model.model_identity;
            best.candidate_identity = candidate.candidate_identity;
            best.cost_policy_identity = candidate.cost_policy_identity;
            best.path = candidate.path;
            best.feature_block_width = candidate.shape.feature_block_width;
            best.storage_bytes = candidate.storage_bytes;
            best.predicted_runtime_nanoseconds = runtime;
            best.objective = objective;
        }
    }
    if (!found) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
            "hardware-autotune has no candidate with a supported width");
    }
    *out = best;
    return validation_ok();
}

} // namespace cellpack
