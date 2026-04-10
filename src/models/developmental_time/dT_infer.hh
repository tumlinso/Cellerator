#pragma once

#include "dT_model.hh"
#include "../../torch/bindings.hh"

#include "../../../extern/CellShard/src/sharded/sharded_host.cuh"

#include <torch/torch.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cellerator::models::developmental_time {

struct DevelopmentalTimeInferConfig {
    torch::Device device = torch::Device(torch::kCPU);
    bool move_model_to_device = true;
    bool drop_fetched_parts = true;
    bool calibrate_output = false;
    const SpeciesTimeCalibrator *calibrator = nullptr;
    std::string species_id;
};

struct DevelopmentalTimeBatchPrediction {
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
};

struct DevelopmentalTimeTable {
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
};

namespace detail {

inline std::int64_t checked_i64_developmental_time_(unsigned long value, const char *label) {
    if (value > static_cast<unsigned long>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit into int64");
    }
    return static_cast<std::int64_t>(value);
}

inline torch::Tensor copy_i64_tensor_developmental_time_(const std::vector<std::int64_t> &values) {
    torch::Tensor tensor = torch::empty(
        { static_cast<std::int64_t>(values.size()) },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
    }
    return tensor;
}

inline torch::Tensor copy_f32_tensor_developmental_time_(const std::vector<float> &values) {
    torch::Tensor tensor = torch::empty(
        { static_cast<std::int64_t>(values.size()) },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<float>(), values.data(), values.size() * sizeof(float));
    }
    return tensor;
}

inline void validate_developmental_time_infer_inputs_(BalancedTimeSampler::matrix_type *matrix) {
    if (matrix == nullptr) throw std::invalid_argument("developmental time inference requires a non-null CellShard matrix");
    if (matrix->num_parts == 0 || matrix->part_offsets == nullptr || matrix->part_rows == nullptr || matrix->part_aux == nullptr) {
        throw std::invalid_argument("developmental time inference requires initialized sharded CSR metadata");
    }
}

inline DevelopmentalTimeTable materialize_developmental_time_batches_(
    const std::vector<DevelopmentalTimeBatchPrediction> &batches) {
    if (batches.empty()) {
        return DevelopmentalTimeTable{
            torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)),
            torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        };
    }

    std::vector<torch::Tensor> all_indices;
    std::vector<torch::Tensor> all_time;
    all_indices.reserve(batches.size());
    all_time.reserve(batches.size());
    for (const DevelopmentalTimeBatchPrediction &batch : batches) {
        all_indices.push_back(batch.cell_indices);
        all_time.push_back(batch.developmental_time);
    }
    return DevelopmentalTimeTable{
        torch::cat(all_indices, 0).contiguous(),
        torch::cat(all_time, 0).contiguous()
    };
}

} // namespace detail

template<typename Callback>
void for_each_developmental_time_batch(
    DevelopmentalStageModel &model,
    BalancedTimeSampler::matrix_type *matrix,
    const BalancedTimeSampler::storage_type *storage,
    Callback &&callback,
    const DevelopmentalTimeInferConfig &config = DevelopmentalTimeInferConfig()) {
    detail::validate_developmental_time_infer_inputs_(matrix);
    if (config.calibrate_output && (config.calibrator == nullptr || config.species_id.empty())) {
        throw std::invalid_argument("calibrated developmental-time inference requires a calibrator and species_id");
    }

    torch::NoGradGuard no_grad;
    model->eval();
    if (config.move_model_to_device) model->to(config.device);

    for (unsigned long part_id = 0; part_id < matrix->num_parts; ++part_id) {
        bool loaded_here = false;
        const unsigned long row_begin_ul = matrix->part_offsets[part_id];
        const unsigned long row_end_ul = matrix->part_offsets[part_id + 1];
        const std::int64_t row_count = detail::checked_i64_developmental_time_(row_end_ul - row_begin_ul, "part row count");

        if (!cellshard::part_loaded(matrix, part_id)) {
            if (storage == nullptr) {
                throw std::runtime_error("developmental time inference encountered an unloaded part without shard storage");
            }
            if (!cellshard::fetch_part(matrix, storage, part_id)) {
                throw std::runtime_error("developmental time inference failed to fetch a CellShard part");
            }
            loaded_here = true;
        }

        const cellshard::sparse::compressed *part = matrix->parts[part_id];
        if (part == nullptr || part->axis != cellshard::sparse::compressed_by_row) {
            throw std::runtime_error("developmental time inference requires loaded row-compressed CSR parts");
        }

        torch::Tensor features = cellerator::torch_bindings::export_as_tensor(*part);
        features = features.to(config.device);
        torch::Tensor predicted_time = model->predict_stage(features);
        if (config.calibrate_output) {
            predicted_time = config.calibrator->stage_to_time(predicted_time, config.species_id);
        }
        predicted_time = predicted_time.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();

        std::vector<std::int64_t> batch_indices(static_cast<std::size_t>(row_count));
        for (std::int64_t local_row = 0; local_row < row_count; ++local_row) {
            const unsigned long global_row = row_begin_ul + static_cast<unsigned long>(local_row);
            batch_indices[static_cast<std::size_t>(local_row)] = detail::checked_i64_developmental_time_(global_row, "global row");
        }

        callback(DevelopmentalTimeBatchPrediction{
            detail::copy_i64_tensor_developmental_time_(batch_indices),
            std::move(predicted_time)
        });

        if (loaded_here && config.drop_fetched_parts) {
            cellshard::drop_part(matrix, part_id);
        }
    }
}

inline DevelopmentalTimeTable infer_developmental_time(
    DevelopmentalStageModel &model,
    BalancedTimeSampler::matrix_type *matrix,
    const BalancedTimeSampler::storage_type *storage = nullptr,
    const DevelopmentalTimeInferConfig &config = DevelopmentalTimeInferConfig()) {
    std::vector<DevelopmentalTimeBatchPrediction> batches;
    for_each_developmental_time_batch(
        model,
        matrix,
        storage,
        [&batches](DevelopmentalTimeBatchPrediction batch) { batches.push_back(std::move(batch)); },
        config);
    return detail::materialize_developmental_time_batches_(batches);
}

} // namespace cellerator::models::developmental_time
