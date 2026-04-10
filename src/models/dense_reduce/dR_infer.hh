#pragma once

#include "dR_model.hh"
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

namespace cellerator::models::dense_reduce {

struct DenseReduceInferConfig {
    torch::Device device = torch::Device(torch::kCPU);
    bool move_model_to_device = true;
    bool drop_fetched_parts = true;
};

struct DenseReduceEncodedBatch {
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
    torch::Tensor latent_unit;
};

struct DenseReduceEmbeddingTable {
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
    torch::Tensor latent_unit;
};

namespace detail {

inline std::int64_t checked_i64_dense_reduce_(unsigned long value, const char *label) {
    if (value > static_cast<unsigned long>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit into int64");
    }
    return static_cast<std::int64_t>(value);
}

inline torch::Tensor copy_i64_tensor_dense_reduce_(const std::vector<std::int64_t> &values) {
    torch::Tensor tensor = torch::empty(
        { static_cast<std::int64_t>(values.size()) },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
    }
    return tensor;
}

inline torch::Tensor copy_f32_tensor_dense_reduce_(const std::vector<float> &values) {
    torch::Tensor tensor = torch::empty(
        { static_cast<std::int64_t>(values.size()) },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<float>(), values.data(), values.size() * sizeof(float));
    }
    return tensor;
}

inline void validate_dense_reduce_infer_inputs_(
    BalancedDenseReduceSampler::matrix_type *matrix,
    const std::vector<float> &developmental_time) {
    if (matrix == nullptr) throw std::invalid_argument("dense reduce inference requires a non-null CellShard matrix");
    if (matrix->num_parts == 0 || matrix->part_offsets == nullptr || matrix->part_rows == nullptr || matrix->part_aux == nullptr) {
        throw std::invalid_argument("dense reduce inference requires initialized sharded CSR metadata");
    }
    if (developmental_time.size() != static_cast<std::size_t>(matrix->rows)) {
        throw std::invalid_argument("developmental_time length must match the number of rows in the CellShard matrix");
    }
}

inline DenseReduceEmbeddingTable materialize_dense_reduce_batches_(
    const std::vector<DenseReduceEncodedBatch> &batches) {
    if (batches.empty()) {
        return DenseReduceEmbeddingTable{
            torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)),
            torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
            torch::empty({ 0, 0 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
        };
    }

    std::vector<torch::Tensor> all_indices;
    std::vector<torch::Tensor> all_times;
    std::vector<torch::Tensor> all_latents;
    all_indices.reserve(batches.size());
    all_times.reserve(batches.size());
    all_latents.reserve(batches.size());
    for (const DenseReduceEncodedBatch &batch : batches) {
        all_indices.push_back(batch.cell_indices);
        all_times.push_back(batch.developmental_time);
        all_latents.push_back(batch.latent_unit);
    }
    return DenseReduceEmbeddingTable{
        torch::cat(all_indices, 0).contiguous(),
        torch::cat(all_times, 0).contiguous(),
        torch::cat(all_latents, 0).contiguous()
    };
}

} // namespace detail

template<typename Callback>
void for_each_dense_reduce_encoded_batch(
    DenseReduceModel &model,
    BalancedDenseReduceSampler::matrix_type *matrix,
    const std::vector<float> &developmental_time,
    const BalancedDenseReduceSampler::storage_type *storage,
    Callback &&callback,
    const DenseReduceInferConfig &config = DenseReduceInferConfig()) {
    detail::validate_dense_reduce_infer_inputs_(matrix, developmental_time);

    torch::NoGradGuard no_grad;
    model->eval();
    if (config.move_model_to_device) model->to(config.device);

    for (unsigned long part_id = 0; part_id < matrix->num_parts; ++part_id) {
        bool loaded_here = false;
        const unsigned long row_begin_ul = matrix->part_offsets[part_id];
        const unsigned long row_end_ul = matrix->part_offsets[part_id + 1];
        const std::int64_t row_count = detail::checked_i64_dense_reduce_(row_end_ul - row_begin_ul, "part row count");

        if (!cellshard::part_loaded(matrix, part_id)) {
            if (storage == nullptr) {
                throw std::runtime_error("dense reduce inference encountered an unloaded part without shard storage");
            }
            if (!cellshard::fetch_part(matrix, storage, part_id)) {
                throw std::runtime_error("dense reduce inference failed to fetch a CellShard part");
            }
            loaded_here = true;
        }

        const cellshard::sparse::compressed *part = matrix->parts[part_id];
        if (part == nullptr || part->axis != cellshard::sparse::compressed_by_row) {
            throw std::runtime_error("dense reduce inference requires loaded row-compressed CSR parts");
        }

        torch::Tensor features = cellerator::torch_bindings::export_as_tensor(*part);
        features = features.to(config.device);
        torch::Tensor latent = model->encode(features).to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();

        std::vector<std::int64_t> batch_indices(static_cast<std::size_t>(row_count));
        std::vector<float> batch_time(static_cast<std::size_t>(row_count));
        for (std::int64_t local_row = 0; local_row < row_count; ++local_row) {
            const unsigned long global_row = row_begin_ul + static_cast<unsigned long>(local_row);
            batch_indices[static_cast<std::size_t>(local_row)] = detail::checked_i64_dense_reduce_(global_row, "global row");
            batch_time[static_cast<std::size_t>(local_row)] = developmental_time[static_cast<std::size_t>(global_row)];
        }

        callback(DenseReduceEncodedBatch{
            detail::copy_i64_tensor_dense_reduce_(batch_indices),
            detail::copy_f32_tensor_dense_reduce_(batch_time),
            std::move(latent)
        });

        if (loaded_here && config.drop_fetched_parts) {
            cellshard::drop_part(matrix, part_id);
        }
    }
}

inline DenseReduceEmbeddingTable infer_dense_reduce_embeddings(
    DenseReduceModel &model,
    BalancedDenseReduceSampler::matrix_type *matrix,
    const std::vector<float> &developmental_time,
    const BalancedDenseReduceSampler::storage_type *storage = nullptr,
    const DenseReduceInferConfig &config = DenseReduceInferConfig()) {
    std::vector<DenseReduceEncodedBatch> batches;
    for_each_dense_reduce_encoded_batch(
        model,
        matrix,
        developmental_time,
        storage,
        [&batches](DenseReduceEncodedBatch batch) { batches.push_back(std::move(batch)); },
        config);
    return detail::materialize_dense_reduce_batches_(batches);
}

} // namespace cellerator::models::dense_reduce
