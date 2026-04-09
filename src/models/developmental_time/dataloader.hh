#pragma once

#include "../../../extern/CellShard/src/formats/compressed.cuh"
#include "../../../extern/CellShard/src/sharded/shard_paths.cuh"
#include "../../../extern/CellShard/src/sharded/sharded.cuh"
#include "../../../extern/CellShard/src/sharded/sharded_host.cuh"

#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator {

struct RandomTimeBatch {
    torch::Tensor features;
    torch::Tensor timepoints;
    torch::Tensor cell_indices;
};

// RandomTimeSampler owns only sampling state and label storage.
// The CellShard matrix and optional shard storage stay owned by the caller.
class RandomTimeSampler {
public:
    using matrix_type = cellshard::sharded<cellshard::sparse::compressed>;
    using part_type = cellshard::sparse::compressed;
    using storage_type = cellshard::shard_storage;

    struct Options {
        bool with_replacement = true;
        bool drop_fetched_parts = true;
        std::uint64_t seed = std::random_device{}();
    };

    RandomTimeSampler(matrix_type *matrix,
                      std::vector<float> timepoints,
                      const storage_type *storage = 0,
                      Options options = Options())
        : matrix_(matrix),
          storage_(storage),
          timepoints_(std::move(timepoints)),
          options_(options),
          rng_(options.seed) {
        validate_constructor_state_();
    }

    std::size_t num_cells() const {
        return matrix_ != 0 ? static_cast<std::size_t>(matrix_->rows) : 0u;
    }

    std::size_t num_features() const {
        return matrix_ != 0 ? static_cast<std::size_t>(matrix_->cols) : 0u;
    }

    RandomTimeBatch sample_dense(std::size_t batch_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        return build_dense_batch_(sample_rows_(batch_size));
    }

    RandomTimeBatch sample_sparse_csr(std::size_t batch_size) {
        std::lock_guard<std::mutex> lock(mutex_);
        return build_sparse_batch_(sample_rows_(batch_size));
    }

private:
    static std::int64_t checked_i64_(unsigned long value, const char *label) {
        if (value > static_cast<unsigned long>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error(std::string(label) + " does not fit into int64");
        }
        return static_cast<std::int64_t>(value);
    }

    static torch::Tensor copy_vector_to_tensor_(const std::vector<std::int64_t> &values) {
        torch::Tensor tensor = torch::empty(
            { static_cast<std::int64_t>(values.size()) },
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        if (!values.empty()) std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
        return tensor;
    }

    static torch::Tensor copy_vector_to_tensor_(const std::vector<float> &values) {
        torch::Tensor tensor = torch::empty(
            { static_cast<std::int64_t>(values.size()) },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        if (!values.empty()) std::memcpy(tensor.data_ptr<float>(), values.data(), values.size() * sizeof(float));
        return tensor;
    }

    void validate_constructor_state_() const {
        if (matrix_ == 0) throw std::invalid_argument("RandomTimeSampler requires a non-null CellShard matrix");
        if (matrix_->num_parts == 0 || matrix_->part_offsets == 0 || matrix_->part_rows == 0 || matrix_->part_aux == 0) {
            throw std::invalid_argument("RandomTimeSampler requires sharded CSR metadata to be initialized");
        }
        if (timepoints_.size() != static_cast<std::size_t>(matrix_->rows)) {
            throw std::invalid_argument("timepoint vector length must match the number of cells in the CellShard matrix");
        }
        checked_i64_(matrix_->rows, "rows");
        checked_i64_(matrix_->cols, "cols");
        for (unsigned long part_id = 0; part_id < matrix_->num_parts; ++part_id) {
            if (matrix_->part_aux[part_id] != cellshard::sparse::compressed_by_row) {
                throw std::invalid_argument("RandomTimeSampler requires CSR parts compressed by row");
            }
        }
    }

    std::vector<unsigned long> sample_rows_(std::size_t batch_size) {
        std::vector<unsigned long> rows;

        if (batch_size == 0) throw std::invalid_argument("batch_size must be greater than zero");
        if (matrix_->rows == 0) throw std::runtime_error("cannot sample from an empty CellShard matrix");
        if (!options_.with_replacement && batch_size > static_cast<std::size_t>(matrix_->rows)) {
            throw std::invalid_argument("batch_size exceeds the number of cells when sampling without replacement");
        }

        rows.reserve(batch_size);
        std::uniform_int_distribution<unsigned long> dist(0, matrix_->rows - 1);
        if (options_.with_replacement) {
            for (std::size_t i = 0; i < batch_size; ++i) rows.push_back(dist(rng_));
            return rows;
        }

        std::unordered_set<unsigned long> seen;
        seen.reserve(batch_size * 2u);
        while (rows.size() < batch_size) {
            const unsigned long row = dist(rng_);
            if (seen.insert(row).second) rows.push_back(row);
        }
        return rows;
    }

    part_type *require_part_(unsigned long part_id, std::vector<unsigned long> *fetched_parts) {
        if (part_id >= matrix_->num_parts) throw std::out_of_range("sampled row resolved to an invalid CellShard part");

        if (!cellshard::part_loaded(matrix_, part_id)) {
            if (storage_ == 0) {
                throw std::runtime_error("sampled row lives in an unloaded CellShard part, but no shard_storage was provided");
            }
            if (!cellshard::fetch_part(matrix_, storage_, part_id)) {
                throw std::runtime_error("failed to fetch CellShard part for sampled row");
            }
            if (fetched_parts != 0) fetched_parts->push_back(part_id);
        }

        part_type *part = matrix_->parts[part_id];
        if (part == 0) throw std::runtime_error("CellShard part is still null after fetch");
        if (part->axis != cellshard::sparse::compressed_by_row) {
            throw std::runtime_error("RandomTimeSampler only supports row-compressed CSR parts");
        }
        if (matrix_->cols != 0 && part->cols != matrix_->cols) {
            throw std::runtime_error("CellShard part column count does not match sharded metadata");
        }
        return part;
    }

    void drop_fetched_parts_(const std::vector<unsigned long> &fetched_parts) {
        if (!options_.drop_fetched_parts) return;
        for (auto it = fetched_parts.rbegin(); it != fetched_parts.rend(); ++it) {
            cellshard::drop_part(matrix_, *it);
        }
    }

    RandomTimeBatch build_dense_batch_(const std::vector<unsigned long> &sampled_rows) {
        const std::int64_t batch_rows = static_cast<std::int64_t>(sampled_rows.size());
        const std::int64_t feature_cols = checked_i64_(matrix_->cols, "cols");
        std::vector<unsigned long> fetched_parts;
        std::vector<float> batch_timepoints;
        std::vector<std::int64_t> batch_indices;
        torch::Tensor features = torch::zeros(
            { batch_rows, feature_cols },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        float *dense = features.data_ptr<float>();

        batch_timepoints.reserve(sampled_rows.size());
        batch_indices.reserve(sampled_rows.size());
        fetched_parts.reserve(std::min<std::size_t>(sampled_rows.size(), static_cast<std::size_t>(matrix_->num_parts)));

        try {
            // Sampling is row-wise over cells, so CSR stays the natural source
            // layout. We densify only at the minibatch boundary.
            for (std::size_t batch_row = 0; batch_row < sampled_rows.size(); ++batch_row) {
                const unsigned long global_row = sampled_rows[batch_row];
                const unsigned long part_id = cellshard::find_part(matrix_, global_row);
                part_type *part = require_part_(part_id, &fetched_parts);
                const unsigned long part_row_base = matrix_->part_offsets[part_id];
                const cellshard::types::dim_t local_row = static_cast<cellshard::types::dim_t>(global_row - part_row_base);
                const cellshard::types::ptr_t row_begin = part->majorPtr[local_row];
                const cellshard::types::ptr_t row_end = part->majorPtr[local_row + 1];
                const std::int64_t dense_offset = static_cast<std::int64_t>(batch_row) * feature_cols;

                for (cellshard::types::ptr_t i = row_begin; i < row_end; ++i) {
                    dense[dense_offset + static_cast<std::int64_t>(part->minorIdx[i])] = __half2float(part->val[i]);
                }
                batch_timepoints.push_back(timepoints_[global_row]);
                batch_indices.push_back(static_cast<std::int64_t>(global_row));
            }
        } catch (...) {
            drop_fetched_parts_(fetched_parts);
            throw;
        }

        drop_fetched_parts_(fetched_parts);
        return RandomTimeBatch{
            std::move(features),
            copy_vector_to_tensor_(batch_timepoints),
            copy_vector_to_tensor_(batch_indices)
        };
    }

    RandomTimeBatch build_sparse_batch_(const std::vector<unsigned long> &sampled_rows) {
        const std::int64_t batch_rows = static_cast<std::int64_t>(sampled_rows.size());
        const std::int64_t feature_cols = checked_i64_(matrix_->cols, "cols");
        std::vector<unsigned long> fetched_parts;
        std::vector<std::int64_t> crow_indices;
        std::vector<std::int64_t> col_indices;
        std::vector<float> values;
        std::vector<float> batch_timepoints;
        std::vector<std::int64_t> batch_indices;
        std::int64_t nnz_cursor = 0;

        crow_indices.reserve(sampled_rows.size() + 1u);
        crow_indices.push_back(0);
        batch_timepoints.reserve(sampled_rows.size());
        batch_indices.reserve(sampled_rows.size());
        fetched_parts.reserve(std::min<std::size_t>(sampled_rows.size(), static_cast<std::size_t>(matrix_->num_parts)));

        try {
            for (const unsigned long global_row : sampled_rows) {
                const unsigned long part_id = cellshard::find_part(matrix_, global_row);
                part_type *part = require_part_(part_id, &fetched_parts);
                const unsigned long part_row_base = matrix_->part_offsets[part_id];
                const cellshard::types::dim_t local_row = static_cast<cellshard::types::dim_t>(global_row - part_row_base);
                const cellshard::types::ptr_t row_begin = part->majorPtr[local_row];
                const cellshard::types::ptr_t row_end = part->majorPtr[local_row + 1];

                col_indices.reserve(col_indices.size() + static_cast<std::size_t>(row_end - row_begin));
                values.reserve(values.size() + static_cast<std::size_t>(row_end - row_begin));
                for (cellshard::types::ptr_t i = row_begin; i < row_end; ++i) {
                    col_indices.push_back(static_cast<std::int64_t>(part->minorIdx[i]));
                    values.push_back(__half2float(part->val[i]));
                }

                nnz_cursor += static_cast<std::int64_t>(row_end - row_begin);
                crow_indices.push_back(nnz_cursor);
                batch_timepoints.push_back(timepoints_[global_row]);
                batch_indices.push_back(static_cast<std::int64_t>(global_row));
            }
        } catch (...) {
            drop_fetched_parts_(fetched_parts);
            throw;
        }

        drop_fetched_parts_(fetched_parts);

        torch::Tensor crow_tensor = copy_vector_to_tensor_(crow_indices);
        torch::Tensor col_tensor = copy_vector_to_tensor_(col_indices);
        torch::Tensor value_tensor = copy_vector_to_tensor_(values);
        torch::Tensor features = torch::sparse_csr_tensor(
            crow_tensor,
            col_tensor,
            value_tensor,
            { batch_rows, feature_cols },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        return RandomTimeBatch{
            std::move(features),
            copy_vector_to_tensor_(batch_timepoints),
            copy_vector_to_tensor_(batch_indices)
        };
    }

    matrix_type *matrix_;
    const storage_type *storage_;
    std::vector<float> timepoints_;
    Options options_;
    std::mt19937_64 rng_;
    std::mutex mutex_;
};

} // namespace cellerator
