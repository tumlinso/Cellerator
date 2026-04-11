#pragma once

#include "../../../extern/CellShard/src/formats/compressed.cuh"
#include "../../../extern/CellShard/src/sharded/shard_paths.cuh"
#include "../../../extern/CellShard/src/sharded/sharded.cuh"
#include "../../../extern/CellShard/src/sharded/sharded_host.cuh"
#include "../rngFetch.hh"

#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cellerator::models::developmental_time {

struct TimeBatch {
    torch::Tensor features;
    torch::Tensor day_labels;
    torch::Tensor day_buckets;
    torch::Tensor cell_indices;
};

class BalancedTimeSampler {
public:
    using matrix_type = cellshard::sharded<cellshard::sparse::compressed>;
    using part_type = cellshard::sparse::compressed;
    using storage_type = cellshard::shard_storage;

    struct Options {
        bool with_replacement = true;
        bool drop_fetched_parts = true;
        std::size_t max_days_per_batch = 0;
        std::uint64_t seed = std::random_device{}();
    };

    BalancedTimeSampler(matrix_type *matrix,
                        std::vector<float> day_labels,
                        const storage_type *storage = 0)
        : BalancedTimeSampler(matrix, std::move(day_labels), storage, Options{}) {}

    BalancedTimeSampler(matrix_type *matrix,
                        std::vector<float> day_labels,
                        const storage_type *storage,
                        Options options)
        : matrix_(matrix),
          storage_(storage),
          day_labels_(std::move(day_labels)),
          options_(options) {
        validate_constructor_state_();
        build_day_buckets_();
    }

    std::size_t num_cells() const {
        return matrix_ != 0 ? static_cast<std::size_t>(matrix_->rows) : 0u;
    }

    std::size_t num_features() const {
        return matrix_ != 0 ? static_cast<std::size_t>(matrix_->cols) : 0u;
    }

    std::size_t num_days() const {
        return day_values_.size();
    }

    const std::vector<float> &day_values() const {
        return day_values_;
    }

    TimeBatch sample_sparse_csr(std::size_t cells_per_day) {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<unsigned long> fetched_parts;
        std::vector<sampled_row_span> sampled_rows;
        std::int64_t total_nnz = 0;

        if (cells_per_day == 0) throw std::invalid_argument("sample_sparse_csr requires cells_per_day > 0");

        // Host-side batch assembly: bucket selection, optional part fetch, then
        // copy sampled rows into a fresh CPU sparse CSR tensor.
        try {
            const std::vector<std::int64_t> selected_days = select_day_buckets_();
            sampled_rows.reserve(selected_days.size() * cells_per_day);

            for (const std::int64_t bucket_id : selected_days) {
                const std::vector<unsigned long> local_positions = row_fetchers_[static_cast<std::size_t>(bucket_id)].next(cells_per_day);
                const std::vector<unsigned long> &bucket_rows = rows_by_bucket_[static_cast<std::size_t>(bucket_id)];
                for (const unsigned long local_pos : local_positions) {
                    const unsigned long global_row = bucket_rows[static_cast<std::size_t>(local_pos)];
                    const unsigned long part_id = cellshard::find_part(matrix_, global_row);
                    part_type *part = require_part_(part_id, &fetched_parts);
                    const unsigned long part_row_base = matrix_->part_offsets[part_id];
                    const cellshard::types::dim_t local_row = static_cast<cellshard::types::dim_t>(global_row - part_row_base);
                    const cellshard::types::ptr_t row_begin = part->majorPtr[local_row];
                    const cellshard::types::ptr_t row_end = part->majorPtr[local_row + 1];

                    sampled_rows.push_back(sampled_row_span{
                        global_row,
                        bucket_id,
                        day_values_[static_cast<std::size_t>(bucket_id)],
                        part,
                        row_begin,
                        row_end
                    });
                    total_nnz += static_cast<std::int64_t>(row_end - row_begin);
                }
            }
        } catch (...) {
            drop_fetched_parts_(fetched_parts);
            throw;
        }

        TimeBatch batch = build_sparse_batch_(sampled_rows, total_nnz);
        drop_fetched_parts_(fetched_parts);
        return batch;
    }

private:
    struct sampled_row_span {
        unsigned long global_row;
        std::int64_t bucket_id;
        float day_label;
        part_type *part;
        cellshard::types::ptr_t row_begin;
        cellshard::types::ptr_t row_end;
    };

    static std::int64_t checked_i64_(unsigned long value, const char *label) {
        if (value > static_cast<unsigned long>(std::numeric_limits<std::int64_t>::max())) {
            throw std::overflow_error(std::string(label) + " does not fit into int64");
        }
        return static_cast<std::int64_t>(value);
    }

    static torch::Tensor copy_i64_tensor_(const std::vector<std::int64_t> &values) {
        torch::Tensor tensor = torch::empty(
            { static_cast<std::int64_t>(values.size()) },
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        if (!values.empty()) std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
        return tensor;
    }

    static torch::Tensor copy_f32_tensor_(const std::vector<float> &values) {
        torch::Tensor tensor = torch::empty(
            { static_cast<std::int64_t>(values.size()) },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        if (!values.empty()) std::memcpy(tensor.data_ptr<float>(), values.data(), values.size() * sizeof(float));
        return tensor;
    }

    void validate_constructor_state_() const {
        if (matrix_ == 0) throw std::invalid_argument("BalancedTimeSampler requires a non-null CellShard matrix");
        if (matrix_->rows == 0) throw std::invalid_argument("BalancedTimeSampler requires a non-empty CellShard matrix");
        if (matrix_->num_parts == 0 || matrix_->part_offsets == 0 || matrix_->part_rows == 0 || matrix_->part_aux == 0) {
            throw std::invalid_argument("BalancedTimeSampler requires sharded CSR metadata to be initialized");
        }
        if (day_labels_.size() != static_cast<std::size_t>(matrix_->rows)) {
            throw std::invalid_argument("day label vector length must match the number of cells in the CellShard matrix");
        }
        checked_i64_(matrix_->rows, "rows");
        checked_i64_(matrix_->cols, "cols");
        for (unsigned long part_id = 0; part_id < matrix_->num_parts; ++part_id) {
            if (matrix_->part_aux[part_id] != cellshard::sparse::compressed_by_row) {
                throw std::invalid_argument("BalancedTimeSampler requires CSR parts compressed by row");
            }
        }
    }

    void build_day_buckets_() {
        std::vector<std::pair<float, unsigned long>> labeled_rows;

        // One-time O(rows log rows) sort so later draws are O(sampled rows).
        labeled_rows.reserve(day_labels_.size());
        for (unsigned long row = 0; row < static_cast<unsigned long>(day_labels_.size()); ++row) {
            labeled_rows.emplace_back(day_labels_[static_cast<std::size_t>(row)], row);
        }
        std::sort(labeled_rows.begin(), labeled_rows.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.first < rhs.first) return true;
            if (lhs.first > rhs.first) return false;
            return lhs.second < rhs.second;
        });

        rows_by_bucket_.clear();
        day_values_.clear();
        bucket_by_row_.assign(day_labels_.size(), 0);

        for (const auto &entry : labeled_rows) {
            if (day_values_.empty() || entry.first != day_values_.back()) {
                day_values_.push_back(entry.first);
                rows_by_bucket_.emplace_back();
            }
            const std::int64_t bucket_id = static_cast<std::int64_t>(day_values_.size() - 1u);
            rows_by_bucket_.back().push_back(entry.second);
            bucket_by_row_[static_cast<std::size_t>(entry.second)] = bucket_id;
        }

        row_fetchers_.clear();
        row_fetchers_.reserve(rows_by_bucket_.size());
        for (std::size_t bucket = 0; bucket < rows_by_bucket_.size(); ++bucket) {
            row_fetchers_.emplace_back(
                static_cast<unsigned long>(rows_by_bucket_[bucket].size()),
                RngFetchOptions{ options_.with_replacement, options_.seed + static_cast<std::uint64_t>(bucket) + 1u });
        }

        day_bucket_fetch_.reset();
        if (options_.max_days_per_batch != 0 && options_.max_days_per_batch < rows_by_bucket_.size()) {
            day_bucket_fetch_ = std::make_unique<RngFetch>(
                static_cast<unsigned long>(rows_by_bucket_.size()),
                RngFetchOptions{ false, options_.seed ^ 0x9e3779b97f4a7c15ULL });
        }
    }

    std::vector<std::int64_t> select_day_buckets_() {
        std::vector<std::int64_t> buckets;

        if (!day_bucket_fetch_) {
            buckets.reserve(day_values_.size());
            for (std::size_t i = 0; i < day_values_.size(); ++i) buckets.push_back(static_cast<std::int64_t>(i));
            return buckets;
        }

        {
            const std::vector<unsigned long> sampled = day_bucket_fetch_->next(options_.max_days_per_batch);
            buckets.reserve(sampled.size());
            for (const unsigned long idx : sampled) buckets.push_back(static_cast<std::int64_t>(idx));
        }
        std::sort(buckets.begin(), buckets.end());
        return buckets;
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
            // Cold-part fetches dominate batch latency more than local row copy.
            if (fetched_parts != 0) fetched_parts->push_back(part_id);
        }

        part_type *part = matrix_->parts[part_id];
        if (part == 0) throw std::runtime_error("CellShard part is still null after fetch");
        if (part->axis != cellshard::sparse::compressed_by_row) {
            throw std::runtime_error("BalancedTimeSampler only supports row-compressed CSR parts");
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

    TimeBatch build_sparse_batch_(const std::vector<sampled_row_span> &sampled_rows, std::int64_t total_nnz) const {
        static_assert(sizeof(at::Half) == sizeof(::real::storage_t), "ATen half type must match CellShard half storage");

        // Expensive boundary: widen CSR metadata to int64 and copy sampled nnz
        // into a brand-new Torch-owned CPU sparse tensor every batch.
        const std::int64_t batch_rows = static_cast<std::int64_t>(sampled_rows.size());
        const std::int64_t feature_cols = checked_i64_(matrix_->cols, "cols");
        std::vector<std::int64_t> crow_indices;
        std::vector<std::int64_t> day_buckets;
        std::vector<std::int64_t> cell_indices;
        std::vector<float> batch_day_labels;
        torch::Tensor col_tensor = torch::empty(
            { total_nnz },
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        torch::Tensor value_tensor = torch::empty(
            { total_nnz },
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));
        std::int64_t *col_ptr = col_tensor.data_ptr<std::int64_t>();
        at::Half *value_ptr = value_tensor.data_ptr<at::Half>();
        std::int64_t nnz_cursor = 0;

        crow_indices.reserve(sampled_rows.size() + 1u);
        crow_indices.push_back(0);
        day_buckets.reserve(sampled_rows.size());
        cell_indices.reserve(sampled_rows.size());
        batch_day_labels.reserve(sampled_rows.size());

        for (const sampled_row_span &row : sampled_rows) {
            const std::int64_t row_nnz = static_cast<std::int64_t>(row.row_end - row.row_begin);
            for (std::int64_t i = 0; i < row_nnz; ++i) {
                col_ptr[nnz_cursor + i] = static_cast<std::int64_t>(row.part->minorIdx[row.row_begin + static_cast<cellshard::types::ptr_t>(i)]);
            }
            if (row_nnz != 0) {
                std::memcpy(
                    value_ptr + nnz_cursor,
                    row.part->val + row.row_begin,
                    static_cast<std::size_t>(row_nnz) * sizeof(::real::storage_t));
            }
            nnz_cursor += row_nnz;
            crow_indices.push_back(nnz_cursor);
            day_buckets.push_back(row.bucket_id);
            cell_indices.push_back(static_cast<std::int64_t>(row.global_row));
            batch_day_labels.push_back(row.day_label);
        }

        torch::Tensor crow_tensor = copy_i64_tensor_(crow_indices);
        torch::Tensor features = torch::sparse_csr_tensor(
            crow_tensor,
            col_tensor,
            value_tensor,
            { batch_rows, feature_cols },
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

        return TimeBatch{
            std::move(features),
            copy_f32_tensor_(batch_day_labels),
            copy_i64_tensor_(day_buckets),
            copy_i64_tensor_(cell_indices)
        };
    }

    matrix_type *matrix_;
    const storage_type *storage_;
    std::vector<float> day_labels_;
    Options options_;
    std::vector<float> day_values_;
    std::vector<std::vector<unsigned long>> rows_by_bucket_;
    std::vector<std::int64_t> bucket_by_row_;
    std::vector<RngFetch> row_fetchers_;
    std::unique_ptr<RngFetch> day_bucket_fetch_;
    std::mutex mutex_;
};

using RandomTimeBatch = TimeBatch;
using RandomTimeSampler = BalancedTimeSampler;

} // namespace cellerator::models::developmental_time
