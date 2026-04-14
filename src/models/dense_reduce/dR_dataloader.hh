#pragma once

#include "../../../extern/CellShard/src/formats/compressed.cuh"
#include "../../../extern/CellShard/src/sharded/shard_paths.cuh"
#include "../../../extern/CellShard/src/sharded/sharded.cuh"
#include "../../../extern/CellShard/src/sharded/sharded_host.cuh"
#include "../rngFetch.hh"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <utility>

namespace cellerator::models::dense_reduce {

struct DenseReduceBatch {
    torch::Tensor features;
    torch::Tensor developmental_time;
    torch::Tensor time_buckets;
    torch::Tensor cell_indices;
};

class BalancedDenseReduceSampler {
public:
    using matrix_type = cellshard::sharded<cellshard::sparse::compressed>;
    using part_type = cellshard::sparse::compressed;
    using storage_type = cellshard::shard_storage;

    struct Options {
        bool with_replacement = true;
        bool drop_fetched_parts = true;
        std::size_t max_buckets_per_batch = 0;
        std::size_t target_bucket_count = 16;
        std::uint64_t seed = std::random_device{}();
    };

    BalancedDenseReduceSampler(
        matrix_type *matrix,
        const float *developmental_time,
        std::size_t developmental_time_count,
        const storage_type *storage = 0)
        : BalancedDenseReduceSampler(matrix, developmental_time, developmental_time_count, storage, Options{}) {}

    BalancedDenseReduceSampler(
        matrix_type *matrix,
        const float *developmental_time,
        std::size_t developmental_time_count,
        const storage_type *storage,
        Options options)
        : matrix_(matrix),
          storage_(storage),
          options_(options) {
        developmental_time_.assign_copy(developmental_time, developmental_time_count);
        validate_constructor_state_();
        build_time_buckets_();
    }

    std::size_t num_cells() const {
        return matrix_ != 0 ? static_cast<std::size_t>(matrix_->rows) : 0u;
    }

    std::size_t num_features() const {
        return matrix_ != 0 ? static_cast<std::size_t>(matrix_->cols) : 0u;
    }

    std::size_t num_buckets() const {
        return bucket_row_offsets_.empty() ? 0u : bucket_row_offsets_.size() - 1u;
    }

    DenseReduceBatch sample_sparse_csr(std::size_t cells_per_bucket) {
        std::lock_guard<std::mutex> lock(mutex_);
        host_buffer<unsigned long> fetched_partitions;
        host_buffer<sampled_row_span> sampled_rows;
        std::int64_t total_nnz = 0;

        if (cells_per_bucket == 0) {
            throw std::invalid_argument("sample_sparse_csr requires cells_per_bucket > 0");
        }

        // Host-side batch assembly: bucket selection, optional part fetch, then
        // fresh sparse-CSR tensor construction.
        try {
            const host_buffer<std::int64_t> selected_buckets = select_time_buckets_();
            sampled_rows.reserve(selected_buckets.size() * cells_per_bucket);

            for (const std::int64_t bucket_id : selected_buckets) {
                const host_buffer<unsigned long> local_positions =
                    row_fetchers_[static_cast<std::size_t>(bucket_id)].next(cells_per_bucket);
                const unsigned long *bucket_rows = bucket_rows_begin_(static_cast<std::size_t>(bucket_id));
                for (const unsigned long local_pos : local_positions) {
                    const unsigned long global_row = bucket_rows[static_cast<std::size_t>(local_pos)];
                    const unsigned long part_id = cellshard::find_partition(matrix_, global_row);
                    part_type *part = require_partition_(part_id, &fetched_partitions);
                    const unsigned long part_row_base = matrix_->partition_offsets[part_id];
                    const cellshard::types::dim_t local_row =
                        static_cast<cellshard::types::dim_t>(global_row - part_row_base);
                    const cellshard::types::ptr_t row_begin = part->majorPtr[local_row];
                    const cellshard::types::ptr_t row_end = part->majorPtr[local_row + 1];

                    sampled_rows.push_back(sampled_row_span{
                        global_row,
                        bucket_id,
                        developmental_time_[static_cast<std::size_t>(global_row)],
                        part,
                        row_begin,
                        row_end
                    });
                    total_nnz += static_cast<std::int64_t>(row_end - row_begin);
                }
            }
        } catch (...) {
            drop_fetched_partitions_(fetched_partitions);
            throw;
        }

        DenseReduceBatch batch = build_sparse_batch_(sampled_rows, total_nnz);
        drop_fetched_partitions_(fetched_partitions);
        return batch;
    }

private:
    struct sampled_row_span {
        unsigned long global_row;
        std::int64_t bucket_id;
        float developmental_time;
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

    static torch::Tensor copy_i64_tensor_(const host_buffer<std::int64_t> &values) {
        torch::Tensor tensor = torch::empty(
            { static_cast<std::int64_t>(values.size()) },
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        if (!values.empty()) {
            std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
        }
        return tensor;
    }

    static torch::Tensor copy_f32_tensor_(const host_buffer<float> &values) {
        torch::Tensor tensor = torch::empty(
            { static_cast<std::int64_t>(values.size()) },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        if (!values.empty()) std::memcpy(tensor.data_ptr<float>(), values.data(), values.size() * sizeof(float));
        return tensor;
    }

    void validate_constructor_state_() const {
        if (matrix_ == 0) throw std::invalid_argument("BalancedDenseReduceSampler requires a non-null CellShard matrix");
        if (matrix_->rows == 0) throw std::invalid_argument("BalancedDenseReduceSampler requires a non-empty CellShard matrix");
        if (matrix_->num_partitions == 0 || matrix_->partition_offsets == 0 || matrix_->partition_rows == 0 || matrix_->partition_aux == 0) {
            throw std::invalid_argument("BalancedDenseReduceSampler requires sharded CSR metadata to be initialized");
        }
        if (developmental_time_.size() != static_cast<std::size_t>(matrix_->rows)) {
            throw std::invalid_argument("developmental_time length must match the number of cells in the CellShard matrix");
        }
        if (options_.target_bucket_count == 0) {
            throw std::invalid_argument("BalancedDenseReduceSampler target_bucket_count must be > 0");
        }
        checked_i64_(matrix_->rows, "rows");
        checked_i64_(matrix_->cols, "cols");
        for (unsigned long part_id = 0; part_id < matrix_->num_partitions; ++part_id) {
            if (matrix_->partition_aux[part_id] != cellshard::sparse::compressed_by_row) {
                throw std::invalid_argument("BalancedDenseReduceSampler requires CSR parts compressed by row");
            }
        }
    }

    void build_time_buckets_() {
        host_buffer<std::pair<float, unsigned long>> labeled_rows;

        // One-time O(rows log rows) sort so later draws are O(sampled rows).
        labeled_rows.reserve(developmental_time_.size());
        for (unsigned long row = 0; row < static_cast<unsigned long>(developmental_time_.size()); ++row) {
            labeled_rows.push_back(std::pair<float, unsigned long>{ developmental_time_[static_cast<std::size_t>(row)], row });
        }
        std::sort(labeled_rows.begin(), labeled_rows.end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.first < rhs.first) return true;
            if (lhs.first > rhs.first) return false;
            return lhs.second < rhs.second;
        });

        const std::size_t row_count = labeled_rows.size();
        const std::size_t bucket_count = std::max<std::size_t>(
            1u,
            std::min<std::size_t>(options_.target_bucket_count, row_count));

        bucket_rows_.clear();
        bucket_row_offsets_.assign_fill(bucket_count + 1u, 0u);
        bucket_by_row_.assign_fill(row_count, 0);
        host_buffer<std::size_t> bucket_counts;
        bucket_counts.assign_fill(bucket_count, 0u);
        for (std::size_t idx = 0; idx < row_count; ++idx) {
            const std::size_t bucket = std::min<std::size_t>((idx * bucket_count) / row_count, bucket_count - 1u);
            const unsigned long row = labeled_rows[idx].second;
            ++bucket_counts[bucket];
            bucket_by_row_[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(bucket);
        }
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
            bucket_row_offsets_[bucket + 1u] = bucket_row_offsets_[bucket] + bucket_counts[bucket];
        }
        bucket_rows_.resize(row_count);
        bucket_counts.assign_fill(bucket_count, 0u);
        for (std::size_t idx = 0; idx < row_count; ++idx) {
            const unsigned long row = labeled_rows[idx].second;
            const std::size_t bucket = static_cast<std::size_t>(bucket_by_row_[static_cast<std::size_t>(row)]);
            const std::size_t write = bucket_row_offsets_[bucket] + bucket_counts[bucket];
            bucket_rows_[write] = row;
            ++bucket_counts[bucket];
        }

        row_fetchers_.resize(bucket_count);
        for (std::size_t bucket = 0; bucket < bucket_count; ++bucket) {
            row_fetchers_[bucket] = RngFetch(
                static_cast<unsigned long>(bucket_row_count_(bucket)),
                RngFetchOptions{ options_.with_replacement, options_.seed + static_cast<std::uint64_t>(bucket) + 1u });
        }

        bucket_fetch_.reset();
        if (options_.max_buckets_per_batch != 0 && options_.max_buckets_per_batch < bucket_count) {
            bucket_fetch_ = std::make_unique<RngFetch>(
                static_cast<unsigned long>(bucket_count),
                RngFetchOptions{ false, options_.seed ^ 0x9e3779b97f4a7c15ULL });
        }
    }

    host_buffer<std::int64_t> select_time_buckets_() {
        host_buffer<std::int64_t> buckets;
        if (!bucket_fetch_) {
            const std::size_t bucket_count = num_buckets();
            buckets.resize(bucket_count);
            for (std::size_t i = 0; i < bucket_count; ++i) buckets[i] = static_cast<std::int64_t>(i);
            return buckets;
        }

        const host_buffer<unsigned long> sampled = bucket_fetch_->next(options_.max_buckets_per_batch);
        buckets.resize(sampled.size());
        for (std::size_t i = 0; i < sampled.size(); ++i) buckets[i] = static_cast<std::int64_t>(sampled[i]);
        std::sort(buckets.begin(), buckets.end());
        return buckets;
    }

    part_type *require_partition_(unsigned long part_id, host_buffer<unsigned long> *fetched_partitions) {
        if (part_id >= matrix_->num_partitions) {
            throw std::out_of_range("sampled row resolved to an invalid CellShard partition");
        }
        if (!cellshard::partition_loaded(matrix_, part_id)) {
            if (storage_ == 0) {
                throw std::runtime_error("sampled row lives in an unloaded CellShard partition, but no shard_storage was provided");
            }
            if (!cellshard::fetch_partition(matrix_, storage_, part_id)) {
                throw std::runtime_error("failed to fetch CellShard partition for sampled row");
            }
            // Cold-partition fetches dominate latency more than local row copy.
            if (fetched_partitions != 0) fetched_partitions->push_back(part_id);
        }

        part_type *part = matrix_->parts[part_id];
        if (part == 0) throw std::runtime_error("CellShard partition is still null after fetch");
        if (part->axis != cellshard::sparse::compressed_by_row) {
            throw std::runtime_error("BalancedDenseReduceSampler only supports row-compressed CSR partitions");
        }
        if (matrix_->cols != 0 && part->cols != matrix_->cols) {
            throw std::runtime_error("CellShard partition column count does not match sharded metadata");
        }
        return part;
    }

    void drop_fetched_partitions_(const host_buffer<unsigned long> &fetched_partitions) {
        if (!options_.drop_fetched_parts) return;
        for (std::size_t i = fetched_partitions.size(); i != 0u; --i) {
            cellshard::drop_partition(matrix_, fetched_partitions[i - 1u]);
        }
    }

    DenseReduceBatch build_sparse_batch_(const host_buffer<sampled_row_span> &sampled_rows, std::int64_t total_nnz) const {
        static_assert(sizeof(at::Half) == sizeof(::real::storage_t), "ATen half type must match CellShard half storage");

        // Expensive boundary: widen CSR metadata to int64 and copy sampled nnz
        // into a new Torch-owned CPU sparse tensor.
        const std::int64_t batch_rows = static_cast<std::int64_t>(sampled_rows.size());
        const std::int64_t feature_cols = checked_i64_(matrix_->cols, "cols");
        host_buffer<std::int64_t> crow_indices;
        host_buffer<std::int64_t> time_buckets;
        host_buffer<std::int64_t> cell_indices;
        host_buffer<float> batch_times;
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
        time_buckets.reserve(sampled_rows.size());
        cell_indices.reserve(sampled_rows.size());
        batch_times.reserve(sampled_rows.size());

        for (const sampled_row_span &row : sampled_rows) {
            const std::int64_t row_nnz = static_cast<std::int64_t>(row.row_end - row.row_begin);
            for (std::int64_t i = 0; i < row_nnz; ++i) {
                col_ptr[nnz_cursor + i] =
                    static_cast<std::int64_t>(row.part->minorIdx[row.row_begin + static_cast<cellshard::types::ptr_t>(i)]);
            }
            if (row_nnz != 0) {
                std::memcpy(
                    value_ptr + nnz_cursor,
                    row.part->val + row.row_begin,
                    static_cast<std::size_t>(row_nnz) * sizeof(::real::storage_t));
            }
            nnz_cursor += row_nnz;
            crow_indices.push_back(nnz_cursor);
            time_buckets.push_back(row.bucket_id);
            cell_indices.push_back(static_cast<std::int64_t>(row.global_row));
            batch_times.push_back(row.developmental_time);
        }

        torch::Tensor crow_tensor = copy_i64_tensor_(crow_indices);
        torch::Tensor features = torch::sparse_csr_tensor(
            crow_tensor,
            col_tensor,
            value_tensor,
            { batch_rows, feature_cols },
            torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCPU));

        return DenseReduceBatch{
            std::move(features),
            copy_f32_tensor_(batch_times),
            copy_i64_tensor_(time_buckets),
            copy_i64_tensor_(cell_indices)
        };
    }

    std::size_t bucket_row_count_(std::size_t bucket) const {
        return bucket_row_offsets_[bucket + 1u] - bucket_row_offsets_[bucket];
    }

    const unsigned long *bucket_rows_begin_(std::size_t bucket) const {
        return bucket_rows_.data() + bucket_row_offsets_[bucket];
    }

    matrix_type *matrix_;
    const storage_type *storage_;
    host_buffer<float> developmental_time_;
    Options options_;
    host_buffer<unsigned long> bucket_rows_;
    host_buffer<std::size_t> bucket_row_offsets_;
    host_buffer<std::int64_t> bucket_by_row_;
    host_buffer<RngFetch> row_fetchers_;
    std::unique_ptr<RngFetch> bucket_fetch_;
    std::mutex mutex_;
};

} // namespace cellerator::models::dense_reduce
