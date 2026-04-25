#pragma once

#include "fn_query.hh"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace cellerator::compute::neighbors::forward_neighbors {

struct ForwardNeighborRecordBatch {
    const_array_view<std::int64_t> cell_indices;
    const_array_view<float> developmental_time;
    const_array_view<std::int64_t> embryo_ids;
    ForwardNeighborMatrixView matrix;
};

struct ForwardNeighborOwnedRecordBatch {
    host_array<std::int64_t> cell_indices;
    host_array<float> developmental_time;
    host_array<std::int64_t> embryo_ids;
    host_array<float> dense_values;
    std::int64_t dense_cols = 0;

    ForwardNeighborRecordBatch view() const {
        return ForwardNeighborRecordBatch{
            view_of(cell_indices),
            view_of(developmental_time),
            view_of(embryo_ids),
            make_forward_neighbor_dense_view(dense_values.data(),
                                             static_cast<std::int64_t>(cell_indices.size()),
                                             dense_cols)
        };
    }
};

struct ForwardNeighborBuildConfig {
    host_array<int> shard_devices;
    std::int64_t target_shard_count = 0;
    std::int64_t max_rows_per_segment = 0;
    std::int64_t ann_rows_per_list = 4096;
    bool renormalize_latent = true;
    bool eager_device_upload = false;
};

struct ForwardNeighborAnnList {
    std::int64_t embryo_id = -1;
    std::int64_t row_begin = 0;
    std::int64_t row_end = 0;
    float time_begin = detail::quiet_nan_();
    float time_end = detail::quiet_nan_();
};

struct ForwardNeighborSegment {
    std::int64_t embryo_id = -1;
    std::int64_t row_begin = 0;
    std::int64_t row_end = 0;
    float time_begin = detail::quiet_nan_();
    float time_end = detail::quiet_nan_();
    std::int64_t ann_list_begin = 0;
    std::int64_t ann_list_end = 0;
};

class ForwardNeighborSearchWorkspace;
class ForwardNeighborSearchExecutor;

struct ForwardNeighborShardSummary {
    int device_id = -1;
    std::int64_t row_begin = 0;
    std::int64_t row_end = 0;
    std::int64_t rows = 0;
    std::int64_t segment_count = 0;
    std::int64_t ann_list_count = 0;
    float time_begin = detail::quiet_nan_();
    float time_end = detail::quiet_nan_();
    std::size_t resident_bytes = 0u;
    bool resident = false;
    ForwardNeighborInputLayout sparse_layout = ForwardNeighborInputLayout::dense;
};

namespace detail {
struct ForwardNeighborIndexStorage;
struct ForwardNeighborSearchWorkspaceStorage;
struct ForwardNeighborExecutorStorage;
ForwardNeighborSearchWorkspaceStorage &workspace_storage_(ForwardNeighborSearchWorkspace *workspace);
}

class ForwardNeighborIndex;
class ForwardNeighborSearchWorkspace {
public:
    ForwardNeighborSearchWorkspace();
    ~ForwardNeighborSearchWorkspace();

    ForwardNeighborSearchWorkspace(const ForwardNeighborSearchWorkspace &) = delete;
    ForwardNeighborSearchWorkspace &operator=(const ForwardNeighborSearchWorkspace &) = delete;

    ForwardNeighborSearchWorkspace(ForwardNeighborSearchWorkspace &&) noexcept;
    ForwardNeighborSearchWorkspace &operator=(ForwardNeighborSearchWorkspace &&) noexcept;

    void reset();

private:
    std::unique_ptr<detail::ForwardNeighborSearchWorkspaceStorage> storage_;

    friend class ForwardNeighborIndex;
    friend detail::ForwardNeighborSearchWorkspaceStorage &detail::workspace_storage_(ForwardNeighborSearchWorkspace *workspace);
};

class ForwardNeighborIndexBuilder {
public:
    explicit ForwardNeighborIndexBuilder(ForwardNeighborBuildConfig config = ForwardNeighborBuildConfig());

    void append(const ForwardNeighborRecordBatch &batch);
    ForwardNeighborIndex finalize() &&;

private:
    ForwardNeighborBuildConfig config_;
    ForwardNeighborOwnedRecordBatch records_;
};

class ForwardNeighborIndex {
public:
    struct CellLocation {
        std::size_t shard = 0;
        std::int64_t local_row = 0;
    };

    ForwardNeighborIndex();
    explicit ForwardNeighborIndex(std::shared_ptr<detail::ForwardNeighborIndexStorage> storage);

    std::size_t shard_count() const;
    std::int64_t latent_dim() const;
    ForwardNeighborShardSummary shard_summary(std::size_t shard) const;

    ForwardNeighborOwnedQueryBatch query_batch_from_cell_indices(const std::int64_t *cell_indices, std::size_t cell_count) const;
    ForwardNeighborRoutingPlan plan_future_neighbor_routes(
        const ForwardNeighborQueryBatch &query,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborRoutingPlan plan_future_neighbor_routes_by_cell_index(
        const std::int64_t *cell_indices,
        std::size_t cell_count,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborSearchResult search_future_neighbors(
        const ForwardNeighborQueryBatch &query,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborSearchResult search_future_neighbors(
        const ForwardNeighborQueryBatch &query,
        ForwardNeighborSearchWorkspace *workspace,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborSearchResult search_future_neighbors_by_cell_index(
        const std::int64_t *cell_indices,
        std::size_t cell_count,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborSearchResult search_future_neighbors_by_cell_index(
        const std::int64_t *cell_indices,
        std::size_t cell_count,
        ForwardNeighborSearchWorkspace *workspace,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;

private:
    std::shared_ptr<detail::ForwardNeighborIndexStorage> storage_;

    friend class ForwardNeighborSearchExecutor;
};

class ForwardNeighborSearchExecutor {
public:
    explicit ForwardNeighborSearchExecutor(
        ForwardNeighborExecutorConfig config = ForwardNeighborExecutorConfig());
    ~ForwardNeighborSearchExecutor();

    ForwardNeighborSearchExecutor(const ForwardNeighborSearchExecutor &) = delete;
    ForwardNeighborSearchExecutor &operator=(const ForwardNeighborSearchExecutor &) = delete;

    ForwardNeighborSearchExecutor(ForwardNeighborSearchExecutor &&) noexcept;
    ForwardNeighborSearchExecutor &operator=(ForwardNeighborSearchExecutor &&) noexcept;

    void reset();
    const ForwardNeighborExecutorConfig &config() const;

    ForwardNeighborRoutingPlan plan_future_neighbor_routes(
        const ForwardNeighborIndex &index,
        const ForwardNeighborQueryBatch &query,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborRoutingPlan plan_future_neighbor_routes_by_cell_index(
        const ForwardNeighborIndex &index,
        const std::int64_t *cell_indices,
        std::size_t cell_count,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborSearchResult search_future_neighbors(
        const ForwardNeighborIndex &index,
        const ForwardNeighborQueryBatch &query,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig());
    ForwardNeighborSearchResult search_future_neighbors_by_cell_index(
        const ForwardNeighborIndex &index,
        const std::int64_t *cell_indices,
        std::size_t cell_count,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig());

private:
    ForwardNeighborSearchWorkspace workspace_;
    std::unique_ptr<detail::ForwardNeighborExecutorStorage> storage_;
};

ForwardNeighborIndex build_forward_neighbor_index(
    const ForwardNeighborRecordBatch &records,
    const ForwardNeighborBuildConfig &config = ForwardNeighborBuildConfig());

namespace detail {

inline void validate_forward_neighbor_record_batch_(const ForwardNeighborRecordBatch &batch) {
    if (matrix_cols_(batch.matrix) <= 0) {
        throw std::invalid_argument("ForwardNeighborRecordBatch matrix cols must be > 0");
    }
    if (batch.cell_indices.size != batch.developmental_time.size) {
        throw std::invalid_argument("ForwardNeighborRecordBatch cell_indices and developmental_time must align");
    }
    if (matrix_rows_(batch.matrix) != checked_i64_(batch.cell_indices.size, "record rows")) {
        throw std::invalid_argument("ForwardNeighborRecordBatch matrix rows must align with record rows");
    }
    if (!batch.embryo_ids.empty() && batch.embryo_ids.size != batch.cell_indices.size) {
        throw std::invalid_argument("ForwardNeighborRecordBatch embryo_ids must be empty or align with rows");
    }
    if (batch.matrix.layout == ForwardNeighborInputLayout::dense) {
        if (batch.matrix.dense_values == nullptr && batch.cell_indices.size != 0u) {
            throw std::invalid_argument("ForwardNeighborRecordBatch dense_values must not be null");
        }
        if (batch.matrix.dense_row_stride < matrix_cols_(batch.matrix)) {
            throw std::invalid_argument("ForwardNeighborRecordBatch dense_row_stride must be >= dense_cols");
        }
    }
    if (batch.matrix.layout == ForwardNeighborInputLayout::blocked_ell && batch.matrix.blocked_ell == nullptr) {
        throw std::invalid_argument("ForwardNeighborRecordBatch blocked_ell view is null");
    }
    if (batch.matrix.layout == ForwardNeighborInputLayout::sliced_ell && batch.matrix.sliced_ell == nullptr) {
        throw std::invalid_argument("ForwardNeighborRecordBatch sliced_ell view is null");
    }
}

inline std::int64_t default_segment_rows_(const ForwardNeighborBuildConfig &config, std::int64_t rows, std::int64_t shard_count) {
    if (config.max_rows_per_segment > 0) return config.max_rows_per_segment;
    if (rows == 0 || shard_count <= 0) return rows;
    const std::int64_t target = std::max<std::int64_t>(1, (rows + shard_count - 1) / shard_count);
    return std::max<std::int64_t>(target, config.ann_rows_per_list > 0 ? config.ann_rows_per_list : 1);
}

inline void append_i64_(host_array<std::int64_t> *dst, const const_array_view<std::int64_t> &src) {
    const std::size_t old_size = dst->size();
    dst->resize(old_size + src.size);
    if (!src.empty()) std::memcpy(dst->data() + old_size, src.data, src.size * sizeof(std::int64_t));
}

inline void append_f32_(host_array<float> *dst, const const_array_view<float> &src) {
    const std::size_t old_size = dst->size();
    dst->resize(old_size + src.size);
    if (!src.empty()) std::memcpy(dst->data() + old_size, src.data, src.size * sizeof(float));
}

} // namespace detail

} // namespace cellerator::compute::neighbors::forward_neighbors
