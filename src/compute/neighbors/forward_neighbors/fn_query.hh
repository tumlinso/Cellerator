#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

#include "../../../../extern/CellShard/include/CellShard/formats/dense.cuh"
#include "../../../../extern/CellShard/include/CellShard/formats/blocked_ell.cuh"
#include "../../../../extern/CellShard/include/CellShard/formats/sliced_ell.cuh"
#include "../host_buffer.hh"

namespace cellerator::compute::neighbors::forward_neighbors {
template<typename T>
using host_array = ::cellerator::compute::neighbors::host_buffer<T>;

template<typename T>
using const_array_view = ::cellerator::compute::neighbors::const_buffer_view<T>;

using ::cellerator::compute::neighbors::view_of;
namespace cs = ::cellshard;
namespace css = ::cellshard::sparse;

enum class ForwardNeighborBackend {
    exact_windowed,
    ann_windowed
};

enum class ForwardNeighborDirection {
    forward
};

enum class ForwardNeighborEmbryoPolicy {
    any_embryo,
    same_embryo_first,
    same_embryo_only
};

struct ForwardTimeWindow {
    float min_delta = 0.0f;
    float max_delta = std::numeric_limits<float>::infinity();
};

enum class ForwardNeighborInputLayout {
    dense,
    blocked_ell,
    sliced_ell
};

struct ForwardNeighborMatrixView {
    ForwardNeighborInputLayout layout = ForwardNeighborInputLayout::dense;
    const float *dense_values = nullptr;
    std::int64_t dense_rows = 0;
    std::int64_t dense_cols = 0;
    std::int64_t dense_row_stride = 0;
    const css::blocked_ell *blocked_ell = nullptr;
    const css::sliced_ell *sliced_ell = nullptr;
};

inline ForwardNeighborMatrixView make_forward_neighbor_dense_view(
    const float *values,
    std::int64_t rows,
    std::int64_t cols,
    std::int64_t row_stride = 0) {
    return ForwardNeighborMatrixView{
        ForwardNeighborInputLayout::dense,
        values,
        rows,
        cols,
        row_stride > 0 ? row_stride : cols,
        nullptr,
        nullptr
    };
}

inline ForwardNeighborMatrixView make_forward_neighbor_blocked_ell_view(const css::blocked_ell *matrix) {
    return ForwardNeighborMatrixView{
        ForwardNeighborInputLayout::blocked_ell,
        nullptr,
        0,
        0,
        0,
        matrix,
        nullptr
    };
}

inline ForwardNeighborMatrixView make_forward_neighbor_sliced_ell_view(const css::sliced_ell *matrix) {
    return ForwardNeighborMatrixView{
        ForwardNeighborInputLayout::sliced_ell,
        nullptr,
        0,
        0,
        0,
        nullptr,
        matrix
    };
}

struct ForwardNeighborQueryBatch {
    const_array_view<std::int64_t> cell_indices;
    const_array_view<float> developmental_time;
    const_array_view<std::int64_t> embryo_ids;
    ForwardNeighborMatrixView matrix;
};

struct ForwardNeighborOwnedQueryBatch {
    host_array<std::int64_t> cell_indices;
    host_array<float> developmental_time;
    host_array<std::int64_t> embryo_ids;
    host_array<float> dense_values;
    std::int64_t dense_cols = 0;

    ForwardNeighborQueryBatch view() const {
        return ForwardNeighborQueryBatch{
            view_of(cell_indices),
            view_of(developmental_time),
            view_of(embryo_ids),
            make_forward_neighbor_dense_view(dense_values.data(),
                                             static_cast<std::int64_t>(cell_indices.size()),
                                             dense_cols)
        };
    }
};

struct ForwardNeighborSearchConfig {
    ForwardNeighborBackend backend = ForwardNeighborBackend::exact_windowed;
    ForwardNeighborDirection direction = ForwardNeighborDirection::forward;
    ForwardNeighborEmbryoPolicy embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
    std::int64_t top_k = 15;
    std::int64_t candidate_k = 15;
    std::int64_t query_block_rows = 256;
    std::int64_t index_block_rows = 16384;
    std::int64_t ann_probe_list_count = 8;
    float strict_future_epsilon = 0.0f;
    ForwardTimeWindow time_window{};
};

struct ForwardNeighborExecutorConfig {
    std::int64_t max_resident_shards_per_device = 2;
    std::size_t resident_bytes_per_device = 0u;
    bool enable_pair_pooling = true;
    bool allow_cross_pair_merge = true;
};

struct ForwardNeighborRoutingPlan {
    host_array<std::int64_t> block_query_begin;
    host_array<std::int64_t> block_query_end;
    host_array<float> block_window_lower;
    host_array<float> block_window_upper;
    host_array<std::uint32_t> block_route_offsets;
    host_array<std::uint32_t> route_shard_indices;
    host_array<int> route_device_ids;
};

struct ForwardNeighborSearchResult {
    std::int64_t query_count = 0;
    std::int64_t top_k = 0;
    host_array<std::int64_t> query_cell_indices;
    host_array<float> query_time;
    host_array<std::int64_t> query_embryo_ids;
    host_array<std::int64_t> neighbor_cell_indices;
    host_array<std::int64_t> neighbor_shard_indices;
    host_array<float> neighbor_time;
    host_array<std::int64_t> neighbor_embryo_ids;
    host_array<float> neighbor_similarity;
    host_array<float> neighbor_sqdist;
    host_array<float> neighbor_distance;
};

namespace detail {

inline std::size_t checked_size_(std::int64_t value, const char *label) {
    if (value < 0) throw std::invalid_argument(std::string(label) + " must be >= 0");
    return static_cast<std::size_t>(value);
}

inline std::size_t result_offset_(std::int64_t row, std::int64_t slot, std::int64_t top_k) {
    return checked_size_(row, "row") * checked_size_(top_k, "top_k") + checked_size_(slot, "slot");
}

inline float negative_infinity_() {
    return -std::numeric_limits<float>::infinity();
}

inline float positive_infinity_() {
    return std::numeric_limits<float>::infinity();
}

inline float quiet_nan_() {
    return std::numeric_limits<float>::quiet_NaN();
}

inline host_array<std::int64_t> make_missing_i64_array_(std::size_t rows) {
    host_array<std::int64_t> out;
    out.assign_fill(rows, static_cast<std::int64_t>(-1));
    return out;
}

inline std::int64_t checked_i64_(std::size_t value, const char *label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit into int64");
    }
    return static_cast<std::int64_t>(value);
}

inline std::int64_t matrix_rows_(const ForwardNeighborMatrixView &matrix) {
    switch (matrix.layout) {
        case ForwardNeighborInputLayout::dense:
            return matrix.dense_rows;
        case ForwardNeighborInputLayout::blocked_ell:
            return matrix.blocked_ell != nullptr ? static_cast<std::int64_t>(matrix.blocked_ell->rows) : 0;
        case ForwardNeighborInputLayout::sliced_ell:
            return matrix.sliced_ell != nullptr ? static_cast<std::int64_t>(matrix.sliced_ell->rows) : 0;
    }
    return 0;
}

inline std::int64_t matrix_cols_(const ForwardNeighborMatrixView &matrix) {
    switch (matrix.layout) {
        case ForwardNeighborInputLayout::dense:
            return matrix.dense_cols;
        case ForwardNeighborInputLayout::blocked_ell:
            return matrix.blocked_ell != nullptr ? static_cast<std::int64_t>(matrix.blocked_ell->cols) : 0;
        case ForwardNeighborInputLayout::sliced_ell:
            return matrix.sliced_ell != nullptr ? static_cast<std::int64_t>(matrix.sliced_ell->cols) : 0;
    }
    return 0;
}

inline void validate_forward_neighbor_search_config_(const ForwardNeighborSearchConfig &config) {
    if (config.top_k <= 0) throw std::invalid_argument("ForwardNeighborSearchConfig.top_k must be > 0");
    if (config.direction != ForwardNeighborDirection::forward) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.direction is not implemented");
    }
    if (config.candidate_k < config.top_k) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.candidate_k must be >= top_k");
    }
    if (config.query_block_rows <= 0) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.query_block_rows must be > 0");
    }
    if (config.index_block_rows <= 0) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.index_block_rows must be > 0");
    }
    if (config.ann_probe_list_count <= 0) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.ann_probe_list_count must be > 0");
    }
    if (config.strict_future_epsilon < 0.0f) {
        throw std::invalid_argument("ForwardNeighborSearchConfig.strict_future_epsilon must be >= 0");
    }
    if (config.time_window.min_delta < 0.0f) {
        throw std::invalid_argument("ForwardTimeWindow.min_delta must be >= 0");
    }
    if (std::isfinite(config.time_window.max_delta) && config.time_window.max_delta < config.time_window.min_delta) {
        throw std::invalid_argument("ForwardTimeWindow.max_delta must be >= min_delta");
    }
}

inline void validate_forward_neighbor_query_batch_(const ForwardNeighborQueryBatch &query) {
    if (matrix_cols_(query.matrix) <= 0) {
        throw std::invalid_argument("ForwardNeighborQueryBatch matrix cols must be > 0");
    }
    if (query.cell_indices.size != query.developmental_time.size) {
        throw std::invalid_argument("ForwardNeighborQueryBatch cell_indices and developmental_time must align");
    }
    if (matrix_rows_(query.matrix) != checked_i64_(query.cell_indices.size, "query rows")) {
        throw std::invalid_argument("ForwardNeighborQueryBatch matrix rows must align with query rows");
    }
    if (!query.embryo_ids.empty() && query.embryo_ids.size != query.cell_indices.size) {
        throw std::invalid_argument("ForwardNeighborQueryBatch embryo_ids must be empty or align with rows");
    }
    if (query.matrix.layout == ForwardNeighborInputLayout::dense) {
        if (query.matrix.dense_values == nullptr && query.cell_indices.size != 0u) {
            throw std::invalid_argument("ForwardNeighborQueryBatch dense_values must not be null");
        }
        if (query.matrix.dense_row_stride < matrix_cols_(query.matrix)) {
            throw std::invalid_argument("ForwardNeighborQueryBatch dense_row_stride must be >= dense_cols");
        }
    }
    if (query.matrix.layout == ForwardNeighborInputLayout::blocked_ell && query.matrix.blocked_ell == nullptr) {
        throw std::invalid_argument("ForwardNeighborQueryBatch blocked_ell view is null");
    }
    if (query.matrix.layout == ForwardNeighborInputLayout::sliced_ell && query.matrix.sliced_ell == nullptr) {
        throw std::invalid_argument("ForwardNeighborQueryBatch sliced_ell view is null");
    }
}

inline ForwardNeighborSearchResult empty_forward_neighbor_result_(std::int64_t top_k) {
    ForwardNeighborSearchResult result;
    result.query_count = 0;
    result.top_k = top_k;
    return result;
}

} // namespace detail

} // namespace cellerator::compute::neighbors::forward_neighbors
