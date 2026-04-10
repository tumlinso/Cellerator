#pragma once

#include <torch/torch.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cellerator::models::forward_neighbors {

enum class ForwardNeighborBackend {
    exact_windowed,
    ann_windowed
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

struct ForwardNeighborQueryBatch {
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
    torch::Tensor latent_unit;
    torch::Tensor embryo_ids;
};

struct ForwardNeighborSearchConfig {
    ForwardNeighborBackend backend = ForwardNeighborBackend::exact_windowed;
    ForwardNeighborEmbryoPolicy embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
    std::int64_t top_k = 15;
    std::int64_t candidate_k = 15;
    std::int64_t query_block_rows = 256;
    std::int64_t index_block_rows = 16384;
    std::int64_t ann_probe_list_count = 8;
    float strict_future_epsilon = 0.0f;
    ForwardTimeWindow time_window{};
};

struct ForwardNeighborSearchResult {
    torch::Tensor query_cell_indices;
    torch::Tensor query_time;
    torch::Tensor query_embryo_ids;
    torch::Tensor neighbor_cell_indices;
    torch::Tensor neighbor_time;
    torch::Tensor neighbor_embryo_ids;
    torch::Tensor neighbor_similarity;
    torch::Tensor neighbor_sqdist;
    torch::Tensor neighbor_distance;
};

namespace detail {

inline void validate_forward_neighbor_search_config_(const ForwardNeighborSearchConfig &config) {
    if (config.top_k <= 0) throw std::invalid_argument("ForwardNeighborSearchConfig.top_k must be > 0");
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
    if (query.cell_indices.dim() != 1) {
        throw std::invalid_argument("ForwardNeighborQueryBatch.cell_indices must be rank-1");
    }
    if (query.developmental_time.dim() != 1) {
        throw std::invalid_argument("ForwardNeighborQueryBatch.developmental_time must be rank-1");
    }
    if (query.latent_unit.dim() != 2) {
        throw std::invalid_argument("ForwardNeighborQueryBatch.latent_unit must be rank-2");
    }
    if (query.cell_indices.size(0) != query.developmental_time.size(0) || query.cell_indices.size(0) != query.latent_unit.size(0)) {
        throw std::invalid_argument("ForwardNeighborQueryBatch tensors must agree on the query row count");
    }
    if (query.embryo_ids.defined()) {
        if (query.embryo_ids.dim() != 1 || query.embryo_ids.size(0) != query.cell_indices.size(0)) {
            throw std::invalid_argument("ForwardNeighborQueryBatch.embryo_ids must be rank-1 and align with the query row count");
        }
    }
}

inline ForwardNeighborSearchResult empty_forward_neighbor_result_(std::int64_t top_k) {
    return ForwardNeighborSearchResult{
        torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)),
        torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
        torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)),
        torch::empty({ 0, top_k }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)),
        torch::empty({ 0, top_k }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
        torch::empty({ 0, top_k }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)),
        torch::empty({ 0, top_k }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
        torch::empty({ 0, top_k }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
        torch::empty({ 0, top_k }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
    };
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

inline torch::Tensor make_missing_i64_tensor_(std::int64_t rows) {
    return torch::full(
        { rows },
        static_cast<std::int64_t>(-1),
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
}

} // namespace detail

} // namespace cellerator::models::forward_neighbors
