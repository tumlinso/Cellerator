#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace cellerator::compute::neighbors::forward_neighbors {

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
    std::vector<std::int64_t> cell_indices;
    std::vector<float> developmental_time;
    std::vector<float> latent_unit;
    std::vector<std::int64_t> embryo_ids;
    std::int64_t latent_dim = 0;
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
    std::int64_t query_count = 0;
    std::int64_t top_k = 0;
    std::vector<std::int64_t> query_cell_indices;
    std::vector<float> query_time;
    std::vector<std::int64_t> query_embryo_ids;
    std::vector<std::int64_t> neighbor_cell_indices;
    std::vector<float> neighbor_time;
    std::vector<std::int64_t> neighbor_embryo_ids;
    std::vector<float> neighbor_similarity;
    std::vector<float> neighbor_sqdist;
    std::vector<float> neighbor_distance;
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

inline std::vector<std::int64_t> make_missing_i64_vector_(std::size_t rows) {
    return std::vector<std::int64_t>(rows, static_cast<std::int64_t>(-1));
}

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
    if (query.latent_dim <= 0) {
        throw std::invalid_argument("ForwardNeighborQueryBatch.latent_dim must be > 0");
    }
    if (query.cell_indices.size() != query.developmental_time.size()) {
        throw std::invalid_argument("ForwardNeighborQueryBatch cell_indices and developmental_time must align");
    }
    if (query.cell_indices.size() * checked_size_(query.latent_dim, "latent_dim") != query.latent_unit.size()) {
        throw std::invalid_argument("ForwardNeighborQueryBatch latent_unit must equal rows * latent_dim");
    }
    if (!query.embryo_ids.empty() && query.embryo_ids.size() != query.cell_indices.size()) {
        throw std::invalid_argument("ForwardNeighborQueryBatch embryo_ids must be empty or align with rows");
    }
}

inline ForwardNeighborSearchResult empty_forward_neighbor_result_(std::int64_t top_k) {
    return ForwardNeighborSearchResult{
        0,
        top_k,
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        {}
    };
}

} // namespace detail

} // namespace cellerator::compute::neighbors::forward_neighbors
