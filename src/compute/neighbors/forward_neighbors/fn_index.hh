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
    host_array<std::int64_t> cell_indices;
    host_array<float> developmental_time;
    host_array<float> latent_unit;
    host_array<std::int64_t> embryo_ids;
    std::int64_t latent_dim = 0;
};

struct ForwardNeighborBuildConfig {
    host_array<int> shard_devices;
    std::int64_t target_shard_count = 0;
    std::int64_t max_rows_per_segment = 0;
    std::int64_t ann_rows_per_list = 4096;
    bool renormalize_latent = true;
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

namespace detail {
struct ForwardNeighborIndexStorage;
}

class ForwardNeighborIndex;

class ForwardNeighborIndexBuilder {
public:
    explicit ForwardNeighborIndexBuilder(ForwardNeighborBuildConfig config = ForwardNeighborBuildConfig());

    void append(const ForwardNeighborRecordBatch &batch);
    ForwardNeighborIndex finalize() &&;

private:
    ForwardNeighborBuildConfig config_;
    ForwardNeighborRecordBatch records_;
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

    ForwardNeighborQueryBatch query_batch_from_cell_indices(const std::int64_t *cell_indices, std::size_t cell_count) const;
    ForwardNeighborSearchResult search_future_neighbors(
        const ForwardNeighborQueryBatch &query,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;
    ForwardNeighborSearchResult search_future_neighbors_by_cell_index(
        const std::int64_t *cell_indices,
        std::size_t cell_count,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const;

private:
    std::shared_ptr<detail::ForwardNeighborIndexStorage> storage_;
};

ForwardNeighborIndex build_forward_neighbor_index(
    const ForwardNeighborRecordBatch &records,
    const ForwardNeighborBuildConfig &config = ForwardNeighborBuildConfig());

namespace detail {

inline void validate_forward_neighbor_record_batch_(const ForwardNeighborRecordBatch &batch) {
    if (batch.latent_dim <= 0) {
        throw std::invalid_argument("ForwardNeighborRecordBatch.latent_dim must be > 0");
    }
    if (batch.cell_indices.size() != batch.developmental_time.size()) {
        throw std::invalid_argument("ForwardNeighborRecordBatch cell_indices and developmental_time must align");
    }
    if (batch.cell_indices.size() * static_cast<std::size_t>(batch.latent_dim) != batch.latent_unit.size()) {
        throw std::invalid_argument("ForwardNeighborRecordBatch latent_unit must equal rows * latent_dim");
    }
    if (!batch.embryo_ids.empty() && batch.embryo_ids.size() != batch.cell_indices.size()) {
        throw std::invalid_argument("ForwardNeighborRecordBatch embryo_ids must be empty or align with rows");
    }
}

inline std::int64_t checked_i64_(std::size_t value, const char *label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit into int64");
    }
    return static_cast<std::int64_t>(value);
}

inline std::int64_t default_segment_rows_(const ForwardNeighborBuildConfig &config, std::int64_t rows, std::int64_t shard_count) {
    if (config.max_rows_per_segment > 0) return config.max_rows_per_segment;
    if (rows == 0 || shard_count <= 0) return rows;
    const std::int64_t target = std::max<std::int64_t>(1, (rows + shard_count - 1) / shard_count);
    return std::max<std::int64_t>(target, config.ann_rows_per_list > 0 ? config.ann_rows_per_list : 1);
}

inline void append_i64_(host_array<std::int64_t> *dst, const host_array<std::int64_t> &src) {
    const std::size_t old_size = dst->size();
    dst->resize(old_size + src.size());
    if (!src.empty()) std::memcpy(dst->data() + old_size, src.data(), src.size() * sizeof(std::int64_t));
}

inline void append_f32_(host_array<float> *dst, const host_array<float> &src) {
    const std::size_t old_size = dst->size();
    dst->resize(old_size + src.size());
    if (!src.empty()) std::memcpy(dst->data() + old_size, src.data(), src.size() * sizeof(float));
}

} // namespace detail

} // namespace cellerator::compute::neighbors::forward_neighbors
