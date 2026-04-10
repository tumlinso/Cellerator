#pragma once

#include "fn_query.hh"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::models::forward_neighbors {

struct ForwardNeighborRecordBatch {
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
    torch::Tensor latent_unit;
    torch::Tensor embryo_ids;
};

struct ForwardNeighborBuildConfig {
    std::vector<torch::Device> shard_devices;
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

struct ForwardNeighborShard {
    torch::Device device = torch::Device(torch::kCPU);
    torch::Tensor cell_indices_cpu;
    torch::Tensor developmental_time_cpu;
    torch::Tensor embryo_ids_cpu;
    torch::Tensor cell_indices;
    torch::Tensor developmental_time;
    torch::Tensor embryo_ids;
    torch::Tensor latent_unit;
    torch::Tensor ann_centroids;
    std::vector<ForwardNeighborSegment> segments;
    std::vector<ForwardNeighborAnnList> ann_lists;
};

namespace detail {

struct Candidate {
    float similarity = detail::negative_infinity_();
    float developmental_time = detail::quiet_nan_();
    std::int64_t embryo_id = -1;
    std::int64_t cell_index = -1;
};

struct SegmentPlan {
    std::int64_t embryo_id = -1;
    std::int64_t source_begin = 0;
    std::int64_t source_end = 0;
    float time_begin = detail::quiet_nan_();
    float time_end = detail::quiet_nan_();
};

inline std::vector<torch::Device> resolve_forward_neighbor_devices_(
    const ForwardNeighborBuildConfig &config) {
    if (!config.shard_devices.empty()) return config.shard_devices;
    return { torch::Device(torch::kCPU) };
}

inline torch::Tensor contiguous_i64_cpu_(const torch::Tensor &tensor) {
    return tensor.to(torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU)).contiguous();
}

inline torch::Tensor contiguous_f32_cpu_(const torch::Tensor &tensor) {
    return tensor.to(torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)).contiguous();
}

inline torch::Tensor optional_i64_cpu_(const torch::Tensor &tensor, std::int64_t rows) {
    if (!tensor.defined()) return detail::make_missing_i64_tensor_(rows);
    return contiguous_i64_cpu_(tensor);
}

inline std::int64_t checked_i64_(std::size_t value, const char *label) {
    if (value > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max())) {
        throw std::overflow_error(std::string(label) + " does not fit into int64");
    }
    return static_cast<std::int64_t>(value);
}

inline torch::Tensor copy_i64_tensor_(const std::vector<std::int64_t> &values) {
    torch::Tensor tensor = torch::empty(
        { checked_i64_(values.size(), "vector size") },
        torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<std::int64_t>(), values.data(), values.size() * sizeof(std::int64_t));
    }
    return tensor;
}

inline torch::Tensor copy_f32_tensor_(const std::vector<float> &values) {
    torch::Tensor tensor = torch::empty(
        { checked_i64_(values.size(), "vector size") },
        torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    if (!values.empty()) {
        std::memcpy(tensor.data_ptr<float>(), values.data(), values.size() * sizeof(float));
    }
    return tensor;
}

inline torch::Tensor normalize_latent_rows_(const torch::Tensor &latent) {
    return torch::nn::functional::normalize(
        latent.to(torch::kFloat32),
        torch::nn::functional::NormalizeFuncOptions().p(2.0).dim(1).eps(1.0e-6));
}

inline void validate_forward_neighbor_record_batch_(const ForwardNeighborRecordBatch &batch) {
    if (batch.cell_indices.dim() != 1) {
        throw std::invalid_argument("ForwardNeighborRecordBatch.cell_indices must be rank-1");
    }
    if (batch.developmental_time.dim() != 1) {
        throw std::invalid_argument("ForwardNeighborRecordBatch.developmental_time must be rank-1");
    }
    if (batch.latent_unit.dim() != 2) {
        throw std::invalid_argument("ForwardNeighborRecordBatch.latent_unit must be rank-2");
    }
    if (batch.cell_indices.size(0) != batch.developmental_time.size(0)
        || batch.cell_indices.size(0) != batch.latent_unit.size(0)) {
        throw std::invalid_argument("ForwardNeighborRecordBatch tensors must agree on the row count");
    }
    if (batch.latent_unit.size(1) <= 0) {
        throw std::invalid_argument("ForwardNeighborRecordBatch.latent_unit must have a positive latent dimension");
    }
    if (batch.embryo_ids.defined()) {
        if (batch.embryo_ids.dim() != 1 || batch.embryo_ids.size(0) != batch.cell_indices.size(0)) {
            throw std::invalid_argument("ForwardNeighborRecordBatch.embryo_ids must be rank-1 and align with the row count");
        }
    }
}

inline bool better_candidate_(const Candidate &lhs, const Candidate &rhs) {
    const bool lhs_valid = std::isfinite(lhs.similarity) && lhs.cell_index >= 0;
    const bool rhs_valid = std::isfinite(rhs.similarity) && rhs.cell_index >= 0;
    if (!lhs_valid) return false;
    if (!rhs_valid) return true;
    if (lhs.similarity > rhs.similarity) return true;
    if (lhs.similarity < rhs.similarity) return false;
    if (lhs.developmental_time < rhs.developmental_time) return true;
    if (lhs.developmental_time > rhs.developmental_time) return false;
    if (lhs.embryo_id < rhs.embryo_id) return true;
    if (lhs.embryo_id > rhs.embryo_id) return false;
    return lhs.cell_index < rhs.cell_index;
}

inline void insert_candidate_(const Candidate &candidate, std::vector<Candidate> *best) {
    if (best == nullptr || best->empty()) return;
    const std::size_t k = best->size();
    if (!better_candidate_(candidate, (*best)[k - 1u])) return;

    std::size_t insert = k - 1u;
    while (insert > 0u && better_candidate_(candidate, (*best)[insert - 1u])) {
        (*best)[insert] = (*best)[insert - 1u];
        --insert;
    }
    (*best)[insert] = candidate;
}

inline float sqdist_from_similarity_(float similarity) {
    if (!std::isfinite(similarity)) return detail::positive_infinity_();
    return std::max(0.0f, 2.0f - 2.0f * similarity);
}

inline std::int64_t default_segment_rows_(const ForwardNeighborBuildConfig &config, std::int64_t rows, std::int64_t shard_count) {
    if (config.max_rows_per_segment > 0) return config.max_rows_per_segment;
    if (rows == 0 || shard_count <= 0) return rows;
    const std::int64_t target = std::max<std::int64_t>(1, (rows + shard_count - 1) / shard_count);
    return std::max<std::int64_t>(target, config.ann_rows_per_list > 0 ? config.ann_rows_per_list : 1);
}

inline std::vector<std::int64_t> sorted_row_order_(
    const torch::Tensor &cell_indices,
    const torch::Tensor &developmental_time,
    const torch::Tensor &embryo_ids) {
    const std::int64_t rows = cell_indices.size(0);
    const std::int64_t *cell_ptr = cell_indices.data_ptr<std::int64_t>();
    const float *time_ptr = developmental_time.data_ptr<float>();
    const std::int64_t *embryo_ptr = embryo_ids.data_ptr<std::int64_t>();

    std::vector<std::int64_t> order(static_cast<std::size_t>(rows));
    for (std::int64_t row = 0; row < rows; ++row) order[static_cast<std::size_t>(row)] = row;
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t lhs, std::int64_t rhs) {
        if (embryo_ptr[lhs] < embryo_ptr[rhs]) return true;
        if (embryo_ptr[lhs] > embryo_ptr[rhs]) return false;
        if (time_ptr[lhs] < time_ptr[rhs]) return true;
        if (time_ptr[lhs] > time_ptr[rhs]) return false;
        return cell_ptr[lhs] < cell_ptr[rhs];
    });
    return order;
}

inline std::vector<SegmentPlan> build_segment_plans_(
    const torch::Tensor &developmental_time,
    const torch::Tensor &embryo_ids,
    std::int64_t max_rows_per_segment) {
    const std::int64_t rows = developmental_time.size(0);
    const float *time_ptr = developmental_time.data_ptr<float>();
    const std::int64_t *embryo_ptr = embryo_ids.data_ptr<std::int64_t>();
    std::vector<SegmentPlan> segments;

    std::int64_t begin = 0;
    while (begin < rows) {
        const std::int64_t embryo_id = embryo_ptr[begin];
        std::int64_t end = begin + 1;
        while (end < rows && embryo_ptr[end] == embryo_id) ++end;

        for (std::int64_t block_begin = begin; block_begin < end; block_begin += max_rows_per_segment) {
            const std::int64_t block_end = std::min(block_begin + max_rows_per_segment, end);
            segments.push_back(SegmentPlan{
                embryo_id,
                block_begin,
                block_end,
                time_ptr[block_begin],
                time_ptr[block_end - 1]
            });
        }
        begin = end;
    }
    return segments;
}

inline torch::Tensor slice_tensor_rows_(const torch::Tensor &tensor, std::int64_t begin, std::int64_t end) {
    return tensor.index({ torch::indexing::Slice(begin, end) }).contiguous();
}

inline std::vector<std::vector<SegmentPlan>> assign_segments_to_shards_(
    const std::vector<SegmentPlan> &segments,
    std::size_t shard_count) {
    std::vector<std::vector<SegmentPlan>> plans(std::max<std::size_t>(1u, shard_count));
    std::vector<std::int64_t> shard_rows(plans.size(), 0);
    std::vector<SegmentPlan> sorted = segments;
    std::sort(sorted.begin(), sorted.end(), [](const SegmentPlan &lhs, const SegmentPlan &rhs) {
        const std::int64_t lhs_rows = lhs.source_end - lhs.source_begin;
        const std::int64_t rhs_rows = rhs.source_end - rhs.source_begin;
        if (lhs_rows > rhs_rows) return true;
        if (lhs_rows < rhs_rows) return false;
        if (lhs.embryo_id < rhs.embryo_id) return true;
        if (lhs.embryo_id > rhs.embryo_id) return false;
        return lhs.source_begin < rhs.source_begin;
    });

    for (const SegmentPlan &segment : sorted) {
        std::size_t best_shard = 0;
        for (std::size_t shard = 1; shard < plans.size(); ++shard) {
            if (shard_rows[shard] < shard_rows[best_shard]) best_shard = shard;
        }
        plans[best_shard].push_back(segment);
        shard_rows[best_shard] += segment.source_end - segment.source_begin;
    }
    return plans;
}

inline std::vector<std::int64_t> eligible_segment_order_(
    const std::vector<ForwardNeighborSegment> &segments,
    const std::unordered_set<std::int64_t> &query_embryos,
    ForwardNeighborEmbryoPolicy policy) {
    std::vector<std::int64_t> ordered;
    std::vector<std::int64_t> deferred;
    ordered.reserve(segments.size());
    deferred.reserve(segments.size());

    for (std::int64_t segment_idx = 0; segment_idx < static_cast<std::int64_t>(segments.size()); ++segment_idx) {
        const bool embryo_match = !query_embryos.empty() && query_embryos.find(segments[static_cast<std::size_t>(segment_idx)].embryo_id) != query_embryos.end();
        if (policy == ForwardNeighborEmbryoPolicy::same_embryo_only) {
            if (query_embryos.empty() || embryo_match) ordered.push_back(segment_idx);
            continue;
        }
        if (policy == ForwardNeighborEmbryoPolicy::same_embryo_first && embryo_match) {
            ordered.push_back(segment_idx);
        } else if (policy == ForwardNeighborEmbryoPolicy::same_embryo_first) {
            deferred.push_back(segment_idx);
        } else {
            ordered.push_back(segment_idx);
        }
    }
    ordered.insert(ordered.end(), deferred.begin(), deferred.end());
    return ordered;
}

inline std::unordered_set<std::int64_t> block_query_embryos_(
    const std::int64_t *query_embryo_ptr,
    std::int64_t query_begin,
    std::int64_t query_end) {
    std::unordered_set<std::int64_t> embryo_ids;
    for (std::int64_t row = query_begin; row < query_end; ++row) {
        if (query_embryo_ptr[row] >= 0) embryo_ids.insert(query_embryo_ptr[row]);
    }
    return embryo_ids;
}

inline bool segment_time_overlaps_block_(
    const ForwardNeighborSegment &segment,
    float block_lower,
    float block_upper) {
    if (segment.row_begin >= segment.row_end) return false;
    if (segment.time_end <= block_lower) return false;
    if (std::isfinite(block_upper) && segment.time_begin > block_upper) return false;
    return true;
}

inline bool ann_list_overlaps_block_(
    const ForwardNeighborAnnList &list,
    float block_lower,
    float block_upper) {
    if (list.row_begin >= list.row_end) return false;
    if (list.time_end <= block_lower) return false;
    if (std::isfinite(block_upper) && list.time_begin > block_upper) return false;
    return true;
}

inline std::pair<std::int64_t, std::int64_t> segment_candidate_bounds_(
    const ForwardNeighborShard &shard,
    const ForwardNeighborSegment &segment,
    float block_lower,
    float block_upper) {
    const float *time_ptr = shard.developmental_time_cpu.data_ptr<float>();
    const float *begin_ptr = time_ptr + segment.row_begin;
    const float *end_ptr = time_ptr + segment.row_end;
    const float *lower_it = std::upper_bound(begin_ptr, end_ptr, block_lower);
    const float *upper_it = std::isfinite(block_upper)
        ? std::upper_bound(lower_it, end_ptr, block_upper)
        : end_ptr;
    return {
        static_cast<std::int64_t>(lower_it - time_ptr),
        static_cast<std::int64_t>(upper_it - time_ptr)
    };
}

inline std::pair<float, float> block_time_limits_(
    const float *query_time_ptr,
    std::int64_t query_begin,
    std::int64_t query_end,
    const ForwardNeighborSearchConfig &config) {
    float block_lower = detail::positive_infinity_();
    float block_upper = -detail::positive_infinity_();
    for (std::int64_t row = query_begin; row < query_end; ++row) {
        const float lower = query_time_ptr[row] + config.strict_future_epsilon + config.time_window.min_delta;
        const float upper = std::isfinite(config.time_window.max_delta)
            ? query_time_ptr[row] + config.time_window.max_delta
            : detail::positive_infinity_();
        if (lower < block_lower) block_lower = lower;
        if (upper > block_upper) block_upper = upper;
    }
    return { block_lower, block_upper };
}

inline std::vector<std::pair<std::int64_t, std::int64_t>> merge_row_intervals_(
    std::vector<std::pair<std::int64_t, std::int64_t>> intervals) {
    if (intervals.empty()) return intervals;
    std::sort(intervals.begin(), intervals.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.first < rhs.first) return true;
        if (lhs.first > rhs.first) return false;
        return lhs.second < rhs.second;
    });

    std::vector<std::pair<std::int64_t, std::int64_t>> merged;
    merged.push_back(intervals[0]);
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, intervals[i].second);
        } else {
            merged.push_back(intervals[i]);
        }
    }
    return merged;
}

} // namespace detail

class ForwardNeighborIndex;

class ForwardNeighborIndexBuilder {
public:
    explicit ForwardNeighborIndexBuilder(ForwardNeighborBuildConfig config = ForwardNeighborBuildConfig())
        : config_(std::move(config)) {}

    void append(const ForwardNeighborRecordBatch &batch) {
        detail::validate_forward_neighbor_record_batch_(batch);
        batches_.push_back(ForwardNeighborRecordBatch{
            detail::contiguous_i64_cpu_(batch.cell_indices),
            detail::contiguous_f32_cpu_(batch.developmental_time),
            detail::contiguous_f32_cpu_(batch.latent_unit),
            detail::optional_i64_cpu_(batch.embryo_ids, batch.cell_indices.size(0))
        });
    }

    ForwardNeighborIndex finalize() &&;

private:
    ForwardNeighborBuildConfig config_;
    std::vector<ForwardNeighborRecordBatch> batches_;
};

class ForwardNeighborIndex {
public:
    struct CellLocation {
        std::size_t shard = 0;
        std::int64_t local_row = 0;
    };

    ForwardNeighborIndex() = default;

    explicit ForwardNeighborIndex(std::vector<ForwardNeighborShard> shards)
        : shards_(std::move(shards)) {
        build_lookup_();
    }

    std::size_t shard_count() const {
        return shards_.size();
    }

    std::int64_t latent_dim() const {
        for (const ForwardNeighborShard &shard : shards_) {
            if (shard.latent_unit.defined()) return shard.latent_unit.size(1);
        }
        return 0;
    }

    ForwardNeighborQueryBatch query_batch_from_cell_indices(const torch::Tensor &cell_indices) const {
        torch::Tensor query_ids = detail::contiguous_i64_cpu_(cell_indices);
        if (query_ids.dim() != 1) throw std::invalid_argument("query cell indices must be rank-1");
        if (query_ids.size(0) == 0) {
            return ForwardNeighborQueryBatch{
                std::move(query_ids),
                torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
                torch::empty({ 0, latent_dim() }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)),
                torch::empty({ 0 }, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU))
            };
        }

        const std::int64_t query_count = query_ids.size(0);
        std::vector<float> query_time(static_cast<std::size_t>(query_count), detail::quiet_nan_());
        std::vector<std::int64_t> query_embryo(static_cast<std::size_t>(query_count), -1);
        torch::Tensor query_latent = torch::empty(
            { query_count, latent_dim() },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        std::unordered_map<std::size_t, std::vector<std::int64_t>> positions_by_shard;
        std::unordered_map<std::size_t, std::vector<std::int64_t>> rows_by_shard;
        const std::int64_t *id_ptr = query_ids.data_ptr<std::int64_t>();
        for (std::int64_t query_row = 0; query_row < query_count; ++query_row) {
            const auto it = row_by_cell_index_.find(id_ptr[query_row]);
            if (it == row_by_cell_index_.end()) {
                throw std::invalid_argument("query cell index is not present in the forward-neighbor index");
            }
            positions_by_shard[it->second.shard].push_back(query_row);
            rows_by_shard[it->second.shard].push_back(it->second.local_row);
            const ForwardNeighborShard &shard = shards_[it->second.shard];
            query_time[static_cast<std::size_t>(query_row)] =
                shard.developmental_time_cpu.data_ptr<float>()[it->second.local_row];
            query_embryo[static_cast<std::size_t>(query_row)] =
                shard.embryo_ids_cpu.data_ptr<std::int64_t>()[it->second.local_row];
        }

        for (const auto &entry : positions_by_shard) {
            const std::size_t shard_idx = entry.first;
            const ForwardNeighborShard &shard = shards_[shard_idx];
            torch::Tensor positions = detail::copy_i64_tensor_(entry.second);
            torch::Tensor rows = detail::copy_i64_tensor_(rows_by_shard[shard_idx]).to(shard.device);
            torch::Tensor gathered = shard.latent_unit.index_select(0, rows).to(torch::kCPU).contiguous();
            query_latent.index_copy_(0, positions, gathered);
        }

        return ForwardNeighborQueryBatch{
            std::move(query_ids),
            detail::copy_f32_tensor_(query_time),
            query_latent,
            detail::copy_i64_tensor_(query_embryo)
        };
    }

    ForwardNeighborSearchResult search_future_neighbors(
        const ForwardNeighborQueryBatch &query,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const {
        detail::validate_forward_neighbor_search_config_(config);
        detail::validate_forward_neighbor_query_batch_(query);
        if (query.cell_indices.size(0) == 0) return detail::empty_forward_neighbor_result_(config.top_k);
        if (latent_dim() != 0 && query.latent_unit.size(1) != latent_dim()) {
            throw std::invalid_argument("query latent dimension does not match the forward-neighbor index latent dimension");
        }

        torch::Tensor query_ids = detail::contiguous_i64_cpu_(query.cell_indices);
        torch::Tensor query_time = detail::contiguous_f32_cpu_(query.developmental_time);
        torch::Tensor query_latent = detail::normalize_latent_rows_(detail::contiguous_f32_cpu_(query.latent_unit)).contiguous();
        torch::Tensor query_embryo = detail::optional_i64_cpu_(query.embryo_ids, query_ids.size(0));

        const std::int64_t query_count = query_ids.size(0);
        const std::int64_t local_k = std::max(config.top_k, config.candidate_k);
        std::vector<std::vector<detail::Candidate>> best(
            static_cast<std::size_t>(query_count),
            std::vector<detail::Candidate>(static_cast<std::size_t>(config.top_k)));
        const float *query_time_ptr = query_time.data_ptr<float>();
        const std::int64_t *query_embryo_ptr = query_embryo.data_ptr<std::int64_t>();

        for (const ForwardNeighborShard &shard : shards_) {
            if (!shard.latent_unit.defined() || shard.latent_unit.size(0) == 0) continue;

            for (std::int64_t query_begin = 0; query_begin < query_count; query_begin += config.query_block_rows) {
                const std::int64_t query_end = std::min(query_begin + config.query_block_rows, query_count);
                const std::int64_t block_queries = query_end - query_begin;
                const auto block_limits = detail::block_time_limits_(query_time_ptr, query_begin, query_end, config);
                const std::unordered_set<std::int64_t> block_embryos =
                    detail::block_query_embryos_(query_embryo_ptr, query_begin, query_end);
                const std::vector<std::int64_t> segment_order =
                    detail::eligible_segment_order_(shard.segments, block_embryos, config.embryo_policy);

                torch::Tensor query_latent_block = query_latent.index({
                    torch::indexing::Slice(query_begin, query_end)
                }).to(shard.device);
                torch::Tensor query_time_block = query_time.index({
                    torch::indexing::Slice(query_begin, query_end)
                }).to(shard.device);
                torch::Tensor query_embryo_block = query_embryo.index({
                    torch::indexing::Slice(query_begin, query_end)
                }).to(shard.device);
                torch::Tensor lower_bound = query_time_block + (config.strict_future_epsilon + config.time_window.min_delta);
                torch::Tensor upper_bound;
                if (std::isfinite(config.time_window.max_delta)) {
                    upper_bound = query_time_block + config.time_window.max_delta;
                }

                torch::Tensor shard_best_similarity = torch::full(
                    { block_queries, local_k },
                    detail::negative_infinity_(),
                    torch::TensorOptions().dtype(torch::kFloat32).device(shard.device));
                torch::Tensor shard_best_time = torch::full(
                    { block_queries, local_k },
                    detail::quiet_nan_(),
                    torch::TensorOptions().dtype(torch::kFloat32).device(shard.device));
                torch::Tensor shard_best_embryo = torch::full(
                    { block_queries, local_k },
                    static_cast<std::int64_t>(-1),
                    torch::TensorOptions().dtype(torch::kInt64).device(shard.device));
                torch::Tensor shard_best_id = torch::full(
                    { block_queries, local_k },
                    static_cast<std::int64_t>(-1),
                    torch::TensorOptions().dtype(torch::kInt64).device(shard.device));

                auto merge_candidate_rows = [&](const std::vector<std::pair<std::int64_t, std::int64_t>> &row_intervals) {
                    for (const auto &interval : row_intervals) {
                        for (std::int64_t index_begin = interval.first; index_begin < interval.second; index_begin += config.index_block_rows) {
                            const std::int64_t index_end = std::min(index_begin + config.index_block_rows, interval.second);
                            const std::int64_t block_rows = index_end - index_begin;
                            const std::int64_t block_k = std::min<std::int64_t>(local_k, block_rows);
                            if (block_k <= 0) continue;

                            torch::Tensor index_latent_block = shard.latent_unit.index({
                                torch::indexing::Slice(index_begin, index_end)
                            });
                            torch::Tensor index_time_block = shard.developmental_time.index({
                                torch::indexing::Slice(index_begin, index_end)
                            });
                            torch::Tensor index_embryo_block = shard.embryo_ids.index({
                                torch::indexing::Slice(index_begin, index_end)
                            });
                            torch::Tensor index_id_block = shard.cell_indices.index({
                                torch::indexing::Slice(index_begin, index_end)
                            });

                            torch::Tensor score_block = torch::matmul(query_latent_block, index_latent_block.transpose(0, 1));
                            torch::Tensor valid_mask = index_time_block.unsqueeze(0) > lower_bound.unsqueeze(1);
                            if (upper_bound.defined()) {
                                valid_mask = valid_mask.logical_and(index_time_block.unsqueeze(0) <= upper_bound.unsqueeze(1));
                            }
                            if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only) {
                                torch::Tensor query_has_embryo = query_embryo_block >= 0;
                                torch::Tensor embryo_match = index_embryo_block.unsqueeze(0) == query_embryo_block.unsqueeze(1);
                                valid_mask = valid_mask.logical_and(torch::where(
                                    query_has_embryo.unsqueeze(1),
                                    embryo_match,
                                    torch::ones_like(embryo_match)));
                            }
                            score_block = score_block.masked_fill(valid_mask.logical_not(), detail::negative_infinity_());

                            auto top = torch::topk(score_block, block_k, 1, true, true);
                            torch::Tensor block_similarity = std::get<0>(top);
                            torch::Tensor block_offset = std::get<1>(top);
                            torch::Tensor flat_offset = block_offset.reshape({ -1 });
                            torch::Tensor block_id = index_id_block.index_select(0, flat_offset).view(block_offset.sizes());
                            torch::Tensor block_time = index_time_block.index_select(0, flat_offset).view(block_offset.sizes());
                            torch::Tensor block_embryo = index_embryo_block.index_select(0, flat_offset).view(block_offset.sizes());
                            torch::Tensor finite_mask = torch::isfinite(block_similarity);
                            block_id = torch::where(
                                finite_mask,
                                block_id,
                                torch::full(block_id.sizes(), static_cast<std::int64_t>(-1), block_id.options()));
                            block_time = torch::where(
                                finite_mask,
                                block_time,
                                torch::full(block_time.sizes(), detail::quiet_nan_(), block_time.options()));
                            block_embryo = torch::where(
                                finite_mask,
                                block_embryo,
                                torch::full(block_embryo.sizes(), static_cast<std::int64_t>(-1), block_embryo.options()));

                            torch::Tensor merged_similarity = torch::cat({ shard_best_similarity, block_similarity }, 1);
                            torch::Tensor merged_time = torch::cat({ shard_best_time, block_time }, 1);
                            torch::Tensor merged_embryo = torch::cat({ shard_best_embryo, block_embryo }, 1);
                            torch::Tensor merged_id = torch::cat({ shard_best_id, block_id }, 1);
                            auto merged_top = torch::topk(merged_similarity, local_k, 1, true, true);
                            torch::Tensor keep_similarity = std::get<0>(merged_top);
                            torch::Tensor keep_offset = std::get<1>(merged_top);
                            shard_best_similarity = keep_similarity;
                            shard_best_time = merged_time.gather(1, keep_offset);
                            shard_best_embryo = merged_embryo.gather(1, keep_offset);
                            shard_best_id = merged_id.gather(1, keep_offset);
                        }
                    }
                };

                if (config.backend == ForwardNeighborBackend::exact_windowed) {
                    std::vector<std::pair<std::int64_t, std::int64_t>> intervals;
                    intervals.reserve(segment_order.size());
                    for (const std::int64_t segment_idx : segment_order) {
                        const ForwardNeighborSegment &segment = shard.segments[static_cast<std::size_t>(segment_idx)];
                        if (!detail::segment_time_overlaps_block_(segment, block_limits.first, block_limits.second)) continue;
                        const auto bounds = detail::segment_candidate_bounds_(shard, segment, block_limits.first, block_limits.second);
                        if (bounds.first < bounds.second) intervals.push_back(bounds);
                    }
                    merge_candidate_rows(detail::merge_row_intervals_(std::move(intervals)));
                } else {
                    std::vector<std::int64_t> eligible_lists;
                    eligible_lists.reserve(shard.ann_lists.size());
                    for (const std::int64_t segment_idx : segment_order) {
                        const ForwardNeighborSegment &segment = shard.segments[static_cast<std::size_t>(segment_idx)];
                        if (!detail::segment_time_overlaps_block_(segment, block_limits.first, block_limits.second)) continue;
                        for (std::int64_t list_idx = segment.ann_list_begin; list_idx < segment.ann_list_end; ++list_idx) {
                            const ForwardNeighborAnnList &list = shard.ann_lists[static_cast<std::size_t>(list_idx)];
                            if (detail::ann_list_overlaps_block_(list, block_limits.first, block_limits.second)) {
                                eligible_lists.push_back(list_idx);
                            }
                        }
                    }

                    if (!eligible_lists.empty()) {
                        torch::Tensor eligible_tensor = detail::copy_i64_tensor_(eligible_lists).to(shard.device);
                        torch::Tensor centroid_block = shard.ann_centroids.index_select(0, eligible_tensor);
                        torch::Tensor list_scores = torch::matmul(query_latent_block, centroid_block.transpose(0, 1));
                        const std::int64_t probe_count = std::min<std::int64_t>(
                            config.ann_probe_list_count,
                            static_cast<std::int64_t>(eligible_lists.size()));
                        auto top = torch::topk(list_scores, probe_count, 1, true, true);
                        torch::Tensor top_offset = std::get<1>(top).to(torch::kCPU).contiguous();
                        const std::int64_t *top_offset_ptr = top_offset.data_ptr<std::int64_t>();
                        std::unordered_set<std::int64_t> selected_list_ids;
                        for (std::int64_t row = 0; row < block_queries; ++row) {
                            for (std::int64_t slot = 0; slot < probe_count; ++slot) {
                                const std::size_t off = static_cast<std::size_t>(row * probe_count + slot);
                                selected_list_ids.insert(eligible_lists[static_cast<std::size_t>(top_offset_ptr[off])]);
                            }
                        }

                        std::vector<std::pair<std::int64_t, std::int64_t>> intervals;
                        intervals.reserve(selected_list_ids.size());
                        for (const std::int64_t list_idx : selected_list_ids) {
                            const ForwardNeighborAnnList &list = shard.ann_lists[static_cast<std::size_t>(list_idx)];
                            intervals.emplace_back(list.row_begin, list.row_end);
                        }
                        merge_candidate_rows(detail::merge_row_intervals_(std::move(intervals)));
                    }
                }

                torch::Tensor shard_best_similarity_cpu = shard_best_similarity.to(torch::kCPU).contiguous();
                torch::Tensor shard_best_time_cpu = shard_best_time.to(torch::kCPU).contiguous();
                torch::Tensor shard_best_embryo_cpu = shard_best_embryo.to(torch::kCPU).contiguous();
                torch::Tensor shard_best_id_cpu = shard_best_id.to(torch::kCPU).contiguous();
                const float *sim_ptr = shard_best_similarity_cpu.data_ptr<float>();
                const float *time_ptr = shard_best_time_cpu.data_ptr<float>();
                const std::int64_t *embryo_ptr = shard_best_embryo_cpu.data_ptr<std::int64_t>();
                const std::int64_t *id_ptr = shard_best_id_cpu.data_ptr<std::int64_t>();

                for (std::int64_t q = 0; q < block_queries; ++q) {
                    std::vector<detail::Candidate> &row_best = best[static_cast<std::size_t>(query_begin + q)];
                    for (std::int64_t slot = 0; slot < local_k; ++slot) {
                        const std::size_t off = static_cast<std::size_t>(q * local_k + slot);
                        detail::insert_candidate_(detail::Candidate{
                            sim_ptr[off],
                            time_ptr[off],
                            embryo_ptr[off],
                            id_ptr[off]
                        }, &row_best);
                    }
                }
            }
        }

        torch::Tensor out_ids = torch::empty(
            { query_count, config.top_k },
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        torch::Tensor out_time = torch::empty(
            { query_count, config.top_k },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        torch::Tensor out_embryo = torch::empty(
            { query_count, config.top_k },
            torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
        torch::Tensor out_similarity = torch::empty(
            { query_count, config.top_k },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        torch::Tensor out_sqdist = torch::empty(
            { query_count, config.top_k },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        torch::Tensor out_distance = torch::empty(
            { query_count, config.top_k },
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        std::int64_t *out_id_ptr = out_ids.data_ptr<std::int64_t>();
        float *out_time_ptr = out_time.data_ptr<float>();
        std::int64_t *out_embryo_ptr = out_embryo.data_ptr<std::int64_t>();
        float *out_similarity_ptr = out_similarity.data_ptr<float>();
        float *out_sqdist_ptr = out_sqdist.data_ptr<float>();
        float *out_distance_ptr = out_distance.data_ptr<float>();
        for (std::int64_t q = 0; q < query_count; ++q) {
            for (std::int64_t slot = 0; slot < config.top_k; ++slot) {
                const std::size_t off = static_cast<std::size_t>(q * config.top_k + slot);
                const detail::Candidate &candidate = best[static_cast<std::size_t>(q)][static_cast<std::size_t>(slot)];
                out_id_ptr[off] = candidate.cell_index;
                out_time_ptr[off] = candidate.developmental_time;
                out_embryo_ptr[off] = candidate.embryo_id;
                out_similarity_ptr[off] = candidate.similarity;
                out_sqdist_ptr[off] = detail::sqdist_from_similarity_(candidate.similarity);
                out_distance_ptr[off] = std::isfinite(out_sqdist_ptr[off])
                    ? std::sqrt(out_sqdist_ptr[off] + 1.0e-12f)
                    : detail::positive_infinity_();
            }
        }

        return ForwardNeighborSearchResult{
            std::move(query_ids),
            std::move(query_time),
            std::move(query_embryo),
            std::move(out_ids),
            std::move(out_time),
            std::move(out_embryo),
            std::move(out_similarity),
            std::move(out_sqdist),
            std::move(out_distance)
        };
    }

    ForwardNeighborSearchResult search_future_neighbors_by_cell_index(
        const torch::Tensor &cell_indices,
        const ForwardNeighborSearchConfig &config = ForwardNeighborSearchConfig()) const {
        return search_future_neighbors(query_batch_from_cell_indices(cell_indices), config);
    }

private:
    void build_lookup_() {
        row_by_cell_index_.clear();
        for (std::size_t shard = 0; shard < shards_.size(); ++shard) {
            const ForwardNeighborShard &current = shards_[shard];
            if (!current.cell_indices_cpu.defined()) continue;
            const std::int64_t *cell_ptr = current.cell_indices_cpu.data_ptr<std::int64_t>();
            const std::int64_t row_count = current.cell_indices_cpu.size(0);
            for (std::int64_t row = 0; row < row_count; ++row) {
                row_by_cell_index_[cell_ptr[row]] = CellLocation{ shard, row };
            }
        }
    }

    std::vector<ForwardNeighborShard> shards_;
    std::unordered_map<std::int64_t, CellLocation> row_by_cell_index_;
};

inline ForwardNeighborIndex ForwardNeighborIndexBuilder::finalize() && {
    std::vector<torch::Tensor> all_cell_indices;
    std::vector<torch::Tensor> all_time;
    std::vector<torch::Tensor> all_latent;
    std::vector<torch::Tensor> all_embryo;
    all_cell_indices.reserve(batches_.size());
    all_time.reserve(batches_.size());
    all_latent.reserve(batches_.size());
    all_embryo.reserve(batches_.size());

    for (const ForwardNeighborRecordBatch &batch : batches_) {
        all_cell_indices.push_back(batch.cell_indices);
        all_time.push_back(batch.developmental_time);
        all_latent.push_back(batch.latent_unit);
        all_embryo.push_back(batch.embryo_ids);
    }

    if (all_cell_indices.empty()) return ForwardNeighborIndex{};

    torch::Tensor cell_indices_cpu = torch::cat(all_cell_indices, 0).contiguous();
    torch::Tensor developmental_time_cpu = torch::cat(all_time, 0).contiguous();
    torch::Tensor latent_unit_cpu = torch::cat(all_latent, 0).contiguous();
    torch::Tensor embryo_ids_cpu = torch::cat(all_embryo, 0).contiguous();
    if (config_.renormalize_latent) {
        latent_unit_cpu = detail::normalize_latent_rows_(latent_unit_cpu).contiguous();
    }

    const std::vector<std::int64_t> order = detail::sorted_row_order_(cell_indices_cpu, developmental_time_cpu, embryo_ids_cpu);
    torch::Tensor order_tensor = detail::copy_i64_tensor_(order);
    cell_indices_cpu = cell_indices_cpu.index_select(0, order_tensor).contiguous();
    developmental_time_cpu = developmental_time_cpu.index_select(0, order_tensor).contiguous();
    latent_unit_cpu = latent_unit_cpu.index_select(0, order_tensor).contiguous();
    embryo_ids_cpu = embryo_ids_cpu.index_select(0, order_tensor).contiguous();

    const std::vector<torch::Device> devices = detail::resolve_forward_neighbor_devices_(config_);
    const std::int64_t shard_count = std::max<std::int64_t>(
        1,
        config_.target_shard_count > 0 ? config_.target_shard_count : static_cast<std::int64_t>(devices.size()));
    const std::int64_t max_rows_per_segment =
        detail::default_segment_rows_(config_, cell_indices_cpu.size(0), shard_count);
    const std::vector<detail::SegmentPlan> segment_plans =
        detail::build_segment_plans_(developmental_time_cpu, embryo_ids_cpu, max_rows_per_segment);
    const std::vector<std::vector<detail::SegmentPlan>> shard_plans =
        detail::assign_segments_to_shards_(segment_plans, static_cast<std::size_t>(shard_count));

    std::vector<ForwardNeighborShard> shards;
    shards.reserve(shard_plans.size());

    for (std::size_t shard_idx = 0; shard_idx < shard_plans.size(); ++shard_idx) {
        const std::vector<detail::SegmentPlan> &plan = shard_plans[shard_idx];
        if (plan.empty()) continue;

        std::vector<torch::Tensor> shard_ids_parts;
        std::vector<torch::Tensor> shard_time_parts;
        std::vector<torch::Tensor> shard_latent_parts;
        std::vector<torch::Tensor> shard_embryo_parts;
        shard_ids_parts.reserve(plan.size());
        shard_time_parts.reserve(plan.size());
        shard_latent_parts.reserve(plan.size());
        shard_embryo_parts.reserve(plan.size());

        std::vector<ForwardNeighborSegment> segments;
        std::vector<ForwardNeighborAnnList> ann_lists;
        std::vector<torch::Tensor> centroid_parts;
        std::int64_t shard_row_cursor = 0;
        const std::int64_t ann_rows_per_list = std::max<std::int64_t>(1, config_.ann_rows_per_list);

        for (const detail::SegmentPlan &segment_plan : plan) {
            const std::int64_t segment_rows = segment_plan.source_end - segment_plan.source_begin;
            shard_ids_parts.push_back(detail::slice_tensor_rows_(cell_indices_cpu, segment_plan.source_begin, segment_plan.source_end));
            shard_time_parts.push_back(detail::slice_tensor_rows_(developmental_time_cpu, segment_plan.source_begin, segment_plan.source_end));
            shard_latent_parts.push_back(detail::slice_tensor_rows_(latent_unit_cpu, segment_plan.source_begin, segment_plan.source_end));
            shard_embryo_parts.push_back(detail::slice_tensor_rows_(embryo_ids_cpu, segment_plan.source_begin, segment_plan.source_end));

            const std::int64_t ann_list_begin = static_cast<std::int64_t>(ann_lists.size());
            for (std::int64_t local_begin = 0; local_begin < segment_rows; local_begin += ann_rows_per_list) {
                const std::int64_t local_end = std::min(local_begin + ann_rows_per_list, segment_rows);
                torch::Tensor centroid = torch::mean(
                    shard_latent_parts.back().index({
                        torch::indexing::Slice(local_begin, local_end)
                    }).to(torch::kFloat32),
                    torch::IntArrayRef{ 0 }).unsqueeze(0);
                centroid = detail::normalize_latent_rows_(centroid).squeeze(0).contiguous();
                centroid_parts.push_back(centroid);
                ann_lists.push_back(ForwardNeighborAnnList{
                    segment_plan.embryo_id,
                    shard_row_cursor + local_begin,
                    shard_row_cursor + local_end,
                    shard_time_parts.back().data_ptr<float>()[local_begin],
                    shard_time_parts.back().data_ptr<float>()[local_end - 1]
                });
            }

            segments.push_back(ForwardNeighborSegment{
                segment_plan.embryo_id,
                shard_row_cursor,
                shard_row_cursor + segment_rows,
                shard_time_parts.back().data_ptr<float>()[0],
                shard_time_parts.back().data_ptr<float>()[segment_rows - 1],
                ann_list_begin,
                static_cast<std::int64_t>(ann_lists.size())
            });
            shard_row_cursor += segment_rows;
        }

        torch::Tensor shard_ids_cpu = torch::cat(shard_ids_parts, 0).contiguous();
        torch::Tensor shard_time_cpu = torch::cat(shard_time_parts, 0).contiguous();
        torch::Tensor shard_latent_cpu = torch::cat(shard_latent_parts, 0).contiguous();
        torch::Tensor shard_embryo_cpu = torch::cat(shard_embryo_parts, 0).contiguous();
        torch::Tensor shard_ann_centroids_cpu = centroid_parts.empty()
            ? torch::empty({ 0, shard_latent_cpu.size(1) }, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU))
            : torch::stack(centroid_parts, 0).contiguous();

        const torch::Device device = devices[shard_idx % devices.size()];
        shards.push_back(ForwardNeighborShard{
            device,
            shard_ids_cpu,
            shard_time_cpu,
            shard_embryo_cpu,
            shard_ids_cpu.to(device),
            shard_time_cpu.to(device),
            shard_embryo_cpu.to(device),
            shard_latent_cpu.to(device),
            shard_ann_centroids_cpu.to(device),
            std::move(segments),
            std::move(ann_lists)
        });
    }

    return ForwardNeighborIndex(std::move(shards));
}

inline ForwardNeighborIndex build_forward_neighbor_index(
    const ForwardNeighborRecordBatch &records,
    const ForwardNeighborBuildConfig &config = ForwardNeighborBuildConfig()) {
    ForwardNeighborIndexBuilder builder(config);
    builder.append(records);
    return std::move(builder).finalize();
}

} // namespace cellerator::models::forward_neighbors
