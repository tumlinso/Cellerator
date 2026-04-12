#include "forwardNeighbors.hh"

#include "../../graph/workspace.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cellerator::compute::neighbors::forward_neighbors {

namespace detail {

namespace cg = ::cellerator::compute::graph;

constexpr int kWarpThreads = 32;
constexpr int kMaxTopK = 32;
constexpr int kMaxProbe = 16;

struct Candidate {
    float similarity = -INFINITY;
    float developmental_time = INFINITY;
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

struct ProbeCandidate {
    float similarity = -INFINITY;
    std::int32_t list_offset = -1;
};

struct DeviceShardStorage {
    int device_id = 0;
    host_array<std::int64_t> cell_indices_cpu;
    host_array<float> developmental_time_cpu;
    host_array<std::int64_t> embryo_ids_cpu;
    host_array<float> latent_unit_cpu;
    host_array<float> ann_centroids_cpu;
    cg::device_buffer<std::int64_t> cell_indices;
    cg::device_buffer<float> developmental_time;
    cg::device_buffer<std::int64_t> embryo_ids;
    cg::device_buffer<__half> latent_unit;
    cg::device_buffer<float> ann_centroids;
    host_array<ForwardNeighborSegment> segments;
    host_array<ForwardNeighborAnnList> ann_lists;
};

struct ForwardNeighborIndexStorage {
    std::int64_t latent_dim = 0;
    host_array<DeviceShardStorage> shards;
    std::unordered_map<std::int64_t, ForwardNeighborIndex::CellLocation> row_by_cell_index;
};

struct ShardPlanTable {
    host_array<std::size_t> shard_offsets;
    host_array<SegmentPlan> segments;
};

__device__ inline bool candidate_valid_(const Candidate &candidate) {
    return isfinite(candidate.similarity) && candidate.cell_index >= 0;
}

__device__ inline bool better_candidate_device_(const Candidate &lhs, const Candidate &rhs) {
    const bool lhs_valid = candidate_valid_(lhs);
    const bool rhs_valid = candidate_valid_(rhs);
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

inline bool better_candidate_host_(const Candidate &lhs, const Candidate &rhs) {
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

__device__ inline void init_candidates_device_(Candidate *best, int k) {
    for (int i = 0; i < k; ++i) best[i] = Candidate{};
}

inline void init_candidates_host_(Candidate *best, int k) {
    for (int i = 0; i < k; ++i) best[i] = Candidate{};
}

__device__ inline void insert_candidate_device_(const Candidate &candidate, Candidate *best, int k) {
    if (!better_candidate_device_(candidate, best[k - 1])) return;

    int insert = k - 1;
    while (insert > 0 && better_candidate_device_(candidate, best[insert - 1])) {
        best[insert] = best[insert - 1];
        --insert;
    }
    best[insert] = candidate;
}

inline void insert_candidate_host_(const Candidate &candidate, Candidate *best, int k) {
    if (best == nullptr || k <= 0) return;
    if (!better_candidate_host_(candidate, best[static_cast<std::size_t>(k) - 1u])) return;

    std::size_t insert = static_cast<std::size_t>(k) - 1u;
    while (insert > 0u && better_candidate_host_(candidate, best[insert - 1u])) {
        best[insert] = best[insert - 1u];
        --insert;
    }
    best[insert] = candidate;
}

__device__ inline void init_probe_candidates_device_(ProbeCandidate *best, int k) {
    for (int i = 0; i < k; ++i) best[i] = ProbeCandidate{};
}

__device__ inline bool better_probe_device_(const ProbeCandidate &lhs, const ProbeCandidate &rhs) {
    const bool lhs_valid = std::isfinite(lhs.similarity) && lhs.list_offset >= 0;
    const bool rhs_valid = std::isfinite(rhs.similarity) && rhs.list_offset >= 0;
    if (!lhs_valid) return false;
    if (!rhs_valid) return true;
    if (lhs.similarity > rhs.similarity) return true;
    if (lhs.similarity < rhs.similarity) return false;
    return lhs.list_offset < rhs.list_offset;
}

__device__ inline void insert_probe_candidate_device_(const ProbeCandidate &candidate, ProbeCandidate *best, int k) {
    if (!better_probe_device_(candidate, best[k - 1])) return;

    int insert = k - 1;
    while (insert > 0 && better_probe_device_(candidate, best[insert - 1])) {
        best[insert] = best[insert - 1];
        --insert;
    }
    best[insert] = candidate;
}

__device__ inline float dot_half_rows_(const __half *lhs, const __half *rhs, int dim) {
    float sum = 0.0f;
    int i = 0;
    for (; i + 1 < dim; i += 2) {
        const __half2 a = *reinterpret_cast<const __half2 *>(lhs + i);
        const __half2 b = *reinterpret_cast<const __half2 *>(rhs + i);
        const float2 af = __half22float2(a);
        const float2 bf = __half22float2(b);
        sum += af.x * bf.x + af.y * bf.y;
    }
    if (i < dim) sum += __half2float(lhs[i]) * __half2float(rhs[i]);
    return sum;
}

__device__ inline float dot_half_float_rows_(const __half *lhs, const float *rhs, int dim) {
    float sum = 0.0f;
    for (int i = 0; i < dim; ++i) sum += __half2float(lhs[i]) * rhs[i];
    return sum;
}

__global__ void init_best_kernel_(
    std::int64_t *best_cell,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity,
    std::int64_t query_rows,
    int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = query_rows * static_cast<std::int64_t>(k);
    if (index >= total) return;
    best_cell[index] = -1;
    best_time[index] = INFINITY;
    best_embryo[index] = -1;
    best_similarity[index] = -INFINITY;
}

__global__ void exact_search_kernel_(
    const __half *query_latent,
    const float *query_lower,
    const float *query_upper,
    const std::int64_t *query_embryo,
    const __half *index_latent,
    const float *index_time,
    const std::int64_t *index_embryo,
    const std::int64_t *index_cell,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t index_begin,
    std::int64_t index_count,
    int hard_same_embryo,
    int k,
    std::int64_t *best_cell,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity) {
    const std::int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_rows || lane >= kWarpThreads) return;

    Candidate local_best[kMaxTopK];
    init_candidates_device_(local_best, k);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const float lower = query_lower[row];
    const float upper = query_upper[row];
    const std::int64_t embryo = query_embryo[row];

    for (std::int64_t off = lane; off < index_count; off += kWarpThreads) {
        const std::int64_t index_row = index_begin + off;
        const float time = index_time[index_row];
        if (!(time > lower)) continue;
        if (std::isfinite(upper) && time > upper) continue;
        if (hard_same_embryo && embryo >= 0 && index_embryo[index_row] != embryo) continue;

        const __half *target_row = index_latent + index_row * static_cast<std::int64_t>(latent_dim);
        insert_candidate_device_(Candidate{
            dot_half_rows_(query_row, target_row, latent_dim),
            time,
            index_embryo[index_row],
            index_cell[index_row]
        }, local_best, k);
    }

    __shared__ Candidate shared_candidates[kWarpThreads * kMaxTopK];
    Candidate *thread_shared = shared_candidates + lane * kMaxTopK;
    for (int i = 0; i < k; ++i) thread_shared[i] = local_best[i];
    __syncthreads();

    if (lane == 0) {
        Candidate merged[kMaxTopK];
        init_candidates_device_(merged, k);
        const std::int64_t row_base = row * static_cast<std::int64_t>(k);
        for (int i = 0; i < k; ++i) {
            merged[i] = Candidate{
                best_similarity[row_base + i],
                best_time[row_base + i],
                best_embryo[row_base + i],
                best_cell[row_base + i]
            };
        }
        for (int thread = 0; thread < kWarpThreads; ++thread) {
            const Candidate *thread_candidates = shared_candidates + thread * kMaxTopK;
            for (int i = 0; i < k; ++i) insert_candidate_device_(thread_candidates[i], merged, k);
        }
        for (int i = 0; i < k; ++i) {
            best_similarity[row_base + i] = merged[i].similarity;
            best_time[row_base + i] = merged[i].developmental_time;
            best_embryo[row_base + i] = merged[i].embryo_id;
            best_cell[row_base + i] = merged[i].cell_index;
        }
    }
}

__global__ void ann_probe_kernel_(
    const __half *query_latent,
    const std::int64_t *query_embryo,
    const float *centroids,
    const std::int64_t *list_embryo,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t list_count,
    int hard_same_embryo,
    int probe_count,
    std::int32_t *selected_list_offsets) {
    const std::int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_rows || lane >= kWarpThreads) return;

    ProbeCandidate local_best[kMaxProbe];
    init_probe_candidates_device_(local_best, probe_count);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const std::int64_t embryo = query_embryo[row];
    for (std::int64_t list_idx = lane; list_idx < list_count; list_idx += kWarpThreads) {
        if (hard_same_embryo && embryo >= 0 && list_embryo[list_idx] != embryo) continue;
        const float *centroid = centroids + list_idx * static_cast<std::int64_t>(latent_dim);
        insert_probe_candidate_device_(ProbeCandidate{
            dot_half_float_rows_(query_row, centroid, latent_dim),
            static_cast<std::int32_t>(list_idx)
        }, local_best, probe_count);
    }

    __shared__ ProbeCandidate shared_candidates[kWarpThreads * kMaxProbe];
    ProbeCandidate *thread_shared = shared_candidates + lane * kMaxProbe;
    for (int i = 0; i < probe_count; ++i) thread_shared[i] = local_best[i];
    __syncthreads();

    if (lane == 0) {
        ProbeCandidate merged[kMaxProbe];
        init_probe_candidates_device_(merged, probe_count);
        for (int thread = 0; thread < kWarpThreads; ++thread) {
            const ProbeCandidate *thread_candidates = shared_candidates + thread * kMaxProbe;
            for (int i = 0; i < probe_count; ++i) insert_probe_candidate_device_(thread_candidates[i], merged, probe_count);
        }
        for (int i = 0; i < probe_count; ++i) {
            selected_list_offsets[row * static_cast<std::int64_t>(probe_count) + i] = merged[i].list_offset;
        }
    }
}

__global__ void ann_refine_kernel_(
    const __half *query_latent,
    const float *query_lower,
    const float *query_upper,
    const std::int64_t *query_embryo,
    const __half *index_latent,
    const float *index_time,
    const std::int64_t *index_embryo,
    const std::int64_t *index_cell,
    const std::int32_t *selected_list_offsets,
    const std::int64_t *list_row_begin,
    const std::int64_t *list_row_end,
    std::int64_t query_rows,
    int latent_dim,
    int hard_same_embryo,
    int probe_count,
    int k,
    std::int64_t *best_cell,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity) {
    const std::int64_t row = blockIdx.x;
    const int lane = threadIdx.x;
    if (row >= query_rows || lane >= kWarpThreads) return;

    Candidate local_best[kMaxTopK];
    init_candidates_device_(local_best, k);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const float lower = query_lower[row];
    const float upper = query_upper[row];
    const std::int64_t embryo = query_embryo[row];

    for (int slot = 0; slot < probe_count; ++slot) {
        const std::int32_t list_offset = selected_list_offsets[row * static_cast<std::int64_t>(probe_count) + slot];
        if (list_offset < 0) continue;
        const std::int64_t begin = list_row_begin[list_offset];
        const std::int64_t end = list_row_end[list_offset];
        for (std::int64_t index_row = begin + lane; index_row < end; index_row += kWarpThreads) {
            const float time = index_time[index_row];
            if (!(time > lower)) continue;
            if (std::isfinite(upper) && time > upper) continue;
            if (hard_same_embryo && embryo >= 0 && index_embryo[index_row] != embryo) continue;

            const __half *target_row = index_latent + index_row * static_cast<std::int64_t>(latent_dim);
            insert_candidate_device_(Candidate{
                dot_half_rows_(query_row, target_row, latent_dim),
                time,
                index_embryo[index_row],
                index_cell[index_row]
            }, local_best, k);
        }
    }

    __shared__ Candidate shared_candidates[kWarpThreads * kMaxTopK];
    Candidate *thread_shared = shared_candidates + lane * kMaxTopK;
    for (int i = 0; i < k; ++i) thread_shared[i] = local_best[i];
    __syncthreads();

    if (lane == 0) {
        Candidate merged[kMaxTopK];
        init_candidates_device_(merged, k);
        const std::int64_t row_base = row * static_cast<std::int64_t>(k);
        for (int i = 0; i < k; ++i) {
            merged[i] = Candidate{
                best_similarity[row_base + i],
                best_time[row_base + i],
                best_embryo[row_base + i],
                best_cell[row_base + i]
            };
        }
        for (int thread = 0; thread < kWarpThreads; ++thread) {
            const Candidate *thread_candidates = shared_candidates + thread * kMaxTopK;
            for (int i = 0; i < k; ++i) insert_candidate_device_(thread_candidates[i], merged, k);
        }
        for (int i = 0; i < k; ++i) {
            best_similarity[row_base + i] = merged[i].similarity;
            best_time[row_base + i] = merged[i].developmental_time;
            best_embryo[row_base + i] = merged[i].embryo_id;
            best_cell[row_base + i] = merged[i].cell_index;
        }
    }
}

inline void require_gpu_() {
    int count = 0;
    cg::cuda_require(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count <= 0) throw std::runtime_error("forward_neighbors requires at least one CUDA device");
}

inline host_array<int> resolve_forward_neighbor_devices_(const ForwardNeighborBuildConfig &config) {
    require_gpu_();
    if (!config.shard_devices.empty()) return config.shard_devices;

    int count = 0;
    cg::cuda_require(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    host_array<int> devices;
    const int use_count = std::max(1, std::min(count, 4));
    devices.resize(static_cast<std::size_t>(use_count));
    for (int device = 0; device < use_count; ++device) devices[static_cast<std::size_t>(device)] = device;
    return devices;
}

inline host_array<float> normalize_latent_rows_(host_array<float> latent, std::int64_t rows, std::int64_t dim) {
    for (std::int64_t row = 0; row < rows; ++row) {
        float norm_sq = 0.0f;
        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(dim);
        for (std::int64_t d = 0; d < dim; ++d) {
            const float value = latent[base + static_cast<std::size_t>(d)];
            norm_sq += value * value;
        }
        const float inv = 1.0f / std::sqrt(std::max(norm_sq, 1.0e-12f));
        for (std::int64_t d = 0; d < dim; ++d) latent[base + static_cast<std::size_t>(d)] *= inv;
    }
    return latent;
}

inline host_array<std::int64_t> sorted_row_order_(
    const host_array<std::int64_t> &cell_indices,
    const host_array<float> &developmental_time,
    const host_array<std::int64_t> &embryo_ids) {
    const std::size_t rows = cell_indices.size();
    host_array<std::int64_t> order(rows);
    for (std::size_t row = 0; row < rows; ++row) order[row] = static_cast<std::int64_t>(row);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t lhs, std::int64_t rhs) {
        if (embryo_ids[static_cast<std::size_t>(lhs)] < embryo_ids[static_cast<std::size_t>(rhs)]) return true;
        if (embryo_ids[static_cast<std::size_t>(lhs)] > embryo_ids[static_cast<std::size_t>(rhs)]) return false;
        if (developmental_time[static_cast<std::size_t>(lhs)] < developmental_time[static_cast<std::size_t>(rhs)]) return true;
        if (developmental_time[static_cast<std::size_t>(lhs)] > developmental_time[static_cast<std::size_t>(rhs)]) return false;
        return cell_indices[static_cast<std::size_t>(lhs)] < cell_indices[static_cast<std::size_t>(rhs)];
    });
    return order;
}

template<typename T>
inline void append_copy_(host_array<T> *dst, const T *src, std::size_t count);

template<typename T>
inline void append_one_(host_array<T> *dst, T value);

inline host_array<SegmentPlan> build_segment_plans_(
    const host_array<float> &developmental_time,
    const host_array<std::int64_t> &embryo_ids,
    std::int64_t max_rows_per_segment) {
    const std::int64_t rows = static_cast<std::int64_t>(developmental_time.size());
    host_array<SegmentPlan> segments;
    std::int64_t begin = 0;
    while (begin < rows) {
        const std::int64_t embryo_id = embryo_ids[static_cast<std::size_t>(begin)];
        std::int64_t end = begin + 1;
        while (end < rows && embryo_ids[static_cast<std::size_t>(end)] == embryo_id) ++end;
        for (std::int64_t block_begin = begin; block_begin < end; block_begin += max_rows_per_segment) {
            const std::int64_t block_end = std::min(block_begin + max_rows_per_segment, end);
            append_one_(&segments, SegmentPlan{
                embryo_id,
                block_begin,
                block_end,
                developmental_time[static_cast<std::size_t>(block_begin)],
                developmental_time[static_cast<std::size_t>(block_end - 1)]
            });
        }
        begin = end;
    }
    return segments;
}

inline ShardPlanTable assign_segments_to_shards_(
    const host_array<SegmentPlan> &segments,
    std::size_t shard_count) {
    const std::size_t used_shards = std::max<std::size_t>(1u, shard_count);
    ShardPlanTable table;
    table.shard_offsets.assign_fill(used_shards + 1u, 0u);
    if (segments.empty()) return table;

    host_array<std::int64_t> shard_rows(used_shards);
    shard_rows.assign_fill(used_shards, 0);
    host_array<std::size_t> shard_counts(used_shards);
    shard_counts.assign_fill(used_shards, 0u);
    host_array<std::size_t> segment_shards(segments.size());
    host_array<SegmentPlan> sorted = segments;
    std::sort(sorted.begin(), sorted.end(), [](const SegmentPlan &lhs, const SegmentPlan &rhs) {
        const std::int64_t lhs_rows = lhs.source_end - lhs.source_begin;
        const std::int64_t rhs_rows = rhs.source_end - rhs.source_begin;
        if (lhs_rows > rhs_rows) return true;
        if (lhs_rows < rhs_rows) return false;
        if (lhs.embryo_id < rhs.embryo_id) return true;
        if (lhs.embryo_id > rhs.embryo_id) return false;
        return lhs.source_begin < rhs.source_begin;
    });

    for (std::size_t idx = 0; idx < sorted.size(); ++idx) {
        const SegmentPlan &segment = sorted[idx];
        std::size_t best_shard = 0;
        for (std::size_t shard = 1; shard < used_shards; ++shard) {
            if (shard_rows[shard] < shard_rows[best_shard]) best_shard = shard;
        }
        segment_shards[idx] = best_shard;
        ++shard_counts[best_shard];
        shard_rows[best_shard] += segment.source_end - segment.source_begin;
    }

    for (std::size_t shard = 0; shard < used_shards; ++shard) {
        table.shard_offsets[shard + 1u] = table.shard_offsets[shard] + shard_counts[shard];
    }
    table.segments.resize(sorted.size());
    host_array<std::size_t> cursor = table.shard_offsets;
    for (std::size_t idx = 0; idx < sorted.size(); ++idx) {
        const std::size_t shard = segment_shards[idx];
        table.segments[cursor[shard]++] = sorted[idx];
    }
    return table;
}

inline host_array<std::int64_t> eligible_segment_order_(
    const host_array<ForwardNeighborSegment> &segments,
    const std::unordered_set<std::int64_t> &query_embryos,
    ForwardNeighborEmbryoPolicy policy) {
    host_array<std::int64_t> ordered;
    host_array<std::int64_t> deferred;
    for (std::int64_t segment_idx = 0; segment_idx < static_cast<std::int64_t>(segments.size()); ++segment_idx) {
        const bool embryo_match = !query_embryos.empty() && query_embryos.find(segments[static_cast<std::size_t>(segment_idx)].embryo_id) != query_embryos.end();
        if (policy == ForwardNeighborEmbryoPolicy::same_embryo_only) {
            if (query_embryos.empty() || embryo_match) append_one_(&ordered, segment_idx);
            continue;
        }
        if (policy == ForwardNeighborEmbryoPolicy::same_embryo_first && embryo_match) {
            append_one_(&ordered, segment_idx);
        } else if (policy == ForwardNeighborEmbryoPolicy::same_embryo_first) {
            append_one_(&deferred, segment_idx);
        } else {
            append_one_(&ordered, segment_idx);
        }
    }
    append_copy_(&ordered, deferred.data(), deferred.size());
    return ordered;
}

inline std::unordered_set<std::int64_t> block_query_embryos_(
    const host_array<std::int64_t> &query_embryo,
    std::int64_t query_begin,
    std::int64_t query_end) {
    std::unordered_set<std::int64_t> embryo_ids;
    for (std::int64_t row = query_begin; row < query_end; ++row) {
        const std::int64_t embryo = query_embryo[static_cast<std::size_t>(row)];
        if (embryo >= 0) embryo_ids.insert(embryo);
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
    const DeviceShardStorage &shard,
    const ForwardNeighborSegment &segment,
    float block_lower,
    float block_upper) {
    const float *begin_ptr = shard.developmental_time_cpu.data() + segment.row_begin;
    const float *end_ptr = shard.developmental_time_cpu.data() + segment.row_end;
    const float *lower_it = std::upper_bound(begin_ptr, end_ptr, block_lower);
    const float *upper_it = std::isfinite(block_upper)
        ? std::upper_bound(lower_it, end_ptr, block_upper)
        : end_ptr;
    return {
        static_cast<std::int64_t>(lower_it - shard.developmental_time_cpu.data()),
        static_cast<std::int64_t>(upper_it - shard.developmental_time_cpu.data())
    };
}

inline std::pair<float, float> block_time_limits_(
    const host_array<float> &query_time,
    std::int64_t query_begin,
    std::int64_t query_end,
    const ForwardNeighborSearchConfig &config) {
    float block_lower = detail::positive_infinity_();
    float block_upper = -detail::positive_infinity_();
    for (std::int64_t row = query_begin; row < query_end; ++row) {
        const float lower = query_time[static_cast<std::size_t>(row)] + config.strict_future_epsilon + config.time_window.min_delta;
        const float upper = std::isfinite(config.time_window.max_delta)
            ? query_time[static_cast<std::size_t>(row)] + config.time_window.max_delta
            : detail::positive_infinity_();
        if (lower < block_lower) block_lower = lower;
        if (upper > block_upper) block_upper = upper;
    }
    return { block_lower, block_upper };
}

inline host_array<std::pair<std::int64_t, std::int64_t>> merge_row_intervals_(
    host_array<std::pair<std::int64_t, std::int64_t>> intervals) {
    if (intervals.empty()) return intervals;
    std::sort(intervals.begin(), intervals.end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.first < rhs.first) return true;
        if (lhs.first > rhs.first) return false;
        return lhs.second < rhs.second;
    });
    host_array<std::pair<std::int64_t, std::int64_t>> merged;
    append_one_(&merged, intervals[0]);
    for (std::size_t i = 1; i < intervals.size(); ++i) {
        if (intervals[i].first <= merged.back().second) {
            merged.back().second = std::max(merged.back().second, intervals[i].second);
        } else {
            append_one_(&merged, intervals[i]);
        }
    }
    return merged;
}

inline host_array<__half> convert_f32_to_half_(const host_array<float> &input) {
    host_array<__half> output(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) output[i] = __float2half_rn(input[i]);
    return output;
}

inline host_array<float> build_list_centroid_(
    const host_array<float> &latent,
    std::int64_t latent_dim,
    std::int64_t row_begin,
    std::int64_t row_end) {
    host_array<float> centroid(static_cast<std::size_t>(latent_dim));
    for (std::size_t i = 0; i < centroid.size(); ++i) centroid[i] = 0.0f;
    if (row_begin >= row_end) return centroid;
    for (std::int64_t row = row_begin; row < row_end; ++row) {
        const std::size_t base = static_cast<std::size_t>(row) * static_cast<std::size_t>(latent_dim);
        for (std::int64_t d = 0; d < latent_dim; ++d) centroid[static_cast<std::size_t>(d)] += latent[base + static_cast<std::size_t>(d)];
    }
    const float inv = 1.0f / static_cast<float>(row_end - row_begin);
    float norm_sq = 0.0f;
    for (float &value : centroid) {
        value *= inv;
        norm_sq += value * value;
    }
    const float inv_norm = 1.0f / std::sqrt(std::max(norm_sq, 1.0e-12f));
    for (float &value : centroid) value *= inv_norm;
    return centroid;
}

inline void set_device_(int device_id) {
    cg::cuda_require(cudaSetDevice(device_id), "cudaSetDevice(forward_neighbors)");
}

inline void init_best_arrays_(
    std::int64_t query_rows,
    int k,
    cg::device_buffer<std::int64_t> *best_cell,
    cg::device_buffer<float> *best_time,
    cg::device_buffer<std::int64_t> *best_embryo,
    cg::device_buffer<float> *best_similarity) {
    const std::size_t total = static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k);
    best_cell->resize(total);
    best_time->resize(total);
    best_embryo->resize(total);
    best_similarity->resize(total);
    const int threads = 128;
    const int blocks = static_cast<int>((total + static_cast<std::size_t>(threads) - 1u) / static_cast<std::size_t>(threads));
    // Initialization launch is pure bookkeeping and followed by a sync, so it is noticeable only for small query batches.
    init_best_kernel_<<<blocks, threads>>>(
        best_cell->data(),
        best_time->data(),
        best_embryo->data(),
        best_similarity->data(),
        query_rows,
        k);
    cg::cuda_require(cudaGetLastError(), "init_best_kernel launch");
    // This barrier keeps later host merges simple but removes any chance to overlap init with query uploads.
    cg::cuda_require(cudaDeviceSynchronize(), "init_best_kernel sync");
}

inline host_array<Candidate> download_best_candidates_(
    const cg::device_buffer<std::int64_t> &best_cell,
    const cg::device_buffer<float> &best_time,
    const cg::device_buffer<std::int64_t> &best_embryo,
    const cg::device_buffer<float> &best_similarity,
    std::int64_t query_rows,
    int k) {
    const std::size_t total = static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k);
    host_array<std::int64_t> cell;
    host_array<float> time;
    host_array<std::int64_t> embryo;
    host_array<float> similarity;
    cell.assign_fill(total, static_cast<std::int64_t>(-1));
    time.assign_fill(total, detail::quiet_nan_());
    embryo.assign_fill(total, static_cast<std::int64_t>(-1));
    similarity.assign_fill(total, detail::negative_infinity_());
    // Full-table downloads are intentional here: merge logic is host-side today, so query blocks should be large enough to amortize D2H latency.
    best_cell.download(cell.data(), total);
    best_time.download(time.data(), total);
    best_embryo.download(embryo.data(), total);
    best_similarity.download(similarity.data(), total);

    host_array<Candidate> out(total);
    for (std::size_t i = 0; i < total; ++i) {
        out[i] = Candidate{ similarity[i], time[i], embryo[i], cell[i] };
    }
    return out;
}

template<typename T>
inline void append_copy_(host_array<T> *dst, const T *src, std::size_t count) {
    const std::size_t old_size = dst->size();
    dst->resize(old_size + count);
    if (count == 0u || src == nullptr) return;
    if constexpr (std::is_trivially_copyable_v<T>) {
        std::memcpy(dst->data() + old_size, src, count * sizeof(T));
    } else {
        for (std::size_t i = 0; i < count; ++i) (*dst)[old_size + i] = src[i];
    }
}

template<typename T>
inline void append_one_(host_array<T> *dst, T value) {
    const std::size_t old_size = dst->size();
    dst->resize(old_size + 1u);
    (*dst)[old_size] = std::move(value);
}

inline void append_batch_(
    const ForwardNeighborRecordBatch &batch,
    host_array<std::int64_t> *cell_indices,
    host_array<float> *developmental_time,
    host_array<float> *latent_unit,
    host_array<std::int64_t> *embryo_ids,
    std::int64_t *latent_dim) {
    detail::validate_forward_neighbor_record_batch_(batch);
    if (*latent_dim == 0) *latent_dim = batch.latent_dim;
    if (*latent_dim != batch.latent_dim) throw std::invalid_argument("all forward-neighbor batches must share latent_dim");

    detail::append_i64_(cell_indices, batch.cell_indices);
    detail::append_f32_(developmental_time, batch.developmental_time);
    detail::append_f32_(latent_unit, batch.latent_unit);
    if (batch.embryo_ids.empty()) {
        const std::size_t old_size = embryo_ids->size();
        embryo_ids->resize(old_size + batch.cell_indices.size());
        for (std::size_t i = 0; i < batch.cell_indices.size(); ++i) {
            (*embryo_ids)[old_size + i] = static_cast<std::int64_t>(-1);
        }
    } else {
        detail::append_i64_(embryo_ids, batch.embryo_ids);
    }
}

inline std::shared_ptr<ForwardNeighborIndexStorage> build_storage_(
    const ForwardNeighborRecordBatch &records,
    const ForwardNeighborBuildConfig &config) {
    auto storage = std::make_shared<ForwardNeighborIndexStorage>();
    host_array<std::int64_t> cell_indices;
    host_array<float> developmental_time;
    host_array<float> latent_unit;
    host_array<std::int64_t> embryo_ids;

    append_batch_(records, &cell_indices, &developmental_time, &latent_unit, &embryo_ids, &storage->latent_dim);

    if (cell_indices.empty()) return storage;
    if (config.renormalize_latent) {
        latent_unit = normalize_latent_rows_(std::move(latent_unit), static_cast<std::int64_t>(cell_indices.size()), storage->latent_dim);
    }

    const host_array<std::int64_t> order = sorted_row_order_(cell_indices, developmental_time, embryo_ids);
    host_array<std::int64_t> sorted_ids(cell_indices.size());
    host_array<float> sorted_time(developmental_time.size());
    host_array<std::int64_t> sorted_embryo(embryo_ids.size());
    host_array<float> sorted_latent(latent_unit.size());
    for (std::size_t row = 0; row < order.size(); ++row) {
        const std::size_t src = static_cast<std::size_t>(order[row]);
        sorted_ids[row] = cell_indices[src];
        sorted_time[row] = developmental_time[src];
        sorted_embryo[row] = embryo_ids[src];
        std::memcpy(
            sorted_latent.data() + row * static_cast<std::size_t>(storage->latent_dim),
            latent_unit.data() + src * static_cast<std::size_t>(storage->latent_dim),
            static_cast<std::size_t>(storage->latent_dim) * sizeof(float));
    }

    const host_array<int> devices = resolve_forward_neighbor_devices_(config);
    const std::int64_t shard_count = std::max<std::int64_t>(
        1,
        config.target_shard_count > 0 ? config.target_shard_count : static_cast<std::int64_t>(devices.size()));
    const std::int64_t max_rows_per_segment =
        detail::default_segment_rows_(config, static_cast<std::int64_t>(sorted_ids.size()), shard_count);
    const host_array<SegmentPlan> segment_plans =
        build_segment_plans_(sorted_time, sorted_embryo, max_rows_per_segment);
    const ShardPlanTable shard_plans =
        assign_segments_to_shards_(segment_plans, static_cast<std::size_t>(shard_count));

    storage->shards.reserve(shard_plans.shard_offsets.size() > 0u ? shard_plans.shard_offsets.size() - 1u : 0u);
    for (std::size_t shard_idx = 0; shard_idx + 1u < shard_plans.shard_offsets.size(); ++shard_idx) {
        const std::size_t plan_begin = shard_plans.shard_offsets[shard_idx];
        const std::size_t plan_end = shard_plans.shard_offsets[shard_idx + 1u];
        if (plan_begin == plan_end) continue;

        DeviceShardStorage shard;
        shard.device_id = devices[shard_idx % devices.size()];
        std::int64_t shard_row_cursor = 0;
        const std::int64_t ann_rows_per_list = std::max<std::int64_t>(1, config.ann_rows_per_list);

        for (std::size_t plan_idx = plan_begin; plan_idx < plan_end; ++plan_idx) {
            const SegmentPlan &segment_plan = shard_plans.segments[plan_idx];
            const std::int64_t segment_rows = segment_plan.source_end - segment_plan.source_begin;
            const std::size_t begin = static_cast<std::size_t>(segment_plan.source_begin);
            const std::size_t end = static_cast<std::size_t>(segment_plan.source_end);
            append_copy_(&shard.cell_indices_cpu, sorted_ids.data() + begin, end - begin);
            append_copy_(&shard.developmental_time_cpu, sorted_time.data() + begin, end - begin);
            append_copy_(&shard.embryo_ids_cpu, sorted_embryo.data() + begin, end - begin);
            append_copy_(
                &shard.latent_unit_cpu,
                sorted_latent.data() + begin * static_cast<std::size_t>(storage->latent_dim),
                (end - begin) * static_cast<std::size_t>(storage->latent_dim));

            const std::int64_t ann_list_begin = static_cast<std::int64_t>(shard.ann_lists.size());
            for (std::int64_t local_begin = 0; local_begin < segment_rows; local_begin += ann_rows_per_list) {
                const std::int64_t local_end = std::min(local_begin + ann_rows_per_list, segment_rows);
                const host_array<float> centroid = build_list_centroid_(
                    shard.latent_unit_cpu,
                    storage->latent_dim,
                    shard_row_cursor + local_begin,
                    shard_row_cursor + local_end);
                append_copy_(&shard.ann_centroids_cpu, centroid.data(), centroid.size());
                append_one_(&shard.ann_lists, ForwardNeighborAnnList{
                    segment_plan.embryo_id,
                    shard_row_cursor + local_begin,
                    shard_row_cursor + local_end,
                    shard.developmental_time_cpu[static_cast<std::size_t>(shard_row_cursor + local_begin)],
                    shard.developmental_time_cpu[static_cast<std::size_t>(shard_row_cursor + local_end - 1)]
                });
            }

            append_one_(&shard.segments, ForwardNeighborSegment{
                segment_plan.embryo_id,
                shard_row_cursor,
                shard_row_cursor + segment_rows,
                shard.developmental_time_cpu[static_cast<std::size_t>(shard_row_cursor)],
                shard.developmental_time_cpu[static_cast<std::size_t>(shard_row_cursor + segment_rows - 1)],
                ann_list_begin,
                static_cast<std::int64_t>(shard.ann_lists.size())
            });
            shard_row_cursor += segment_rows;
        }

        set_device_(shard.device_id);
        shard.cell_indices.resize(shard.cell_indices_cpu.size());
        shard.developmental_time.resize(shard.developmental_time_cpu.size());
        shard.embryo_ids.resize(shard.embryo_ids_cpu.size());
        const host_array<__half> latent_half = convert_f32_to_half_(shard.latent_unit_cpu);
        shard.latent_unit.resize(latent_half.size());
        shard.ann_centroids.resize(shard.ann_centroids_cpu.size());

        shard.cell_indices.upload(shard.cell_indices_cpu.data(), shard.cell_indices_cpu.size());
        shard.developmental_time.upload(shard.developmental_time_cpu.data(), shard.developmental_time_cpu.size());
        shard.embryo_ids.upload(shard.embryo_ids_cpu.data(), shard.embryo_ids_cpu.size());
        shard.latent_unit.upload(latent_half.data(), latent_half.size());
        if (!shard.ann_centroids_cpu.empty()) {
            shard.ann_centroids.upload(shard.ann_centroids_cpu.data(), shard.ann_centroids_cpu.size());
        }

        append_one_(&storage->shards, std::move(shard));
    }

    for (std::size_t shard_idx = 0; shard_idx < storage->shards.size(); ++shard_idx) {
        const DeviceShardStorage &shard = storage->shards[shard_idx];
        for (std::size_t row = 0; row < shard.cell_indices_cpu.size(); ++row) {
            storage->row_by_cell_index[shard.cell_indices_cpu[row]] = ForwardNeighborIndex::CellLocation{
                shard_idx,
                static_cast<std::int64_t>(row)
            };
        }
    }
    return storage;
}

inline host_array<float> normalize_query_latent_(const host_array<float> &latent, std::int64_t rows, std::int64_t dim) {
    host_array<float> copy;
    copy.assign_copy(latent.data(), latent.size());
    return normalize_latent_rows_(std::move(copy), rows, dim);
}

inline ForwardNeighborSearchResult search_core_(
    const ForwardNeighborIndexStorage &storage,
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config,
    bool hard_same_embryo) {
    detail::validate_forward_neighbor_search_config_(config);
    detail::validate_forward_neighbor_query_batch_(query);
    if (query.cell_indices.empty()) return detail::empty_forward_neighbor_result_(config.top_k);
    if (storage.latent_dim != 0 && storage.latent_dim != query.latent_dim) {
        throw std::invalid_argument("query latent dimension does not match the forward-neighbor index latent dimension");
    }
    if (config.candidate_k > kMaxTopK || config.ann_probe_list_count > kMaxProbe) {
        throw std::invalid_argument("forward-neighbor config exceeds native kernel limits");
    }

    ForwardNeighborSearchResult result;
    result.query_count = static_cast<std::int64_t>(query.cell_indices.size());
    result.top_k = config.top_k;
    result.query_cell_indices = query.cell_indices;
    result.query_time = query.developmental_time;
    result.query_embryo_ids = query.embryo_ids.empty()
        ? detail::make_missing_i64_array_(query.cell_indices.size())
        : query.embryo_ids;

    const host_array<float> normalized_query_latent =
        normalize_query_latent_(query.latent_unit, result.query_count, query.latent_dim);
    host_array<Candidate> best(static_cast<std::size_t>(result.query_count * result.top_k));
    for (std::size_t i = 0; i < best.size(); ++i) best[i] = Candidate{};

    for (const DeviceShardStorage &shard : storage.shards) {
        set_device_(shard.device_id);

        for (std::int64_t query_begin = 0; query_begin < result.query_count; query_begin += config.query_block_rows) {
            const std::int64_t query_end = std::min(query_begin + config.query_block_rows, result.query_count);
            const std::int64_t block_queries = query_end - query_begin;
            const auto block_limits = block_time_limits_(result.query_time, query_begin, query_end, config);
            const std::unordered_set<std::int64_t> block_embryos =
                block_query_embryos_(result.query_embryo_ids, query_begin, query_end);
            const host_array<std::int64_t> segment_order =
                eligible_segment_order_(shard.segments, block_embryos, hard_same_embryo ? ForwardNeighborEmbryoPolicy::same_embryo_only : config.embryo_policy);

            host_array<float> block_latent_f32(static_cast<std::size_t>(block_queries) * static_cast<std::size_t>(query.latent_dim));
            std::memcpy(
                block_latent_f32.data(),
                normalized_query_latent.data() + static_cast<std::size_t>(query_begin) * static_cast<std::size_t>(query.latent_dim),
                block_latent_f32.size() * sizeof(float));
            const host_array<__half> block_latent = convert_f32_to_half_(block_latent_f32);
            host_array<float> block_lower;
            host_array<float> block_upper;
            host_array<std::int64_t> block_embryo;
            block_lower.assign_fill(static_cast<std::size_t>(block_queries), detail::positive_infinity_());
            block_upper.assign_fill(static_cast<std::size_t>(block_queries), detail::positive_infinity_());
            block_embryo.assign_fill(static_cast<std::size_t>(block_queries), static_cast<std::int64_t>(-1));

            for (std::int64_t row = 0; row < block_queries; ++row) {
                const std::int64_t global = query_begin + row;
                block_lower[static_cast<std::size_t>(row)] =
                    result.query_time[static_cast<std::size_t>(global)] + config.strict_future_epsilon + config.time_window.min_delta;
                block_upper[static_cast<std::size_t>(row)] = std::isfinite(config.time_window.max_delta)
                    ? result.query_time[static_cast<std::size_t>(global)] + config.time_window.max_delta
                    : detail::positive_infinity_();
                block_embryo[static_cast<std::size_t>(row)] = result.query_embryo_ids[static_cast<std::size_t>(global)];
            }

            cg::device_buffer<__half> d_query_latent(block_latent.size());
            cg::device_buffer<float> d_query_lower(block_lower.size());
            cg::device_buffer<float> d_query_upper(block_upper.size());
            cg::device_buffer<std::int64_t> d_query_embryo(block_embryo.size());
            // Per-block query uploads are fixed overhead every shard visit, so larger query blocks reduce PCIe and launch tax.
            d_query_latent.upload(block_latent.data(), block_latent.size());
            d_query_lower.upload(block_lower.data(), block_lower.size());
            d_query_upper.upload(block_upper.data(), block_upper.size());
            d_query_embryo.upload(block_embryo.data(), block_embryo.size());

            cg::device_buffer<std::int64_t> d_best_cell;
            cg::device_buffer<float> d_best_time;
            cg::device_buffer<std::int64_t> d_best_embryo;
            cg::device_buffer<float> d_best_similarity;
            init_best_arrays_(block_queries, static_cast<int>(config.candidate_k), &d_best_cell, &d_best_time, &d_best_embryo, &d_best_similarity);

            if (config.backend == ForwardNeighborBackend::exact_windowed) {
                host_array<std::pair<std::int64_t, std::int64_t>> intervals;
                for (const std::int64_t segment_idx : segment_order) {
                    const ForwardNeighborSegment &segment = shard.segments[static_cast<std::size_t>(segment_idx)];
                    if (!segment_time_overlaps_block_(segment, block_limits.first, block_limits.second)) continue;
                    const auto bounds = segment_candidate_bounds_(shard, segment, block_limits.first, block_limits.second);
                    if (bounds.first < bounds.second) append_one_(&intervals, bounds);
                }

                const host_array<std::pair<std::int64_t, std::int64_t>> merged_intervals = merge_row_intervals_(std::move(intervals));
                for (const auto &interval : merged_intervals) {
                    for (std::int64_t index_begin = interval.first; index_begin < interval.second; index_begin += config.index_block_rows) {
                        const std::int64_t index_end = std::min(index_begin + config.index_block_rows, interval.second);
                        // Exact search pays one launch per query block x index block tile; too-small index tiles become launch-bound very quickly.
                        exact_search_kernel_<<<block_queries, kWarpThreads>>>(
                            d_query_latent.data(),
                            d_query_lower.data(),
                            d_query_upper.data(),
                            d_query_embryo.data(),
                            shard.latent_unit.data(),
                            shard.developmental_time.data(),
                            shard.embryo_ids.data(),
                            shard.cell_indices.data(),
                            block_queries,
                            static_cast<int>(storage.latent_dim),
                            index_begin,
                            index_end - index_begin,
                            hard_same_embryo ? 1 : 0,
                            static_cast<int>(config.candidate_k),
                            d_best_cell.data(),
                            d_best_time.data(),
                            d_best_embryo.data(),
                            d_best_similarity.data());
                        cg::cuda_require(cudaGetLastError(), "exact_search_kernel launch");
                    }
                }
                // Required because the next step reads the candidate buffers on host; this is the main overlap barrier in the exact path.
                cg::cuda_require(cudaDeviceSynchronize(), "exact_search_kernel sync");
            } else {
                host_array<std::int64_t> eligible_lists;
                for (const std::int64_t segment_idx : segment_order) {
                    const ForwardNeighborSegment &segment = shard.segments[static_cast<std::size_t>(segment_idx)];
                    if (!segment_time_overlaps_block_(segment, block_limits.first, block_limits.second)) continue;
                    for (std::int64_t list_idx = segment.ann_list_begin; list_idx < segment.ann_list_end; ++list_idx) {
                        const ForwardNeighborAnnList &list = shard.ann_lists[static_cast<std::size_t>(list_idx)];
                        if (ann_list_overlaps_block_(list, block_limits.first, block_limits.second)) append_one_(&eligible_lists, list_idx);
                    }
                }

                if (!eligible_lists.empty()) {
                    host_array<float> eligible_centroids;
                    host_array<std::int64_t> eligible_embryo;
                    host_array<std::int64_t> eligible_row_begin;
                    host_array<std::int64_t> eligible_row_end;
                    for (const std::int64_t list_idx : eligible_lists) {
                        const ForwardNeighborAnnList &list = shard.ann_lists[static_cast<std::size_t>(list_idx)];
                        append_copy_(
                            &eligible_centroids,
                            shard.ann_centroids_cpu.data() + static_cast<std::size_t>(list_idx) * static_cast<std::size_t>(storage.latent_dim),
                            static_cast<std::size_t>(storage.latent_dim));
                        append_one_(&eligible_embryo, list.embryo_id);
                        append_one_(&eligible_row_begin, list.row_begin);
                        append_one_(&eligible_row_end, list.row_end);
                    }

                    cg::device_buffer<float> d_centroids(eligible_centroids.size());
                    cg::device_buffer<std::int64_t> d_list_embryo(eligible_embryo.size());
                    cg::device_buffer<std::int64_t> d_list_row_begin(eligible_row_begin.size());
                    cg::device_buffer<std::int64_t> d_list_row_end(eligible_row_end.size());
                    cg::device_buffer<std::int32_t> d_selected_lists(static_cast<std::size_t>(block_queries) * static_cast<std::size_t>(config.ann_probe_list_count));
                    // ANN metadata upload is paid once per shard/query block and can dominate when the eligible list set is tiny.
                    d_centroids.upload(eligible_centroids.data(), eligible_centroids.size());
                    d_list_embryo.upload(eligible_embryo.data(), eligible_embryo.size());
                    d_list_row_begin.upload(eligible_row_begin.data(), eligible_row_begin.size());
                    d_list_row_end.upload(eligible_row_end.data(), eligible_row_end.size());

                    // Probe is intentionally cheap: it scores list centroids only, so it should stay much smaller than refine.
                    ann_probe_kernel_<<<block_queries, kWarpThreads>>>(
                        d_query_latent.data(),
                        d_query_embryo.data(),
                        d_centroids.data(),
                        d_list_embryo.data(),
                        block_queries,
                        static_cast<int>(storage.latent_dim),
                        static_cast<std::int64_t>(eligible_lists.size()),
                        hard_same_embryo ? 1 : 0,
                        static_cast<int>(config.ann_probe_list_count),
                        d_selected_lists.data());
                    cg::cuda_require(cudaGetLastError(), "ann_probe_kernel launch");

                    // Refine does the expensive candidate scan inside the selected lists; this is the hot kernel in ANN mode.
                    ann_refine_kernel_<<<block_queries, kWarpThreads>>>(
                        d_query_latent.data(),
                        d_query_lower.data(),
                        d_query_upper.data(),
                        d_query_embryo.data(),
                        shard.latent_unit.data(),
                        shard.developmental_time.data(),
                        shard.embryo_ids.data(),
                        shard.cell_indices.data(),
                        d_selected_lists.data(),
                        d_list_row_begin.data(),
                        d_list_row_end.data(),
                        block_queries,
                        static_cast<int>(storage.latent_dim),
                        hard_same_embryo ? 1 : 0,
                        static_cast<int>(config.ann_probe_list_count),
                        static_cast<int>(config.candidate_k),
                        d_best_cell.data(),
                        d_best_time.data(),
                        d_best_embryo.data(),
                        d_best_similarity.data());
                    cg::cuda_require(cudaGetLastError(), "ann_refine_kernel launch");
                    // Like the exact path, this sync exists only because results are merged on host immediately after.
                    cg::cuda_require(cudaDeviceSynchronize(), "ann_refine_kernel sync");
                }
            }

            const host_array<Candidate> shard_best = download_best_candidates_(
                d_best_cell,
                d_best_time,
                d_best_embryo,
                d_best_similarity,
                block_queries,
                static_cast<int>(config.candidate_k));

            for (std::int64_t row = 0; row < block_queries; ++row) {
                Candidate row_best[kMaxTopK];
                init_candidates_host_(row_best, static_cast<int>(result.top_k));
                std::memcpy(
                    row_best,
                    best.data() + static_cast<std::size_t>(query_begin + row) * static_cast<std::size_t>(result.top_k),
                    static_cast<std::size_t>(result.top_k) * sizeof(Candidate));
                for (std::int64_t slot = 0; slot < config.candidate_k; ++slot) {
                    const std::size_t off = static_cast<std::size_t>(row) * static_cast<std::size_t>(config.candidate_k)
                        + static_cast<std::size_t>(slot);
                    insert_candidate_host_(shard_best[off], row_best, static_cast<int>(result.top_k));
                }
                std::memcpy(
                    best.data() + static_cast<std::size_t>(query_begin + row) * static_cast<std::size_t>(result.top_k),
                    row_best,
                    static_cast<std::size_t>(result.top_k) * sizeof(Candidate));
            }
        }
    }

    result.neighbor_cell_indices.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), static_cast<std::int64_t>(-1));
    result.neighbor_time.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::quiet_nan_());
    result.neighbor_embryo_ids.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), static_cast<std::int64_t>(-1));
    result.neighbor_similarity.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::negative_infinity_());
    result.neighbor_sqdist.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::positive_infinity_());
    result.neighbor_distance.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::positive_infinity_());

    for (std::int64_t row = 0; row < result.query_count; ++row) {
        for (std::int64_t slot = 0; slot < result.top_k; ++slot) {
            const Candidate &candidate = best[static_cast<std::size_t>(row) * static_cast<std::size_t>(result.top_k) + static_cast<std::size_t>(slot)];
            const std::size_t off = detail::result_offset_(row, slot, result.top_k);
            result.neighbor_cell_indices[off] = candidate.cell_index;
            result.neighbor_time[off] = candidate.developmental_time;
            result.neighbor_embryo_ids[off] = candidate.embryo_id;
            result.neighbor_similarity[off] = candidate.similarity;
            result.neighbor_sqdist[off] = std::isfinite(candidate.similarity)
                ? std::max(0.0f, 2.0f - 2.0f * candidate.similarity)
                : detail::positive_infinity_();
            result.neighbor_distance[off] = std::isfinite(result.neighbor_sqdist[off])
                ? std::sqrt(result.neighbor_sqdist[off] + 1.0e-12f)
                : detail::positive_infinity_();
        }
    }
    return result;
}

} // namespace detail

ForwardNeighborIndexBuilder::ForwardNeighborIndexBuilder(ForwardNeighborBuildConfig config)
    : config_(std::move(config)) {}

void ForwardNeighborIndexBuilder::append(const ForwardNeighborRecordBatch &batch) {
    detail::validate_forward_neighbor_record_batch_(batch);
    detail::append_batch_(batch, &records_.cell_indices, &records_.developmental_time, &records_.latent_unit, &records_.embryo_ids, &records_.latent_dim);
}

ForwardNeighborIndex ForwardNeighborIndexBuilder::finalize() && {
    return ForwardNeighborIndex(detail::build_storage_(records_, config_));
}

ForwardNeighborIndex::ForwardNeighborIndex() = default;

ForwardNeighborIndex::ForwardNeighborIndex(std::shared_ptr<detail::ForwardNeighborIndexStorage> storage)
    : storage_(std::move(storage)) {}

std::size_t ForwardNeighborIndex::shard_count() const {
    return storage_ ? storage_->shards.size() : 0u;
}

std::int64_t ForwardNeighborIndex::latent_dim() const {
    return storage_ ? storage_->latent_dim : 0;
}

ForwardNeighborQueryBatch ForwardNeighborIndex::query_batch_from_cell_indices(const std::int64_t *cell_indices, std::size_t cell_count) const {
    if (!storage_) return ForwardNeighborQueryBatch{};
    ForwardNeighborQueryBatch query;
    query.cell_indices.assign_copy(cell_indices, cell_count);
    query.developmental_time.assign_fill(cell_count, detail::quiet_nan_());
    query.embryo_ids.assign_fill(cell_count, static_cast<std::int64_t>(-1));
    query.latent_dim = storage_->latent_dim;
    query.latent_unit.assign_fill(cell_count * static_cast<std::size_t>(storage_->latent_dim), 0.0f);

    for (std::size_t row = 0; row < cell_count; ++row) {
        const auto it = storage_->row_by_cell_index.find(cell_indices[row]);
        if (it == storage_->row_by_cell_index.end()) {
            throw std::invalid_argument("query cell index is not present in the forward-neighbor index");
        }
        const detail::DeviceShardStorage &shard = storage_->shards[it->second.shard];
        const std::size_t local_row = static_cast<std::size_t>(it->second.local_row);
        query.developmental_time[row] = shard.developmental_time_cpu[local_row];
        query.embryo_ids[row] = shard.embryo_ids_cpu[local_row];
        std::memcpy(
            query.latent_unit.data() + row * static_cast<std::size_t>(storage_->latent_dim),
            shard.latent_unit_cpu.data() + local_row * static_cast<std::size_t>(storage_->latent_dim),
            static_cast<std::size_t>(storage_->latent_dim) * sizeof(float));
    }

    return query;
}

ForwardNeighborSearchResult ForwardNeighborIndex::search_future_neighbors(
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config) const {
    if (!storage_) return detail::empty_forward_neighbor_result_(config.top_k);
    if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_first) {
        ForwardNeighborSearchConfig same_config = config;
        same_config.embryo_policy = ForwardNeighborEmbryoPolicy::same_embryo_only;
        ForwardNeighborSearchResult same_result = detail::search_core_(*storage_, query, same_config, true);
        ForwardNeighborSearchConfig any_config = config;
        any_config.embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
        ForwardNeighborSearchResult any_result = detail::search_core_(*storage_, query, any_config, false);

        for (std::int64_t row = 0; row < same_result.query_count; ++row) {
            std::size_t fill = 0u;
            std::unordered_set<std::int64_t> seen;
            for (; fill < static_cast<std::size_t>(same_result.top_k); ++fill) {
                const std::size_t off = detail::result_offset_(row, static_cast<std::int64_t>(fill), same_result.top_k);
                if (same_result.neighbor_cell_indices[off] < 0) break;
                seen.insert(same_result.neighbor_cell_indices[off]);
            }
            for (std::size_t slot = 0; slot < static_cast<std::size_t>(any_result.top_k) && fill < static_cast<std::size_t>(same_result.top_k); ++slot) {
                const std::size_t src = detail::result_offset_(row, static_cast<std::int64_t>(slot), any_result.top_k);
                if (any_result.neighbor_cell_indices[src] < 0) continue;
                if (!seen.insert(any_result.neighbor_cell_indices[src]).second) continue;
                const std::size_t dst = detail::result_offset_(row, static_cast<std::int64_t>(fill), same_result.top_k);
                same_result.neighbor_cell_indices[dst] = any_result.neighbor_cell_indices[src];
                same_result.neighbor_time[dst] = any_result.neighbor_time[src];
                same_result.neighbor_embryo_ids[dst] = any_result.neighbor_embryo_ids[src];
                same_result.neighbor_similarity[dst] = any_result.neighbor_similarity[src];
                same_result.neighbor_sqdist[dst] = any_result.neighbor_sqdist[src];
                same_result.neighbor_distance[dst] = any_result.neighbor_distance[src];
                ++fill;
            }
        }
        return same_result;
    }
    return detail::search_core_(
        *storage_,
        query,
        config,
        config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only);
}

ForwardNeighborSearchResult ForwardNeighborIndex::search_future_neighbors_by_cell_index(
    const std::int64_t *cell_indices,
    std::size_t cell_count,
    const ForwardNeighborSearchConfig &config) const {
    return search_future_neighbors(query_batch_from_cell_indices(cell_indices, cell_count), config);
}

ForwardNeighborIndex build_forward_neighbor_index(
    const ForwardNeighborRecordBatch &records,
    const ForwardNeighborBuildConfig &config) {
    ForwardNeighborIndexBuilder builder(config);
    builder.append(records);
    return std::move(builder).finalize();
}

} // namespace cellerator::compute::neighbors::forward_neighbors
