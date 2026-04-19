#include "forwardNeighbors.hh"
#include "cellerator_cuda_mode.hh"

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
#include <utility>

namespace cellerator::compute::neighbors::forward_neighbors {

namespace detail {

namespace cg = ::cellerator::compute::graph;
namespace css = ::cellshard::sparse;

constexpr int kWarpThreads = 32;
constexpr int kForwardNeighborWarpsPerBlock = 1;
constexpr int kForwardNeighborThreadsPerBlock = kForwardNeighborWarpsPerBlock * kWarpThreads;
constexpr int kMaxTopK = 32;
constexpr int kMaxProbe = 16;

struct Candidate {
    float similarity = -INFINITY;
    float developmental_time = INFINITY;
    std::int64_t embryo_id = -1;
    std::int64_t cell_index = -1;
    std::int64_t shard_index = -1;
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
    std::int64_t row_begin = 0;
    std::int64_t row_end = 0;
    float time_begin = detail::quiet_nan_();
    float time_end = detail::quiet_nan_();
    std::size_t resident_bytes = 0u;
    bool resident = false;
    host_array<std::int64_t> cell_indices_cpu;
    host_array<float> developmental_time_cpu;
    host_array<std::int64_t> embryo_ids_cpu;
    host_array<float> latent_unit_cpu;
    host_array<__half> latent_unit_half_cpu;
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

struct ForwardNeighborSearchWorkspaceStorage {
    int active_device_id = -1;
    host_array<float> normalized_query_latent;
    host_array<Candidate> merged_best;
    host_array<Candidate> shard_best;
    host_array<std::int64_t> block_query_embryos;
    host_array<std::int64_t> deferred_segment_order;
    host_array<std::int64_t> segment_order;
    host_array<std::pair<std::int64_t, std::int64_t>> intervals;
    host_array<std::pair<std::int64_t, std::int64_t>> merged_intervals;
    host_array<float> block_latent_f32;
    host_array<__half> block_latent_half;
    host_array<float> block_lower;
    host_array<float> block_upper;
    host_array<std::int64_t> block_embryo;
    host_array<std::int64_t> eligible_lists;
    host_array<float> eligible_centroids;
    host_array<std::int64_t> eligible_embryo;
    host_array<std::int64_t> eligible_row_begin;
    host_array<std::int64_t> eligible_row_end;
    host_array<std::int64_t> download_cell;
    host_array<std::int64_t> download_shard;
    host_array<float> download_time;
    host_array<std::int64_t> download_embryo;
    host_array<float> download_similarity;
    host_array<std::int64_t> seen_neighbor_ids;
    host_array<int> block_device_ids;
    host_array<std::uint32_t> block_device_offsets;
    host_array<std::uint32_t> block_device_shards;
    cg::device_buffer<__half> d_query_latent;
    cg::device_buffer<float> d_query_lower;
    cg::device_buffer<float> d_query_upper;
    cg::device_buffer<std::int64_t> d_query_embryo;
    cg::device_buffer<std::int64_t> d_best_cell;
    cg::device_buffer<std::int64_t> d_best_shard;
    cg::device_buffer<float> d_best_time;
    cg::device_buffer<std::int64_t> d_best_embryo;
    cg::device_buffer<float> d_best_similarity;
    cg::device_buffer<float> d_centroids;
    cg::device_buffer<std::int64_t> d_list_embryo;
    cg::device_buffer<std::int64_t> d_list_row_begin;
    cg::device_buffer<std::int64_t> d_list_row_end;
    cg::device_buffer<std::int32_t> d_selected_lists;
};

struct DeviceResidencyState {
    int device_id = -1;
    std::size_t resident_bytes = 0u;
    host_array<std::uint32_t> resident_shards;
};

struct ForwardNeighborExecutorStorage {
    ForwardNeighborExecutorConfig config;
    host_array<DeviceResidencyState> residency;
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
    if (lhs.cell_index < rhs.cell_index) return true;
    if (lhs.cell_index > rhs.cell_index) return false;
    return lhs.shard_index < rhs.shard_index;
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
    if (lhs.cell_index < rhs.cell_index) return true;
    if (lhs.cell_index > rhs.cell_index) return false;
    return lhs.shard_index < rhs.shard_index;
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

__device__ inline std::int64_t shfl_i64_device_(unsigned mask, std::int64_t value, int src_lane) {
    const std::uint64_t bits = static_cast<std::uint64_t>(value);
    const std::uint32_t lo = __shfl_sync(mask, static_cast<std::uint32_t>(bits), src_lane);
    const std::uint32_t hi = __shfl_sync(mask, static_cast<std::uint32_t>(bits >> 32), src_lane);
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(hi) << 32) | static_cast<std::uint64_t>(lo));
}

__device__ inline Candidate shfl_candidate_device_(unsigned mask, const Candidate &candidate, int src_lane) {
    Candidate shuffled;
    shuffled.similarity = __shfl_sync(mask, candidate.similarity, src_lane);
    shuffled.developmental_time = __shfl_sync(mask, candidate.developmental_time, src_lane);
    shuffled.embryo_id = shfl_i64_device_(mask, candidate.embryo_id, src_lane);
    shuffled.cell_index = shfl_i64_device_(mask, candidate.cell_index, src_lane);
    shuffled.shard_index = shfl_i64_device_(mask, candidate.shard_index, src_lane);
    return shuffled;
}

__device__ inline ProbeCandidate shfl_probe_candidate_device_(unsigned mask, const ProbeCandidate &candidate, int src_lane) {
    ProbeCandidate shuffled;
    shuffled.similarity = __shfl_sync(mask, candidate.similarity, src_lane);
    shuffled.list_offset = __shfl_sync(mask, candidate.list_offset, src_lane);
    return shuffled;
}

inline std::int64_t query_blocks_for_rows_(std::int64_t query_rows) {
    return (query_rows + static_cast<std::int64_t>(kForwardNeighborWarpsPerBlock) - 1) /
        static_cast<std::int64_t>(kForwardNeighborWarpsPerBlock);
}

#include "kernels/init_best_kernel_.cuh"
#include "kernels/exact_search_kernel_.cuh"
#include "kernels/ann_probe_kernel_.cuh"
#include "kernels/ann_refine_kernel_.cuh"

inline void require_gpu_() {
    int count = 0;
    cg::cuda_require(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count <= 0) throw std::runtime_error("forward_neighbors requires at least one CUDA device");
}

inline int peer_rank_for_devices_(int src_device, int dst_device) {
    if (src_device == dst_device) return 0;
    int can_access = 0;
    cg::cuda_require(cudaDeviceCanAccessPeer(&can_access, src_device, dst_device), "cudaDeviceCanAccessPeer");
    if (can_access == 0) return -1;
    int rank = -1;
    cg::cuda_require(
        cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, src_device, dst_device),
        "cudaDeviceGetP2PAttribute");
    return rank;
}

inline bool visible_devices_match_native_topology_(int count) {
    if (count < 4) return false;

    int best_rank = std::numeric_limits<int>::max();
    for (int src = 0; src < 4; ++src) {
        for (int dst = 0; dst < 4; ++dst) {
            if (src == dst) continue;
            const int rank = peer_rank_for_devices_(src, dst);
            if (rank >= 0 && rank < best_rank) best_rank = rank;
        }
    }
    if (best_rank == std::numeric_limits<int>::max()) return false;

    const bool pair02_best = peer_rank_for_devices_(0, 2) == best_rank && peer_rank_for_devices_(2, 0) == best_rank;
    const bool pair13_best = peer_rank_for_devices_(1, 3) == best_rank && peer_rank_for_devices_(3, 1) == best_rank;
    const bool leaders_worse =
        peer_rank_for_devices_(0, 1) > best_rank
        && peer_rank_for_devices_(1, 0) > best_rank
        && peer_rank_for_devices_(2, 3) > best_rank
        && peer_rank_for_devices_(3, 2) > best_rank;
    const bool cross_worse =
        peer_rank_for_devices_(0, 3) > best_rank
        && peer_rank_for_devices_(3, 0) > best_rank
        && peer_rank_for_devices_(1, 2) > best_rank
        && peer_rank_for_devices_(2, 1) > best_rank;
    return pair02_best && pair13_best && leaders_worse && cross_worse;
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

    // Generic mode is intentionally ordinal and topology-agnostic. Native
    // modes may opt into the V100 pair-local order only after discovery proves
    // the visible device set matches that host profile.
    if (!build::cuda_mode_is_generic && count >= 4 && visible_devices_match_native_topology_(count)) {
        devices[0] = 0;
        devices[1] = 2;
        devices[2] = 1;
        devices[3] = 3;
    }
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

inline host_array<float> materialize_dense_matrix_(const ForwardNeighborMatrixView &matrix) {
    const std::int64_t rows = matrix_rows_(matrix);
    const std::int64_t cols = matrix_cols_(matrix);
    host_array<float> dense(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    dense.assign_fill(dense.size(), 0.0f);
    if (rows == 0 || cols == 0) return dense;

    switch (matrix.layout) {
        case ForwardNeighborInputLayout::dense:
            for (std::int64_t row = 0; row < rows; ++row) {
                std::memcpy(
                    dense.data() + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols),
                    matrix.dense_values + static_cast<std::size_t>(row) * static_cast<std::size_t>(matrix.dense_row_stride),
                    static_cast<std::size_t>(cols) * sizeof(float));
            }
            break;
        case ForwardNeighborInputLayout::blocked_ell: {
            const css::blocked_ell *src = matrix.blocked_ell;
            for (std::int64_t row = 0; row < rows; ++row) {
                for (std::int64_t col = 0; col < cols; ++col) {
                    const __half *value = css::at(src,
                                                  static_cast<cellshard::types::dim_t>(row),
                                                  static_cast<cellshard::types::idx_t>(col));
                    if (value != nullptr) {
                        dense[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col)] =
                            __half2float(*value);
                    }
                }
            }
            break;
        }
        case ForwardNeighborInputLayout::sliced_ell: {
            const css::sliced_ell *src = matrix.sliced_ell;
            for (std::int64_t row = 0; row < rows; ++row) {
                for (std::int64_t col = 0; col < cols; ++col) {
                    const __half *value = css::at(src,
                                                  static_cast<cellshard::types::dim_t>(row),
                                                  static_cast<cellshard::types::idx_t>(col));
                    if (value != nullptr) {
                        dense[static_cast<std::size_t>(row) * static_cast<std::size_t>(cols) + static_cast<std::size_t>(col)] =
                            __half2float(*value);
                    }
                }
            }
            break;
        }
    }
    return dense;
}

inline host_array<std::int64_t> sorted_row_order_(
    const host_array<std::int64_t> &cell_indices,
    const host_array<float> &developmental_time,
    const host_array<std::int64_t> &embryo_ids) {
    const std::size_t rows = cell_indices.size();
    host_array<std::int64_t> order(rows);
    for (std::size_t row = 0; row < rows; ++row) order[row] = static_cast<std::int64_t>(row);
    std::stable_sort(order.begin(), order.end(), [&](std::int64_t lhs, std::int64_t rhs) {
        if (developmental_time[static_cast<std::size_t>(lhs)] < developmental_time[static_cast<std::size_t>(rhs)]) return true;
        if (developmental_time[static_cast<std::size_t>(lhs)] > developmental_time[static_cast<std::size_t>(rhs)]) return false;
        if (embryo_ids[static_cast<std::size_t>(lhs)] < embryo_ids[static_cast<std::size_t>(rhs)]) return true;
        if (embryo_ids[static_cast<std::size_t>(lhs)] > embryo_ids[static_cast<std::size_t>(rhs)]) return false;
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

    table.segments = segments;
    std::int64_t total_rows = 0;
    for (const SegmentPlan &segment : segments) total_rows += segment.source_end - segment.source_begin;
    const std::int64_t target_rows = std::max<std::int64_t>(
        1,
        (total_rows + static_cast<std::int64_t>(used_shards) - 1) / static_cast<std::int64_t>(used_shards));

    std::size_t segment_cursor = 0u;
    for (std::size_t shard = 0; shard < used_shards; ++shard) {
        table.shard_offsets[shard] = segment_cursor;
        if (segment_cursor >= segments.size()) continue;

        std::int64_t shard_rows = 0;
        const std::size_t remaining_shards = used_shards - shard;
        const std::size_t remaining_segments = segments.size() - segment_cursor;
        const std::size_t min_segments_to_leave = remaining_shards > 1u ? remaining_shards - 1u : 0u;

        while (segment_cursor < segments.size()) {
            const SegmentPlan &segment = segments[segment_cursor];
            const std::int64_t segment_rows = segment.source_end - segment.source_begin;
            const bool must_leave_segment = (segments.size() - (segment_cursor + 1u)) < min_segments_to_leave;
            if (shard_rows > 0 && shard_rows + segment_rows > target_rows && !must_leave_segment) break;
            shard_rows += segment_rows;
            ++segment_cursor;
            if (must_leave_segment) break;
        }
    }
    table.shard_offsets[used_shards] = segments.size();
    for (std::size_t shard = 0; shard < used_shards; ++shard) {
        if (table.shard_offsets[shard + 1u] < table.shard_offsets[shard]) {
            table.shard_offsets[shard + 1u] = table.shard_offsets[shard];
        }
    }
    return table;
}

inline void eligible_segment_order_(
    const host_array<ForwardNeighborSegment> &segments,
    const host_array<std::int64_t> &query_embryos,
    ForwardNeighborEmbryoPolicy policy,
    host_array<std::int64_t> *ordered,
    host_array<std::int64_t> *deferred) {
    ordered->clear();
    deferred->clear();
    const auto contains_embryo = [&](std::int64_t embryo_id) {
        for (std::size_t i = 0; i < query_embryos.size(); ++i) {
            if (query_embryos[i] == embryo_id) return true;
        }
        return false;
    };

    for (std::int64_t segment_idx = 0; segment_idx < static_cast<std::int64_t>(segments.size()); ++segment_idx) {
        const bool embryo_match = !query_embryos.empty() && contains_embryo(segments[static_cast<std::size_t>(segment_idx)].embryo_id);
        if (policy == ForwardNeighborEmbryoPolicy::same_embryo_only) {
            if (query_embryos.empty() || embryo_match) append_one_(ordered, segment_idx);
            continue;
        }
        if (policy == ForwardNeighborEmbryoPolicy::same_embryo_first && embryo_match) {
            append_one_(ordered, segment_idx);
        } else if (policy == ForwardNeighborEmbryoPolicy::same_embryo_first) {
            append_one_(deferred, segment_idx);
        } else {
            append_one_(ordered, segment_idx);
        }
    }
    append_copy_(ordered, deferred->data(), deferred->size());
}

inline void collect_block_query_embryos_(
    const host_array<std::int64_t> &query_embryo,
    std::int64_t query_begin,
    std::int64_t query_end,
    host_array<std::int64_t> *embryo_ids) {
    embryo_ids->clear();
    for (std::int64_t row = query_begin; row < query_end; ++row) {
        const std::int64_t embryo = query_embryo[static_cast<std::size_t>(row)];
        if (embryo < 0) continue;
        bool seen = false;
        for (std::size_t i = 0; i < embryo_ids->size(); ++i) {
            if ((*embryo_ids)[i] == embryo) {
                seen = true;
                break;
            }
        }
        if (!seen) append_one_(embryo_ids, embryo);
    }
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
    const const_array_view<float> &query_time,
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

inline void merge_row_intervals_(
    host_array<std::pair<std::int64_t, std::int64_t>> *intervals,
    host_array<std::pair<std::int64_t, std::int64_t>> *merged) {
    merged->clear();
    if (intervals->empty()) return;
    std::sort(intervals->begin(), intervals->end(), [](const auto &lhs, const auto &rhs) {
        if (lhs.first < rhs.first) return true;
        if (lhs.first > rhs.first) return false;
        return lhs.second < rhs.second;
    });
    append_one_(merged, (*intervals)[0]);
    for (std::size_t i = 1; i < intervals->size(); ++i) {
        if ((*intervals)[i].first <= merged->back().second) {
            merged->back().second = std::max(merged->back().second, (*intervals)[i].second);
        } else {
            append_one_(merged, (*intervals)[i]);
        }
    }
}

inline void convert_f32_to_half_(const host_array<float> &input, host_array<__half> *output) {
    output->resize(input.size());
    for (std::size_t i = 0; i < input.size(); ++i) (*output)[i] = __float2half_rn(input[i]);
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
    ForwardNeighborSearchWorkspaceStorage *workspace) {
    const std::size_t total = static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k);
    workspace->d_best_cell.resize(total);
    workspace->d_best_shard.resize(total);
    workspace->d_best_time.resize(total);
    workspace->d_best_embryo.resize(total);
    workspace->d_best_similarity.resize(total);
    const int threads = 128;
    const int blocks = static_cast<int>((total + static_cast<std::size_t>(threads) - 1u) / static_cast<std::size_t>(threads));
    // Initialization launch is pure bookkeeping and followed by a sync, so it is noticeable only for small query batches.
    init_best_kernel_<<<blocks, threads>>>(
        workspace->d_best_cell.data(),
        workspace->d_best_shard.data(),
        workspace->d_best_time.data(),
        workspace->d_best_embryo.data(),
        workspace->d_best_similarity.data(),
        query_rows,
        k);
    cg::cuda_require(cudaGetLastError(), "init_best_kernel launch");
}

inline void download_best_candidates_(
    ForwardNeighborSearchWorkspaceStorage *workspace,
    std::int64_t query_rows,
    int k) {
    const std::size_t total = static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k);
    workspace->download_cell.assign_fill(total, static_cast<std::int64_t>(-1));
    workspace->download_shard.assign_fill(total, static_cast<std::int64_t>(-1));
    workspace->download_time.assign_fill(total, detail::quiet_nan_());
    workspace->download_embryo.assign_fill(total, static_cast<std::int64_t>(-1));
    workspace->download_similarity.assign_fill(total, detail::negative_infinity_());
    // Full-table downloads are intentional here: merge logic is host-side today, so query blocks should be large enough to amortize D2H latency.
    workspace->d_best_cell.download(workspace->download_cell.data(), total);
    workspace->d_best_shard.download(workspace->download_shard.data(), total);
    workspace->d_best_time.download(workspace->download_time.data(), total);
    workspace->d_best_embryo.download(workspace->download_embryo.data(), total);
    workspace->d_best_similarity.download(workspace->download_similarity.data(), total);

    workspace->shard_best.resize(total);
    for (std::size_t i = 0; i < total; ++i) {
        workspace->shard_best[i] = Candidate{
            workspace->download_similarity[i],
            workspace->download_time[i],
            workspace->download_embryo[i],
            workspace->download_cell[i],
            workspace->download_shard[i]
        };
    }
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
    const std::int64_t batch_cols = matrix_cols_(batch.matrix);
    if (*latent_dim == 0) *latent_dim = batch_cols;
    if (*latent_dim != batch_cols) throw std::invalid_argument("all forward-neighbor batches must share matrix cols");

    detail::append_i64_(cell_indices, batch.cell_indices);
    detail::append_f32_(developmental_time, batch.developmental_time);
    const host_array<float> dense_batch = materialize_dense_matrix_(batch.matrix);
    append_copy_(latent_unit, dense_batch.data(), dense_batch.size());
    if (batch.embryo_ids.empty()) {
        const std::size_t old_size = embryo_ids->size();
        embryo_ids->resize(old_size + batch.cell_indices.size);
        for (std::size_t i = 0; i < batch.cell_indices.size; ++i) {
            (*embryo_ids)[old_size + i] = static_cast<std::int64_t>(-1);
        }
    } else {
        detail::append_i64_(embryo_ids, batch.embryo_ids);
    }
}

inline std::size_t estimate_shard_resident_bytes_(const DeviceShardStorage &shard) {
    return shard.cell_indices_cpu.size() * sizeof(std::int64_t)
        + shard.developmental_time_cpu.size() * sizeof(float)
        + shard.embryo_ids_cpu.size() * sizeof(std::int64_t)
        + shard.latent_unit_half_cpu.size() * sizeof(__half)
        + shard.ann_centroids_cpu.size() * sizeof(float);
}

inline void release_shard_device_buffers_(DeviceShardStorage *shard) {
    if (shard == nullptr || !shard->resident) return;
    set_device_(shard->device_id);
    shard->cell_indices.reset();
    shard->developmental_time.reset();
    shard->embryo_ids.reset();
    shard->latent_unit.reset();
    shard->ann_centroids.reset();
    shard->resident = false;
}

inline void upload_shard_to_device_(DeviceShardStorage *shard) {
    if (shard == nullptr || shard->resident) return;
    set_device_(shard->device_id);
    shard->cell_indices.resize(shard->cell_indices_cpu.size());
    shard->developmental_time.resize(shard->developmental_time_cpu.size());
    shard->embryo_ids.resize(shard->embryo_ids_cpu.size());
    shard->latent_unit.resize(shard->latent_unit_half_cpu.size());
    shard->ann_centroids.resize(shard->ann_centroids_cpu.size());
    shard->cell_indices.upload(shard->cell_indices_cpu.data(), shard->cell_indices_cpu.size());
    shard->developmental_time.upload(shard->developmental_time_cpu.data(), shard->developmental_time_cpu.size());
    shard->embryo_ids.upload(shard->embryo_ids_cpu.data(), shard->embryo_ids_cpu.size());
    shard->latent_unit.upload(shard->latent_unit_half_cpu.data(), shard->latent_unit_half_cpu.size());
    if (!shard->ann_centroids_cpu.empty()) {
        shard->ann_centroids.upload(shard->ann_centroids_cpu.data(), shard->ann_centroids_cpu.size());
    }
    shard->resident = true;
}

inline DeviceResidencyState &device_residency_state_(
    ForwardNeighborExecutorStorage *executor,
    int device_id) {
    for (std::size_t i = 0; i < executor->residency.size(); ++i) {
        if (executor->residency[i].device_id == device_id) return executor->residency[i];
    }
    const std::size_t old_size = executor->residency.size();
    executor->residency.resize(old_size + 1u);
    executor->residency[old_size].device_id = device_id;
    executor->residency[old_size].resident_bytes = 0u;
    executor->residency[old_size].resident_shards.clear();
    return executor->residency[old_size];
}

inline void erase_resident_shard_at_(DeviceResidencyState *state, std::size_t idx) {
    if (idx >= state->resident_shards.size()) return;
    for (std::size_t i = idx + 1u; i < state->resident_shards.size(); ++i) {
        state->resident_shards[i - 1u] = state->resident_shards[i];
    }
    state->resident_shards.resize(state->resident_shards.size() - 1u);
}

inline void touch_resident_shard_(
    DeviceResidencyState *state,
    std::uint32_t shard_idx) {
    for (std::size_t i = 0; i < state->resident_shards.size(); ++i) {
        if (state->resident_shards[i] != shard_idx) continue;
        const std::uint32_t value = state->resident_shards[i];
        erase_resident_shard_at_(state, i);
        append_one_(&state->resident_shards, value);
        return;
    }
}

inline void evict_oldest_resident_shard_(
    ForwardNeighborIndexStorage *storage,
    DeviceResidencyState *state) {
    if (state->resident_shards.empty()) return;
    const std::uint32_t shard_idx = state->resident_shards[0];
    if (shard_idx >= storage->shards.size()) {
        erase_resident_shard_at_(state, 0u);
        return;
    }
    DeviceShardStorage &shard = storage->shards[shard_idx];
    if (shard.resident && state->resident_bytes >= shard.resident_bytes) {
        state->resident_bytes -= shard.resident_bytes;
    } else {
        state->resident_bytes = 0u;
    }
    release_shard_device_buffers_(&shard);
    erase_resident_shard_at_(state, 0u);
}

inline void ensure_shard_resident_(
    ForwardNeighborIndexStorage *storage,
    std::uint32_t shard_idx,
    ForwardNeighborExecutorStorage *executor) {
    DeviceShardStorage &shard = storage->shards[shard_idx];
    DeviceResidencyState &state = device_residency_state_(executor, shard.device_id);
    if (shard.resident) {
        touch_resident_shard_(&state, shard_idx);
        return;
    }

    const std::int64_t max_resident_shards = executor->config.max_resident_shards_per_device;
    while (max_resident_shards > 0
           && state.resident_shards.size() >= static_cast<std::size_t>(max_resident_shards)
           && !state.resident_shards.empty()) {
        evict_oldest_resident_shard_(storage, &state);
    }
    while (executor->config.resident_bytes_per_device > 0u
           && state.resident_bytes + shard.resident_bytes > executor->config.resident_bytes_per_device
           && !state.resident_shards.empty()) {
        evict_oldest_resident_shard_(storage, &state);
    }

    upload_shard_to_device_(&shard);
    state.resident_bytes += shard.resident_bytes;
    append_one_(&state.resident_shards, shard_idx);
}

inline bool shard_time_overlaps_block_(
    const DeviceShardStorage &shard,
    float block_lower,
    float block_upper) {
    if (shard.row_begin >= shard.row_end) return false;
    if (shard.time_end <= block_lower) return false;
    if (std::isfinite(block_upper) && shard.time_begin > block_upper) return false;
    return true;
}

inline ForwardNeighborRoutingPlan build_routing_plan_(
    const ForwardNeighborIndexStorage &storage,
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config) {
    ForwardNeighborRoutingPlan plan;
    if (query.cell_indices.empty()) {
        plan.block_route_offsets.assign_fill(1u, 0u);
        return plan;
    }

    std::uint32_t route_cursor = 0u;
    const std::int64_t block_rows = std::max<std::int64_t>(1, config.query_block_rows);
    const std::int64_t query_count = static_cast<std::int64_t>(query.cell_indices.size);
    for (std::int64_t query_begin = 0; query_begin < query_count; query_begin += block_rows) {
        const std::int64_t query_end = std::min(query_begin + block_rows, query_count);
        const auto block_limits = block_time_limits_(query.developmental_time, query_begin, query_end, config);
        append_one_(&plan.block_query_begin, query_begin);
        append_one_(&plan.block_query_end, query_end);
        append_one_(&plan.block_window_lower, block_limits.first);
        append_one_(&plan.block_window_upper, block_limits.second);
        append_one_(&plan.block_route_offsets, route_cursor);
        for (std::uint32_t shard_idx = 0u; shard_idx < storage.shards.size(); ++shard_idx) {
            const DeviceShardStorage &shard = storage.shards[shard_idx];
            if (!shard_time_overlaps_block_(shard, block_limits.first, block_limits.second)) continue;
            append_one_(&plan.route_shard_indices, shard_idx);
            append_one_(&plan.route_device_ids, shard.device_id);
            ++route_cursor;
        }
    }
    append_one_(&plan.block_route_offsets, route_cursor);
    return plan;
}

inline void build_block_device_routes_(
    const ForwardNeighborRoutingPlan &plan,
    std::size_t block_idx,
    ForwardNeighborSearchWorkspaceStorage *workspace) {
    workspace->block_device_ids.clear();
    workspace->block_device_offsets.clear();
    workspace->block_device_shards.clear();
    if (block_idx + 1u >= plan.block_route_offsets.size()) return;

    const std::uint32_t route_begin = plan.block_route_offsets[block_idx];
    const std::uint32_t route_end = plan.block_route_offsets[block_idx + 1u];
    for (std::uint32_t route = route_begin; route < route_end; ++route) {
        const int device_id = plan.route_device_ids[route];
        bool seen = false;
        for (std::size_t idx = 0; idx < workspace->block_device_ids.size(); ++idx) {
            if (workspace->block_device_ids[idx] == device_id) {
                seen = true;
                break;
            }
        }
        if (!seen) append_one_(&workspace->block_device_ids, device_id);
    }

    workspace->block_device_offsets.assign_fill(workspace->block_device_ids.size() + 1u, 0u);
    for (std::uint32_t route = route_begin; route < route_end; ++route) {
        const int device_id = plan.route_device_ids[route];
        for (std::size_t idx = 0; idx < workspace->block_device_ids.size(); ++idx) {
            if (workspace->block_device_ids[idx] != device_id) continue;
            ++workspace->block_device_offsets[idx + 1u];
            break;
        }
    }
    for (std::size_t idx = 0; idx + 1u < workspace->block_device_offsets.size(); ++idx) {
        workspace->block_device_offsets[idx + 1u] += workspace->block_device_offsets[idx];
    }
    workspace->block_device_shards.resize(route_end - route_begin);
    host_array<std::uint32_t> cursor = workspace->block_device_offsets;
    for (std::uint32_t route = route_begin; route < route_end; ++route) {
        const int device_id = plan.route_device_ids[route];
        for (std::size_t idx = 0; idx < workspace->block_device_ids.size(); ++idx) {
            if (workspace->block_device_ids[idx] != device_id) continue;
            workspace->block_device_shards[cursor[idx]++] = plan.route_shard_indices[route];
            break;
        }
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
        shard.row_begin = shard_plans.segments[plan_begin].source_begin;
        shard.row_end = shard_plans.segments[plan_end - 1u].source_end;
        shard.time_begin = shard_plans.segments[plan_begin].time_begin;
        shard.time_end = shard_plans.segments[plan_end - 1u].time_end;
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

        convert_f32_to_half_(shard.latent_unit_cpu, &shard.latent_unit_half_cpu);
        shard.resident_bytes = estimate_shard_resident_bytes_(shard);
        if (config.eager_device_upload) upload_shard_to_device_(&shard);

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

inline ForwardNeighborSearchWorkspaceStorage &workspace_storage_(ForwardNeighborSearchWorkspace *workspace) {
    if (workspace == nullptr || workspace->storage_ == nullptr) {
        static thread_local ForwardNeighborSearchWorkspaceStorage local_workspace;
        return local_workspace;
    }
    return *workspace->storage_;
}

inline void reset_workspace_device_buffers_(ForwardNeighborSearchWorkspaceStorage *workspace) {
    workspace->d_query_latent.reset();
    workspace->d_query_lower.reset();
    workspace->d_query_upper.reset();
    workspace->d_query_embryo.reset();
    workspace->d_best_cell.reset();
    workspace->d_best_shard.reset();
    workspace->d_best_time.reset();
    workspace->d_best_embryo.reset();
    workspace->d_best_similarity.reset();
    workspace->d_centroids.reset();
    workspace->d_list_embryo.reset();
    workspace->d_list_row_begin.reset();
    workspace->d_list_row_end.reset();
    workspace->d_selected_lists.reset();
}

inline void prepare_workspace_device_(ForwardNeighborSearchWorkspaceStorage *workspace, int device_id) {
    if (workspace->active_device_id == device_id) return;
    reset_workspace_device_buffers_(workspace);
    workspace->active_device_id = device_id;
}

inline ForwardNeighborSearchResult search_core_(
    ForwardNeighborIndexStorage *storage,
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config,
    bool hard_same_embryo,
    ForwardNeighborSearchWorkspace *workspace,
    ForwardNeighborExecutorStorage *executor) {
    detail::validate_forward_neighbor_search_config_(config);
    detail::validate_forward_neighbor_query_batch_(query);
    if (query.cell_indices.empty()) return detail::empty_forward_neighbor_result_(config.top_k);
    if (storage == nullptr) return detail::empty_forward_neighbor_result_(config.top_k);
    const std::int64_t query_latent_dim = matrix_cols_(query.matrix);
    if (storage->latent_dim != 0 && storage->latent_dim != query_latent_dim) {
        throw std::invalid_argument("query latent dimension does not match the forward-neighbor index latent dimension");
    }
    if (config.candidate_k > kMaxTopK || config.ann_probe_list_count > kMaxProbe) {
        throw std::invalid_argument("forward-neighbor config exceeds native kernel limits");
    }

    ForwardNeighborSearchResult result;
    result.query_count = static_cast<std::int64_t>(query.cell_indices.size);
    result.top_k = config.top_k;
    result.query_cell_indices.assign_copy(query.cell_indices.data, query.cell_indices.size);
    result.query_time.assign_copy(query.developmental_time.data, query.developmental_time.size);
    result.query_embryo_ids = query.embryo_ids.empty()
        ? detail::make_missing_i64_array_(query.cell_indices.size)
        : host_array<std::int64_t>{};
    if (!query.embryo_ids.empty()) result.query_embryo_ids.assign_copy(query.embryo_ids.data, query.embryo_ids.size);

    ForwardNeighborSearchWorkspaceStorage &scratch = workspace_storage_(workspace);
    scratch.normalized_query_latent = normalize_query_latent_(materialize_dense_matrix_(query.matrix), result.query_count, query_latent_dim);
    scratch.merged_best.resize(static_cast<std::size_t>(result.query_count * result.top_k));
    for (std::size_t i = 0; i < scratch.merged_best.size(); ++i) scratch.merged_best[i] = Candidate{};
    const ForwardNeighborRoutingPlan routing = build_routing_plan_(*storage, query, config);

    for (std::size_t block_idx = 0; block_idx < routing.block_query_begin.size(); ++block_idx) {
        const std::int64_t query_begin = routing.block_query_begin[block_idx];
        const std::int64_t query_end = routing.block_query_end[block_idx];
        const std::int64_t block_queries = query_end - query_begin;
        if (block_queries <= 0) continue;

        collect_block_query_embryos_(result.query_embryo_ids, query_begin, query_end, &scratch.block_query_embryos);
        build_block_device_routes_(routing, block_idx, &scratch);
        if (scratch.block_device_ids.empty()) continue;

        scratch.block_latent_f32.resize(static_cast<std::size_t>(block_queries) * static_cast<std::size_t>(query_latent_dim));
        std::memcpy(
            scratch.block_latent_f32.data(),
            scratch.normalized_query_latent.data() + static_cast<std::size_t>(query_begin) * static_cast<std::size_t>(query_latent_dim),
            scratch.block_latent_f32.size() * sizeof(float));
        convert_f32_to_half_(scratch.block_latent_f32, &scratch.block_latent_half);
        scratch.block_lower.assign_fill(static_cast<std::size_t>(block_queries), detail::positive_infinity_());
        scratch.block_upper.assign_fill(static_cast<std::size_t>(block_queries), detail::positive_infinity_());
        scratch.block_embryo.assign_fill(static_cast<std::size_t>(block_queries), static_cast<std::int64_t>(-1));
        for (std::int64_t row = 0; row < block_queries; ++row) {
            const std::int64_t global = query_begin + row;
            scratch.block_lower[static_cast<std::size_t>(row)] =
                result.query_time[static_cast<std::size_t>(global)] + config.strict_future_epsilon + config.time_window.min_delta;
            scratch.block_upper[static_cast<std::size_t>(row)] = std::isfinite(config.time_window.max_delta)
                ? result.query_time[static_cast<std::size_t>(global)] + config.time_window.max_delta
                : detail::positive_infinity_();
            scratch.block_embryo[static_cast<std::size_t>(row)] = result.query_embryo_ids[static_cast<std::size_t>(global)];
        }

        for (std::size_t device_group = 0; device_group < scratch.block_device_ids.size(); ++device_group) {
            const int device_id = scratch.block_device_ids[device_group];
            const std::uint32_t shard_begin = scratch.block_device_offsets[device_group];
            const std::uint32_t shard_end = scratch.block_device_offsets[device_group + 1u];
            if (shard_begin == shard_end) continue;

            set_device_(device_id);
            prepare_workspace_device_(&scratch, device_id);
            scratch.d_query_latent.resize(scratch.block_latent_half.size());
            scratch.d_query_lower.resize(scratch.block_lower.size());
            scratch.d_query_upper.resize(scratch.block_upper.size());
            scratch.d_query_embryo.resize(scratch.block_embryo.size());
            scratch.d_query_latent.upload(scratch.block_latent_half.data(), scratch.block_latent_half.size());
            scratch.d_query_lower.upload(scratch.block_lower.data(), scratch.block_lower.size());
            scratch.d_query_upper.upload(scratch.block_upper.data(), scratch.block_upper.size());
            scratch.d_query_embryo.upload(scratch.block_embryo.data(), scratch.block_embryo.size());

            init_best_arrays_(block_queries, static_cast<int>(config.candidate_k), &scratch);

            for (std::uint32_t shard_slot = shard_begin; shard_slot < shard_end; ++shard_slot) {
                const std::uint32_t shard_idx = scratch.block_device_shards[shard_slot];
                ensure_shard_resident_(storage, shard_idx, executor);
                DeviceShardStorage &shard = storage->shards[shard_idx];
                eligible_segment_order_(
                    shard.segments,
                    scratch.block_query_embryos,
                    hard_same_embryo ? ForwardNeighborEmbryoPolicy::same_embryo_only : config.embryo_policy,
                    &scratch.segment_order,
                    &scratch.deferred_segment_order);

                if (config.backend == ForwardNeighborBackend::exact_windowed) {
                    scratch.intervals.clear();
                    for (const std::int64_t segment_idx : scratch.segment_order) {
                        const ForwardNeighborSegment &segment = shard.segments[static_cast<std::size_t>(segment_idx)];
                        if (!segment_time_overlaps_block_(segment, routing.block_window_lower[block_idx], routing.block_window_upper[block_idx])) continue;
                        const auto bounds = segment_candidate_bounds_(
                            shard,
                            segment,
                            routing.block_window_lower[block_idx],
                            routing.block_window_upper[block_idx]);
                        if (bounds.first < bounds.second) append_one_(&scratch.intervals, bounds);
                    }

                    merge_row_intervals_(&scratch.intervals, &scratch.merged_intervals);
                    for (const auto &interval : scratch.merged_intervals) {
                        for (std::int64_t index_begin = interval.first; index_begin < interval.second; index_begin += config.index_block_rows) {
                            const std::int64_t index_end = std::min(index_begin + config.index_block_rows, interval.second);
                            exact_search_kernel_<<<query_blocks_for_rows_(block_queries), kForwardNeighborThreadsPerBlock>>>(
                                scratch.d_query_latent.data(),
                                scratch.d_query_lower.data(),
                                scratch.d_query_upper.data(),
                                scratch.d_query_embryo.data(),
                                shard.latent_unit.data(),
                                shard.developmental_time.data(),
                                shard.embryo_ids.data(),
                                shard.cell_indices.data(),
                                block_queries,
                                static_cast<int>(storage->latent_dim),
                                static_cast<std::int64_t>(shard_idx),
                                index_begin,
                                index_end - index_begin,
                                hard_same_embryo ? 1 : 0,
                                static_cast<int>(config.candidate_k),
                                scratch.d_best_cell.data(),
                                scratch.d_best_shard.data(),
                                scratch.d_best_time.data(),
                                scratch.d_best_embryo.data(),
                                scratch.d_best_similarity.data());
                            cg::cuda_require(cudaGetLastError(), "exact_search_kernel launch");
                        }
                    }
                } else {
                    scratch.eligible_lists.clear();
                    for (const std::int64_t segment_idx : scratch.segment_order) {
                        const ForwardNeighborSegment &segment = shard.segments[static_cast<std::size_t>(segment_idx)];
                        if (!segment_time_overlaps_block_(segment, routing.block_window_lower[block_idx], routing.block_window_upper[block_idx])) continue;
                        for (std::int64_t list_idx = segment.ann_list_begin; list_idx < segment.ann_list_end; ++list_idx) {
                            const ForwardNeighborAnnList &list = shard.ann_lists[static_cast<std::size_t>(list_idx)];
                            if (ann_list_overlaps_block_(list, routing.block_window_lower[block_idx], routing.block_window_upper[block_idx])) {
                                append_one_(&scratch.eligible_lists, list_idx);
                            }
                        }
                    }

                    if (scratch.eligible_lists.empty()) continue;
                    scratch.eligible_centroids.clear();
                    scratch.eligible_embryo.clear();
                    scratch.eligible_row_begin.clear();
                    scratch.eligible_row_end.clear();
                    for (const std::int64_t list_idx : scratch.eligible_lists) {
                        const ForwardNeighborAnnList &list = shard.ann_lists[static_cast<std::size_t>(list_idx)];
                        append_copy_(
                            &scratch.eligible_centroids,
                            shard.ann_centroids_cpu.data() + static_cast<std::size_t>(list_idx) * static_cast<std::size_t>(storage->latent_dim),
                            static_cast<std::size_t>(storage->latent_dim));
                        append_one_(&scratch.eligible_embryo, list.embryo_id);
                        append_one_(&scratch.eligible_row_begin, list.row_begin);
                        append_one_(&scratch.eligible_row_end, list.row_end);
                    }

                    scratch.d_centroids.resize(scratch.eligible_centroids.size());
                    scratch.d_list_embryo.resize(scratch.eligible_embryo.size());
                    scratch.d_list_row_begin.resize(scratch.eligible_row_begin.size());
                    scratch.d_list_row_end.resize(scratch.eligible_row_end.size());
                    scratch.d_selected_lists.resize(static_cast<std::size_t>(block_queries) * static_cast<std::size_t>(config.ann_probe_list_count));
                    scratch.d_centroids.upload(scratch.eligible_centroids.data(), scratch.eligible_centroids.size());
                    scratch.d_list_embryo.upload(scratch.eligible_embryo.data(), scratch.eligible_embryo.size());
                    scratch.d_list_row_begin.upload(scratch.eligible_row_begin.data(), scratch.eligible_row_begin.size());
                    scratch.d_list_row_end.upload(scratch.eligible_row_end.data(), scratch.eligible_row_end.size());

                    const std::int64_t query_blocks = query_blocks_for_rows_(block_queries);

                    ann_probe_kernel_<<<query_blocks, kForwardNeighborThreadsPerBlock>>>(
                        scratch.d_query_latent.data(),
                        scratch.d_query_embryo.data(),
                        scratch.d_centroids.data(),
                        scratch.d_list_embryo.data(),
                        block_queries,
                        static_cast<int>(storage->latent_dim),
                        static_cast<std::int64_t>(scratch.eligible_lists.size()),
                        hard_same_embryo ? 1 : 0,
                        static_cast<int>(config.ann_probe_list_count),
                        scratch.d_selected_lists.data());
                    cg::cuda_require(cudaGetLastError(), "ann_probe_kernel launch");

                    ann_refine_kernel_<<<query_blocks, kForwardNeighborThreadsPerBlock>>>(
                        scratch.d_query_latent.data(),
                        scratch.d_query_lower.data(),
                        scratch.d_query_upper.data(),
                        scratch.d_query_embryo.data(),
                        shard.latent_unit.data(),
                        shard.developmental_time.data(),
                        shard.embryo_ids.data(),
                        shard.cell_indices.data(),
                        scratch.d_selected_lists.data(),
                        scratch.d_list_row_begin.data(),
                        scratch.d_list_row_end.data(),
                        block_queries,
                        static_cast<int>(storage->latent_dim),
                        static_cast<std::int64_t>(shard_idx),
                        hard_same_embryo ? 1 : 0,
                        static_cast<int>(config.ann_probe_list_count),
                        static_cast<int>(config.candidate_k),
                        scratch.d_best_cell.data(),
                        scratch.d_best_shard.data(),
                        scratch.d_best_time.data(),
                        scratch.d_best_embryo.data(),
                        scratch.d_best_similarity.data());
                    cg::cuda_require(cudaGetLastError(), "ann_refine_kernel launch");
                }
            }

            cg::cuda_require(cudaDeviceSynchronize(), "forward_neighbors device sync");
            download_best_candidates_(&scratch, block_queries, static_cast<int>(config.candidate_k));
            for (std::int64_t row = 0; row < block_queries; ++row) {
                Candidate row_best[kMaxTopK];
                init_candidates_host_(row_best, static_cast<int>(result.top_k));
                std::memcpy(
                    row_best,
                    scratch.merged_best.data() + static_cast<std::size_t>(query_begin + row) * static_cast<std::size_t>(result.top_k),
                    static_cast<std::size_t>(result.top_k) * sizeof(Candidate));
                for (std::int64_t slot = 0; slot < config.candidate_k; ++slot) {
                    const std::size_t off = static_cast<std::size_t>(row) * static_cast<std::size_t>(config.candidate_k)
                        + static_cast<std::size_t>(slot);
                    insert_candidate_host_(scratch.shard_best[off], row_best, static_cast<int>(result.top_k));
                }
                std::memcpy(
                    scratch.merged_best.data() + static_cast<std::size_t>(query_begin + row) * static_cast<std::size_t>(result.top_k),
                    row_best,
                    static_cast<std::size_t>(result.top_k) * sizeof(Candidate));
            }
        }
    }

    result.neighbor_cell_indices.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), static_cast<std::int64_t>(-1));
    result.neighbor_shard_indices.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), static_cast<std::int64_t>(-1));
    result.neighbor_time.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::quiet_nan_());
    result.neighbor_embryo_ids.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), static_cast<std::int64_t>(-1));
    result.neighbor_similarity.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::negative_infinity_());
    result.neighbor_sqdist.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::positive_infinity_());
    result.neighbor_distance.assign_fill(static_cast<std::size_t>(result.query_count * result.top_k), detail::positive_infinity_());

    for (std::int64_t row = 0; row < result.query_count; ++row) {
        for (std::int64_t slot = 0; slot < result.top_k; ++slot) {
            const Candidate &candidate = scratch.merged_best[static_cast<std::size_t>(row) * static_cast<std::size_t>(result.top_k) + static_cast<std::size_t>(slot)];
            const std::size_t off = detail::result_offset_(row, slot, result.top_k);
            result.neighbor_cell_indices[off] = candidate.cell_index;
            result.neighbor_shard_indices[off] = candidate.shard_index;
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

ForwardNeighborSearchWorkspace::ForwardNeighborSearchWorkspace()
    : storage_(std::make_unique<detail::ForwardNeighborSearchWorkspaceStorage>()) {}

ForwardNeighborSearchWorkspace::~ForwardNeighborSearchWorkspace() = default;

ForwardNeighborSearchWorkspace::ForwardNeighborSearchWorkspace(ForwardNeighborSearchWorkspace &&) noexcept = default;

ForwardNeighborSearchWorkspace &ForwardNeighborSearchWorkspace::operator=(ForwardNeighborSearchWorkspace &&) noexcept = default;

void ForwardNeighborSearchWorkspace::reset() {
    storage_ = std::make_unique<detail::ForwardNeighborSearchWorkspaceStorage>();
}

ForwardNeighborIndexBuilder::ForwardNeighborIndexBuilder(ForwardNeighborBuildConfig config)
    : config_(std::move(config)) {}

void ForwardNeighborIndexBuilder::append(const ForwardNeighborRecordBatch &batch) {
    detail::validate_forward_neighbor_record_batch_(batch);
    detail::append_batch_(batch,
                          &records_.cell_indices,
                          &records_.developmental_time,
                          &records_.dense_values,
                          &records_.embryo_ids,
                          &records_.dense_cols);
}

ForwardNeighborIndex ForwardNeighborIndexBuilder::finalize() && {
    const ForwardNeighborRecordBatch records_view{
        view_of(records_.cell_indices),
        view_of(records_.developmental_time),
        view_of(records_.embryo_ids),
        make_forward_neighbor_dense_view(records_.dense_values.data(),
                                         static_cast<std::int64_t>(records_.cell_indices.size()),
                                         records_.dense_cols)
    };
    return ForwardNeighborIndex(detail::build_storage_(records_view, config_));
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

ForwardNeighborShardSummary ForwardNeighborIndex::shard_summary(std::size_t shard) const {
    if (!storage_ || shard >= storage_->shards.size()) {
        throw std::out_of_range("forward-neighbor shard index out of range");
    }
    const detail::DeviceShardStorage &src = storage_->shards[shard];
    return ForwardNeighborShardSummary{
        src.device_id,
        src.row_begin,
        src.row_end,
        src.row_end - src.row_begin,
        static_cast<std::int64_t>(src.segments.size()),
        static_cast<std::int64_t>(src.ann_lists.size()),
        src.time_begin,
        src.time_end,
        src.resident_bytes,
        src.resident
    };
}

ForwardNeighborOwnedQueryBatch ForwardNeighborIndex::query_batch_from_cell_indices(const std::int64_t *cell_indices, std::size_t cell_count) const {
    if (!storage_) return ForwardNeighborOwnedQueryBatch{};
    ForwardNeighborOwnedQueryBatch query;
    query.cell_indices.assign_copy(cell_indices, cell_count);
    query.developmental_time.assign_fill(cell_count, detail::quiet_nan_());
    query.embryo_ids.assign_fill(cell_count, static_cast<std::int64_t>(-1));
    query.dense_cols = storage_->latent_dim;
    query.dense_values.assign_fill(cell_count * static_cast<std::size_t>(storage_->latent_dim), 0.0f);

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
            query.dense_values.data() + row * static_cast<std::size_t>(storage_->latent_dim),
            shard.latent_unit_cpu.data() + local_row * static_cast<std::size_t>(storage_->latent_dim),
            static_cast<std::size_t>(storage_->latent_dim) * sizeof(float));
    }

    return query;
}

ForwardNeighborRoutingPlan ForwardNeighborIndex::plan_future_neighbor_routes(
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config) const {
    if (!storage_) return ForwardNeighborRoutingPlan{};
    detail::validate_forward_neighbor_search_config_(config);
    detail::validate_forward_neighbor_query_batch_(query);
    return detail::build_routing_plan_(*storage_, query, config);
}

ForwardNeighborRoutingPlan ForwardNeighborIndex::plan_future_neighbor_routes_by_cell_index(
    const std::int64_t *cell_indices,
    std::size_t cell_count,
    const ForwardNeighborSearchConfig &config) const {
    const ForwardNeighborOwnedQueryBatch query = query_batch_from_cell_indices(cell_indices, cell_count);
    return plan_future_neighbor_routes(query.view(), config);
}

ForwardNeighborSearchResult ForwardNeighborIndex::search_future_neighbors(
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config) const {
    return search_future_neighbors(query, nullptr, config);
}

ForwardNeighborSearchResult ForwardNeighborIndex::search_future_neighbors(
    const ForwardNeighborQueryBatch &query,
    ForwardNeighborSearchWorkspace *workspace,
    const ForwardNeighborSearchConfig &config) const {
    if (!storage_) return detail::empty_forward_neighbor_result_(config.top_k);
    detail::ForwardNeighborExecutorStorage executor;
    executor.config = ForwardNeighborExecutorConfig{};
    if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_first) {
        ForwardNeighborSearchConfig same_config = config;
        same_config.embryo_policy = ForwardNeighborEmbryoPolicy::same_embryo_only;
        ForwardNeighborSearchResult same_result = detail::search_core_(storage_.get(), query, same_config, true, workspace, &executor);
        ForwardNeighborSearchConfig any_config = config;
        any_config.embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
        ForwardNeighborSearchResult any_result = detail::search_core_(storage_.get(), query, any_config, false, workspace, &executor);

        for (std::int64_t row = 0; row < same_result.query_count; ++row) {
            std::size_t fill = 0u;
            detail::ForwardNeighborSearchWorkspaceStorage &scratch = detail::workspace_storage_(workspace);
            scratch.seen_neighbor_ids.clear();
            for (; fill < static_cast<std::size_t>(same_result.top_k); ++fill) {
                const std::size_t off = detail::result_offset_(row, static_cast<std::int64_t>(fill), same_result.top_k);
                if (same_result.neighbor_cell_indices[off] < 0) break;
                detail::append_one_(&scratch.seen_neighbor_ids, same_result.neighbor_cell_indices[off]);
            }
            for (std::size_t slot = 0; slot < static_cast<std::size_t>(any_result.top_k) && fill < static_cast<std::size_t>(same_result.top_k); ++slot) {
                const std::size_t src = detail::result_offset_(row, static_cast<std::int64_t>(slot), any_result.top_k);
                if (any_result.neighbor_cell_indices[src] < 0) continue;
                bool seen = false;
                for (std::size_t seen_idx = 0; seen_idx < scratch.seen_neighbor_ids.size(); ++seen_idx) {
                    if (scratch.seen_neighbor_ids[seen_idx] == any_result.neighbor_cell_indices[src]) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;
                detail::append_one_(&scratch.seen_neighbor_ids, any_result.neighbor_cell_indices[src]);
                const std::size_t dst = detail::result_offset_(row, static_cast<std::int64_t>(fill), same_result.top_k);
                same_result.neighbor_cell_indices[dst] = any_result.neighbor_cell_indices[src];
                same_result.neighbor_shard_indices[dst] = any_result.neighbor_shard_indices[src];
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
        storage_.get(),
        query,
        config,
        config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only,
        workspace,
        &executor);
}

ForwardNeighborSearchResult ForwardNeighborIndex::search_future_neighbors_by_cell_index(
    const std::int64_t *cell_indices,
    std::size_t cell_count,
    const ForwardNeighborSearchConfig &config) const {
    const ForwardNeighborOwnedQueryBatch query = query_batch_from_cell_indices(cell_indices, cell_count);
    return search_future_neighbors(query.view(), nullptr, config);
}

ForwardNeighborSearchResult ForwardNeighborIndex::search_future_neighbors_by_cell_index(
    const std::int64_t *cell_indices,
    std::size_t cell_count,
    ForwardNeighborSearchWorkspace *workspace,
    const ForwardNeighborSearchConfig &config) const {
    const ForwardNeighborOwnedQueryBatch query = query_batch_from_cell_indices(cell_indices, cell_count);
    return search_future_neighbors(query.view(), workspace, config);
}

ForwardNeighborSearchExecutor::ForwardNeighborSearchExecutor(
    ForwardNeighborExecutorConfig config)
    : storage_(std::make_unique<detail::ForwardNeighborExecutorStorage>()) {
    storage_->config = std::move(config);
}

ForwardNeighborSearchExecutor::~ForwardNeighborSearchExecutor() = default;

ForwardNeighborSearchExecutor::ForwardNeighborSearchExecutor(ForwardNeighborSearchExecutor &&) noexcept = default;

ForwardNeighborSearchExecutor &ForwardNeighborSearchExecutor::operator=(ForwardNeighborSearchExecutor &&) noexcept = default;

void ForwardNeighborSearchExecutor::reset() {
    const ForwardNeighborExecutorConfig current = storage_ ? storage_->config : ForwardNeighborExecutorConfig{};
    workspace_.reset();
    storage_ = std::make_unique<detail::ForwardNeighborExecutorStorage>();
    storage_->config = current;
}

const ForwardNeighborExecutorConfig &ForwardNeighborSearchExecutor::config() const {
    return storage_->config;
}

ForwardNeighborRoutingPlan ForwardNeighborSearchExecutor::plan_future_neighbor_routes(
    const ForwardNeighborIndex &index,
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config) const {
    return index.plan_future_neighbor_routes(query, config);
}

ForwardNeighborRoutingPlan ForwardNeighborSearchExecutor::plan_future_neighbor_routes_by_cell_index(
    const ForwardNeighborIndex &index,
    const std::int64_t *cell_indices,
    std::size_t cell_count,
    const ForwardNeighborSearchConfig &config) const {
    return index.plan_future_neighbor_routes_by_cell_index(cell_indices, cell_count, config);
}

ForwardNeighborSearchResult ForwardNeighborSearchExecutor::search_future_neighbors(
    const ForwardNeighborIndex &index,
    const ForwardNeighborQueryBatch &query,
    const ForwardNeighborSearchConfig &config) {
    if (!index.storage_) return detail::empty_forward_neighbor_result_(config.top_k);
    if (config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_first) {
        ForwardNeighborSearchConfig same_config = config;
        same_config.embryo_policy = ForwardNeighborEmbryoPolicy::same_embryo_only;
        ForwardNeighborSearchResult same_result =
            detail::search_core_(index.storage_.get(), query, same_config, true, &workspace_, storage_.get());
        ForwardNeighborSearchConfig any_config = config;
        any_config.embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
        ForwardNeighborSearchResult any_result =
            detail::search_core_(index.storage_.get(), query, any_config, false, &workspace_, storage_.get());
        detail::ForwardNeighborSearchWorkspaceStorage &scratch = detail::workspace_storage_(&workspace_);
        for (std::int64_t row = 0; row < same_result.query_count; ++row) {
            std::size_t fill = 0u;
            scratch.seen_neighbor_ids.clear();
            for (; fill < static_cast<std::size_t>(same_result.top_k); ++fill) {
                const std::size_t off = detail::result_offset_(row, static_cast<std::int64_t>(fill), same_result.top_k);
                if (same_result.neighbor_cell_indices[off] < 0) break;
                detail::append_one_(&scratch.seen_neighbor_ids, same_result.neighbor_cell_indices[off]);
            }
            for (std::size_t slot = 0; slot < static_cast<std::size_t>(any_result.top_k) && fill < static_cast<std::size_t>(same_result.top_k); ++slot) {
                const std::size_t src = detail::result_offset_(row, static_cast<std::int64_t>(slot), any_result.top_k);
                if (any_result.neighbor_cell_indices[src] < 0) continue;
                bool seen = false;
                for (std::size_t seen_idx = 0; seen_idx < scratch.seen_neighbor_ids.size(); ++seen_idx) {
                    if (scratch.seen_neighbor_ids[seen_idx] == any_result.neighbor_cell_indices[src]) {
                        seen = true;
                        break;
                    }
                }
                if (seen) continue;
                detail::append_one_(&scratch.seen_neighbor_ids, any_result.neighbor_cell_indices[src]);
                const std::size_t dst = detail::result_offset_(row, static_cast<std::int64_t>(fill), same_result.top_k);
                same_result.neighbor_cell_indices[dst] = any_result.neighbor_cell_indices[src];
                same_result.neighbor_shard_indices[dst] = any_result.neighbor_shard_indices[src];
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
        index.storage_.get(),
        query,
        config,
        config.embryo_policy == ForwardNeighborEmbryoPolicy::same_embryo_only,
        &workspace_,
        storage_.get());
}

ForwardNeighborSearchResult ForwardNeighborSearchExecutor::search_future_neighbors_by_cell_index(
    const ForwardNeighborIndex &index,
    const std::int64_t *cell_indices,
    std::size_t cell_count,
    const ForwardNeighborSearchConfig &config) {
    const ForwardNeighborOwnedQueryBatch query = index.query_batch_from_cell_indices(cell_indices, cell_count);
    return search_future_neighbors(index, query.view(), config);
}

ForwardNeighborIndex build_forward_neighbor_index(
    const ForwardNeighborRecordBatch &records,
    const ForwardNeighborBuildConfig &config) {
    ForwardNeighborIndexBuilder builder(config);
    builder.append(records);
    return std::move(builder).finalize();
}

} // namespace cellerator::compute::neighbors::forward_neighbors
