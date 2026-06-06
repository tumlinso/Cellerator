#include "exact_search.hh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace cellerator::compute::neighbors::exact_search {

namespace detail {

namespace cg = ::cellerator::compute::graph;

constexpr int kWarpThreads = 32;
constexpr int kWarpsPerBlock = 1;
constexpr int kThreadsPerBlock = kWarpThreads * kWarpsPerBlock;

struct Candidate {
    float similarity = -INFINITY;
    float developmental_time = INFINITY;
    std::int64_t embryo_id = -1;
    std::int64_t cell_index = -1;
    std::int64_t shard_index = -1;
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

__device__ inline void init_candidates_device_(Candidate *best, int k) {
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

__device__ inline float dot_query_sliced_row_(
    const __half *query_row,
    const std::uint32_t *row_slot_offsets,
    const std::uint32_t *row_widths,
    const std::uint32_t *col_idx,
    const __half *values,
    std::int64_t index_row) {
    float sum = 0.0f;
    const std::size_t slot_begin = static_cast<std::size_t>(row_slot_offsets[index_row]);
    const std::uint32_t width = row_widths[index_row];
    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::size_t off = slot_begin + static_cast<std::size_t>(slot);
        sum += __half2float(query_row[col_idx[off]]) * __half2float(values[off]);
    }
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

inline std::int64_t query_blocks_for_rows_(std::int64_t query_rows) {
    return (query_rows + static_cast<std::int64_t>(kWarpsPerBlock) - 1) / static_cast<std::int64_t>(kWarpsPerBlock);
}

inline void validate_common_(const ExactSearchQueryDeviceView &query, std::int64_t segment_count, int k) {
    if (query.query_rows < 0) throw std::invalid_argument("query_rows must be >= 0");
    if (query.latent_dim <= 0) throw std::invalid_argument("latent_dim must be > 0");
    if (segment_count < 0) throw std::invalid_argument("segment_count must be >= 0");
    if (k <= 0 || k > kExactSearchMaxTopK) {
        throw std::invalid_argument("exact_search top-k exceeds native limit");
    }
    if (query.query_rows > 0 && (query.query_latent == nullptr || query.query_lower == nullptr || query.query_upper == nullptr)) {
        throw std::invalid_argument("query view is missing required device buffers");
    }
}

inline void validate_result_(const ExactSearchResultDeviceView &result) {
    if (result.best_cell == nullptr || result.best_shard == nullptr || result.best_time == nullptr
        || result.best_embryo == nullptr || result.best_similarity == nullptr) {
        throw std::invalid_argument("result view is missing required device buffers");
    }
}

__global__ void init_best_kernel_(
    std::int64_t *best_cell,
    std::int64_t *best_shard,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity,
    std::int64_t query_rows,
    int k) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t total = query_rows * static_cast<std::int64_t>(k);
    if (index >= total) return;
    best_cell[index] = -1;
    best_shard[index] = -1;
    best_time[index] = INFINITY;
    best_embryo[index] = -1;
    best_similarity[index] = -INFINITY;
}

__global__ void routed_dense_topk_kernel_(
    const __half *query_latent,
    const float *query_lower,
    const float *query_upper,
    const std::int64_t *query_embryo,
    const __half *index_latent,
    const float *index_time,
    const std::int64_t *index_embryo,
    const std::int64_t *index_cell,
    const std::int64_t *segment_begin,
    const std::int64_t *segment_end,
    std::int64_t segment_count,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t shard_index,
    int hard_same_embryo,
    int k,
    std::int64_t *best_cell,
    std::int64_t *best_shard,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity) {
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(kWarpsPerBlock)
        + static_cast<std::int64_t>(warp);
    if (row >= query_rows) return;
    const unsigned warp_mask = __activemask();
    __shared__ Candidate merged_shared[kWarpsPerBlock * kExactSearchMaxTopK];
    Candidate *merged = merged_shared + warp * kExactSearchMaxTopK;

    Candidate local_best[kExactSearchMaxTopK];
    init_candidates_device_(local_best, k);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const float lower = query_lower[row];
    const float upper = query_upper[row];
    const std::int64_t embryo = query_embryo != nullptr ? query_embryo[row] : static_cast<std::int64_t>(-1);

    for (std::int64_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const std::int64_t begin = segment_begin[segment_idx];
        const std::int64_t end = segment_end[segment_idx];
        for (std::int64_t index_row = begin + lane; index_row < end; index_row += kWarpThreads) {
            const float time = index_time[index_row];
            if (!(time > lower)) continue;
            if (isfinite(upper) && time > upper) continue;
            if (hard_same_embryo && embryo >= 0 && index_embryo[index_row] != embryo) continue;
            insert_candidate_device_(Candidate{
                dot_half_rows_(query_row, index_latent + index_row * static_cast<std::int64_t>(latent_dim), latent_dim),
                time,
                index_embryo[index_row],
                index_cell[index_row],
                shard_index
            }, local_best, k);
        }
    }

    if (lane == 0) init_candidates_device_(merged, k);
    for (int src_lane = 0; src_lane < kWarpThreads; ++src_lane) {
        for (int i = 0; i < k; ++i) {
            const Candidate candidate = shfl_candidate_device_(warp_mask, local_best[i], src_lane);
            if (lane == 0) insert_candidate_device_(candidate, merged, k);
        }
    }
    if (lane == 0) {
        const std::int64_t row_base = row * static_cast<std::int64_t>(k);
        for (int i = 0; i < k; ++i) {
            best_similarity[row_base + i] = merged[i].similarity;
            best_time[row_base + i] = merged[i].developmental_time;
            best_embryo[row_base + i] = merged[i].embryo_id;
            best_cell[row_base + i] = merged[i].cell_index;
            best_shard[row_base + i] = merged[i].shard_index;
        }
    }
}

__global__ void routed_sliced_topk_kernel_(
    const __half *query_latent,
    const float *query_lower,
    const float *query_upper,
    const std::int64_t *query_embryo,
    const std::uint32_t *row_slot_offsets,
    const std::uint32_t *row_widths,
    const std::uint32_t *col_idx,
    const __half *values,
    const float *index_time,
    const std::int64_t *index_embryo,
    const std::int64_t *index_cell,
    const std::int64_t *segment_begin,
    const std::int64_t *segment_end,
    std::int64_t segment_count,
    std::int64_t query_rows,
    int latent_dim,
    std::int64_t shard_index,
    int hard_same_embryo,
    int k,
    std::int64_t *best_cell,
    std::int64_t *best_shard,
    float *best_time,
    std::int64_t *best_embryo,
    float *best_similarity) {
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(kWarpsPerBlock)
        + static_cast<std::int64_t>(warp);
    if (row >= query_rows) return;
    const unsigned warp_mask = __activemask();
    __shared__ Candidate merged_shared[kWarpsPerBlock * kExactSearchMaxTopK];
    Candidate *merged = merged_shared + warp * kExactSearchMaxTopK;

    Candidate local_best[kExactSearchMaxTopK];
    init_candidates_device_(local_best, k);

    const __half *query_row = query_latent + row * static_cast<std::int64_t>(latent_dim);
    const float lower = query_lower[row];
    const float upper = query_upper[row];
    const std::int64_t embryo = query_embryo != nullptr ? query_embryo[row] : static_cast<std::int64_t>(-1);

    for (std::int64_t segment_idx = 0; segment_idx < segment_count; ++segment_idx) {
        const std::int64_t begin = segment_begin[segment_idx];
        const std::int64_t end = segment_end[segment_idx];
        for (std::int64_t index_row = begin + lane; index_row < end; index_row += kWarpThreads) {
            const float time = index_time[index_row];
            if (!(time > lower)) continue;
            if (isfinite(upper) && time > upper) continue;
            if (hard_same_embryo && embryo >= 0 && index_embryo[index_row] != embryo) continue;
            insert_candidate_device_(Candidate{
                dot_query_sliced_row_(query_row, row_slot_offsets, row_widths, col_idx, values, index_row),
                time,
                index_embryo[index_row],
                index_cell[index_row],
                shard_index
            }, local_best, k);
        }
    }

    if (lane == 0) init_candidates_device_(merged, k);
    for (int src_lane = 0; src_lane < kWarpThreads; ++src_lane) {
        for (int i = 0; i < k; ++i) {
            const Candidate candidate = shfl_candidate_device_(warp_mask, local_best[i], src_lane);
            if (lane == 0) insert_candidate_device_(candidate, merged, k);
        }
    }
    if (lane == 0) {
        const std::int64_t row_base = row * static_cast<std::int64_t>(k);
        for (int i = 0; i < k; ++i) {
            best_similarity[row_base + i] = merged[i].similarity;
            best_time[row_base + i] = merged[i].developmental_time;
            best_embryo[row_base + i] = merged[i].embryo_id;
            best_cell[row_base + i] = merged[i].cell_index;
            best_shard[row_base + i] = merged[i].shard_index;
        }
    }
}

__global__ void merge_topk_kernel_(
    const std::int64_t *local_cell,
    const std::int64_t *local_shard,
    const float *local_time,
    const std::int64_t *local_embryo,
    const float *local_similarity,
    std::int64_t *global_cell,
    std::int64_t *global_shard,
    float *global_time,
    std::int64_t *global_embryo,
    float *global_similarity,
    std::int64_t query_rows,
    int k) {
    const int warp = threadIdx.x / kWarpThreads;
    const int lane = threadIdx.x & (kWarpThreads - 1);
    const std::int64_t row = static_cast<std::int64_t>(blockIdx.x) * static_cast<std::int64_t>(kWarpsPerBlock)
        + static_cast<std::int64_t>(warp);
    if (row >= query_rows || lane != 0) return;

    Candidate merged[kExactSearchMaxTopK];
    init_candidates_device_(merged, k);
    const std::int64_t row_base = row * static_cast<std::int64_t>(k);
    for (int i = 0; i < k; ++i) {
        insert_candidate_device_(Candidate{
            global_similarity[row_base + i],
            global_time[row_base + i],
            global_embryo[row_base + i],
            global_cell[row_base + i],
            global_shard[row_base + i]
        }, merged, k);
    }
    for (int i = 0; i < k; ++i) {
        insert_candidate_device_(Candidate{
            local_similarity[row_base + i],
            local_time[row_base + i],
            local_embryo[row_base + i],
            local_cell[row_base + i],
            local_shard[row_base + i]
        }, merged, k);
    }
    for (int i = 0; i < k; ++i) {
        global_similarity[row_base + i] = merged[i].similarity;
        global_time[row_base + i] = merged[i].developmental_time;
        global_embryo[row_base + i] = merged[i].embryo_id;
        global_cell[row_base + i] = merged[i].cell_index;
        global_shard[row_base + i] = merged[i].shard_index;
    }
}

} // namespace detail

void init_result_arrays(
    const ExactSearchResultDeviceView &result,
    std::int64_t query_rows,
    int k) {
    detail::validate_result_(result);
    if (query_rows == 0) return;
    if (k <= 0 || k > kExactSearchMaxTopK) {
        throw std::invalid_argument("exact_search top-k exceeds native limit");
    }
    const std::size_t total = static_cast<std::size_t>(query_rows) * static_cast<std::size_t>(k);
    const int threads = 128;
    const int blocks = static_cast<int>((total + static_cast<std::size_t>(threads) - 1u) / static_cast<std::size_t>(threads));
    detail::init_best_kernel_<<<blocks, threads>>>(
        result.best_cell,
        result.best_shard,
        result.best_time,
        result.best_embryo,
        result.best_similarity,
        query_rows,
        k);
    detail::cg::cuda_require(cudaGetLastError(), "exact_search init_best_kernel launch");
}

void routed_dense_topk(
    const ExactSearchQueryDeviceView &query,
    const ExactSearchDenseIndexDeviceView &index,
    const std::int64_t *segment_begin,
    const std::int64_t *segment_end,
    std::int64_t segment_count,
    int hard_same_embryo,
    int k,
    const ExactSearchResultDeviceView &result) {
    detail::validate_common_(query, segment_count, k);
    detail::validate_result_(result);
    if (query.query_rows == 0 || segment_count == 0) return;
    if (index.latent == nullptr || index.time == nullptr || index.embryo == nullptr || index.cell == nullptr
        || segment_begin == nullptr || segment_end == nullptr) {
        throw std::invalid_argument("dense exact-search view is missing required device buffers");
    }
    detail::routed_dense_topk_kernel_<<<detail::query_blocks_for_rows_(query.query_rows), detail::kThreadsPerBlock>>>(
        query.query_latent,
        query.query_lower,
        query.query_upper,
        query.query_embryo,
        index.latent,
        index.time,
        index.embryo,
        index.cell,
        segment_begin,
        segment_end,
        segment_count,
        query.query_rows,
        query.latent_dim,
        index.shard_index,
        hard_same_embryo,
        k,
        result.best_cell,
        result.best_shard,
        result.best_time,
        result.best_embryo,
        result.best_similarity);
    detail::cg::cuda_require(cudaGetLastError(), "exact_search routed_dense_topk launch");
}

void routed_sliced_ell_topk(
    const ExactSearchQueryDeviceView &query,
    const ExactSearchSlicedEllIndexDeviceView &index,
    const std::int64_t *segment_begin,
    const std::int64_t *segment_end,
    std::int64_t segment_count,
    int hard_same_embryo,
    int k,
    const ExactSearchResultDeviceView &result) {
    detail::validate_common_(query, segment_count, k);
    detail::validate_result_(result);
    if (query.query_rows == 0 || segment_count == 0) return;
    if (index.row_slot_offsets == nullptr || index.row_widths == nullptr || index.col_idx == nullptr || index.values == nullptr
        || index.time == nullptr || index.embryo == nullptr || index.cell == nullptr
        || segment_begin == nullptr || segment_end == nullptr) {
        throw std::invalid_argument("sliced exact-search view is missing required device buffers");
    }
    detail::routed_sliced_topk_kernel_<<<detail::query_blocks_for_rows_(query.query_rows), detail::kThreadsPerBlock>>>(
        query.query_latent,
        query.query_lower,
        query.query_upper,
        query.query_embryo,
        index.row_slot_offsets,
        index.row_widths,
        index.col_idx,
        index.values,
        index.time,
        index.embryo,
        index.cell,
        segment_begin,
        segment_end,
        segment_count,
        query.query_rows,
        query.latent_dim,
        index.shard_index,
        hard_same_embryo,
        k,
        result.best_cell,
        result.best_shard,
        result.best_time,
        result.best_embryo,
        result.best_similarity);
    detail::cg::cuda_require(cudaGetLastError(), "exact_search routed_sliced_ell_topk launch");
}

void merge_result_arrays(
    const ExactSearchResultDeviceView &local_result,
    const ExactSearchResultDeviceView &global_result,
    std::int64_t query_rows,
    int k) {
    detail::validate_result_(local_result);
    detail::validate_result_(global_result);
    if (query_rows == 0) return;
    if (k <= 0 || k > kExactSearchMaxTopK) {
        throw std::invalid_argument("exact_search top-k exceeds native limit");
    }
    detail::merge_topk_kernel_<<<detail::query_blocks_for_rows_(query_rows), detail::kThreadsPerBlock>>>(
        local_result.best_cell,
        local_result.best_shard,
        local_result.best_time,
        local_result.best_embryo,
        local_result.best_similarity,
        global_result.best_cell,
        global_result.best_shard,
        global_result.best_time,
        global_result.best_embryo,
        global_result.best_similarity,
        query_rows,
        k);
    detail::cg::cuda_require(cudaGetLastError(), "exact_search merge_topk_kernel launch");
}

} // namespace cellerator::compute::neighbors::exact_search
