#pragma once

#include "record_table.cuh"
#include "slab_index.cuh"
#include "workspace.cuh"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace cellerator::compute::graph {

struct ForwardNeighborConfig {
    std::uint32_t candidate_k = 4;
    std::uint32_t max_out_degree = 4;
    float min_time_delta = 0.01f;
    float max_time_delta = 0.20f;
    float strict_future_epsilon = 1.0e-6f;
    float min_similarity = -1.0f;
    float shortcut_ratio = 1.75f;
    float shortcut_similarity_gap = 0.05f;
};

struct CandidateEdgeTable {
    std::uint32_t rows = 0;
    std::uint32_t k = 0;
    std::vector<std::uint32_t> dst;
    std::vector<float> similarity;
    std::vector<float> delta_t;
};

namespace detail {

struct device_candidate {
    float similarity;
    float delta_t;
    std::uint32_t dst;
};

constexpr int max_candidate_k = 8;
constexpr int candidate_block_threads = 128;

__device__ inline void insert_candidate_device_(
    device_candidate candidate,
    device_candidate *best,
    int k) {
    if (k <= 0) return;
    if (candidate.dst == std::numeric_limits<std::uint32_t>::max()) return;
    if (candidate.similarity <= best[k - 1].similarity) return;

    int insert = k - 1;
    while (insert > 0 && candidate.similarity > best[insert - 1].similarity) {
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

__global__ inline void score_forward_candidates_kernel_(
    const __half *latent,
    const float *time,
    const std::uint32_t *window_begin,
    const std::uint32_t *window_end,
    std::uint32_t rows,
    int latent_dim,
    int top_k,
    float min_similarity,
    std::uint32_t *out_dst,
    float *out_similarity,
    float *out_dt) {
    const std::uint32_t row = static_cast<std::uint32_t>(blockIdx.x);
    if (row >= rows) return;

    const __half *query = latent + static_cast<std::size_t>(row) * static_cast<std::size_t>(latent_dim);
    const std::uint32_t begin = window_begin[row];
    const std::uint32_t end = window_end[row];
    const float query_time = time[row];

    device_candidate local_best[max_candidate_k];
    for (int i = 0; i < top_k; ++i) {
        local_best[i] = device_candidate{
            -INFINITY,
            INFINITY,
            std::numeric_limits<std::uint32_t>::max()
        };
    }

    for (std::uint32_t candidate = begin + static_cast<std::uint32_t>(threadIdx.x); candidate < end; candidate += blockDim.x) {
        const __half *target = latent + static_cast<std::size_t>(candidate) * static_cast<std::size_t>(latent_dim);
        const float similarity = dot_half_rows_(query, target, latent_dim);
        if (!isfinite(similarity) || similarity < min_similarity) continue;
        insert_candidate_device_(device_candidate{
            similarity,
            time[candidate] - query_time,
            candidate
        }, local_best, top_k);
    }

    __shared__ device_candidate shared[candidate_block_threads * max_candidate_k];
    device_candidate *thread_out = shared + threadIdx.x * max_candidate_k;
    for (int i = 0; i < top_k; ++i) thread_out[i] = local_best[i];
    __syncthreads();

    if (threadIdx.x == 0) {
        device_candidate block_best[max_candidate_k];
        for (int i = 0; i < top_k; ++i) {
            block_best[i] = device_candidate{
                -INFINITY,
                INFINITY,
                std::numeric_limits<std::uint32_t>::max()
            };
        }
        for (int lane = 0; lane < blockDim.x; ++lane) {
            const device_candidate *lane_best = shared + lane * max_candidate_k;
            for (int i = 0; i < top_k; ++i) insert_candidate_device_(lane_best[i], block_best, top_k);
        }

        for (int i = 0; i < top_k; ++i) {
            const std::size_t off = static_cast<std::size_t>(row) * static_cast<std::size_t>(top_k) + static_cast<std::size_t>(i);
            out_dst[off] = block_best[i].dst;
            out_similarity[off] = block_best[i].similarity;
            out_dt[off] = block_best[i].delta_t;
        }
    }
}

} // namespace detail

inline CandidateEdgeTable build_forward_candidates_cuda(
    const TrajectoryRecordTable &table,
    const FutureWindowBounds &bounds,
    const ForwardNeighborConfig &config = ForwardNeighborConfig()) {
    table.validate();
    if (bounds.row_begin.size() != table.rows || bounds.row_end.size() != table.rows) {
        throw std::invalid_argument("FutureWindowBounds must align with the record table");
    }
    if (config.candidate_k == 0 || config.candidate_k > detail::max_candidate_k) {
        throw std::invalid_argument("candidate_k must be in [1, 8]");
    }

    CandidateEdgeTable result;
    result.rows = table.rows;
    result.k = config.candidate_k;
    result.dst.assign(static_cast<std::size_t>(table.rows) * static_cast<std::size_t>(config.candidate_k),
                      std::numeric_limits<std::uint32_t>::max());
    result.similarity.assign(result.dst.size(), -std::numeric_limits<float>::infinity());
    result.delta_t.assign(result.dst.size(), std::numeric_limits<float>::infinity());
    if (table.rows == 0) return result;

    device_buffer<__half> d_latent(table.latent.size());
    device_buffer<float> d_time(table.developmental_time.size());
    device_buffer<std::uint32_t> d_begin(bounds.row_begin.size());
    device_buffer<std::uint32_t> d_end(bounds.row_end.size());
    device_buffer<std::uint32_t> d_dst(result.dst.size());
    device_buffer<float> d_similarity(result.similarity.size());
    device_buffer<float> d_dt(result.delta_t.size());

    d_latent.upload(table.latent.data(), table.latent.size());
    d_time.upload(table.developmental_time.data(), table.developmental_time.size());
    d_begin.upload(bounds.row_begin.data(), bounds.row_begin.size());
    d_end.upload(bounds.row_end.data(), bounds.row_end.size());

    // One block per anchor row keeps the interface simple, but on small tables the fixed launch plus full-device sync cost dominates.
    detail::score_forward_candidates_kernel_<<<table.rows, detail::candidate_block_threads>>>(
        d_latent.data(),
        d_time.data(),
        d_begin.data(),
        d_end.data(),
        table.rows,
        table.latent_dim,
        static_cast<int>(config.candidate_k),
        config.min_similarity,
        d_dst.data(),
        d_similarity.data(),
        d_dt.data());
    cuda_require(cudaGetLastError(), "score_forward_candidates_kernel launch");
    // This explicit sync makes the helper easy to consume, but it prevents overlap with later transfers or kernels.
    cuda_require(cudaDeviceSynchronize(), "score_forward_candidates_kernel sync");

    // These downloads are the full candidate table, so the helper is intentionally batch-oriented rather than row-at-a-time.
    d_dst.download(result.dst.data(), result.dst.size());
    d_similarity.download(result.similarity.data(), result.similarity.size());
    d_dt.download(result.delta_t.data(), result.delta_t.size());
    return result;
}

} // namespace cellerator::compute::graph
