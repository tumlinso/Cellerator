#pragma once

#include "../../graph/workspace.cuh"

#include <cuda_fp16.h>

#include <cstdint>

namespace cellerator::compute::neighbors::exact_search {

struct ExactSearchQueryDeviceView {
    const __half *query_latent = nullptr;
    const float *query_lower = nullptr;
    const float *query_upper = nullptr;
    const std::int64_t *query_embryo = nullptr;
    std::int64_t query_rows = 0;
    int latent_dim = 0;
};

struct ExactSearchDenseIndexDeviceView {
    const __half *latent = nullptr;
    const float *time = nullptr;
    const std::int64_t *embryo = nullptr;
    const std::int64_t *cell = nullptr;
    std::int64_t shard_index = -1;
};

struct ExactSearchSlicedEllIndexDeviceView {
    const std::uint32_t *row_slot_offsets = nullptr;
    const std::uint32_t *row_widths = nullptr;
    const std::uint32_t *col_idx = nullptr;
    const __half *values = nullptr;
    const float *time = nullptr;
    const std::int64_t *embryo = nullptr;
    const std::int64_t *cell = nullptr;
    std::int64_t shard_index = -1;
};

struct ExactSearchResultDeviceView {
    std::int64_t *best_cell = nullptr;
    std::int64_t *best_shard = nullptr;
    float *best_time = nullptr;
    std::int64_t *best_embryo = nullptr;
    float *best_similarity = nullptr;
};

constexpr int kExactSearchMaxTopK = 32;

void init_result_arrays(
    const ExactSearchResultDeviceView &result,
    std::int64_t query_rows,
    int k);

void routed_dense_topk(
    const ExactSearchQueryDeviceView &query,
    const ExactSearchDenseIndexDeviceView &index,
    const std::int64_t *segment_begin,
    const std::int64_t *segment_end,
    std::int64_t segment_count,
    int hard_same_embryo,
    int k,
    const ExactSearchResultDeviceView &result);

void routed_sliced_ell_topk(
    const ExactSearchQueryDeviceView &query,
    const ExactSearchSlicedEllIndexDeviceView &index,
    const std::int64_t *segment_begin,
    const std::int64_t *segment_end,
    std::int64_t segment_count,
    int hard_same_embryo,
    int k,
    const ExactSearchResultDeviceView &result);

void merge_result_arrays(
    const ExactSearchResultDeviceView &local_result,
    const ExactSearchResultDeviceView &global_result,
    std::int64_t query_rows,
    int k);

} // namespace cellerator::compute::neighbors::exact_search
