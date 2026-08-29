#pragma once

#include <cuda_fp16.h>

#include <cstddef>
#include <cstdint>

namespace cellerator::compute::neighbors::forward_neighbors {

constexpr int kForwardNeighborMaxTopK = 32;
constexpr int kForwardNeighborMaxProbeCount = 32;

enum class ForwardNeighborEmbryoPolicy : std::uint8_t {
    any_embryo = 0,
    same_embryo_only = 1
};

struct ForwardNeighborQueryDeviceView {
    const __half *latent = nullptr;
    const float *window_lower = nullptr;
    const float *window_upper = nullptr;
    const std::int64_t *embryo_ids = nullptr;
    std::int64_t rows = 0;
    int latent_dim = 0;
};

struct ForwardNeighborAnnListDeviceView {
    const float *centroids = nullptr;
    const std::int64_t *embryo_ids = nullptr;
    const std::int64_t *row_begin = nullptr;
    const std::int64_t *row_end = nullptr;
    std::int64_t count = 0;
    int latent_dim = 0;
};

struct ForwardNeighborDenseIndexDeviceView {
    const __half *latent = nullptr;
    const float *developmental_time = nullptr;
    const std::int64_t *embryo_ids = nullptr;
    const std::int64_t *cell_indices = nullptr;
    std::int64_t rows = 0;
    int latent_dim = 0;
    std::int64_t shard_index = -1;
};

struct ForwardNeighborBlockedEllIndexDeviceView {
    const std::uint32_t *block_col_indices = nullptr;
    const __half *values = nullptr;
    const float *developmental_time = nullptr;
    const std::int64_t *embryo_ids = nullptr;
    const std::int64_t *cell_indices = nullptr;
    std::int64_t rows = 0;
    int latent_dim = 0;
    int block_size = 0;
    int ell_cols = 0;
    std::int64_t shard_index = -1;
};

struct ForwardNeighborSlicedEllIndexDeviceView {
    const std::uint32_t *row_slot_offsets = nullptr;
    const std::uint32_t *row_widths = nullptr;
    const std::uint32_t *col_indices = nullptr;
    const __half *values = nullptr;
    const float *developmental_time = nullptr;
    const std::int64_t *embryo_ids = nullptr;
    const std::int64_t *cell_indices = nullptr;
    std::int64_t rows = 0;
    int latent_dim = 0;
    std::int64_t shard_index = -1;
};

struct ForwardNeighborResultDeviceView {
    std::int64_t *cell_indices = nullptr;
    std::int64_t *shard_indices = nullptr;
    float *developmental_time = nullptr;
    std::int64_t *embryo_ids = nullptr;
    float *similarity = nullptr;
    std::size_t capacity = 0;
};

struct ForwardNeighborAnnWorkspaceDeviceView {
    std::int32_t *selected_list_offsets = nullptr;
    std::size_t selected_list_capacity = 0;
};

struct ForwardNeighborAnnSearchConfig {
    int probe_count = 8;
    int top_k = 15;
    ForwardNeighborEmbryoPolicy embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
};

std::size_t forward_neighbor_result_elements(std::int64_t query_rows, int top_k);
std::size_t forward_neighbor_ann_workspace_elements(std::int64_t query_rows, int probe_count);

} // namespace cellerator::compute::neighbors::forward_neighbors
