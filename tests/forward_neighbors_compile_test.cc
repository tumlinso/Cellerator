#include "../src/models/forward_neighbors/forwardNeighbors.hh"

#include <torch/torch.h>

#include <cmath>

int main() {
    using namespace cellerator::models::forward_neighbors;

    ForwardNeighborRecordBatch records{
        torch::tensor({10, 11, 12, 13, 14, 15}, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor({0.10f, 0.16f, 0.30f, 0.12f, 0.18f, 0.40f}, torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor(
            {
                {1.00f, 0.00f},
                {0.98f, 0.02f},
                {0.10f, 0.90f},
                {0.85f, 0.15f},
                {0.70f, 0.30f},
                {0.00f, 1.00f},
            },
            torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor({0, 0, 0, 1, 1, 1}, torch::TensorOptions().dtype(torch::kInt64))
    };

    ForwardNeighborBuildConfig build_config;
    build_config.target_shard_count = 2;
    build_config.max_rows_per_segment = 3;
    build_config.ann_rows_per_list = 1;
    ForwardNeighborIndex index = build_forward_neighbor_index(records, build_config);

    ForwardNeighborSearchConfig exact_config;
    exact_config.backend = ForwardNeighborBackend::exact_windowed;
    exact_config.embryo_policy = ForwardNeighborEmbryoPolicy::same_embryo_only;
    exact_config.top_k = 2;
    exact_config.candidate_k = 3;
    exact_config.query_block_rows = 2;
    exact_config.index_block_rows = 2;
    exact_config.time_window.min_delta = 0.0f;
    exact_config.time_window.max_delta = 0.10f;

    ForwardNeighborSearchResult exact_by_id = index.search_future_neighbors_by_cell_index(
        torch::tensor({10, 12}, torch::TensorOptions().dtype(torch::kInt64)),
        exact_config);

    ForwardNeighborSearchConfig ann_config = exact_config;
    ann_config.backend = ForwardNeighborBackend::ann_windowed;
    ann_config.embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
    ann_config.ann_probe_list_count = 2;

    ForwardNeighborQueryBatch explicit_query{
        torch::tensor({10}, torch::TensorOptions().dtype(torch::kInt64)),
        torch::tensor({0.10f}, torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor({ {1.0f, 0.0f} }, torch::TensorOptions().dtype(torch::kFloat32)),
        torch::tensor({0}, torch::TensorOptions().dtype(torch::kInt64))
    };
    ForwardNeighborSearchResult ann_result = index.search_future_neighbors(explicit_query, ann_config);

    const auto exact_first_neighbor = exact_by_id.neighbor_cell_indices.index({0, 0}).item<std::int64_t>();
    const auto exact_no_neighbor = exact_by_id.neighbor_cell_indices.index({1, 0}).item<std::int64_t>();
    const auto ann_first_neighbor = ann_result.neighbor_cell_indices.index({0, 0}).item<std::int64_t>();
    const float ann_sqdist = ann_result.neighbor_sqdist.index({0, 0}).item<float>();
    const auto ann_embryo = ann_result.neighbor_embryo_ids.index({0, 0}).item<std::int64_t>();

    const bool ok = index.shard_count() == 2
        && exact_first_neighbor == 11
        && exact_no_neighbor == -1
        && ann_first_neighbor == 11
        && ann_embryo == 0
        && std::isfinite(ann_sqdist);
    return ok ? 0 : 1;
}
