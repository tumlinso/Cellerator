#include <Cellerator/compute/neighbors/forward_neighbors.hh>

#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>

int main() {
    using namespace cellerator::compute::neighbors::forward_neighbors;
    namespace css = ::cellshard::sparse;

    auto populate_blocked_ell = [](css::blocked_ell *matrix,
                                   const std::vector<float> &dense,
                                   unsigned int rows,
                                   unsigned int cols) {
        css::init(matrix, rows, cols, rows * cols, 1u, cols);
        if (!css::allocate(matrix)) return false;
        for (unsigned int row = 0; row < rows; ++row) {
            for (unsigned int col = 0; col < cols; ++col) {
                matrix->blockColIdx[static_cast<std::size_t>(row) * cols + col] = col;
                matrix->val[static_cast<std::size_t>(row) * cols + col] =
                    __float2half_rn(dense[static_cast<std::size_t>(row) * cols + col]);
            }
        }
        return true;
    };

    auto populate_sliced_ell = [](css::sliced_ell *matrix,
                                  const std::vector<float> &dense,
                                  unsigned int rows,
                                  unsigned int cols) {
        std::vector<unsigned int> slice_offsets(rows + 1u, 0u);
        std::vector<unsigned int> slice_widths(rows, cols);
        for (unsigned int row = 0; row <= rows; ++row) slice_offsets[row] = row;
        css::init(matrix, rows, cols, rows * cols);
        if (!css::allocate(matrix, rows, slice_offsets.data(), slice_widths.data())) return false;
        for (unsigned int row = 0; row < rows; ++row) {
            const std::size_t slot_base = static_cast<std::size_t>(row) * cols;
            for (unsigned int col = 0; col < cols; ++col) {
                matrix->col_idx[slot_base + col] = col;
                matrix->val[slot_base + col] = __float2half_rn(dense[slot_base + col]);
            }
        }
        return true;
    };

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        std::fprintf(stderr, "forwardNeighborsCompileTest: cudaGetDeviceCount failed\n");
        return 1;
    }
    if (device_count <= 0) {
        std::fprintf(stderr, "forwardNeighborsCompileTest requires at least one visible CUDA device\n");
        return 1;
    }

    ForwardNeighborOwnedRecordBatch records;
    records.cell_indices = {10, 11, 12, 13, 14, 15};
    records.developmental_time = {0.10f, 0.16f, 0.30f, 0.12f, 0.18f, 0.40f};
    records.dense_values = {
        1.00f, 0.00f,
        0.98f, 0.02f,
        0.10f, 0.90f,
        0.85f, 0.15f,
        0.70f, 0.30f,
        0.00f, 1.00f,
    };
    records.embryo_ids = {0, 0, 0, 1, 1, 1};
    records.dense_cols = 2;

    ForwardNeighborBuildConfig build_config;
    build_config.target_shard_count = 4;
    build_config.max_rows_per_segment = 2;
    build_config.ann_rows_per_list = 1;
    ForwardNeighborIndex index = build_forward_neighbor_index(records.view(), build_config);

    const std::vector<float> dense_values{
        1.00f, 0.00f,
        0.98f, 0.02f,
        0.10f, 0.90f,
        0.85f, 0.15f,
        0.70f, 0.30f,
        0.00f, 1.00f,
    };
    css::blocked_ell blocked_records_matrix{};
    css::blocked_ell blocked_query_matrix{};
    css::sliced_ell sliced_records_matrix{};
    css::sliced_ell sliced_query_matrix{};
    if (!populate_blocked_ell(&blocked_records_matrix, dense_values, 6u, 2u)
        || !populate_blocked_ell(&blocked_query_matrix, {1.0f, 0.0f}, 1u, 2u)
        || !populate_sliced_ell(&sliced_records_matrix, dense_values, 6u, 2u)
        || !populate_sliced_ell(&sliced_query_matrix, {1.0f, 0.0f}, 1u, 2u)) {
        return 1;
    }

    ForwardNeighborRecordBatch blocked_records{
        view_of(records.cell_indices),
        view_of(records.developmental_time),
        view_of(records.embryo_ids),
        make_forward_neighbor_blocked_ell_view(&blocked_records_matrix)
    };
    ForwardNeighborRecordBatch sliced_records{
        view_of(records.cell_indices),
        view_of(records.developmental_time),
        view_of(records.embryo_ids),
        make_forward_neighbor_sliced_ell_view(&sliced_records_matrix)
    };
    ForwardNeighborIndex blocked_index = build_forward_neighbor_index(blocked_records, build_config);
    ForwardNeighborIndex sliced_index = build_forward_neighbor_index(sliced_records, build_config);

    ForwardNeighborSearchConfig exact_config;
    exact_config.backend = ForwardNeighborBackend::exact_windowed;
    exact_config.embryo_policy = ForwardNeighborEmbryoPolicy::same_embryo_only;
    exact_config.top_k = 2;
    exact_config.candidate_k = 3;
    exact_config.query_block_rows = 2;
    exact_config.index_block_rows = 2;
    exact_config.time_window.min_delta = 0.0f;
    exact_config.time_window.max_delta = 0.10f;

    ForwardNeighborOwnedQueryBatch routed_query;
    routed_query.cell_indices = {10};
    routed_query.developmental_time = {0.10f};
    routed_query.dense_values = {1.0f, 0.0f};
    routed_query.embryo_ids = {0};
    routed_query.dense_cols = 2;
    const ForwardNeighborRoutingPlan routing = index.plan_future_neighbor_routes(routed_query.view(), exact_config);
    const ForwardNeighborQueryBatch blocked_query{
        view_of(routed_query.cell_indices),
        view_of(routed_query.developmental_time),
        view_of(routed_query.embryo_ids),
        make_forward_neighbor_blocked_ell_view(&blocked_query_matrix)
    };
    const ForwardNeighborQueryBatch sliced_query{
        view_of(routed_query.cell_indices),
        view_of(routed_query.developmental_time),
        view_of(routed_query.embryo_ids),
        make_forward_neighbor_sliced_ell_view(&sliced_query_matrix)
    };
    const ForwardNeighborShardSummary shard0 = index.shard_summary(0u);

    const std::int64_t exact_query_ids[] = {10, 12};
    ForwardNeighborSearchWorkspace workspace;
    ForwardNeighborSearchResult exact_by_id = index.search_future_neighbors_by_cell_index(
        exact_query_ids,
        2u,
        &workspace,
        exact_config);

    ForwardNeighborSearchExecutor executor;
    ForwardNeighborSearchResult exact_via_executor = executor.search_future_neighbors(index, routed_query.view(), exact_config);

    ForwardNeighborSearchConfig ann_config = exact_config;
    ann_config.backend = ForwardNeighborBackend::ann_windowed;
    ann_config.embryo_policy = ForwardNeighborEmbryoPolicy::any_embryo;
    ann_config.ann_probe_list_count = 2;

    ForwardNeighborSearchResult ann_result = index.search_future_neighbors(routed_query.view(), &workspace, ann_config);
    ForwardNeighborSearchResult ann_repeat = index.search_future_neighbors(routed_query.view(), &workspace, ann_config);
    ForwardNeighborSearchResult blocked_exact = blocked_index.search_future_neighbors(blocked_query, exact_config);
    ForwardNeighborSearchResult sliced_exact = sliced_index.search_future_neighbors(sliced_query, exact_config);

    const auto exact_first_neighbor = exact_by_id.neighbor_cell_indices[detail::result_offset_(0, 0, exact_by_id.top_k)];
    const auto exact_no_neighbor = exact_by_id.neighbor_cell_indices[detail::result_offset_(1, 0, exact_by_id.top_k)];
    const auto exact_executor_neighbor = exact_via_executor.neighbor_cell_indices[detail::result_offset_(0, 0, exact_via_executor.top_k)];
    const auto exact_executor_shard = exact_via_executor.neighbor_shard_indices[detail::result_offset_(0, 0, exact_via_executor.top_k)];
    const auto ann_first_neighbor = ann_result.neighbor_cell_indices[detail::result_offset_(0, 0, ann_result.top_k)];
    const auto ann_repeat_neighbor = ann_repeat.neighbor_cell_indices[detail::result_offset_(0, 0, ann_repeat.top_k)];
    const float ann_sqdist = ann_result.neighbor_sqdist[detail::result_offset_(0, 0, ann_result.top_k)];
    const auto ann_embryo = ann_result.neighbor_embryo_ids[detail::result_offset_(0, 0, ann_result.top_k)];
    const auto blocked_neighbor = blocked_exact.neighbor_cell_indices[detail::result_offset_(0, 0, blocked_exact.top_k)];
    const auto sliced_neighbor = sliced_exact.neighbor_cell_indices[detail::result_offset_(0, 0, sliced_exact.top_k)];
    const std::size_t routed_shards = routing.block_route_offsets.size() >= 2u
        ? static_cast<std::size_t>(routing.block_route_offsets[1] - routing.block_route_offsets[0])
        : 0u;

    const bool ok = index.shard_count() >= 1u
        && index.shard_count() <= 4u
        && shard0.rows > 0
        && shard0.time_begin <= shard0.time_end
        && routing.block_query_begin.size() == 1u
        && routed_shards >= 1u
        && exact_first_neighbor == 11
        && exact_no_neighbor == -1
        && exact_executor_neighbor == 11
        && exact_executor_shard >= 0
        && ann_first_neighbor == 11
        && ann_repeat_neighbor == 11
        && blocked_neighbor == 11
        && sliced_neighbor == 11
        && ann_embryo == 0
        && std::isfinite(ann_sqdist);
    css::clear(&blocked_records_matrix);
    css::clear(&blocked_query_matrix);
    css::clear(&sliced_records_matrix);
    css::clear(&sliced_query_matrix);
    return ok ? 0 : 1;
}
