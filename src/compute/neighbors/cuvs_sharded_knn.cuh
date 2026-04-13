#pragma once

#include "../../../extern/CellShard/src/CellShard.hh"
#include "host_buffer.hh"

#include <cstddef>
#include <cstdint>

namespace cellerator {
namespace compute {
namespace neighbors {

enum class metric_kind {
    l2_unexpanded,
    cosine_expanded,
    inner_product
};

struct knn_result_host {
    unsigned long rows;
    int k;
    host_buffer<std::int64_t> neighbors;
    host_buffer<float> distances;
};

struct sparse_exact_params {
    int k;
    metric_kind metric;
    float metric_arg;
    int exclude_self;
    int gpu_limit;
    int batch_size_index;
    int batch_size_query;
    int drop_host_parts_after_index_pack;
    int drop_host_parts_after_query_use;
};

struct dense_ann_params {
    int k;
    metric_kind metric;
    float metric_arg;
    int exclude_self;
    int gpu_limit;
    unsigned int n_lists;
    unsigned int n_probes;
    int use_cagra;
    unsigned int intermediate_graph_degree;
    unsigned int graph_degree;
    std::int64_t rows_per_batch;
};

struct proprietary_dense_params {
    int k;
    metric_kind metric;
    float metric_arg;
    int exclude_self;
    int gpu_limit;
    int query_block_rows;
    int index_block_rows;
};

void init(sparse_exact_params *params);
void init(dense_ann_params *params);
void init(proprietary_dense_params *params);
void init(knn_result_host *result);
void clear(knn_result_host *result);

int sparse_exact_self_knn(const ::cellshard::sharded< ::cellshard::sparse::compressed > *view,
                          const ::cellshard::shard_storage *storage,
                          const sparse_exact_params *params,
                          knn_result_host *result);

int dense_ann_self_knn(const ::cellshard::sharded< ::cellshard::dense > *view,
                       const dense_ann_params *params,
                       knn_result_host *result);

int proprietary_dense_self_knn(const ::cellshard::sharded< ::cellshard::dense > *view,
                               const proprietary_dense_params *params,
                               knn_result_host *result);

} // namespace neighbors
} // namespace compute
} // namespace cellerator
