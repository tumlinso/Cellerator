#include "cuvs_sharded_knn.cuh"

#include "../../../extern/CellShard/include/CellShard/runtime/distributed/distributed.cuh"
#include "../../../extern/CellShard/include/CellShard/runtime/host/sharded_host.cuh"

#include <cublas_v2.h>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/brute_force.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/knn_merge_parts.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources_snmg.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/cuda_stream_view.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace cellerator {
namespace compute {
namespace neighbors {
namespace {

using ShardedCsr = ::cellshard::sharded< ::cellshard::sparse::compressed >;
using ShardedDense = ::cellshard::sharded< ::cellshard::dense >;

enum {
    proprietary_metric_l2 = 0,
    proprietary_metric_cosine = 1,
    proprietary_metric_inner_product = 2
};

struct device_sparse_csr {
    int device_id;
    int rows;
    int cols;
    int nnz;
    unsigned long row_begin;
    unsigned long row_end;
    void *allocation;
    int *row_ptr;
    int *col_idx;
    float *val;
};

struct device_dense_exact_index {
    int device_id;
    int rows;
    int cols;
    int ld;
    unsigned long row_begin;
    unsigned long row_end;
    void *allocation;
    __half *val;
    float *norms;
};

struct device_shard_plan {
    host_buffer<std::size_t> offsets;
    host_buffer<unsigned long> shards;
};

static inline void init(device_sparse_csr *m)
{
    m->device_id = -1;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->row_begin = 0;
    m->row_end = 0;
    m->allocation = 0;
    m->row_ptr = 0;
    m->col_idx = 0;
    m->val = 0;
}

static inline void init(device_dense_exact_index *m)
{
    m->device_id = -1;
    m->rows = 0;
    m->cols = 0;
    m->ld = 0;
    m->row_begin = 0;
    m->row_end = 0;
    m->allocation = 0;
    m->val = 0;
    m->norms = 0;
}

static inline void clear(device_sparse_csr *m)
{
    if (m->allocation != 0) {
        cudaSetDevice(m->device_id >= 0 ? m->device_id : 0);
        cudaFree(m->allocation);
    }
    init(m);
}

static inline void clear(device_dense_exact_index *m)
{
    if (m->allocation != 0) {
        cudaSetDevice(m->device_id >= 0 ? m->device_id : 0);
        cudaFree(m->allocation);
    }
    init(m);
}

static inline int cuda_ok(cudaError_t err, const char *label)
{
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static inline int cublas_ok(cublasStatus_t status, const char *label)
{
    if (status == CUBLAS_STATUS_SUCCESS) return 1;
    std::fprintf(stderr, "cuBLAS error at %s: %d\n", label, (int) status);
    return 0;
}

static inline int round_up_int(int value, int alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

static inline int metric_code(metric_kind metric)
{
    switch (metric) {
        case metric_kind::l2_unexpanded: return proprietary_metric_l2;
        case metric_kind::cosine_expanded: return proprietary_metric_cosine;
        case metric_kind::inner_product: return proprietary_metric_inner_product;
    }
    return proprietary_metric_l2;
}

static inline cuvs::distance::DistanceType to_cuvs_metric(metric_kind metric)
{
    switch (metric) {
        case metric_kind::l2_unexpanded: return cuvs::distance::DistanceType::L2Unexpanded;
        case metric_kind::cosine_expanded: return cuvs::distance::DistanceType::CosineExpanded;
        case metric_kind::inner_product: return cuvs::distance::DistanceType::InnerProduct;
    }
    return cuvs::distance::DistanceType::L2Unexpanded;
}

static inline int metric_selects_min(metric_kind metric)
{
    return cuvs::distance::is_min_close(to_cuvs_metric(metric)) ? 1 : 0;
}

static inline float worst_value(int select_min)
{
    return select_min ? CUDART_INF_F : -CUDART_INF_F;
}

__host__ __device__ static inline int better_value(float candidate, float current, int select_min)
{
    if (!isfinite(candidate)) return 0;
    if (!isfinite(current)) return 1;
    return select_min ? candidate < current : candidate > current;
}

static inline int active_device_ids(int gpu_limit, host_buffer<int> *out)
{
    int count = 0;
    int i = 0;

    out->clear();
    if (!cuda_ok(cudaGetDeviceCount(&count), "cudaGetDeviceCount")) return 0;
    if (count < 1) return 0;
    if (gpu_limit > 0 && gpu_limit < count) count = gpu_limit;
    out->resize((std::size_t) count);
    for (i = 0; i < count; ++i) (*out)[(std::size_t) i] = i;
    return 1;
}

static inline int ensure_sparse_shard_loaded(ShardedCsr *view,
                                             const ::cellshard::shard_storage *storage,
                                             unsigned long shard_id,
                                             int *loaded_here)
{
    *loaded_here = 0;
    if (::cellshard::shard_loaded(view, shard_id)) return 1;
    if (storage == 0) return 0;
    if (!::cellshard::fetch_shard(view, storage, shard_id)) return 0;
    *loaded_here = 1;
    return 1;
}

static inline int shard_rows(const ShardedCsr *view, unsigned long shard_id, int *rows)
{
    const unsigned long row_begin = ::cellshard::first_row_in_shard(view, shard_id);
    const unsigned long row_end = ::cellshard::last_row_in_shard(view, shard_id);
    if (row_end < row_begin) return 0;
    if (row_end - row_begin > (unsigned long) std::numeric_limits<int>::max()) return 0;
    *rows = (int) (row_end - row_begin);
    return 1;
}

static inline int shard_nnz(const ShardedCsr *view, unsigned long shard_id, int *nnz)
{
    const unsigned long total = ::cellshard::nnz_in_shard(view, shard_id);
    if (total > (unsigned long) std::numeric_limits<int>::max()) return 0;
    *nnz = (int) total;
    return 1;
}

static inline int pack_sparse_shard_to_device(ShardedCsr *view,
                                              const ::cellshard::shard_storage *storage,
                                              unsigned long shard_id,
                                              int device_id,
                                              int drop_host_after_pack,
                                              device_sparse_csr *out)
{
    const unsigned long part_begin = ::cellshard::first_part_in_shard(view, shard_id);
    const unsigned long part_end = ::cellshard::last_part_in_shard(view, shard_id);
    const unsigned long row_begin = ::cellshard::first_row_in_shard(view, shard_id);
    const unsigned long row_end = ::cellshard::last_row_in_shard(view, shard_id);
    unsigned long part = 0;
    std::size_t total_bytes = 0;
    std::size_t row_ptr_bytes = 0;
    std::size_t col_idx_bytes = 0;
    std::size_t val_bytes = 0;
    char *allocation = 0;
    int *host_row_ptr = 0;
    int *host_col_idx = 0;
    float *host_val = 0;
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int loaded_here = 0;
    unsigned long row_cursor = 0;
    unsigned long nnz_cursor = 0;

    clear(out);
    if (!ensure_sparse_shard_loaded(view, storage, shard_id, &loaded_here)) return 0;
    if (!shard_rows(view, shard_id, &rows)) goto fail;
    if (!shard_nnz(view, shard_id, &nnz)) goto fail;
    if (view->cols > (unsigned long) std::numeric_limits<int>::max()) goto fail;
    cols = (int) view->cols;

    row_ptr_bytes = (std::size_t) (rows + 1) * sizeof(int);
    col_idx_bytes = (std::size_t) nnz * sizeof(int);
    val_bytes = (std::size_t) nnz * sizeof(float);

    if (row_ptr_bytes != 0 && cudaMallocHost((void **) &host_row_ptr, row_ptr_bytes) != cudaSuccess) goto fail;
    if (col_idx_bytes != 0 && cudaMallocHost((void **) &host_col_idx, col_idx_bytes) != cudaSuccess) goto fail;
    if (val_bytes != 0 && cudaMallocHost((void **) &host_val, val_bytes) != cudaSuccess) goto fail;

    if (rows >= 0) host_row_ptr[0] = 0;
    for (part = part_begin; part < part_end; ++part) {
        const ::cellshard::sparse::compressed *src = view->parts[part];
        unsigned int local_row = 0;
        unsigned int local_nnz = 0;
        if (src == 0 || src->axis != ::cellshard::sparse::compressed_by_row) goto fail;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (local_nnz = 0; local_nnz < src->nnz; ++local_nnz) {
            host_col_idx[nnz_cursor + local_nnz] = (int) src->minorIdx[local_nnz];
            host_val[nnz_cursor + local_nnz] = __half2float(src->val[local_nnz]);
        }
        for (local_row = 0; local_row < src->rows; ++local_row) {
            host_row_ptr[row_cursor + local_row + 1ul] =
                (int) (nnz_cursor + (unsigned long) src->majorPtr[local_row + 1u]);
        }
        row_cursor += src->rows;
        nnz_cursor += src->nnz;
    }
    if (row_cursor != (unsigned long) rows || nnz_cursor != (unsigned long) nnz) goto fail;

    total_bytes = row_ptr_bytes + col_idx_bytes + val_bytes;
    if (!cuda_ok(cudaSetDevice(device_id), "cudaSetDevice(pack_sparse_shard_to_device)")) goto fail;
    if (!cuda_ok(cudaMalloc((void **) &allocation, total_bytes != 0 ? total_bytes : 1u), "cudaMalloc(pack_sparse_shard_to_device)")) goto fail;

    out->device_id = device_id;
    out->rows = rows;
    out->cols = cols;
    out->nnz = nnz;
    out->row_begin = row_begin;
    out->row_end = row_end;
    out->allocation = allocation;
    out->row_ptr = row_ptr_bytes != 0 ? (int *) allocation : 0;
    out->col_idx = col_idx_bytes != 0 ? (int *) (allocation + row_ptr_bytes) : 0;
    out->val = val_bytes != 0 ? (float *) (allocation + row_ptr_bytes + col_idx_bytes) : 0;

    if (row_ptr_bytes != 0 &&
        !cuda_ok(cudaMemcpy(out->row_ptr, host_row_ptr, row_ptr_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(row_ptr)")) goto fail;
    if (col_idx_bytes != 0 &&
        !cuda_ok(cudaMemcpy(out->col_idx, host_col_idx, col_idx_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(col_idx)")) goto fail;
    if (val_bytes != 0 &&
        !cuda_ok(cudaMemcpy(out->val, host_val, val_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(val)")) goto fail;

    if (drop_host_after_pack && loaded_here) ::cellshard::drop_shard(view, shard_id);
    if (host_row_ptr != 0) cudaFreeHost(host_row_ptr);
    if (host_col_idx != 0) cudaFreeHost(host_col_idx);
    if (host_val != 0) cudaFreeHost(host_val);
    return 1;

fail:
    if (allocation != 0) cudaFree(allocation);
    if (host_row_ptr != 0) cudaFreeHost(host_row_ptr);
    if (host_col_idx != 0) cudaFreeHost(host_col_idx);
    if (host_val != 0) cudaFreeHost(host_val);
    init(out);
    return 0;
}

static inline int pack_dense_matrix_to_host_padded(const ShardedDense *view,
                                                   int padded_cols,
                                                   host_buffer<__half> *packed)
{
    unsigned long part = 0;
    unsigned long row_cursor = 0;
    const std::size_t total = (std::size_t) view->rows * (std::size_t) padded_cols;

    packed->assign(total, __float2half(0.0f));
    for (part = 0; part < view->num_parts; ++part) {
        const ::cellshard::dense *src = view->parts[part];
        unsigned int local_row = 0;
        if (src == 0) return 0;
        if ((int) src->cols > padded_cols) return 0;
        if (padded_cols == (int) src->cols) {
            std::memcpy(packed->data() + (std::size_t) row_cursor * (std::size_t) padded_cols,
                        src->val,
                        (std::size_t) src->rows * (std::size_t) src->cols * sizeof(__half));
        } else {
            for (local_row = 0; local_row < src->rows; ++local_row) {
                std::memcpy(packed->data() + ((std::size_t) row_cursor + local_row) * (std::size_t) padded_cols,
                            src->val + (std::size_t) local_row * (std::size_t) src->cols,
                            (std::size_t) src->cols * sizeof(__half));
            }
        }
        row_cursor += src->rows;
    }
    return row_cursor == view->rows;
}

static inline int pack_dense_shard_to_host(const ShardedDense *view,
                                           unsigned long shard_id,
                                           host_buffer<__half> *packed)
{
    const unsigned long part_begin = ::cellshard::first_part_in_shard(view, shard_id);
    const unsigned long part_end = ::cellshard::last_part_in_shard(view, shard_id);
    unsigned long part = 0;
    std::size_t write_pos = 0;
    const unsigned long row_count = ::cellshard::rows_in_shard(view, shard_id);
    const std::size_t count = (std::size_t) row_count * (std::size_t) view->cols;

    packed->assign(count, __float2half(0.0f));
    for (part = part_begin; part < part_end; ++part) {
        const ::cellshard::dense *src = view->parts[part];
        const std::size_t part_count = (std::size_t) src->rows * (std::size_t) src->cols;
        if (src == 0 || src->cols != view->cols) return 0;
        std::memcpy(packed->data() + write_pos, src->val, part_count * sizeof(__half));
        write_pos += part_count;
    }
    return write_pos == count;
}

static inline int split_rows_evenly(unsigned long rows,
                                    int num_devices,
                                    host_buffer<unsigned long> *offsets)
{
    int device = 0;
    offsets->assign((std::size_t) num_devices + 1u, 0ul);
    for (device = 0; device < num_devices; ++device) {
        const unsigned long begin = (rows * (unsigned long) device) / (unsigned long) num_devices;
        const unsigned long end = (rows * (unsigned long) (device + 1)) / (unsigned long) num_devices;
        (*offsets)[(std::size_t) device] = begin;
        (*offsets)[(std::size_t) device + 1u] = end;
    }
    return 1;
}

static inline int build_device_shard_plan(unsigned long num_shards,
                                          unsigned int device_count,
                                          const int *device_slot,
                                          device_shard_plan *out)
{
    host_buffer<std::size_t> counts;
    std::size_t running = 0u;
    unsigned long shard = 0;

    out->offsets.clear();
    out->shards.clear();
    counts.assign((std::size_t) device_count, 0u);
    for (shard = 0; shard < num_shards; ++shard) {
        if (device_slot[shard] < 0 || device_slot[shard] >= (int) device_count) return 0;
        ++counts[(std::size_t) device_slot[shard]];
    }

    out->offsets.resize((std::size_t) device_count + 1u);
    out->offsets[0] = 0u;
    for (std::size_t device = 0; device < (std::size_t) device_count; ++device) {
        running += counts[device];
        out->offsets[device + 1u] = running;
    }

    out->shards.resize(running);
    counts.assign((std::size_t) device_count, 0u);
    for (shard = 0; shard < num_shards; ++shard) {
        const std::size_t device = (std::size_t) device_slot[shard];
        const std::size_t write = out->offsets[device] + counts[device];
        out->shards[write] = shard;
        ++counts[device];
    }
    return 1;
}

static inline const unsigned long *device_shards_begin(const device_shard_plan &plan, int device)
{
    return plan.shards.data() + plan.offsets[(std::size_t) device];
}

static inline std::size_t device_shards_count(const device_shard_plan &plan, int device)
{
    return plan.offsets[(std::size_t) device + 1u] - plan.offsets[(std::size_t) device];
}

static inline int upload_dense_slice_to_device(const __half *host_packed,
                                               int cols,
                                               int padded_cols,
                                               unsigned long row_begin,
                                               unsigned long row_end,
                                               int device_id,
                                               int metric,
                                               device_dense_exact_index *out)
{
    const int rows = (int) (row_end - row_begin);
    const std::size_t value_bytes = (std::size_t) rows * (std::size_t) padded_cols * sizeof(__half);
    const std::size_t norm_bytes = metric == proprietary_metric_l2 ? (std::size_t) rows * sizeof(float) : 0u;
    char *allocation = 0;

    clear(out);
    if (rows < 0) return 0;
    if (!cuda_ok(cudaSetDevice(device_id), "cudaSetDevice(upload_dense_slice_to_device)")) return 0;
    if (!cuda_ok(cudaMalloc((void **) &allocation, value_bytes + norm_bytes + 1u), "cudaMalloc(upload_dense_slice_to_device)")) return 0;

    out->device_id = device_id;
    out->rows = rows;
    out->cols = cols;
    out->ld = padded_cols;
    out->row_begin = row_begin;
    out->row_end = row_end;
    out->allocation = allocation;
    out->val = (__half *) allocation;
    out->norms = norm_bytes != 0 ? (float *) (allocation + value_bytes) : 0;

    if (value_bytes != 0 &&
        !cuda_ok(cudaMemcpy(out->val,
                            host_packed + (std::size_t) row_begin * (std::size_t) padded_cols,
                            value_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy(upload_dense_slice_to_device)")) {
        clear(out);
        return 0;
    }
    return 1;
}

static inline int merge_host_candidates(unsigned long rows,
                                        int k,
                                        int select_min,
                                        int negate_back,
                                        int source_count,
                                        const std::int64_t *candidate_ids,
                                        const float *candidate_values,
                                        std::size_t source_stride,
                                        knn_result_host *out)
{
    unsigned long row = 0;
#ifdef _OPENMP
    const int thread_count = omp_get_max_threads();
#else
    const int thread_count = 1;
#endif
    host_buffer<float> scratch_values((std::size_t) thread_count * (std::size_t) k);
    host_buffer<std::int64_t> scratch_ids((std::size_t) thread_count * (std::size_t) k);

    init(out);
    out->rows = rows;
    out->k = k;
    out->neighbors.assign((std::size_t) rows * (std::size_t) k, (std::int64_t) -1);
    out->distances.assign((std::size_t) rows * (std::size_t) k, worst_value(select_min));

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (row = 0; row < rows; ++row) {
        int thread_idx = 0;
        float *best_values = 0;
        std::int64_t *best_ids = 0;
        for (int slot = 0; slot < k; ++slot) {
            const std::size_t off = (std::size_t) row * (std::size_t) k + (std::size_t) slot;
            out->neighbors[off] = (std::int64_t) -1;
            out->distances[off] = worst_value(select_min);
        }
#ifdef _OPENMP
        thread_idx = omp_get_thread_num();
#endif
        best_values = scratch_values.data() + (std::size_t) thread_idx * (std::size_t) k;
        best_ids = scratch_ids.data() + (std::size_t) thread_idx * (std::size_t) k;
        for (int slot = 0; slot < k; ++slot) {
            best_values[(std::size_t) slot] = worst_value(select_min);
            best_ids[(std::size_t) slot] = (std::int64_t) -1;
        }
        for (int source = 0; source < source_count; ++source) {
            const std::int64_t *ids = candidate_ids + (std::size_t) source * source_stride;
            const float *values = candidate_values + (std::size_t) source * source_stride;
            int slot = 0;
            for (slot = 0; slot < k; ++slot) {
                const std::size_t off = (std::size_t) row * (std::size_t) k + (std::size_t) slot;
                const std::int64_t id = ids[off];
                const float value = values[off];
                int insert = 0;
                if (id < 0 || !std::isfinite(value)) continue;
                if (!better_value(value, best_values[(std::size_t) k - 1u], select_min)) continue;
                insert = k - 1;
                while (insert > 0 && better_value(value, best_values[(std::size_t) insert - 1u], select_min)) {
                    best_values[(std::size_t) insert] = best_values[(std::size_t) insert - 1u];
                    best_ids[(std::size_t) insert] = best_ids[(std::size_t) insert - 1u];
                    --insert;
                }
                best_values[(std::size_t) insert] = value;
                best_ids[(std::size_t) insert] = id;
            }
        }
        for (int slot = 0; slot < k; ++slot) {
            const std::size_t off = (std::size_t) row * (std::size_t) k + (std::size_t) slot;
            out->neighbors[off] = best_ids[(std::size_t) slot];
            out->distances[off] =
                negate_back && isfinite(best_values[(std::size_t) slot]) ? -best_values[(std::size_t) slot] : best_values[(std::size_t) slot];
        }
    }
    return 1;
}

static inline void filter_self_from_host_result(knn_result_host *out)
{
    unsigned long row = 0;
    for (row = 0; row < out->rows; ++row) {
        const std::size_t base = (std::size_t) row * (std::size_t) out->k;
        int write = 0;
        int read = 0;
        for (read = 0; read < out->k; ++read) {
            if (out->neighbors[base + (std::size_t) read] == (std::int64_t) row) continue;
            if (write != read) {
                out->neighbors[base + (std::size_t) write] = out->neighbors[base + (std::size_t) read];
                out->distances[base + (std::size_t) write] = out->distances[base + (std::size_t) read];
            }
            ++write;
        }
        while (write < out->k) {
            out->neighbors[base + (std::size_t) write] = (std::int64_t) -1;
            out->distances[base + (std::size_t) write] = CUDART_INF_F;
            ++write;
        }
    }
}

#include "cuvs_sharded_knn/finalize_sparse_partition_kernel.cuh"
#include "cuvs_sharded_knn/init_topk_kernel.cuh"
#include "cuvs_sharded_knn/compute_row_norms_kernel.cuh"
#include "cuvs_sharded_knn/normalize_rows_in_place_kernel.cuh"
#include "cuvs_sharded_knn/postprocess_score_tile_kernel.cuh"
#include "cuvs_sharded_knn/merge_score_tile_topk_kernel.cuh"

} // namespace

void init(sparse_exact_params *params)
{
    params->k = 15;
    params->metric = metric_kind::cosine_expanded;
    params->metric_arg = 0.0f;
    params->exclude_self = 1;
    params->gpu_limit = 0;
    params->batch_size_index = 1 << 15;
    params->batch_size_query = 1 << 15;
    params->drop_host_parts_after_index_pack = 1;
    params->drop_host_parts_after_query_use = 0;
}

void init(dense_ann_params *params)
{
    params->k = 15;
    params->metric = metric_kind::cosine_expanded;
    params->metric_arg = 0.0f;
    params->exclude_self = 1;
    params->gpu_limit = 0;
    params->n_lists = 1024u;
    params->n_probes = 32u;
    params->use_cagra = 0;
    params->intermediate_graph_degree = 128u;
    params->graph_degree = 64u;
    params->rows_per_batch = 1ll << 20;
}

void init(proprietary_dense_params *params)
{
    params->k = 15;
    params->metric = metric_kind::cosine_expanded;
    params->metric_arg = 0.0f;
    params->exclude_self = 1;
    params->gpu_limit = 0;
    params->query_block_rows = 2048;
    params->index_block_rows = 4096;
}

void init(knn_result_host *result)
{
    result->rows = 0;
    result->k = 0;
    result->neighbors.clear();
    result->distances.clear();
}

void clear(knn_result_host *result)
{
    init(result);
}

int sparse_exact_self_knn(const ShardedCsr *view_const,
                          const ::cellshard::shard_storage *storage,
                          const sparse_exact_params *params_in,
                          knn_result_host *result)
{
    sparse_exact_params params;
    ::cellshard::distributed::local_context ctx;
    ::cellshard::distributed::shard_map map;
    ShardedCsr *view = const_cast<ShardedCsr *>(view_const);
    host_buffer<int> device_ids;
    host_buffer<device_sparse_csr> index_cache;
    device_shard_plan shard_plan;
    unsigned long shard = 0;
    int device = 0;
    int local_k = 0;
    int negate_distance = 0;

    init(&params);
    if (params_in != 0) params = *params_in;
    init(result);
    ::cellshard::distributed::init(&ctx);
    ::cellshard::distributed::init(&map);

    if (view == 0 || result == 0 || params.k <= 0) goto fail;
    if (view->num_shards == 0 || view->cols == 0) goto fail;
    if (!active_device_ids(params.gpu_limit, &device_ids)) goto fail;
    if (!cuda_ok(::cellshard::distributed::discover_local(&ctx, 1, cudaStreamNonBlocking), "discover_local")) goto fail;
    if (ctx.device_count > device_ids.size()) ctx.device_count = (unsigned int) device_ids.size();
    if (ctx.device_count == 0) goto fail;
    for (device = 0; device < (int) ctx.device_count; ++device) ctx.device_ids[device] = device_ids[(std::size_t) device];
    if (!::cellshard::distributed::assign_shards_by_bytes(&map, view, &ctx)) goto fail;

    index_cache.resize((std::size_t) view->num_shards);
    for (shard = 0; shard < view->num_shards; ++shard) {
        init(&index_cache[(std::size_t) shard]);
    }
    if (!build_device_shard_plan(view->num_shards, ctx.device_count, map.device_slot, &shard_plan)) goto fail;

    for (device = 0; device < (int) ctx.device_count; ++device) {
        const unsigned long *owned = device_shards_begin(shard_plan, device);
        const std::size_t owned_count = device_shards_count(shard_plan, device);
        for (std::size_t owned_idx = 0; owned_idx < owned_count; ++owned_idx) {
            if (!pack_sparse_shard_to_device(view,
                                             storage,
                                             owned[owned_idx],
                                             ctx.device_ids[device],
                                             params.drop_host_parts_after_index_pack,
                                             &index_cache[(std::size_t) owned[owned_idx]])) goto fail;
        }
    }

    result->rows = view->rows;
    result->k = params.k;
    result->neighbors.assign((std::size_t) view->rows * (std::size_t) params.k, (std::int64_t) -1);
    result->distances.assign((std::size_t) view->rows * (std::size_t) params.k, CUDART_INF_F);

    local_k = params.k + (params.exclude_self ? 1 : 0);
    negate_distance = metric_selects_min(params.metric) ? 0 : 1;

    for (shard = 0; shard < view->num_shards; ++shard) {
        const unsigned long query_row_begin = ::cellshard::first_row_in_shard(view, shard);
        const unsigned long query_rows = ::cellshard::rows_in_shard(view, shard);
        const std::size_t final_count = (std::size_t) query_rows * (std::size_t) params.k;
        host_buffer<std::int64_t> final_ids_by_device;
        host_buffer<float> final_dist_by_device;
        final_ids_by_device.assign((std::size_t) ctx.device_count * final_count, (std::int64_t) -1);
        final_dist_by_device.assign((std::size_t) ctx.device_count * final_count, CUDART_INF_F);

        for (device = 0; device < (int) ctx.device_count; ++device) {
            const unsigned long *owned = device_shards_begin(shard_plan, device);
            const std::size_t owned_count = device_shards_count(shard_plan, device);
            raft::resources res;
            float *d_part_distances = 0;
            int *d_part_neighbors = 0;
            std::int64_t *d_part_neighbors64 = 0;
            float *d_merge_in_distances = 0;
            std::int64_t *d_merge_in_neighbors = 0;
            float *d_final_distances = 0;
            std::int64_t *d_final_neighbors = 0;
            std::int64_t *d_zero_translations = 0;
            device_sparse_csr query_on_device;

            init(&query_on_device);
            if (owned_count == 0u) continue;
            if (!pack_sparse_shard_to_device(view,
                                             storage,
                                             shard,
                                             ctx.device_ids[device],
                                             params.drop_host_parts_after_query_use,
                                             &query_on_device)) goto device_fail;

            raft::resource::set_cuda_stream(res, rmm::cuda_stream_view(ctx.streams[device]));

            if (!cuda_ok(cudaSetDevice(ctx.device_ids[device]), "cudaSetDevice(sparse query loop)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_part_distances, (std::size_t) query_rows * (std::size_t) local_k * sizeof(float)), "cudaMalloc(d_part_distances)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_part_neighbors, (std::size_t) query_rows * (std::size_t) local_k * sizeof(int)), "cudaMalloc(d_part_neighbors)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_part_neighbors64, (std::size_t) query_rows * (std::size_t) local_k * sizeof(std::int64_t)), "cudaMalloc(d_part_neighbors64)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_merge_in_distances,
                                    (std::size_t) query_rows * (std::size_t) local_k * owned_count * sizeof(float)),
                         "cudaMalloc(d_merge_in_distances)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_merge_in_neighbors,
                                    (std::size_t) query_rows * (std::size_t) local_k * owned_count * sizeof(std::int64_t)),
                         "cudaMalloc(d_merge_in_neighbors)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_final_distances, final_count * sizeof(float)), "cudaMalloc(d_final_distances)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_final_neighbors, final_count * sizeof(std::int64_t)), "cudaMalloc(d_final_neighbors)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_zero_translations, owned_count * sizeof(std::int64_t)), "cudaMalloc(d_zero_translations)")) goto device_fail;
            if (!cuda_ok(cudaMemset(d_zero_translations, 0, owned_count * sizeof(std::int64_t)), "cudaMemset(d_zero_translations)")) goto device_fail;

            for (std::size_t local_shard = 0; local_shard < owned_count; ++local_shard) {
                const device_sparse_csr &index_pack = index_cache[(std::size_t) owned[local_shard]];
                const int same_shard = owned[local_shard] == shard ? 1 : 0;
                cuvs::neighbors::brute_force::index_params index_params;
                cuvs::neighbors::brute_force::search_params search_params;
                auto query_structure =
                    raft::make_device_compressed_structure_view<int, int, int>(
                        query_on_device.row_ptr, query_on_device.col_idx, query_on_device.cols, query_on_device.rows, query_on_device.nnz);
                auto index_structure =
                    raft::make_device_compressed_structure_view<int, int, int>(
                        index_pack.row_ptr, index_pack.col_idx, index_pack.cols, index_pack.rows, index_pack.nnz);
                auto query_view =
                    raft::make_device_csr_matrix_view<const float, int, int, int>(query_on_device.val, query_structure);
                auto index_view =
                    raft::make_device_csr_matrix_view<const float, int, int, int>(index_pack.val, index_structure);
                auto neighbors_view = raft::make_device_matrix_view<int, std::int64_t, raft::row_major>(
                    d_part_neighbors, (std::int64_t) query_on_device.rows, (std::int64_t) local_k);
                auto distances_view = raft::make_device_matrix_view<float, std::int64_t, raft::row_major>(
                    d_part_distances, (std::int64_t) query_on_device.rows, (std::int64_t) local_k);
                index_params.metric = to_cuvs_metric(params.metric);
                index_params.metric_arg = params.metric_arg;
                search_params.batch_size_index = params.batch_size_index;
                search_params.batch_size_query = params.batch_size_query;
                auto index = cuvs::neighbors::brute_force::build(res, index_params, index_view);
                cuvs::neighbors::brute_force::search(res, search_params, index, query_view, neighbors_view, distances_view);
                {
                    const int blocks = (query_on_device.rows * local_k + 255) / 256;
                    finalize_sparse_partition_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                        query_on_device.rows,
                        local_k,
                        index_pack.row_begin,
                        params.exclude_self && same_shard ? 1 : 0,
                        negate_distance,
                        d_part_neighbors,
                        d_part_distances,
                        d_part_neighbors64);
                }
                if (!cuda_ok(cudaGetLastError(), "finalize_sparse_partition_kernel")) goto device_fail;
                if (!cuda_ok(cudaMemcpyAsync(
                                 d_merge_in_distances + (std::size_t) local_shard * (std::size_t) query_rows * (std::size_t) local_k,
                                 d_part_distances,
                                 (std::size_t) query_rows * (std::size_t) local_k * sizeof(float),
                                 cudaMemcpyDeviceToDevice,
                                 ctx.streams[device]),
                             "cudaMemcpyAsync(part distances -> merge)")) goto device_fail;
                if (!cuda_ok(cudaMemcpyAsync(
                                 d_merge_in_neighbors + (std::size_t) local_shard * (std::size_t) query_rows * (std::size_t) local_k,
                                 d_part_neighbors64,
                                 (std::size_t) query_rows * (std::size_t) local_k * sizeof(std::int64_t),
                                 cudaMemcpyDeviceToDevice,
                                 ctx.streams[device]),
                             "cudaMemcpyAsync(part neighbors -> merge)")) goto device_fail;
            }

            cuvs::neighbors::knn_merge_parts(
                res,
                raft::make_device_matrix_view<const float, std::int64_t, raft::row_major>(
                    d_merge_in_distances,
                    (std::int64_t) query_rows,
                    (std::int64_t) local_k * (std::int64_t) owned_count),
                raft::make_device_matrix_view<const std::int64_t, std::int64_t, raft::row_major>(
                    d_merge_in_neighbors,
                    (std::int64_t) query_rows,
                    (std::int64_t) local_k * (std::int64_t) owned_count),
                raft::make_device_matrix_view<float, std::int64_t, raft::row_major>(
                    d_final_distances,
                    (std::int64_t) query_rows,
                    (std::int64_t) params.k),
                raft::make_device_matrix_view<std::int64_t, std::int64_t, raft::row_major>(
                    d_final_neighbors,
                    (std::int64_t) query_rows,
                    (std::int64_t) params.k),
                raft::make_device_vector_view<std::int64_t, std::int64_t>(d_zero_translations, (std::int64_t) owned_count));

            if (!cuda_ok(cudaStreamSynchronize(ctx.streams[device]), "cudaStreamSynchronize(local merge)")) goto device_fail;
            if (!cuda_ok(cudaMemcpy(final_ids_by_device.data() + (std::size_t) device * final_count,
                                    d_final_neighbors,
                                    final_count * sizeof(std::int64_t),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy(final ids -> host)")) goto device_fail;
            if (!cuda_ok(cudaMemcpy(final_dist_by_device.data() + (std::size_t) device * final_count,
                                    d_final_distances,
                                    final_count * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy(final distances -> host)")) goto device_fail;

            clear(&query_on_device);
            cudaFree(d_part_distances);
            cudaFree(d_part_neighbors);
            cudaFree(d_part_neighbors64);
            cudaFree(d_merge_in_distances);
            cudaFree(d_merge_in_neighbors);
            cudaFree(d_final_distances);
            cudaFree(d_final_neighbors);
            cudaFree(d_zero_translations);
            continue;

device_fail:
            clear(&query_on_device);
            if (d_part_distances != 0) cudaFree(d_part_distances);
            if (d_part_neighbors != 0) cudaFree(d_part_neighbors);
            if (d_part_neighbors64 != 0) cudaFree(d_part_neighbors64);
            if (d_merge_in_distances != 0) cudaFree(d_merge_in_distances);
            if (d_merge_in_neighbors != 0) cudaFree(d_merge_in_neighbors);
            if (d_final_distances != 0) cudaFree(d_final_distances);
            if (d_final_neighbors != 0) cudaFree(d_final_neighbors);
            if (d_zero_translations != 0) cudaFree(d_zero_translations);
            goto fail;
        }

        {
            knn_result_host shard_result;
            std::size_t base = 0;
            init(&shard_result);
            if (!merge_host_candidates(query_rows,
                                       params.k,
                                       1,
                                       negate_distance,
                                       (int) ctx.device_count,
                                       final_ids_by_device.data(),
                                       final_dist_by_device.data(),
                                       final_count,
                                       &shard_result)) goto fail;
            base = (std::size_t) query_row_begin * (std::size_t) params.k;
            std::memcpy(result->neighbors.data() + base, shard_result.neighbors.data(), shard_result.neighbors.size() * sizeof(std::int64_t));
            std::memcpy(result->distances.data() + base, shard_result.distances.data(), shard_result.distances.size() * sizeof(float));
        }
    }

    for (shard = 0; shard < index_cache.size(); ++shard) clear(&index_cache[(std::size_t) shard]);
    ::cellshard::distributed::clear(&map);
    ::cellshard::distributed::clear(&ctx);
    return 1;

fail:
    for (shard = 0; shard < index_cache.size(); ++shard) clear(&index_cache[(std::size_t) shard]);
    ::cellshard::distributed::clear(&map);
    ::cellshard::distributed::clear(&ctx);
    clear(result);
    return 0;
}

int dense_ann_self_knn(const ShardedDense *view,
                       const dense_ann_params *params_in,
                       knn_result_host *result)
{
    dense_ann_params params;
    host_buffer<int> device_ids;
    host_buffer<__half> packed;
    host_buffer<std::int64_t> neighbors;
    host_buffer<float> distances;
    int padded_cols = 0;
    const std::int64_t local_k = (std::int64_t) params.k + (params.exclude_self ? 1ll : 0ll);

    init(&params);
    if (params_in != 0) params = *params_in;
    init(result);
    if (view == 0 || view->num_parts == 0 || view->cols == 0 || params.k <= 0) return 0;
    if (!active_device_ids(params.gpu_limit, &device_ids)) return 0;
    if (view->rows > (unsigned long) std::numeric_limits<std::int64_t>::max()) return 0;
    if (view->cols > (unsigned long) std::numeric_limits<std::int64_t>::max()) return 0;

    padded_cols = round_up_int((int) view->cols, 8);
    if (!pack_dense_matrix_to_host_padded(view, padded_cols, &packed)) return 0;

    neighbors.assign((std::size_t) view->rows * (std::size_t) local_k, (std::int64_t) -1);
    distances.assign((std::size_t) view->rows * (std::size_t) local_k, CUDART_INF_F);

    {
        std::vector<int> clique_device_ids(device_ids.data(), device_ids.data() + device_ids.size());
        raft::device_resources_snmg clique(clique_device_ids);
        auto dataset_view = raft::make_host_matrix_view<const __half, std::int64_t, raft::row_major>(
            packed.data(),
            (std::int64_t) view->rows,
            (std::int64_t) padded_cols);
        auto query_view = raft::make_host_matrix_view<const __half, std::int64_t, raft::row_major>(
            packed.data(),
            (std::int64_t) view->rows,
            (std::int64_t) padded_cols);
        auto neighbors_view = raft::make_host_matrix_view<std::int64_t, std::int64_t, raft::row_major>(
            neighbors.data(),
            (std::int64_t) view->rows,
            local_k);
        auto distances_view = raft::make_host_matrix_view<float, std::int64_t, raft::row_major>(
            distances.data(),
            (std::int64_t) view->rows,
            local_k);

        clique.set_memory_pool(80);

        if (params.use_cagra) {
            cuvs::neighbors::mg_index_params<cuvs::neighbors::cagra::index_params> index_params;
            cuvs::neighbors::mg_search_params<cuvs::neighbors::cagra::search_params> search_params;
            index_params.metric = to_cuvs_metric(params.metric);
            index_params.metric_arg = params.metric_arg;
            index_params.intermediate_graph_degree = params.intermediate_graph_degree;
            index_params.graph_degree = params.graph_degree;
            search_params.itopk_size = params.graph_degree;
            search_params.max_queries = 4096;
            search_params.n_rows_per_batch = params.rows_per_batch;
            auto index = cuvs::neighbors::cagra::build(clique, index_params, dataset_view);
            cuvs::neighbors::cagra::search(clique, index, search_params, query_view, neighbors_view, distances_view);
        } else {
            cuvs::neighbors::mg_index_params<cuvs::neighbors::ivf_flat::index_params> index_params;
            cuvs::neighbors::mg_search_params<cuvs::neighbors::ivf_flat::search_params> search_params;
            index_params.metric = to_cuvs_metric(params.metric);
            index_params.metric_arg = params.metric_arg;
            index_params.n_lists = params.n_lists;
            search_params.n_probes = params.n_probes;
            search_params.n_rows_per_batch = params.rows_per_batch;
            auto index = cuvs::neighbors::ivf_flat::build(clique, index_params, dataset_view);
            cuvs::neighbors::ivf_flat::search(clique, index, search_params, query_view, neighbors_view, distances_view);
        }
    }

    result->rows = view->rows;
    result->k = (int) local_k;
    result->neighbors.swap(neighbors);
    result->distances.swap(distances);
    if (params.exclude_self) {
        filter_self_from_host_result(result);
        result->k = params.k;
        result->neighbors.resize((std::size_t) result->rows * (std::size_t) result->k);
        result->distances.resize((std::size_t) result->rows * (std::size_t) result->k);
    }
    return 1;
}

int proprietary_dense_self_knn(const ShardedDense *view,
                               const proprietary_dense_params *params_in,
                               knn_result_host *result)
{
    proprietary_dense_params params;
    ::cellshard::distributed::local_context ctx;
    host_buffer<int> device_ids;
    host_buffer<unsigned long> row_offsets;
    host_buffer<device_dense_exact_index> indices;
    host_buffer<__half> packed_host;
    host_buffer<cublasHandle_t> cublas_handles;
    int device = 0;
    int padded_cols = 0;
    int select_min = 0;
    int metric = 0;
    int query_block_rows = 0;
    int index_block_rows = 0;
    unsigned long query_begin = 0;

    init(&params);
    if (params_in != 0) params = *params_in;
    init(result);
    ::cellshard::distributed::init(&ctx);

    if (view == 0 || result == 0 || params.k <= 0) goto fail;
    if (view->num_parts == 0 || view->rows == 0 || view->cols == 0) goto fail;
    if (!active_device_ids(params.gpu_limit, &device_ids)) goto fail;
    if (!cuda_ok(::cellshard::distributed::discover_local(&ctx, 1, cudaStreamNonBlocking), "discover_local(proprietary)")) goto fail;
    if (ctx.device_count > device_ids.size()) ctx.device_count = (unsigned int) device_ids.size();
    if (ctx.device_count == 0) goto fail;
    for (device = 0; device < (int) ctx.device_count; ++device) ctx.device_ids[device] = device_ids[(std::size_t) device];
    if (!cuda_ok(::cellshard::distributed::enable_peer_access(&ctx), "enable_peer_access")) goto fail;

    padded_cols = round_up_int((int) view->cols, 8);
    select_min = metric_selects_min(params.metric);
    metric = metric_code(params.metric);
    query_block_rows = round_up_int(std::max(params.query_block_rows, 32), 8);
    index_block_rows = round_up_int(std::max(params.index_block_rows, 32), 8);

    if (!pack_dense_matrix_to_host_padded(view, padded_cols, &packed_host)) goto fail;
    if (!split_rows_evenly(view->rows, (int) ctx.device_count, &row_offsets)) goto fail;

    indices.resize((std::size_t) ctx.device_count);
    cublas_handles.assign((std::size_t) ctx.device_count, (cublasHandle_t) 0);
    for (device = 0; device < (int) ctx.device_count; ++device) {
        init(&indices[(std::size_t) device]);
        if (!upload_dense_slice_to_device(packed_host.data(),
                                          (int) view->cols,
                                          padded_cols,
                                          row_offsets[(std::size_t) device],
                                          row_offsets[(std::size_t) device + 1u],
                                          ctx.device_ids[device],
                                          metric,
                                          &indices[(std::size_t) device])) goto fail;
        if (!cublas_ok(cublasCreate(&cublas_handles[(std::size_t) device]), "cublasCreate")) goto fail;
        if (!cublas_ok(cublasSetStream(cublas_handles[(std::size_t) device], ctx.streams[device]), "cublasSetStream")) goto fail;
        if (!cublas_ok(cublasSetMathMode(cublas_handles[(std::size_t) device], CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode")) goto fail;
        if (metric == proprietary_metric_cosine) {
            const int blocks = (indices[(std::size_t) device].rows + 255) / 256;
            normalize_rows_in_place_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                indices[(std::size_t) device].val,
                indices[(std::size_t) device].rows,
                (int) view->cols,
                padded_cols);
            if (!cuda_ok(cudaGetLastError(), "normalize_rows_in_place_kernel(index)")) goto fail;
        } else if (metric == proprietary_metric_l2) {
            const int blocks = (indices[(std::size_t) device].rows + 255) / 256;
            compute_row_norms_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                indices[(std::size_t) device].val,
                indices[(std::size_t) device].rows,
                (int) view->cols,
                padded_cols,
                indices[(std::size_t) device].norms);
            if (!cuda_ok(cudaGetLastError(), "compute_row_norms_kernel(index)")) goto fail;
        }
    }
    for (device = 0; device < (int) ctx.device_count; ++device) {
        if (!cuda_ok(cudaStreamSynchronize(ctx.streams[device]), "cudaStreamSynchronize(index setup)")) goto fail;
    }

    result->rows = view->rows;
    result->k = params.k;
    result->neighbors.assign((std::size_t) view->rows * (std::size_t) params.k, (std::int64_t) -1);
    result->distances.assign((std::size_t) view->rows * (std::size_t) params.k, worst_value(select_min));

    for (query_begin = 0; query_begin < view->rows; query_begin += (unsigned long) query_block_rows) {
        const int query_rows = (int) std::min((unsigned long) query_block_rows, view->rows - query_begin);
        const std::size_t final_count = (std::size_t) query_rows * (std::size_t) params.k;
        host_buffer<std::int64_t> final_ids_by_device;
        host_buffer<float> final_values_by_device;
        final_ids_by_device.assign((std::size_t) ctx.device_count * final_count, (std::int64_t) -1);
        final_values_by_device.assign((std::size_t) ctx.device_count * final_count, worst_value(select_min));

        for (device = 0; device < (int) ctx.device_count; ++device) {
            const device_dense_exact_index &index_pack = indices[(std::size_t) device];
            __half *d_query = 0;
            float *d_query_norms = 0;
            float *d_scores = 0;
            float *d_best_values = 0;
            std::int64_t *d_best_indices = 0;

            if (index_pack.rows == 0) continue;
            if (!cuda_ok(cudaSetDevice(ctx.device_ids[device]), "cudaSetDevice(proprietary query)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_query, (std::size_t) query_rows * (std::size_t) padded_cols * sizeof(__half)), "cudaMalloc(d_query)")) goto device_fail;
            if (metric == proprietary_metric_l2 &&
                !cuda_ok(cudaMalloc((void **) &d_query_norms, (std::size_t) query_rows * sizeof(float)), "cudaMalloc(d_query_norms)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_scores, (std::size_t) query_rows * (std::size_t) index_block_rows * sizeof(float)), "cudaMalloc(d_scores)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_best_values, (std::size_t) query_rows * (std::size_t) params.k * sizeof(float)), "cudaMalloc(d_best_values)")) goto device_fail;
            if (!cuda_ok(cudaMalloc((void **) &d_best_indices, (std::size_t) query_rows * (std::size_t) params.k * sizeof(std::int64_t)), "cudaMalloc(d_best_indices)")) goto device_fail;

            if (!cuda_ok(cudaMemcpyAsync(d_query,
                                         packed_host.data() + (std::size_t) query_begin * (std::size_t) padded_cols,
                                         (std::size_t) query_rows * (std::size_t) padded_cols * sizeof(__half),
                                         cudaMemcpyHostToDevice,
                                         ctx.streams[device]),
                         "cudaMemcpyAsync(query block)")) goto device_fail;
            {
                const int blocks = (query_rows * params.k + 255) / 256;
                init_topk_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                    query_rows * params.k,
                    select_min,
                    d_best_values,
                    d_best_indices);
            }
            if (!cuda_ok(cudaGetLastError(), "init_topk_kernel")) goto device_fail;

            if (metric == proprietary_metric_cosine) {
                const int blocks = (query_rows + 255) / 256;
                normalize_rows_in_place_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                    d_query, query_rows, (int) view->cols, padded_cols);
                if (!cuda_ok(cudaGetLastError(), "normalize_rows_in_place_kernel(query)")) goto device_fail;
            } else if (metric == proprietary_metric_l2) {
                const int blocks = (query_rows + 255) / 256;
                compute_row_norms_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                    d_query, query_rows, (int) view->cols, padded_cols, d_query_norms);
                if (!cuda_ok(cudaGetLastError(), "compute_row_norms_kernel(query)")) goto device_fail;
            }

            for (int index_offset = 0; index_offset < index_pack.rows; index_offset += index_block_rows) {
                const int tile_rows = std::min(index_block_rows, index_pack.rows - index_offset);
                const float alpha = 1.0f;
                const float beta = 0.0f;

                if (!cublas_ok(cublasGemmEx(
                                   cublas_handles[(std::size_t) device],
                                   CUBLAS_OP_T,
                                   CUBLAS_OP_N,
                                   tile_rows,
                                   query_rows,
                                   padded_cols,
                                   &alpha,
                                   index_pack.val + (std::size_t) index_offset * (std::size_t) index_pack.ld,
                                   CUDA_R_16F,
                                   index_pack.ld,
                                   d_query,
                                   CUDA_R_16F,
                                   padded_cols,
                                   &beta,
                                   d_scores,
                                   CUDA_R_32F,
                                   tile_rows,
                                   CUBLAS_COMPUTE_32F,
                                   CUBLAS_GEMM_DEFAULT_TENSOR_OP),
                               "cublasGemmEx(proprietary)")) goto device_fail;

                if (metric != proprietary_metric_inner_product) {
                    const int blocks = (query_rows * tile_rows + 255) / 256;
                    postprocess_score_tile_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                        d_scores,
                        d_query_norms,
                        index_pack.norms != 0 ? index_pack.norms + index_offset : 0,
                        query_rows,
                        tile_rows,
                        metric);
                    if (!cuda_ok(cudaGetLastError(), "postprocess_score_tile_kernel")) goto device_fail;
                }

                {
                    const int blocks = (query_rows + 255) / 256;
                    merge_score_tile_topk_kernel<<<blocks, 256, 0, ctx.streams[device]>>>(
                        d_scores,
                        query_rows,
                        tile_rows,
                        params.k,
                        select_min,
                        query_begin,
                        index_pack.row_begin + (unsigned long) index_offset,
                        params.exclude_self,
                        d_best_values,
                        d_best_indices);
                }
                if (!cuda_ok(cudaGetLastError(), "merge_score_tile_topk_kernel")) goto device_fail;
            }

            if (!cuda_ok(cudaStreamSynchronize(ctx.streams[device]), "cudaStreamSynchronize(proprietary device)")) goto device_fail;
            if (!cuda_ok(cudaMemcpy(final_ids_by_device.data() + (std::size_t) device * final_count,
                                    d_best_indices,
                                    final_count * sizeof(std::int64_t),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy(best indices -> host)")) goto device_fail;
            if (!cuda_ok(cudaMemcpy(final_values_by_device.data() + (std::size_t) device * final_count,
                                    d_best_values,
                                    final_count * sizeof(float),
                                    cudaMemcpyDeviceToHost),
                         "cudaMemcpy(best values -> host)")) goto device_fail;

            if (d_query != 0) cudaFree(d_query);
            if (d_query_norms != 0) cudaFree(d_query_norms);
            if (d_scores != 0) cudaFree(d_scores);
            if (d_best_values != 0) cudaFree(d_best_values);
            if (d_best_indices != 0) cudaFree(d_best_indices);
            continue;

device_fail:
            if (d_query != 0) cudaFree(d_query);
            if (d_query_norms != 0) cudaFree(d_query_norms);
            if (d_scores != 0) cudaFree(d_scores);
            if (d_best_values != 0) cudaFree(d_best_values);
            if (d_best_indices != 0) cudaFree(d_best_indices);
            goto fail;
        }

        {
            knn_result_host block_result;
            const std::size_t base = (std::size_t) query_begin * (std::size_t) params.k;
            init(&block_result);
            if (!merge_host_candidates((unsigned long) query_rows,
                                       params.k,
                                       select_min,
                                       0,
                                       (int) ctx.device_count,
                                       final_ids_by_device.data(),
                                       final_values_by_device.data(),
                                       final_count,
                                       &block_result)) goto fail;
            std::memcpy(result->neighbors.data() + base,
                        block_result.neighbors.data(),
                        block_result.neighbors.size() * sizeof(std::int64_t));
            std::memcpy(result->distances.data() + base,
                        block_result.distances.data(),
                        block_result.distances.size() * sizeof(float));
        }
    }

    for (device = 0; device < (int) cublas_handles.size(); ++device) {
        if (cublas_handles[(std::size_t) device] != 0) cublasDestroy(cublas_handles[(std::size_t) device]);
    }
    for (device = 0; device < (int) indices.size(); ++device) clear(&indices[(std::size_t) device]);
    ::cellshard::distributed::clear(&ctx);
    return 1;

fail:
    for (device = 0; device < (int) cublas_handles.size(); ++device) {
        if (cublas_handles[(std::size_t) device] != 0) cublasDestroy(cublas_handles[(std::size_t) device]);
    }
    for (device = 0; device < (int) indices.size(); ++device) clear(&indices[(std::size_t) device]);
    ::cellshard::distributed::clear(&ctx);
    clear(result);
    return 0;
}

} // namespace neighbors
} // namespace compute
} // namespace cellerator
