#pragma once

#include <cuda_runtime.h>

#include "matrix_io.cuh"

namespace matrix {
namespace device {

template<typename MatrixT>
struct alignas(16) part_record {
    void *view;
    void *a0;
    void *a1;
    void *a2;
    int device_id;
};

template<typename MatrixT>
struct shard_cache {
    typedef Index index_type;

    Index capacity;
    part_record<MatrixT> *parts;
};

template<typename MatrixT>
inline cudaError_t release(part_record<MatrixT> *record);

template<typename MatrixT>
inline cudaError_t release_part(shard_cache<MatrixT> *cache, Index partId);

template<typename ValueT>
struct alignas(16) dense_view {
    Index rows;
    Index cols;
    Index nnz;
    Index ld;
    ValueT *val;
};

template<typename ValueT>
struct alignas(16) csr_view {
    Index rows;
    Index cols;
    Index nnz;
    Index *rowPtr;
    Index *colIdx;
    ValueT *val;
};

template<typename ValueT>
struct alignas(16) coo_view {
    Index rows;
    Index cols;
    Index nnz;
    Index *rowIdx;
    Index *colIdx;
    ValueT *val;
};

template<typename ValueT>
struct alignas(16) dia_view {
    Index rows;
    Index cols;
    Index nnz;
    Index num_diagonals;
    DiagIndex *offsets;
    ValueT *val;
};

template<typename MatrixT>
__host__ __forceinline__ void init(shard_cache<MatrixT> *c) {
    c->capacity = 0;
    c->parts = 0;
}

template<typename MatrixT>
__host__ __forceinline__ void clear(shard_cache<MatrixT> *c) {
    Index i = 0;
    if (c->parts != 0) {
        for (i = 0; i < c->capacity; ++i) {
            if (c->parts[i].view != 0) release_part(c, i);
        }
    }
    std::free(c->parts);
    c->capacity = 0;
    c->parts = 0;
}

template<typename MatrixT>
__host__ __forceinline__ int reserve(shard_cache<MatrixT> *c, Index capacity) {
    part_record<MatrixT> *records = 0;
    Index i = 0;

    if (capacity <= c->capacity) return 1;
    records = (part_record<MatrixT> *) std::malloc((std::size_t) capacity * sizeof(part_record<MatrixT>));
    if (records == 0) return 0;
    std::memset(records, 0, (std::size_t) capacity * sizeof(part_record<MatrixT>));
    for (i = 0; i < capacity; ++i) records[i].device_id = -1;
    for (i = 0; i < c->capacity; ++i) records[i] = c->parts[i];
    std::free(c->parts);
    c->parts = records;
    c->capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ void zero_record(part_record<MatrixT> *record) {
    record->view = 0;
    record->a0 = 0;
    record->a1 = 0;
    record->a2 = 0;
    record->device_id = -1;
}

template<typename ValueT>
__host__ __forceinline__ cudaError_t upload(const ::matrix::dense<ValueT> *src, part_record< ::matrix::dense<ValueT> > *record) {
    dense_view<ValueT> host;
    dense_view<ValueT> *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.ld = src->ld;
    host.val = 0;

    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(ValueT));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(ValueT), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    zero_record(record);
    return err;
}

template<typename ValueT>
__host__ __forceinline__ cudaError_t upload(const ::matrix::sparse::csr<ValueT> *src, part_record< ::matrix::sparse::csr<ValueT> > *record) {
    csr_view<ValueT> host;
    csr_view<ValueT> *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.rowPtr = 0;
    host.colIdx = 0;
    host.val = 0;

    if (src->rows != 0) {
        err = cudaMalloc((void **) &host.rowPtr, (src->rows + 1) * sizeof(Index));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.rowPtr, src->rowPtr, (src->rows + 1) * sizeof(Index), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.colIdx, src->nnz * sizeof(Index));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.colIdx, src->colIdx, src->nnz * sizeof(Index), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(ValueT));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(ValueT), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.rowPtr;
    record->a1 = host.colIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    if (host.colIdx != 0) cudaFree(host.colIdx);
    if (host.rowPtr != 0) cudaFree(host.rowPtr);
    zero_record(record);
    return err;
}

template<typename ValueT>
__host__ __forceinline__ cudaError_t upload(const ::matrix::sparse::coo<ValueT> *src, part_record< ::matrix::sparse::coo<ValueT> > *record) {
    coo_view<ValueT> host;
    coo_view<ValueT> *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.rowIdx = 0;
    host.colIdx = 0;
    host.val = 0;

    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.rowIdx, src->nnz * sizeof(Index));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.rowIdx, src->rowIdx, src->nnz * sizeof(Index), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc((void **) &host.colIdx, src->nnz * sizeof(Index));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.colIdx, src->colIdx, src->nnz * sizeof(Index), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(ValueT));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(ValueT), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.rowIdx;
    record->a1 = host.colIdx;
    record->a2 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    if (host.colIdx != 0) cudaFree(host.colIdx);
    if (host.rowIdx != 0) cudaFree(host.rowIdx);
    zero_record(record);
    return err;
}

template<typename ValueT>
__host__ __forceinline__ cudaError_t upload(const ::matrix::sparse::dia<ValueT> *src, part_record< ::matrix::sparse::dia<ValueT> > *record) {
    dia_view<ValueT> host;
    dia_view<ValueT> *deviceView = 0;
    cudaError_t err = cudaSuccess;

    zero_record(record);
    host.rows = src->rows;
    host.cols = src->cols;
    host.nnz = src->nnz;
    host.num_diagonals = src->num_diagonals;
    host.offsets = 0;
    host.val = 0;

    if (src->num_diagonals != 0) {
        err = cudaMalloc((void **) &host.offsets, src->num_diagonals * sizeof(DiagIndex));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.offsets, src->offsets, src->num_diagonals * sizeof(DiagIndex), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }
    if (src->nnz != 0) {
        err = cudaMalloc((void **) &host.val, src->nnz * sizeof(ValueT));
        if (err != cudaSuccess) goto fail;
        err = cudaMemcpy(host.val, src->val, src->nnz * sizeof(ValueT), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) goto fail;
    }

    err = cudaMalloc((void **) &deviceView, sizeof(host));
    if (err != cudaSuccess) goto fail;
    err = cudaMemcpy(deviceView, &host, sizeof(host), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) goto fail;

    record->view = deviceView;
    record->a0 = host.offsets;
    record->a1 = host.val;
    return cudaSuccess;

fail:
    if (deviceView != 0) cudaFree(deviceView);
    if (host.val != 0) cudaFree(host.val);
    if (host.offsets != 0) cudaFree(host.offsets);
    zero_record(record);
    return err;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release(part_record<MatrixT> *record) {
    cudaError_t err = cudaSuccess;

    if (record->a2 != 0) {
        err = cudaFree(record->a2);
        if (err != cudaSuccess) return err;
    }
    if (record->a1 != 0) {
        err = cudaFree(record->a1);
        if (err != cudaSuccess) return err;
    }
    if (record->a0 != 0) {
        err = cudaFree(record->a0);
        if (err != cudaSuccess) return err;
    }
    if (record->view != 0) {
        err = cudaFree(record->view);
        if (err != cudaSuccess) return err;
    }
    zero_record(record);
    return cudaSuccess;
}

template<typename ValueT, typename ShardedIndexT>
__host__ __forceinline__ std::size_t device_part_bytes(const ::matrix::sharded< ::matrix::dense<ValueT>, ShardedIndexT > *view, ShardedIndexT partId) {
    return sizeof(dense_view<ValueT>) + (std::size_t) view->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT, typename ShardedIndexT>
__host__ __forceinline__ std::size_t device_part_bytes(const ::matrix::sharded< ::matrix::sparse::csr<ValueT>, ShardedIndexT > *view, ShardedIndexT partId) {
    return sizeof(csr_view<ValueT>)
        + (std::size_t) (view->part_rows[partId] + 1) * sizeof(Index)
        + (std::size_t) view->part_nnz[partId] * sizeof(Index)
        + (std::size_t) view->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT, typename ShardedIndexT>
__host__ __forceinline__ std::size_t device_part_bytes(const ::matrix::sharded< ::matrix::sparse::coo<ValueT>, ShardedIndexT > *view, ShardedIndexT partId) {
    return sizeof(coo_view<ValueT>)
        + (std::size_t) view->part_nnz[partId] * sizeof(Index)
        + (std::size_t) view->part_nnz[partId] * sizeof(Index)
        + (std::size_t) view->part_nnz[partId] * sizeof(ValueT);
}

template<typename ValueT, typename ShardedIndexT>
__host__ __forceinline__ std::size_t device_part_bytes(const ::matrix::sharded< ::matrix::sparse::dia<ValueT>, ShardedIndexT > *view, ShardedIndexT partId) {
    return sizeof(dia_view<ValueT>)
        + (std::size_t) view->part_aux[partId] * sizeof(DiagIndex)
        + (std::size_t) view->part_nnz[partId] * sizeof(ValueT);
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ std::size_t device_shard_bytes(const ::matrix::sharded<MatrixT, ShardedIndexT> *view, ShardedIndexT shardId) {
    ShardedIndexT begin = 0;
    ShardedIndexT end = 0;
    ShardedIndexT i = 0;
    std::size_t total = 0;

    if (shardId >= view->num_shards) return 0;
    begin = ::matrix::first_part_in_shard(view, shardId);
    end = ::matrix::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) total += device_part_bytes(view, i);
    return total;
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ int set_shards_by_device_bytes(::matrix::sharded<MatrixT, ShardedIndexT> *view, std::size_t max_bytes) {
    std::size_t used = 0;
    std::size_t bytes = 0;
    ShardedIndexT shardCount = 0;
    ShardedIndexT i = 0;

    if (max_bytes == 0) return ::matrix::set_shards_to_parts(view);
    if (!::matrix::reserve_shards(view, view->num_parts)) return 0;

    view->shard_offsets[0] = 0;
    for (i = 0; i < view->num_parts; ++i) {
        bytes = device_part_bytes(view, i);
        if (bytes == 0) continue;
        if (used != 0 && used + bytes > max_bytes) {
            ++shardCount;
            view->shard_offsets[shardCount] = view->part_offsets[i];
            used = 0;
        }
        used += bytes;
    }
    if (view->num_parts != 0) {
        ++shardCount;
        view->shard_offsets[shardCount] = view->rows;
    }
    view->num_shards = shardCount;
    return 1;
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ cudaError_t upload_part(shard_cache<MatrixT> *cache, const ::matrix::sharded<MatrixT, ShardedIndexT> *view, ShardedIndexT partId, int deviceId) {
    cudaError_t err = cudaSuccess;

    if (partId >= view->num_parts || partId >= cache->capacity || view->parts[partId] == 0) return cudaErrorInvalidValue;
    if (cache->parts[partId].view != 0) {
        if (cache->parts[partId].device_id == deviceId) return cudaSuccess;
        err = release_part(cache, partId);
        if (err != cudaSuccess) return err;
    }
    err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) return err;
    err = upload(view->parts[partId], &cache->parts[partId]);
    if (err != cudaSuccess) return err;
    cache->parts[partId].device_id = deviceId;
    return cudaSuccess;
}

template<typename MatrixT>
__host__ __forceinline__ cudaError_t release_part(shard_cache<MatrixT> *cache, Index partId) {
    cudaError_t err = cudaSuccess;

    if (partId >= cache->capacity || cache->parts[partId].view == 0) return cudaSuccess;
    err = cudaSetDevice(cache->parts[partId].device_id >= 0 ? cache->parts[partId].device_id : 0);
    if (err != cudaSuccess) return err;
    return release(&cache->parts[partId]);
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ cudaError_t upload_shard(shard_cache<MatrixT> *cache, const ::matrix::sharded<MatrixT, ShardedIndexT> *view, ShardedIndexT shardId, int deviceId) {
    ShardedIndexT begin = 0;
    ShardedIndexT end = 0;
    ShardedIndexT i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::matrix::first_part_in_shard(view, shardId);
    end = ::matrix::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) {
        err = upload_part(cache, view, i, deviceId);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ cudaError_t release_shard(shard_cache<MatrixT> *cache, const ::matrix::sharded<MatrixT, ShardedIndexT> *view, ShardedIndexT shardId) {
    ShardedIndexT begin = 0;
    ShardedIndexT end = 0;
    ShardedIndexT i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::matrix::first_part_in_shard(view, shardId);
    end = ::matrix::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) {
        err = release_part(cache, i);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ cudaError_t stage_part(shard_cache<MatrixT> *cache,
                              ::matrix::sharded<MatrixT, ShardedIndexT> *view,
                              const ::matrix::shard_storage *files,
                              ShardedIndexT partId,
                              int deviceId,
                              int drop_host_after_upload) {
    cudaError_t err = cudaSuccess;

    if (partId >= view->num_parts || partId >= cache->capacity) return cudaErrorInvalidValue;
    if (cache->parts[partId].view != 0 && cache->parts[partId].device_id == deviceId) {
        if (drop_host_after_upload && view->parts[partId] != 0) {
            if (!::matrix::drop_part(view, partId)) return cudaErrorInvalidValue;
        }
        return cudaSuccess;
    }
    if (view->parts[partId] == 0) {
        if (!::matrix::fetch_part(view, files, partId)) return cudaErrorInvalidValue;
    }
    err = upload_part(cache, view, partId, deviceId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_upload) {
        if (!::matrix::drop_part(view, partId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ cudaError_t stage_shard(shard_cache<MatrixT> *cache,
                               ::matrix::sharded<MatrixT, ShardedIndexT> *view,
                               const ::matrix::shard_storage *files,
                               ShardedIndexT shardId,
                               int deviceId,
                               int drop_host_after_upload) {
    ShardedIndexT begin = 0;
    ShardedIndexT end = 0;
    ShardedIndexT i = 0;
    cudaError_t err = cudaSuccess;

    if (shardId >= view->num_shards) return cudaErrorInvalidValue;
    begin = ::matrix::first_part_in_shard(view, shardId);
    end = ::matrix::last_part_in_shard(view, shardId);
    for (i = begin; i < end; ++i) {
        err = stage_part(cache, view, files, i, deviceId, drop_host_after_upload);
        if (err != cudaSuccess) return err;
    }
    return cudaSuccess;
}

template<typename MatrixT, typename ShardedIndexT>
__host__ __forceinline__ cudaError_t swap_shard(shard_cache<MatrixT> *cache,
                              ::matrix::sharded<MatrixT, ShardedIndexT> *view,
                              const ::matrix::shard_storage *files,
                              ShardedIndexT outShardId,
                              ShardedIndexT inShardId,
                              int deviceId,
                              int drop_host_after_upload,
                              int drop_host_after_release) {
    cudaError_t err = cudaSuccess;

    err = stage_shard(cache, view, files, inShardId, deviceId, drop_host_after_upload);
    if (err != cudaSuccess) return err;
    err = release_shard(cache, view, outShardId);
    if (err != cudaSuccess) return err;
    if (drop_host_after_release) {
        if (!::matrix::drop_shard(view, outShardId)) return cudaErrorInvalidValue;
    }
    return cudaSuccess;
}

} // namespace device
} // namespace matrix
