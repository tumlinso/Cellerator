#pragma once

#include "kernels.cuh"

#include "../../../extern/CellShard/src/convert/cusparse_utils.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cub/cub.cuh>

namespace cellerator {
namespace compute {
namespace preprocess {

namespace cscu = ::cellshard::convert::cusparse_utils;

static inline void init(device_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
    ws->device = -1;
}

static inline void clear(device_workspace *ws) {
    if (ws->device >= 0) cudaSetDevice(ws->device);
    if (ws->owns_stream && ws->stream != (cudaStream_t) 0) cudaStreamDestroy(ws->stream);
    if (ws->d_reduce_tmp != 0) cudaFree(ws->d_reduce_tmp);
    if (ws->d_spmv_tmp != 0) cudaFree(ws->d_spmv_tmp);
    if (ws->d_gene_block != 0) cudaFree(ws->d_gene_block);
    if (ws->d_cell_block != 0) cudaFree(ws->d_cell_block);
    init(ws);
}

static inline int setup(device_workspace *ws, int device, cudaStream_t stream = (cudaStream_t) 0) {
    clear(ws);
    if (!cscu::cuda_check(cudaSetDevice(device), "cudaSetDevice preprocess workspace")) return 0;
    ws->device = device;
    if (stream == (cudaStream_t) 0) {
        if (!cscu::cuda_check(cudaStreamCreateWithFlags(&ws->stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags preprocess")) return 0;
        ws->owns_stream = 1;
    } else {
        ws->stream = stream;
        ws->owns_stream = 0;
    }
    return 1;
}

static inline std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static inline int reserve(device_workspace *ws, unsigned int rows, unsigned int cols, unsigned int nnz) {
    std::size_t cell_bytes = 0;
    std::size_t gene_bytes = 0;
    char *cell_base = 0;
    char *gene_base = 0;

    if (!cscu::cuda_check(cudaSetDevice(ws->device >= 0 ? ws->device : 0), "cudaSetDevice preprocess reserve")) return 0;

    if (rows > ws->rows_capacity || nnz > ws->nnz_capacity) {
        if (ws->d_cell_block != 0) cudaFree(ws->d_cell_block);
        ws->d_cell_block = 0;
        ws->d_total_counts = 0;
        ws->d_mito_counts = 0;
        ws->d_max_counts = 0;
        ws->d_detected_genes = 0;
        ws->d_keep_cells = 0;
        ws->d_ones_rows = 0;
        ws->d_tmp_nnz = 0;
        ws->d_kept_cells_tmp = 0;
        ws->d_active_rows = 0;

        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(unsigned int));
        cell_bytes += (std::size_t) rows * sizeof(unsigned int);
        cell_bytes = align_up_bytes(cell_bytes, alignof(unsigned char));
        cell_bytes += (std::size_t) rows * sizeof(unsigned char);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += (std::size_t) nnz * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        cell_bytes += sizeof(float);

        if (cell_bytes != 0 && !cscu::cuda_check(cudaMalloc(&ws->d_cell_block, cell_bytes), "cudaMalloc preprocess cell block")) return 0;
        cell_base = (char *) ws->d_cell_block;
        cell_bytes = 0;
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_total_counts = (float *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_mito_counts = (float *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_max_counts = (float *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(unsigned int));
        ws->d_detected_genes = (unsigned int *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) rows * sizeof(unsigned int);
        cell_bytes = align_up_bytes(cell_bytes, alignof(unsigned char));
        ws->d_keep_cells = (unsigned char *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) rows * sizeof(unsigned char);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_ones_rows = (float *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) rows * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_tmp_nnz = (float *) (cell_base + cell_bytes);
        cell_bytes += (std::size_t) nnz * sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_kept_cells_tmp = (float *) (cell_base + cell_bytes);
        cell_bytes += sizeof(float);
        cell_bytes = align_up_bytes(cell_bytes, alignof(float));
        ws->d_active_rows = (float *) (cell_base + cell_bytes);

        ws->rows_capacity = rows;
        ws->nnz_capacity = nnz;
    }

    if (cols > ws->cols_capacity) {
        if (ws->d_gene_block != 0) cudaFree(ws->d_gene_block);
        ws->d_gene_block = 0;
        ws->d_gene_sum = 0;
        ws->d_gene_sq_sum = 0;
        ws->d_gene_detected = 0;
        ws->d_keep_genes = 0;
        ws->d_gene_flags = 0;

        gene_bytes = align_up_bytes(gene_bytes, alignof(float));
        gene_bytes += (std::size_t) cols * sizeof(float);
        gene_bytes = align_up_bytes(gene_bytes, alignof(float));
        gene_bytes += (std::size_t) cols * sizeof(float);
        gene_bytes = align_up_bytes(gene_bytes, alignof(float));
        gene_bytes += (std::size_t) cols * sizeof(float);
        gene_bytes = align_up_bytes(gene_bytes, alignof(unsigned char));
        gene_bytes += (std::size_t) cols * sizeof(unsigned char);
        gene_bytes = align_up_bytes(gene_bytes, alignof(unsigned char));
        gene_bytes += (std::size_t) cols * sizeof(unsigned char);

        if (gene_bytes != 0 && !cscu::cuda_check(cudaMalloc(&ws->d_gene_block, gene_bytes), "cudaMalloc preprocess gene block")) return 0;
        gene_base = (char *) ws->d_gene_block;
        gene_bytes = 0;
        gene_bytes = align_up_bytes(gene_bytes, alignof(float));
        ws->d_gene_sum = (float *) (gene_base + gene_bytes);
        gene_bytes += (std::size_t) cols * sizeof(float);
        gene_bytes = align_up_bytes(gene_bytes, alignof(float));
        ws->d_gene_sq_sum = (float *) (gene_base + gene_bytes);
        gene_bytes += (std::size_t) cols * sizeof(float);
        gene_bytes = align_up_bytes(gene_bytes, alignof(float));
        ws->d_gene_detected = (float *) (gene_base + gene_bytes);
        gene_bytes += (std::size_t) cols * sizeof(float);
        gene_bytes = align_up_bytes(gene_bytes, alignof(unsigned char));
        ws->d_keep_genes = (unsigned char *) (gene_base + gene_bytes);
        gene_bytes += (std::size_t) cols * sizeof(unsigned char);
        gene_bytes = align_up_bytes(gene_bytes, alignof(unsigned char));
        ws->d_gene_flags = (unsigned char *) (gene_base + gene_bytes);

        ws->cols_capacity = cols;
    }

    return 1;
}

static inline void bind_cell_metrics(device_workspace *ws, unsigned int rows, cell_metrics_view *out) {
    out->rows = rows;
    out->total_counts = ws->d_total_counts;
    out->mito_counts = ws->d_mito_counts;
    out->max_counts = ws->d_max_counts;
    out->detected_genes = ws->d_detected_genes;
    out->keep_cells = ws->d_keep_cells;
}

static inline void bind_gene_metrics(device_workspace *ws, unsigned int cols, gene_metrics_view *out) {
    out->cols = cols;
    out->sum = ws->d_gene_sum;
    out->sq_sum = ws->d_gene_sq_sum;
    out->detected_cells = ws->d_gene_detected;
    out->keep_genes = ws->d_keep_genes;
}

static inline int upload_gene_flags(device_workspace *ws, unsigned int cols, const unsigned char *host_flags) {
    if (!reserve(ws, ws->rows_capacity, cols, ws->nnz_capacity)) return 0;
    if (cols == 0) return 1;
    if (host_flags == 0) {
        return cscu::cuda_check(cudaMemsetAsync(ws->d_gene_flags, 0, (std::size_t) cols * sizeof(unsigned char), ws->stream), "cudaMemsetAsync gene flags");
    }
    // Asynchronous H2D for per-gene flags: low arithmetic value, so batching many tiny uploads is usually a net loss.
    return cscu::cuda_check(cudaMemcpyAsync(ws->d_gene_flags,
                                            host_flags,
                                            (std::size_t) cols * sizeof(unsigned char),
                                            cudaMemcpyHostToDevice,
                                            ws->stream),
                            "cudaMemcpyAsync gene flags");
}

static inline int zero_gene_metrics(device_workspace *ws, unsigned int cols) {
    if (!reserve(ws, ws->rows_capacity, cols, ws->nnz_capacity)) return 0;
    if (!cscu::cuda_check(cudaMemsetAsync(ws->d_gene_sum, 0, (std::size_t) cols * sizeof(float), ws->stream), "cudaMemsetAsync gene sum")) return 0;
    if (!cscu::cuda_check(cudaMemsetAsync(ws->d_gene_sq_sum, 0, (std::size_t) cols * sizeof(float), ws->stream), "cudaMemsetAsync gene sq sum")) return 0;
    if (!cscu::cuda_check(cudaMemsetAsync(ws->d_gene_detected, 0, (std::size_t) cols * sizeof(float), ws->stream), "cudaMemsetAsync gene detected")) return 0;
    if (!cscu::cuda_check(cudaMemsetAsync(ws->d_active_rows, 0, sizeof(float), ws->stream), "cudaMemsetAsync active rows")) return 0;
    return cscu::cuda_check(cudaMemsetAsync(ws->d_keep_genes, 0, (std::size_t) cols * sizeof(unsigned char), ws->stream), "cudaMemsetAsync gene keep");
}

static inline int zero_cell_metrics(device_workspace *ws, unsigned int rows) {
    if (!reserve(ws, rows, ws->cols_capacity, ws->nnz_capacity)) return 0;
    if (!cscu::cuda_check(cudaMemsetAsync(ws->d_keep_cells, 0, (std::size_t) rows * sizeof(unsigned char), ws->stream), "cudaMemsetAsync keep cells")) return 0;
    return 1;
}

static inline int ensure_spmv_tmp(device_workspace *ws,
                                  const csv::compressed_view *src,
                                  cudaDataType value_type,
                                  const void *values,
                                  const float *x,
                                  float *y) {
    cusparseHandle_t handle = 0;
    cusparseSpMatDescr_t mat = 0;
    cusparseDnVecDescr_t xvec = 0;
    cusparseDnVecDescr_t yvec = 0;
    std::size_t bytes = 0;
    const float alpha = 1.0f;
    const float beta = 1.0f;

    if (!cscu::acquire(ws->stream, &handle)) return 0;
    if (!cscu::check(cusparseCreateCsr(&mat,
                                       (int64_t) src->rows,
                                       (int64_t) src->cols,
                                       (int64_t) src->nnz,
                                       (void *) src->majorPtr,
                                       (void *) src->minorIdx,
                                       (void *) values,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       value_type),
                     "cusparseCreateCsr preprocess")) return 0;
    if (!cscu::check(cusparseCreateDnVec(&xvec, (int64_t) src->rows, (void *) x, CUDA_R_32F), "cusparseCreateDnVec x preprocess")) {
        (void) cusparseDestroySpMat(mat);
        return 0;
    }
    if (!cscu::check(cusparseCreateDnVec(&yvec, (int64_t) src->cols, (void *) y, CUDA_R_32F), "cusparseCreateDnVec y preprocess")) {
        (void) cusparseDestroyDnVec(xvec);
        (void) cusparseDestroySpMat(mat);
        return 0;
    }

    if (!cscu::check(cusparseSpMV_bufferSize(handle,
                                             CUSPARSE_OPERATION_TRANSPOSE,
                                             &alpha,
                                             mat,
                                             xvec,
                                             &beta,
                                             yvec,
                                             CUDA_R_32F,
                                             CUSPARSE_SPMV_ALG_DEFAULT,
                                             &bytes),
                     "cusparseSpMV_bufferSize preprocess")) {
        (void) cusparseDestroyDnVec(yvec);
        (void) cusparseDestroyDnVec(xvec);
        (void) cusparseDestroySpMat(mat);
        return 0;
    }

    (void) cusparseDestroyDnVec(yvec);
    (void) cusparseDestroyDnVec(xvec);
    (void) cusparseDestroySpMat(mat);

    // Workspace sizing can force cudaMalloc/free churn; the fast path is to reuse one buffer across many equal-or-smaller parts.
    if (bytes > ws->d_spmv_tmp_bytes) {
        if (ws->d_spmv_tmp != 0) cudaFree(ws->d_spmv_tmp);
        ws->d_spmv_tmp = 0;
        if (bytes != 0 && !cscu::cuda_check(cudaMalloc(&ws->d_spmv_tmp, bytes), "cudaMalloc preprocess spmv tmp")) return 0;
        ws->d_spmv_tmp_bytes = bytes;
    }
    return 1;
}

static inline int ensure_reduce_tmp(device_workspace *ws, unsigned int rows) {
    std::size_t bytes = 0;
    if (rows == 0) return 1;
    if (cub::DeviceReduce::Sum(0, bytes, ws->d_ones_rows, ws->d_kept_cells_tmp, rows, ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at DeviceReduce::Sum preprocess workspace sizing\n");
        return 0;
    }
    // Same rule as SpMV scratch: resizing here is amortized if callers keep part sizes stable.
    if (bytes > ws->d_reduce_tmp_bytes) {
        if (ws->d_reduce_tmp != 0) cudaFree(ws->d_reduce_tmp);
        ws->d_reduce_tmp = 0;
        if (bytes != 0 && !cscu::cuda_check(cudaMalloc(&ws->d_reduce_tmp, bytes), "cudaMalloc preprocess reduce tmp")) return 0;
        ws->d_reduce_tmp_bytes = bytes;
    }
    return 1;
}

static inline int run_spmv_transpose(device_workspace *ws,
                                     const csv::compressed_view *src,
                                     cudaDataType value_type,
                                     const void *values,
                                     const float *x,
                                     float *y) {
    cusparseHandle_t handle = 0;
    cusparseSpMatDescr_t mat = 0;
    cusparseDnVecDescr_t xvec = 0;
    cusparseDnVecDescr_t yvec = 0;
    const float alpha = 1.0f;
    const float beta = 1.0f;
    int ok = 0;

    if (!ensure_spmv_tmp(ws, src, value_type, values, x, y)) return 0;
    if (!cscu::acquire(ws->stream, &handle)) return 0;
    if (!cscu::check(cusparseCreateCsr(&mat,
                                       (int64_t) src->rows,
                                       (int64_t) src->cols,
                                       (int64_t) src->nnz,
                                       (void *) src->majorPtr,
                                       (void *) src->minorIdx,
                                       (void *) values,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_32I,
                                       CUSPARSE_INDEX_BASE_ZERO,
                                       value_type),
                     "cusparseCreateCsr preprocess run")) goto done;
    if (!cscu::check(cusparseCreateDnVec(&xvec, (int64_t) src->rows, (void *) x, CUDA_R_32F), "cusparseCreateDnVec x preprocess run")) goto done;
    if (!cscu::check(cusparseCreateDnVec(&yvec, (int64_t) src->cols, (void *) y, CUDA_R_32F), "cusparseCreateDnVec y preprocess run")) goto done;
    // Library path for the heavy lift in preprocess: expected to be bandwidth-bound and worth far more than the small helper launches around it.
    if (!cscu::check(cusparseSpMV(handle,
                                  CUSPARSE_OPERATION_TRANSPOSE,
                                  &alpha,
                                  mat,
                                  xvec,
                                  &beta,
                                  yvec,
                                  CUDA_R_32F,
                                  CUSPARSE_SPMV_ALG_DEFAULT,
                                  ws->d_spmv_tmp),
                     "cusparseSpMV preprocess")) goto done;
    ok = 1;

done:
    if (yvec != 0) (void) cusparseDestroyDnVec(yvec);
    if (xvec != 0) (void) cusparseDestroyDnVec(xvec);
    if (mat != 0) (void) cusparseDestroySpMat(mat);
    return ok;
}

static inline void init(distributed_workspace *ws) {
    std::memset(ws, 0, sizeof(*ws));
}

static inline void clear(distributed_workspace *ws) {
    if (ws->devices != 0) {
        for (unsigned int i = 0; i < ws->device_count; ++i) clear(ws->devices + i);
    }
    std::free(ws->devices);
    if (ws->h_block != 0) cudaFreeHost(ws->h_block);
    init(ws);
}

static inline int setup(distributed_workspace *ws, const csd::local_context *ctx) {
    if (ws == 0 || ctx == 0) return 0;
    clear(ws);
    if (ctx->device_count == 0) return 1;
    ws->devices = (device_workspace *) std::calloc((std::size_t) ctx->device_count, sizeof(device_workspace));
    if (ws->devices == 0) return 0;
    ws->device_count = ctx->device_count;
    for (unsigned int i = 0; i < ctx->device_count; ++i) {
        init(ws->devices + i);
        if (!setup(ws->devices + i, ctx->device_ids[i], ctx->streams != 0 ? ctx->streams[i] : (cudaStream_t) 0)) return 0;
    }
    return 1;
}

static inline int reserve(distributed_workspace *ws,
                          const csd::local_context *ctx,
                          unsigned int cols,
                          unsigned int max_rows_per_part,
                          unsigned int max_nnz_per_part) {
    std::size_t host_bytes = 0;
    char *host_base = 0;

    if (ws == 0 || ctx == 0) return 0;
    if (ws->devices == 0 && !setup(ws, ctx)) return 0;

    for (unsigned int i = 0; i < ws->device_count; ++i) {
        if (!reserve(ws->devices + i, max_rows_per_part, cols, max_nnz_per_part)) return 0;
    }

    if (cols > ws->cols_capacity) {
        if (ws->h_block != 0) cudaFreeHost(ws->h_block);
        ws->h_block = 0;
        ws->h_gene_sum = 0;
        ws->h_gene_sq_sum = 0;
        ws->h_gene_detected = 0;
        ws->h_active_rows = 0;

        host_bytes = align_up_bytes(host_bytes, alignof(float));
        host_bytes += (std::size_t) ws->device_count * cols * sizeof(float);
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        host_bytes += (std::size_t) ws->device_count * cols * sizeof(float);
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        host_bytes += (std::size_t) ws->device_count * cols * sizeof(float);
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        host_bytes += (std::size_t) ws->device_count * sizeof(float);
        if (host_bytes != 0 && !cscu::cuda_check(cudaMallocHost(&ws->h_block, host_bytes), "cudaMallocHost preprocess host reduce block")) return 0;

        host_base = (char *) ws->h_block;
        host_bytes = 0;
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        ws->h_gene_sum = (float *) (host_base + host_bytes);
        host_bytes += (std::size_t) ws->device_count * cols * sizeof(float);
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        ws->h_gene_sq_sum = (float *) (host_base + host_bytes);
        host_bytes += (std::size_t) ws->device_count * cols * sizeof(float);
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        ws->h_gene_detected = (float *) (host_base + host_bytes);
        host_bytes += (std::size_t) ws->device_count * cols * sizeof(float);
        host_bytes = align_up_bytes(host_bytes, alignof(float));
        ws->h_active_rows = (float *) (host_base + host_bytes);

        ws->cols_capacity = cols;
    }

    return 1;
}

static inline int upload_gene_flags(distributed_workspace *ws, unsigned int cols, const unsigned char *host_flags) {
    if (ws == 0) return 0;
    for (unsigned int i = 0; i < ws->device_count; ++i) {
        if (!upload_gene_flags(ws->devices + i, cols, host_flags)) return 0;
    }
    return 1;
}

static inline int zero_gene_metrics(distributed_workspace *ws, unsigned int cols) {
    if (ws == 0) return 0;
    for (unsigned int i = 0; i < ws->device_count; ++i) {
        if (!zero_gene_metrics(ws->devices + i, cols)) return 0;
    }
    return 1;
}

static inline int synchronize(const distributed_workspace *ws) {
    if (ws == 0) return 0;
    for (unsigned int i = 0; i < ws->device_count; ++i) {
        if (!cscu::cuda_check(cudaSetDevice(ws->devices[i].device), "cudaSetDevice preprocess sync")) return 0;
        if (!cscu::cuda_check(cudaStreamSynchronize(ws->devices[i].stream), "cudaStreamSynchronize preprocess")) return 0;
    }
    return 1;
}

static inline int allreduce_gene_metrics(distributed_workspace *ws, csd::local_context *ctx, unsigned int cols) {
    if (ws == 0 || ctx == 0 || cols == 0) return 0;

#if CELLSHARD_HAS_NCCL
    if (ctx->comms != 0 && ctx->nccl_ready != 0u) {
        ncclGroupStart();
        for (unsigned int i = 0; i < ws->device_count; ++i) {
            if (ncclAllReduce((const void *) ws->devices[i].d_gene_sum,
                              (void *) ws->devices[i].d_gene_sum,
                              (size_t) cols,
                              ncclFloat32,
                              ncclSum,
                              ctx->comms[i],
                              ws->devices[i].stream) != ncclSuccess) {
                ncclGroupEnd();
                return 0;
            }
            if (ncclAllReduce((const void *) ws->devices[i].d_gene_sq_sum,
                              (void *) ws->devices[i].d_gene_sq_sum,
                              (size_t) cols,
                              ncclFloat32,
                              ncclSum,
                              ctx->comms[i],
                              ws->devices[i].stream) != ncclSuccess) {
                ncclGroupEnd();
                return 0;
            }
            if (ncclAllReduce((const void *) ws->devices[i].d_gene_detected,
                              (void *) ws->devices[i].d_gene_detected,
                              (size_t) cols,
                              ncclFloat32,
                              ncclSum,
                              ctx->comms[i],
                              ws->devices[i].stream) != ncclSuccess) {
                ncclGroupEnd();
                return 0;
            }
            if (ncclAllReduce((const void *) ws->devices[i].d_active_rows,
                              (void *) ws->devices[i].d_active_rows,
                              1,
                              ncclFloat32,
                              ncclSum,
                              ctx->comms[i],
                              ws->devices[i].stream) != ncclSuccess) {
                ncclGroupEnd();
                return 0;
            }
        }
        if (ncclGroupEnd() != ncclSuccess) return 0;
        return synchronize(ws);
    }
#endif

    if (!synchronize(ws)) return 0;

    for (unsigned int i = 0; i < ws->device_count; ++i) {
        const std::size_t bytes = (std::size_t) cols * sizeof(float);
        if (!cscu::cuda_check(cudaSetDevice(ws->devices[i].device), "cudaSetDevice preprocess host reduce")) return 0;
        // Fallback multi-GPU reduction pays four blocking D2H copies per device; this is simple but PCIe-bound at scale.
        if (!cscu::cuda_check(cudaMemcpy(ws->h_gene_sum + (std::size_t) i * cols,
                                         ws->devices[i].d_gene_sum,
                                         bytes,
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy gene sum fallback")) return 0;
        if (!cscu::cuda_check(cudaMemcpy(ws->h_gene_sq_sum + (std::size_t) i * cols,
                                         ws->devices[i].d_gene_sq_sum,
                                         bytes,
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy gene sq sum fallback")) return 0;
        if (!cscu::cuda_check(cudaMemcpy(ws->h_gene_detected + (std::size_t) i * cols,
                                         ws->devices[i].d_gene_detected,
                                         bytes,
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy gene detected fallback")) return 0;
        if (!cscu::cuda_check(cudaMemcpy(ws->h_active_rows + i,
                                         ws->devices[i].d_active_rows,
                                         sizeof(float),
                                         cudaMemcpyDeviceToHost),
                              "cudaMemcpy active rows fallback")) return 0;
    }

    if (ws->device_count > 1u) {
        // Host accumulation is linear in device_count * cols and exists only as the non-NCCL fallback path.
        for (unsigned int dev = 1; dev < ws->device_count; ++dev) {
            for (unsigned int gene = 0; gene < cols; ++gene) {
                ws->h_gene_sum[gene] += ws->h_gene_sum[(std::size_t) dev * cols + gene];
                ws->h_gene_sq_sum[gene] += ws->h_gene_sq_sum[(std::size_t) dev * cols + gene];
                ws->h_gene_detected[gene] += ws->h_gene_detected[(std::size_t) dev * cols + gene];
            }
            ws->h_active_rows[0] += ws->h_active_rows[dev];
        }
    }

    for (unsigned int i = 0; i < ws->device_count; ++i) {
        const std::size_t bytes = (std::size_t) cols * sizeof(float);
        if (!cscu::cuda_check(cudaSetDevice(ws->devices[i].device), "cudaSetDevice preprocess host scatter")) return 0;
        // Scatter sends the reduced vectors back to every device; acceptable for one global barrier, expensive if done repeatedly per micro-batch.
        if (!cscu::cuda_check(cudaMemcpyAsync(ws->devices[i].d_gene_sum,
                                              ws->h_gene_sum,
                                              bytes,
                                              cudaMemcpyHostToDevice,
                                              ws->devices[i].stream),
                              "cudaMemcpyAsync gene sum scatter")) return 0;
        if (!cscu::cuda_check(cudaMemcpyAsync(ws->devices[i].d_gene_sq_sum,
                                              ws->h_gene_sq_sum,
                                              bytes,
                                              cudaMemcpyHostToDevice,
                                              ws->devices[i].stream),
                              "cudaMemcpyAsync gene sq sum scatter")) return 0;
        if (!cscu::cuda_check(cudaMemcpyAsync(ws->devices[i].d_gene_detected,
                                              ws->h_gene_detected,
                                              bytes,
                                              cudaMemcpyHostToDevice,
                                              ws->devices[i].stream),
                              "cudaMemcpyAsync gene detected scatter")) return 0;
        if (!cscu::cuda_check(cudaMemcpyAsync(ws->devices[i].d_active_rows,
                                              ws->h_active_rows,
                                              sizeof(float),
                                              cudaMemcpyHostToDevice,
                                              ws->devices[i].stream),
                              "cudaMemcpyAsync active rows scatter")) return 0;
    }

    return synchronize(ws);
}

} // namespace preprocess
} // namespace compute
} // namespace cellerator
