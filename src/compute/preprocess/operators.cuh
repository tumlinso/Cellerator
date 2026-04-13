#pragma once

#include "workspace.cuh"

namespace cellerator {
namespace compute {
namespace preprocess {

static inline int compute_cell_metrics(const csv::compressed_view *src,
                                       device_workspace *ws,
                                       const cell_filter_params &filter,
                                       cell_metrics_view *out) {
    unsigned int blocks = 0;
    const unsigned int threads = 256;

    if (src == 0 || ws == 0) return 0;
    if (src->axis != cs::sparse::compressed_by_row) return 0;
    if (!reserve(ws, src->rows, src->cols, src->nnz)) return 0;
    if (!zero_cell_metrics(ws, src->rows)) return 0;

    blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;

    // Row-parallel pass over CSR metadata and values; usually memory-bound, so launch overhead matters when parts are tiny.
    kernels::compute_cell_metrics_kernel<<<blocks, threads, 0, ws->stream>>>(
        *src,
        ws->d_gene_flags,
        filter,
        ws->d_total_counts,
        ws->d_mito_counts,
        ws->d_max_counts,
        ws->d_detected_genes,
        ws->d_keep_cells
    );
    if (!cscu::cuda_check(cudaGetLastError(), "compute_cell_metrics_kernel")) return 0;

    if (out != 0) bind_cell_metrics(ws, src->rows, out);
    return 1;
}

static inline int normalize_log1p_inplace(csv::compressed_view *src,
                                          device_workspace *ws,
                                          const float *d_total_counts,
                                          const unsigned char *d_keep_cells,
                                          float target_sum) {
    unsigned int blocks = 0;
    const unsigned int threads = 256;

    if (src == 0 || ws == 0 || d_total_counts == 0) return 0;
    if (src->axis != cs::sparse::compressed_by_row) return 0;

    blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;

    // In-place nnz transform with one row block schedule; cost scales with row span and nnz traffic, not with much arithmetic.
    kernels::normalize_log1p_kernel<<<blocks, threads, 0, ws->stream>>>(
        *src,
        d_total_counts,
        d_keep_cells,
        target_sum
    );
    return cscu::cuda_check(cudaGetLastError(), "normalize_log1p_kernel");
}

static inline int accumulate_gene_metrics(const csv::compressed_view *src,
                                          device_workspace *ws,
                                          const unsigned char *d_keep_cells,
                                          gene_metrics_view *out) {
    unsigned int blocks_rows = 0;
    unsigned int blocks_nnz = 0;
    const unsigned int threads = 256;

    if (src == 0 || ws == 0) return 0;
    if (src->axis != cs::sparse::compressed_by_row) return 0;
    if (!reserve(ws, src->rows, src->cols, src->nnz)) return 0;

    blocks_rows = (src->rows + threads - 1u) / threads;
    if (blocks_rows < 1u) blocks_rows = 1u;
    if (blocks_rows > 4096u) blocks_rows = 4096u;

    if (d_keep_cells != 0) {
        // Cheap mask expansion, but still a full-row launch before the heavier cuSPARSE reductions.
        kernels::expand_keep_mask_kernel<<<blocks_rows, threads, 0, ws->stream>>>(src->rows, d_keep_cells, ws->d_ones_rows);
        if (!cscu::cuda_check(cudaGetLastError(), "expand_keep_mask_kernel")) return 0;
    } else {
        // Pure fill pass; launch overhead dominates unless the part has enough rows to amortize it.
        kernels::fill_ones_kernel<<<blocks_rows, threads, 0, ws->stream>>>(src->rows, ws->d_ones_rows);
        if (!cscu::cuda_check(cudaGetLastError(), "fill_ones_kernel rows")) return 0;
    }

    if (!ensure_reduce_tmp(ws, src->rows)) return 0;
    if (cub::DeviceReduce::Sum(ws->d_reduce_tmp,
                               ws->d_reduce_tmp_bytes,
                               ws->d_ones_rows,
                               ws->d_kept_cells_tmp,
                               src->rows,
                               ws->stream) != cudaSuccess) {
        std::fprintf(stderr, "CUB error at DeviceReduce::Sum kept rows\n");
        return 0;
    }
    // Single-warp scalar fold after CUB; the math is trivial, so this launch mostly exists to keep the result on device.
    kernels::add_scalar_kernel<<<1, 32, 0, ws->stream>>>(ws->d_active_rows, ws->d_kept_cells_tmp);
    if (!cscu::cuda_check(cudaGetLastError(), "add_scalar_kernel active rows")) return 0;

    blocks_nnz = (src->nnz + threads - 1u) / threads;
    if (blocks_nnz < 1u) blocks_nnz = 1u;
    if (blocks_nnz > 4096u) blocks_nnz = 4096u;

    // Dense staging of nnz values for SpMV; bandwidth-bound and paid three times in this routine.
    kernels::convert_values_kernel<<<blocks_nnz, threads, 0, ws->stream>>>(src->nnz, src->val, ws->d_tmp_nnz);
    if (!cscu::cuda_check(cudaGetLastError(), "convert_values_kernel")) return 0;
    if (!run_spmv_transpose(ws, src, CUDA_R_32F, ws->d_tmp_nnz, ws->d_ones_rows, ws->d_gene_sum)) return 0;

    // Second full nnz pass for squared values; this is another read/write sweep before the transpose SpMV.
    kernels::square_values_kernel<<<blocks_nnz, threads, 0, ws->stream>>>(src->nnz, src->val, ws->d_tmp_nnz);
    if (!cscu::cuda_check(cudaGetLastError(), "square_values_kernel")) return 0;
    if (!run_spmv_transpose(ws, src, CUDA_R_32F, ws->d_tmp_nnz, ws->d_ones_rows, ws->d_gene_sq_sum)) return 0;

    // Third nnz pass builds detection counts; if this path gets hot, fusion pressure should focus here first.
    kernels::fill_ones_kernel<<<blocks_nnz, threads, 0, ws->stream>>>(src->nnz, ws->d_tmp_nnz);
    if (!cscu::cuda_check(cudaGetLastError(), "fill_ones_kernel nnz")) return 0;
    if (!run_spmv_transpose(ws, src, CUDA_R_32F, ws->d_tmp_nnz, ws->d_ones_rows, ws->d_gene_detected)) return 0;

    if (out != 0) bind_gene_metrics(ws, src->cols, out);
    return 1;
}

static inline int bind_uploaded_part_view(csv::compressed_view *out,
                                          const cs::sharded<cs::sparse::compressed> *host,
                                          const csv::partition_record<cs::sparse::compressed> *record,
                                          unsigned long part_id) {
    if (out == 0 || host == 0 || record == 0) return 0;
    if (part_id >= host->num_partitions) return 0;
    if (record->a0 == 0 && host->partition_rows[part_id] != 0) return 0;
    if (record->a2 == 0 && host->partition_nnz[part_id] != 0) return 0;
    out->rows = (unsigned int) host->partition_rows[part_id];
    out->cols = (unsigned int) host->cols;
    out->nnz = (unsigned int) host->partition_nnz[part_id];
    out->axis = (unsigned int) host->partition_aux[part_id];
    out->majorPtr = (unsigned int *) record->a0;
    out->minorIdx = (unsigned int *) record->a1;
    out->val = (__half *) record->a2;
    return 1;
}

static inline int preprocess_part_inplace(csv::compressed_view *src,
                                          device_workspace *ws,
                                          const cell_filter_params &cell_filter,
                                          float target_sum,
                                          part_preprocess_result *out) {
    part_preprocess_result local;

    if (src == 0 || ws == 0) return 0;
    if (!compute_cell_metrics(src, ws, cell_filter, &local.cell)) return 0;
    if (!normalize_log1p_inplace(src, ws, local.cell.total_counts, local.cell.keep_cells, target_sum)) return 0;
    if (!accumulate_gene_metrics(src, ws, local.cell.keep_cells, &local.gene)) return 0;
    local.active_rows = ws->d_active_rows;
    if (out != 0) *out = local;
    return 1;
}

static inline int build_gene_filter_mask(device_workspace *ws,
                                         unsigned int cols,
                                         const gene_filter_params &filter,
                                         gene_metrics_view *out) {
    unsigned int blocks = 0;
    float active_rows = 0.0f;
    float inv_cells = 0.0f;

    if (ws == 0) return 0;
    if (!reserve(ws, ws->rows_capacity, cols, ws->nnz_capacity)) return 0;
    if (!cscu::cuda_check(cudaMemsetAsync(ws->d_keep_genes, 0, (std::size_t) cols * sizeof(unsigned char), ws->stream), "cudaMemsetAsync gene keep rebuild")) return 0;
    // Host readback of one scalar forces stream completion; cheap in bytes, expensive in overlap.
    if (!cscu::cuda_check(cudaMemcpyAsync(&active_rows,
                                          ws->d_active_rows,
                                          sizeof(float),
                                          cudaMemcpyDeviceToHost,
                                          ws->stream),
                          "cudaMemcpyAsync active rows host")) return 0;
    if (!cscu::cuda_check(cudaStreamSynchronize(ws->stream), "cudaStreamSynchronize active rows host")) return 0;
    inv_cells = active_rows > 0.0f ? 1.0f / active_rows : 0.0f;

    blocks = (cols + 255u) >> 8;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;

    // Column-parallel thresholding over aggregate metrics; usually light compared with the earlier SpMV passes.
    kernels::build_gene_filter_mask_kernel<<<blocks, 256, 0, ws->stream>>>(
        cols,
        inv_cells,
        filter,
        ws->d_gene_sum,
        ws->d_gene_sq_sum,
        ws->d_gene_detected,
        ws->d_keep_genes
    );
    if (!cscu::cuda_check(cudaGetLastError(), "build_gene_filter_mask_kernel")) return 0;

    if (out != 0) bind_gene_metrics(ws, cols, out);
    return 1;
}

} // namespace preprocess
} // namespace compute
} // namespace cellerator
