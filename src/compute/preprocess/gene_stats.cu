/*
Preprocess gene-stat kernel note, 2026-05-06, Tesla V100-SXM2-16GB sm_70.
These kernels remain the separate primitive/reference path for tests, partial
workflows, and comparisons against the fused high-level traversal. Validated
after the split with celleratorPreprocessRuntimeTest, QCFleetTest,
QCMetricsEquivalenceTest, and QCMaskGroupsTest; no numerical divergence beyond
existing test tolerances.
*/
#include "preprocess_internal.cuh"

#include "kernels/preprocess_math.cuh"

#include <cuda_fp16.h>

namespace cellerator::compute::preprocess {

namespace {

__global__ void accumulate_gene_metrics_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned long tid = (unsigned long) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = (unsigned long) (gridDim.x * blockDim.x);
    const unsigned long total = (unsigned long) src.rows * (unsigned long) src.ell_cols;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned long linear = tid;

    while (linear < total) {
        const unsigned int row = (unsigned int) (linear / src.ell_cols);
        const unsigned int ell_col = (unsigned int) (linear % src.ell_cols);
        const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
        const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
        const unsigned int lane = block_size != 0u ? ell_col % block_size : 0u;
        const unsigned int block_col = ell_width_blocks != 0u
            ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
            : cellshard::sparse::blocked_ell_invalid_col;
        const unsigned int col = block_col != cellshard::sparse::blocked_ell_invalid_col
            ? block_col * block_size + lane
            : src.cols;
        if ((keep_cells == nullptr || keep_cells[row] != 0u) && col < src.cols) {
            const float value = __half2float(src.val[linear]);
            if (value != 0.0f) {
                atomicAdd(gene_sum + col, value);
                atomicAdd(gene_detected + col, 1.0f);
                atomicAdd(gene_sq_sum + col, value * value);
            }
        }
        linear += stride;
    }
}

__global__ void accumulate_gene_metrics_sliced_ell_kernel(
    cs_device::sliced_ell_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        unsigned int slice = 0u, row_begin = 0u, width = 0u;
        unsigned long slot_base = 0ul;
        if (src.slice_count != 0u) {
            if (src.slice_rows == 32u) {
                slice = row >> 5;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else if (src.slice_rows != 0u) {
                slice = row / src.slice_rows;
                if (slice >= src.slice_count) slice = src.slice_count - 1u;
            } else {
                while (slice + 1u < src.slice_count && row >= src.slice_row_offsets[slice + 1u]) ++slice;
            }
            row_begin = src.slice_row_offsets[slice];
            width = src.slice_widths[slice];
            slot_base = (unsigned long) src.slice_slot_offsets[slice]
                + (unsigned long) (row - row_begin) * (unsigned long) width;
        }

        if (keep_cells == nullptr || keep_cells[row] != 0u) {
            for (unsigned int slot = lane; slot < width; slot += 32u) {
                const unsigned int col = src.col_idx[slot_base + slot];
                const float value = __half2float(src.val[slot_base + slot]);
                if (col < src.cols && value != 0.0f) {
                    atomicAdd(gene_sum + col, value);
                    atomicAdd(gene_detected + col, 1.0f);
                    atomicAdd(gene_sq_sum + col, value * value);
                }
            }
        }
        row += warp_stride;
    }
}

__global__ void accumulate_gene_metrics_compressed_kernel(
    cs_device::compressed_view src,
    const unsigned char *__restrict__ keep_cells,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum
) {
    const unsigned long tid = (unsigned long) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned long stride = (unsigned long) (gridDim.x * blockDim.x);
    unsigned long row = (unsigned long) blockIdx.y;

    while (row < src.rows) {
        if (keep_cells == nullptr || keep_cells[row] != 0u) {
            const unsigned int begin = src.majorPtr[row];
            const unsigned int end = src.majorPtr[row + 1u];
            for (unsigned long idx = begin + tid; idx < end; idx += stride) {
                const unsigned int col = src.minorIdx[idx];
                const float value = __half2float(src.val[idx]);
                if (col < src.cols && value != 0.0f) {
                    atomicAdd(gene_sum + col, value);
                    atomicAdd(gene_detected + col, 1.0f);
                    atomicAdd(gene_sq_sum + col, value * value);
                }
            }
        }
        row += (unsigned long) gridDim.y;
    }
}

__global__ void count_active_rows_kernel(unsigned int rows,
                                         const unsigned char *__restrict__ keep_cells,
                                         float *__restrict__ active_rows) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int row = tid;
    float local = 0.0f;
    while (row < rows) {
        local += keep_cells == nullptr || keep_cells[row] != 0u ? 1.0f : 0.0f;
        row += stride;
    }
    if (local != 0.0f) atomicAdd(active_rows, local);
}

__global__ void build_gene_filter_mask_kernel(unsigned int cols,
                                              const float *__restrict__ active_rows,
                                              gene_filter_params filter,
                                              const float *__restrict__ sum,
                                              const float *__restrict__ sq_sum,
                                              const float *__restrict__ detected_cells,
                                              unsigned char *__restrict__ keep) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    const float active = active_rows != nullptr ? active_rows[0] : 0.0f;
    const float inv_cells = active > 0.0f ? 1.0f / active : 0.0f;
    unsigned int gene = tid;

    while (gene < cols) {
        const float mean = sum[gene] * inv_cells;
        const float var = fmaxf(sq_sum[gene] * inv_cells - mean * mean, 0.0f);
        keep[gene] = (unsigned char) (sum[gene] >= filter.min_sum
                                      && detected_cells[gene] >= filter.min_detected_cells
                                      && var >= filter.min_variance);
        gene += stride;
    }
}

int update_active_rows(preprocess_workspace *workspace, unsigned int rows, const unsigned char *device_keep_cells) {
    if (workspace == nullptr || workspace->active_rows == nullptr) return 0;
    if (!cuda_ok(cudaMemsetAsync(workspace->active_rows, 0, sizeof(float), workspace->stream),
                 "cudaMemsetAsync active rows")) return 0;
    unsigned int blocks = (rows + 255u) >> 8;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    count_active_rows_kernel<<<blocks, 256, 0, workspace->stream>>>(rows, device_keep_cells, workspace->active_rows);
    return cuda_ok(cudaGetLastError(), "count_active_rows_kernel");
}

} // namespace

int accumulate_gene_metrics(const cs_device::blocked_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->rows * src->ell_cols)) return 0;
    if (!update_active_rows(workspace, src->rows, device_keep_cells)) return 0;
    unsigned int blocks = (unsigned int) ((((unsigned long) src->rows * (unsigned long) src->ell_cols) + 255ul) >> 8);
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    accumulate_gene_metrics_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_blocked_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int accumulate_gene_metrics(const cs_device::sliced_ell_view *src,
                            preprocess_workspace *workspace,
                            const unsigned char *device_keep_cells,
                            gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->nnz)) return 0;
    if (!update_active_rows(workspace, src->rows, device_keep_cells)) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    accumulate_gene_metrics_sliced_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_sliced_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int accumulate_gene_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                                preprocess_workspace *workspace,
                                                const unsigned char *device_keep_cells,
                                                gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    if (!reserve(workspace, src->rows, src->cols, src->nnz)) return 0;
    if (!update_active_rows(workspace, src->rows, device_keep_cells)) return 0;
    unsigned int blocks_x = (src->nnz + 255u) >> 8;
    if (blocks_x < 1u) blocks_x = 1u;
    if (blocks_x > 1024u) blocks_x = 1024u;
    unsigned int blocks_y = src->rows < 4096u ? src->rows : 4096u;
    if (blocks_y < 1u) blocks_y = 1u;
    accumulate_gene_metrics_compressed_kernel<<<dim3(blocks_x, blocks_y), 256, 0, workspace->stream>>>(
        *src,
        device_keep_cells,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum);
    if (!cuda_ok(cudaGetLastError(), "accumulate_gene_metrics_compressed_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int build_gene_filter_mask(preprocess_workspace *workspace,
                           unsigned int cols,
                           const gene_filter_params *filter,
                           gene_metrics_view *out) {
    if (workspace == nullptr || filter == nullptr) return 0;
    if (!reserve(workspace, workspace->rows_capacity, cols, workspace->values_capacity)) return 0;
    unsigned int blocks = (cols + 255u) >> 8;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    build_gene_filter_mask_kernel<<<blocks, 256, 0, workspace->stream>>>(
        cols,
        workspace->active_rows,
        *filter,
        workspace->gene_sum,
        workspace->gene_sq_sum,
        workspace->gene_detected,
        workspace->keep_genes);
    if (!cuda_ok(cudaGetLastError(), "build_gene_filter_mask_kernel")) return 0;
    bind_gene_metrics(workspace, cols, out);
    return 1;
}

int finalize_gene_keep_mask_host(const float *gene_sum,
                                 const float *gene_sq_sum,
                                 const float *gene_detected,
                                 unsigned int cols,
                                 float kept_cells,
                                 const gene_filter_params *filter,
                                 unsigned char *keep_genes,
                                 unsigned int *kept_genes) {
    if (gene_sum == nullptr || gene_sq_sum == nullptr || gene_detected == nullptr || filter == nullptr || keep_genes == nullptr) return 0;
    const float inv_cells = kept_cells > 0.0f ? 1.0f / kept_cells : 0.0f;
    unsigned int kept = 0u;
    for (unsigned int gene = 0u; gene < cols; ++gene) {
        const float mean = gene_sum[gene] * inv_cells;
        float var = gene_sq_sum[gene] * inv_cells - mean * mean;
        if (var < 0.0f) var = 0.0f;
        const unsigned char keep = (unsigned char) (gene_sum[gene] >= filter->min_sum
                                                    && gene_detected[gene] >= filter->min_detected_cells
                                                    && var >= filter->min_variance);
        keep_genes[gene] = keep;
        kept += keep != 0u ? 1u : 0u;
    }
    if (kept_genes != nullptr) *kept_genes = kept;
    return 1;
}

} // namespace cellerator::compute::preprocess
