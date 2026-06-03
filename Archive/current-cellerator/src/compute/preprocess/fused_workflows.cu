/*
Preprocess fused sparse traversal note, 2026-05-06, Tesla V100-SXM2-16GB sm_70.
Reference path is the existing compute_qc_metrics -> normalize_log1p_inplace ->
accumulate_gene_metrics primitive sequence. The fused high-level path keeps the
same QC semantics and separate primitive APIs, then combines normalize/log1p,
gene-stat accumulation, and active-row count in one layout-owned sparse pass.
Validated with celleratorPreprocessRuntimeTest, QCFleetTest,
QCMetricsEquivalenceTest, QCMaskGroupsTest, and CellShardSessionApiTest; no
numerical divergence beyond the existing test tolerances.
*/
#include "preprocess_internal.cuh"

#include "kernels/preprocess_math.cuh"

#include <cuda_fp16.h>

namespace cellerator::compute::preprocess {

namespace {

__global__ void fused_normalize_gene_metrics_blocked_ell_kernel(
    cs_device::blocked_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum,
    float *__restrict__ active_rows
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    const unsigned int block_size = src.block_size;
    const unsigned int ell_width_blocks = block_size != 0u ? src.ell_cols / block_size : 0u;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const int keep = kernels::row_is_kept(keep_cells, row);
        if (keep && lane == 0u) atomicAdd(active_rows, 1.0f);
        if (keep) {
            for (unsigned int ell_col = lane; ell_col < src.ell_cols; ell_col += 32u) {
                const unsigned long offset = (unsigned long) row * src.ell_cols + ell_col;
                const float value = __half2float(src.val[offset]);
                const float normalized = kernels::normalized_log1p_value(value, total_counts[row], target_sum);
                src.val[offset] = __float2half(normalized);
                const unsigned int row_block = block_size != 0u ? row / block_size : 0u;
                const unsigned int slot = block_size != 0u ? ell_col / block_size : 0u;
                const unsigned int col_lane = block_size != 0u ? ell_col % block_size : 0u;
                const unsigned int block_col = ell_width_blocks != 0u
                    ? src.blockColIdx[(unsigned long) row_block * ell_width_blocks + slot]
                    : cellerator::core::matrix::blocked_ell_invalid_col;
                const unsigned int col = block_col != cellerator::core::matrix::blocked_ell_invalid_col
                    ? block_col * block_size + col_lane
                    : src.cols;
                kernels::accumulate_gene_stat(normalized, col, src.cols, gene_sum, gene_detected, gene_sq_sum);
            }
        }
        row += warp_stride;
    }
}

__global__ void fused_normalize_gene_metrics_sliced_ell_kernel(
    cs_device::sliced_ell_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum,
    float *__restrict__ active_rows
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

        const int keep = kernels::row_is_kept(keep_cells, row);
        if (keep && lane == 0u) atomicAdd(active_rows, 1.0f);
        if (keep) {
            for (unsigned int slot = lane; slot < width; slot += 32u) {
                const unsigned long offset = slot_base + slot;
                const unsigned int col = src.col_idx[offset];
                const float value = __half2float(src.val[offset]);
                const float normalized = kernels::normalized_log1p_value(value, total_counts[row], target_sum);
                if (col < src.cols) src.val[offset] = __float2half(normalized);
                kernels::accumulate_gene_stat(normalized, col, src.cols, gene_sum, gene_detected, gene_sq_sum);
            }
        }
        row += warp_stride;
    }
}

__global__ void fused_normalize_gene_metrics_compressed_kernel(
    cs_device::compressed_view src,
    const float *__restrict__ total_counts,
    const unsigned char *__restrict__ keep_cells,
    float target_sum,
    float *__restrict__ gene_sum,
    float *__restrict__ gene_detected,
    float *__restrict__ gene_sq_sum,
    float *__restrict__ active_rows
) {
    const unsigned int lane = (unsigned int) (threadIdx.x & 31);
    const unsigned int warps_per_block = (unsigned int) (blockDim.x >> 5);
    const unsigned int warp_in_block = (unsigned int) (threadIdx.x >> 5);
    const unsigned int warp_global = blockIdx.x * warps_per_block + warp_in_block;
    const unsigned int warp_stride = gridDim.x * warps_per_block;
    unsigned int row = warp_global;

    while (row < src.rows) {
        const int keep = kernels::row_is_kept(keep_cells, row);
        if (keep && lane == 0u) atomicAdd(active_rows, 1.0f);
        if (keep) {
            const unsigned int end = src.majorPtr[row + 1u];
            for (unsigned int idx = src.majorPtr[row] + lane; idx < end; idx += 32u) {
                const unsigned int col = src.minorIdx[idx];
                const float value = __half2float(src.val[idx]);
                const float normalized = kernels::normalized_log1p_value(value, total_counts[row], target_sum);
                src.val[idx] = __float2half(normalized);
                kernels::accumulate_gene_stat(normalized, col, src.cols, gene_sum, gene_detected, gene_sq_sum);
            }
        }
        row += warp_stride;
    }
}

int fused_normalize_gene_metrics(cs_device::blocked_ell_view *src,
                                 preprocess_workspace *workspace,
                                 const float *device_total_counts,
                                 const unsigned char *device_keep_cells,
                                 float target_sum,
                                 gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess fused blocked ell")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    fused_normalize_gene_metrics_blocked_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum,
        workspace->active_rows);
    if (!cuda_ok(cudaGetLastError(), "fused_normalize_gene_metrics_blocked_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int fused_normalize_gene_metrics(cs_device::sliced_ell_view *src,
                                 preprocess_workspace *workspace,
                                 const float *device_total_counts,
                                 const unsigned char *device_keep_cells,
                                 float target_sum,
                                 gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess fused sliced ell")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    fused_normalize_gene_metrics_sliced_ell_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum,
        workspace->active_rows);
    if (!cuda_ok(cudaGetLastError(), "fused_normalize_gene_metrics_sliced_ell_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

int fused_normalize_gene_metrics_compressed_fallback(cs_device::compressed_view *src,
                                                     preprocess_workspace *workspace,
                                                     const float *device_total_counts,
                                                     const unsigned char *device_keep_cells,
                                                     float target_sum,
                                                     gene_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || device_total_counts == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    if (!cuda_ok(cudaSetDevice(workspace->device >= 0 ? workspace->device : 0), "cudaSetDevice preprocess fused compressed")) return 0;
    unsigned int blocks = (src->rows + 7u) >> 3;
    if (blocks < 1u) blocks = 1u;
    if (blocks > 4096u) blocks = 4096u;
    fused_normalize_gene_metrics_compressed_kernel<<<blocks, 256, 0, workspace->stream>>>(
        *src,
        device_total_counts,
        device_keep_cells,
        target_sum,
        workspace->gene_sum,
        workspace->gene_detected,
        workspace->gene_sq_sum,
        workspace->active_rows);
    if (!cuda_ok(cudaGetLastError(), "fused_normalize_gene_metrics_compressed_kernel")) return 0;
    bind_gene_metrics(workspace, src->cols, out);
    return 1;
}

} // namespace

int preprocess_blocked_ell_inplace(cs_device::blocked_ell_view *src,
                                   preprocess_workspace *workspace,
                                   const cell_filter_params *cell_filter,
                                   float target_sum,
                                   part_preprocess_result *out) {
    if (cell_filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace != nullptr ? workspace->feature_group_masks : nullptr;
    cell_qc_filter_params generic_filter{cell_filter->min_counts, cell_filter->min_genes, cell_filter->max_mito_fraction, qc_group_mt};
    return preprocess_blocked_ell_qc_groups_inplace(src, workspace, &groups, &generic_filter, target_sum, out);
}

int preprocess_sliced_ell_inplace(cs_device::sliced_ell_view *src,
                                  preprocess_workspace *workspace,
                                  const cell_filter_params *cell_filter,
                                  float target_sum,
                                  part_preprocess_result *out) {
    if (cell_filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace != nullptr ? workspace->feature_group_masks : nullptr;
    cell_qc_filter_params generic_filter{cell_filter->min_counts, cell_filter->min_genes, cell_filter->max_mito_fraction, qc_group_mt};
    return preprocess_sliced_ell_qc_groups_inplace(src, workspace, &groups, &generic_filter, target_sum, out);
}

int preprocess_compressed_fallback_inplace(cs_device::compressed_view *src,
                                           preprocess_workspace *workspace,
                                           const cell_filter_params *cell_filter,
                                           float target_sum,
                                           part_preprocess_result *out) {
    if (cell_filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace != nullptr ? workspace->feature_group_masks : nullptr;
    cell_qc_filter_params generic_filter{cell_filter->min_counts, cell_filter->min_genes, cell_filter->max_mito_fraction, qc_group_mt};
    return preprocess_compressed_fallback_qc_groups_inplace(src, workspace, &groups, &generic_filter, target_sum, out);
}

int preprocess_blocked_ell_qc_groups_inplace(cs_device::blocked_ell_view *src,
                                             preprocess_workspace *workspace,
                                             const qc_group_config_view *groups,
                                             const cell_qc_filter_params *cell_filter,
                                             float target_sum,
                                             part_preprocess_result *out) {
    return preprocess_blocked_ell_qc_groups_plan_inplace(
        src, workspace, groups, cell_filter, target_sum, preprocess_execution_default, out);
}

int preprocess_blocked_ell_qc_groups_plan_inplace(cs_device::blocked_ell_view *src,
                                                  preprocess_workspace *workspace,
                                                  const qc_group_config_view *groups,
                                                  const cell_qc_filter_params *cell_filter,
                                                  float target_sum,
                                                  preprocess_execution_plan plan,
                                                  part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_qc_metrics(src, workspace, groups, cell_filter, &cell)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (plan == preprocess_execution_separate) {
        if (!normalize_log1p_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
        if (!accumulate_gene_metrics(src, workspace, cell.keep_cells, &gene)) return 0;
    } else if (!fused_normalize_gene_metrics(src, workspace, cell.total_counts, cell.keep_cells, target_sum, &gene)) {
        return 0;
    }
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int preprocess_sliced_ell_qc_groups_inplace(cs_device::sliced_ell_view *src,
                                            preprocess_workspace *workspace,
                                            const qc_group_config_view *groups,
                                            const cell_qc_filter_params *cell_filter,
                                            float target_sum,
                                            part_preprocess_result *out) {
    return preprocess_sliced_ell_qc_groups_plan_inplace(
        src, workspace, groups, cell_filter, target_sum, preprocess_execution_default, out);
}

int preprocess_sliced_ell_qc_groups_plan_inplace(cs_device::sliced_ell_view *src,
                                                 preprocess_workspace *workspace,
                                                 const qc_group_config_view *groups,
                                                 const cell_qc_filter_params *cell_filter,
                                                 float target_sum,
                                                 preprocess_execution_plan plan,
                                                 part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_qc_metrics(src, workspace, groups, cell_filter, &cell)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (plan == preprocess_execution_separate) {
        if (!normalize_log1p_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
        if (!accumulate_gene_metrics(src, workspace, cell.keep_cells, &gene)) return 0;
    } else if (!fused_normalize_gene_metrics(src, workspace, cell.total_counts, cell.keep_cells, target_sum, &gene)) {
        return 0;
    }
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int preprocess_compressed_fallback_qc_groups_inplace(cs_device::compressed_view *src,
                                                     preprocess_workspace *workspace,
                                                     const qc_group_config_view *groups,
                                                     const cell_qc_filter_params *cell_filter,
                                                     float target_sum,
                                                     part_preprocess_result *out) {
    return preprocess_compressed_fallback_qc_groups_plan_inplace(
        src, workspace, groups, cell_filter, target_sum, preprocess_execution_default, out);
}

int preprocess_compressed_fallback_qc_groups_plan_inplace(cs_device::compressed_view *src,
                                                          preprocess_workspace *workspace,
                                                          const qc_group_config_view *groups,
                                                          const cell_qc_filter_params *cell_filter,
                                                          float target_sum,
                                                          preprocess_execution_plan plan,
                                                          part_preprocess_result *out) {
    cell_metrics_view cell{};
    gene_metrics_view gene{};
    if (!compute_qc_metrics_compressed_fallback(src, workspace, groups, cell_filter, &cell)) return 0;
    if (!zero_gene_metrics(workspace, src->cols)) return 0;
    if (plan == preprocess_execution_separate) {
        if (!normalize_log1p_compressed_fallback_inplace(src, workspace, cell.total_counts, cell.keep_cells, target_sum)) return 0;
        if (!accumulate_gene_metrics_compressed_fallback(src, workspace, cell.keep_cells, &gene)) return 0;
    } else if (!fused_normalize_gene_metrics_compressed_fallback(src, workspace, cell.total_counts, cell.keep_cells, target_sum, &gene)) {
        return 0;
    }
    if (out != nullptr) {
        out->cell = cell;
        out->gene = gene;
    }
    return 1;
}

int preprocess_blocked_ell_qc_groups_fleet_inplace(cs_device::blocked_ell_view *src_by_slot,
                                                   preprocess_fleet_workspace *fleet,
                                                   const qc_group_config_view *groups,
                                                   const cell_qc_filter_params *cell_filter,
                                                   float target_sum,
                                                   preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || cell_filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
    }
    if (!compute_qc_metrics_fleet(src_by_slot, fleet, groups, cell_filter, nullptr)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (!zero_gene_metrics(fleet->devices + i, cols)) return 0;
        if (!fused_normalize_gene_metrics(src_by_slot + i,
                                          fleet->devices + i,
                                          fleet->results[i].cell.total_counts,
                                          fleet->results[i].cell.keep_cells,
                                          target_sum,
                                          &fleet->results[i].gene)) {
            return 0;
        }
    }
    if (!reduce_gene_metrics_to_leader(fleet, cols, 0u)) return 0;
    bind_fleet_result(fleet, 0u, cols, out);
    return 1;
}

int preprocess_sliced_ell_qc_groups_fleet_inplace(cs_device::sliced_ell_view *src_by_slot,
                                                  preprocess_fleet_workspace *fleet,
                                                  const qc_group_config_view *groups,
                                                  const cell_qc_filter_params *cell_filter,
                                                  float target_sum,
                                                  preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || cell_filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
    }
    if (!compute_qc_metrics_fleet(src_by_slot, fleet, groups, cell_filter, nullptr)) return 0;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (!zero_gene_metrics(fleet->devices + i, cols)) return 0;
        if (!fused_normalize_gene_metrics(src_by_slot + i,
                                          fleet->devices + i,
                                          fleet->results[i].cell.total_counts,
                                          fleet->results[i].cell.keep_cells,
                                          target_sum,
                                          &fleet->results[i].gene)) {
            return 0;
        }
    }
    if (!reduce_gene_metrics_to_leader(fleet, cols, 0u)) return 0;
    bind_fleet_result(fleet, 0u, cols, out);
    return 1;
}

} // namespace cellerator::compute::preprocess
