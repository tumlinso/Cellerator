#pragma once

#include <Cellerator/compute/preprocess/preprocess.cuh>

#include <CellShard/runtime/mask_groups.cuh>

#include <cstdio>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace cellerator::compute::preprocess {

namespace cs_runtime = ::cellshard::runtime;

inline std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

inline int cuda_ok(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "Cellerator preprocess CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

inline void bind_cell_metrics(preprocess_workspace *workspace, unsigned int rows, unsigned int group_count, cell_metrics_view *out) {
    if (out == nullptr) return;
    out->rows = rows;
    out->group_count = group_count;
    out->total_counts = workspace->total_counts;
    out->mito_counts = group_count != 0u ? workspace->cell_group_counts : workspace->mito_counts;
    out->max_counts = workspace->max_counts;
    out->detected_genes = workspace->detected_genes;
    out->keep_cells = workspace->keep_cells;
    out->cell_group_counts = workspace->cell_group_counts;
    out->cell_group_pct = workspace->cell_group_pct;
}

inline void bind_gene_metrics(preprocess_workspace *workspace, unsigned int cols, gene_metrics_view *out) {
    if (out == nullptr) return;
    out->cols = cols;
    out->sum = workspace->gene_sum;
    out->sq_sum = workspace->gene_sq_sum;
    out->detected_cells = workspace->gene_detected;
    out->keep_genes = workspace->keep_genes;
    out->feature_group_masks = workspace->feature_group_masks;
    out->gene_flags = workspace->gene_flags;
}

inline int ensure_runtime_mask_workspace(preprocess_workspace *workspace) {
    if (workspace == nullptr) return 0;
    if (workspace->mask_groups.device >= 0) return 1;
    if (workspace->device < 0) return 0;
    return cs_runtime::setup(&workspace->mask_groups, workspace->device, workspace->stream);
}

inline void alias_runtime_group_outputs(preprocess_workspace *workspace,
                                        const cs_runtime::sparse_group_reduce_result *runtime) {
    if (workspace == nullptr || runtime == nullptr) return;
    workspace->total_counts = runtime->row_totals;
    workspace->max_counts = runtime->max_values;
    workspace->detected_genes = runtime->detected_features;
    workspace->keep_cells = runtime->row_keep;
    workspace->cell_group_counts = runtime->group_counts;
    workspace->cell_group_pct = runtime->group_percentages;
}

inline cs_runtime::group_mask_config_view translate_groups(const qc_group_config_view *groups,
                                                           const preprocess_workspace *workspace) {
    cs_runtime::group_mask_config_view out{};
    if (groups == nullptr) return out;
    out.group_count = groups->group_count;
    out.group_names = groups->group_names;
    out.feature_group_masks = workspace != nullptr ? workspace->mask_groups.feature_group_masks : nullptr;
    return out;
}

inline cs_runtime::sparse_group_filter_params translate_filter(const cell_qc_filter_params *filter) {
    cs_runtime::sparse_group_filter_params out{};
    if (filter == nullptr) return out;
    out.min_total = filter->min_counts;
    out.min_detected_features = filter->min_features;
    out.max_group_fraction = filter->max_group_fraction;
    out.fraction_group_index = filter->fraction_group_index;
    return out;
}

inline int prepare_qc_metric_buffers(preprocess_workspace *workspace,
                                     unsigned int rows,
                                     unsigned int cols,
                                     unsigned int values,
                                     const qc_group_config_view *groups,
                                     unsigned int *requested_groups) {
    if (workspace == nullptr || requested_groups == nullptr) return 0;
    *requested_groups = groups != nullptr ? groups->group_count : 0u;
    if (*requested_groups > CELLERATOR_PREPROCESS_MAX_QC_GROUPS) return 0;
    if (!reserve_qc_groups(workspace, rows, cols, values, *requested_groups)) return 0;
    if (groups != nullptr && groups->feature_group_masks != nullptr
        && groups->feature_group_masks != workspace->feature_group_masks
        && !upload_feature_group_masks(workspace, cols, groups->feature_group_masks)) {
        return 0;
    }
    if (groups != nullptr && groups->explicit_feature_group_masks != nullptr) {
        if (!upload_feature_group_masks(workspace, cols, groups->explicit_feature_group_masks)) return 0;
    } else if (groups == nullptr || groups->feature_group_masks == nullptr) {
        if (!upload_feature_group_masks(workspace, cols, nullptr)) return 0;
    }
    if (!cuda_ok(cudaMemsetAsync(workspace->keep_cells, 0, (std::size_t) rows, workspace->stream),
                 "cudaMemsetAsync keep cells")) return 0;
    if (*requested_groups != 0u) {
        const std::size_t group_values = (std::size_t) rows * *requested_groups;
        if (!cuda_ok(cudaMemsetAsync(workspace->cell_group_counts,
                                     0,
                                     group_values * sizeof(float),
                                     workspace->stream),
                     "cudaMemsetAsync cell group counts")) return 0;
        if (!cuda_ok(cudaMemsetAsync(workspace->cell_group_pct,
                                     0,
                                     group_values * sizeof(float),
                                     workspace->stream),
                     "cudaMemsetAsync cell group pct")) return 0;
    }
    return 1;
}

void bind_fleet_result(preprocess_fleet_workspace *fleet, unsigned int leader_index, unsigned int cols, preprocess_fleet_result *out);
int reduce_gene_metrics_to_leader(preprocess_fleet_workspace *fleet, unsigned int cols, unsigned int leader_index);

} // namespace cellerator::compute::preprocess
