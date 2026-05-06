#include "preprocess_internal.cuh"

namespace cellerator::compute::preprocess {

int compute_cell_metrics(const cs_device::blocked_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace->feature_group_masks;
    cell_qc_filter_params generic_filter{filter->min_counts, filter->min_genes, filter->max_mito_fraction, qc_group_mt};
    return compute_qc_metrics(src, workspace, &groups, &generic_filter, out);
}

int compute_cell_metrics(const cs_device::sliced_ell_view *src,
                         preprocess_workspace *workspace,
                         const cell_filter_params *filter,
                         cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace->feature_group_masks;
    cell_qc_filter_params generic_filter{filter->min_counts, filter->min_genes, filter->max_mito_fraction, qc_group_mt};
    return compute_qc_metrics(src, workspace, &groups, &generic_filter, out);
}

int compute_cell_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                             preprocess_workspace *workspace,
                                             const cell_filter_params *filter,
                                             cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    qc_group_config_view groups{};
    groups.group_count = 1u;
    groups.feature_group_masks = workspace->feature_group_masks;
    cell_qc_filter_params generic_filter{filter->min_counts, filter->min_genes, filter->max_mito_fraction, qc_group_mt};
    return compute_qc_metrics_compressed_fallback(src, workspace, &groups, &generic_filter, out);
}

int compute_qc_metrics(const cs_device::blocked_ell_view *src,
                       preprocess_workspace *workspace,
                       const qc_group_config_view *groups,
                       const cell_qc_filter_params *filter,
                       cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    unsigned int requested_groups = 0u;
    if (!prepare_qc_metric_buffers(workspace, src->rows, src->cols, src->rows * src->ell_cols, groups, &requested_groups)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    cs_runtime::group_mask_config_view runtime_groups = translate_groups(groups, workspace);
    runtime_groups.group_count = requested_groups;
    cs_runtime::sparse_group_filter_params runtime_filter = translate_filter(filter);
    cs_runtime::sparse_group_reduce_result runtime_result{};
    if (!cs_runtime::compute_sparse_group_reduce(src,
                                                 &workspace->mask_groups,
                                                 &runtime_groups,
                                                 nullptr,
                                                 &runtime_filter,
                                                 &runtime_result)) return 0;
    alias_runtime_group_outputs(workspace, &runtime_result);
    bind_cell_metrics(workspace, src->rows, runtime_result.group_count, out);
    return 1;
}

int compute_qc_metrics(const cs_device::sliced_ell_view *src,
                       preprocess_workspace *workspace,
                       const qc_group_config_view *groups,
                       const cell_qc_filter_params *filter,
                       cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    unsigned int requested_groups = 0u;
    if (!prepare_qc_metric_buffers(workspace, src->rows, src->cols, src->nnz, groups, &requested_groups)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    cs_runtime::group_mask_config_view runtime_groups = translate_groups(groups, workspace);
    runtime_groups.group_count = requested_groups;
    cs_runtime::sparse_group_filter_params runtime_filter = translate_filter(filter);
    cs_runtime::sparse_group_reduce_result runtime_result{};
    if (!cs_runtime::compute_sparse_group_reduce(src,
                                                 &workspace->mask_groups,
                                                 &runtime_groups,
                                                 nullptr,
                                                 &runtime_filter,
                                                 &runtime_result)) return 0;
    alias_runtime_group_outputs(workspace, &runtime_result);
    bind_cell_metrics(workspace, src->rows, runtime_result.group_count, out);
    return 1;
}

int compute_qc_metrics_compressed_fallback(const cs_device::compressed_view *src,
                                           preprocess_workspace *workspace,
                                           const qc_group_config_view *groups,
                                           const cell_qc_filter_params *filter,
                                           cell_metrics_view *out) {
    if (src == nullptr || workspace == nullptr || filter == nullptr) return 0;
    if (src->axis != cellshard::sparse::compressed_by_row) return 0;
    unsigned int requested_groups = 0u;
    if (!prepare_qc_metric_buffers(workspace, src->rows, src->cols, src->nnz, groups, &requested_groups)) return 0;
    if (!ensure_runtime_mask_workspace(workspace)) return 0;
    cs_runtime::group_mask_config_view runtime_groups = translate_groups(groups, workspace);
    runtime_groups.group_count = requested_groups;
    cs_runtime::sparse_group_filter_params runtime_filter = translate_filter(filter);
    cs_runtime::sparse_group_reduce_result runtime_result{};
    if (!cs_runtime::compute_sparse_group_reduce_compressed_fallback(src,
                                                                    &workspace->mask_groups,
                                                                    &runtime_groups,
                                                                    nullptr,
                                                                    &runtime_filter,
                                                                    &runtime_result)) return 0;
    alias_runtime_group_outputs(workspace, &runtime_result);
    bind_cell_metrics(workspace, src->rows, runtime_result.group_count, out);
    return 1;
}

int compute_qc_metrics_fleet(const cs_device::blocked_ell_view *src_by_slot,
                             preprocess_fleet_workspace *fleet,
                             const qc_group_config_view *groups,
                             const cell_qc_filter_params *filter,
                             preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
        if (!compute_qc_metrics(src_by_slot + i, fleet->devices + i, groups, filter, &fleet->results[i].cell)) return 0;
        fleet->results[i].gene = gene_metrics_view{};
    }
    bind_fleet_result(fleet, 0u, 0u, out);
    return 1;
}

int compute_qc_metrics_fleet(const cs_device::sliced_ell_view *src_by_slot,
                             preprocess_fleet_workspace *fleet,
                             const qc_group_config_view *groups,
                             const cell_qc_filter_params *filter,
                             preprocess_fleet_result *out) {
    if (src_by_slot == nullptr || fleet == nullptr || filter == nullptr || fleet->slot_count == 0u || fleet->devices == nullptr) return 0;
    const unsigned int cols = src_by_slot[0].cols;
    for (unsigned int i = 0u; i < fleet->slot_count; ++i) {
        if (src_by_slot[i].cols != cols) return 0;
        if (!compute_qc_metrics(src_by_slot + i, fleet->devices + i, groups, filter, &fleet->results[i].cell)) return 0;
        fleet->results[i].gene = gene_metrics_view{};
    }
    bind_fleet_result(fleet, 0u, 0u, out);
    return 1;
}

} // namespace cellerator::compute::preprocess
