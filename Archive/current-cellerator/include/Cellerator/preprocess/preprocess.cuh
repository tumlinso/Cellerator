#pragma once

#include <Cellerator/compute/preprocess/preprocess.cuh>
#include <Cellerator/preprocess/aliases.hh>

namespace cellerator::preprocess {

using ::cellerator::compute::preprocess::CELLERATOR_PREPROCESS_MAX_QC_GROUPS;
using ::cellerator::compute::preprocess::cell_filter_params;
using ::cellerator::compute::preprocess::cell_metrics_view;
using ::cellerator::compute::preprocess::cell_qc_filter_params;
using ::cellerator::compute::preprocess::device_qc_group_config_view;
using ::cellerator::compute::preprocess::gene_filter_params;
using ::cellerator::compute::preprocess::gene_flag;
using ::cellerator::compute::preprocess::gene_flag_mito;
using ::cellerator::compute::preprocess::gene_flag_none;
using ::cellerator::compute::preprocess::gene_metric_packet_float_count;
using ::cellerator::compute::preprocess::gene_metric_packet_is_contiguous;
using ::cellerator::compute::preprocess::gene_metric_packet_view;
using ::cellerator::compute::preprocess::gene_metrics_view;
using ::cellerator::compute::preprocess::part_preprocess_result;
using ::cellerator::compute::preprocess::preprocess_fleet_config;
using ::cellerator::compute::preprocess::preprocess_fleet_result;
using ::cellerator::compute::preprocess::preprocess_fleet_workspace;
using ::cellerator::compute::preprocess::preprocess_ranked_nccl_config;
using ::cellerator::compute::preprocess::preprocess_execution_default;
using ::cellerator::compute::preprocess::preprocess_execution_fused;
using ::cellerator::compute::preprocess::preprocess_execution_plan;
using ::cellerator::compute::preprocess::preprocess_execution_separate;
using ::cellerator::compute::preprocess::preprocess_workspace;
using ::cellerator::compute::preprocess::qc_group_bit;
using ::cellerator::compute::preprocess::qc_group_config_view;
using ::cellerator::compute::preprocess::qc_group_hb;
using ::cellerator::compute::preprocess::qc_group_index;
using ::cellerator::compute::preprocess::qc_group_mt;
using ::cellerator::compute::preprocess::qc_group_ribo;

using ::cellerator::compute::preprocess::accumulate_gene_metrics;
using ::cellerator::compute::preprocess::accumulate_gene_metrics_compressed_fallback;
using ::cellerator::compute::preprocess::build_gene_filter_mask;
using ::cellerator::compute::preprocess::clear;
using ::cellerator::compute::preprocess::compute_cell_metrics;
using ::cellerator::compute::preprocess::compute_cell_metrics_compressed_fallback;
using ::cellerator::compute::preprocess::compute_qc_metrics;
using ::cellerator::compute::preprocess::compute_qc_metrics_compressed_fallback;
using ::cellerator::compute::preprocess::compute_qc_metrics_fleet;
using ::cellerator::compute::preprocess::finalize_gene_keep_mask_host;
using ::cellerator::compute::preprocess::init;
using ::cellerator::compute::preprocess::normalize_log1p_compressed_fallback_inplace;
using ::cellerator::compute::preprocess::normalize_log1p_inplace;
using ::cellerator::compute::preprocess::preprocess_blocked_ell_inplace;
using ::cellerator::compute::preprocess::preprocess_blocked_ell_qc_groups_fleet_inplace;
using ::cellerator::compute::preprocess::preprocess_blocked_ell_qc_groups_inplace;
using ::cellerator::compute::preprocess::preprocess_blocked_ell_qc_groups_plan_inplace;
using ::cellerator::compute::preprocess::preprocess_compressed_fallback_inplace;
using ::cellerator::compute::preprocess::preprocess_compressed_fallback_qc_groups_inplace;
using ::cellerator::compute::preprocess::preprocess_compressed_fallback_qc_groups_plan_inplace;
using ::cellerator::compute::preprocess::preprocess_sliced_ell_inplace;
using ::cellerator::compute::preprocess::preprocess_sliced_ell_qc_groups_fleet_inplace;
using ::cellerator::compute::preprocess::preprocess_sliced_ell_qc_groups_inplace;
using ::cellerator::compute::preprocess::preprocess_sliced_ell_qc_groups_plan_inplace;
using ::cellerator::compute::preprocess::reserve;
using ::cellerator::compute::preprocess::reserve_qc_groups;
using ::cellerator::compute::preprocess::setup;
using ::cellerator::compute::preprocess::setup_fleet;
using ::cellerator::compute::preprocess::upload_feature_group_masks;
using ::cellerator::compute::preprocess::upload_gene_flags;
using ::cellerator::compute::preprocess::zero_gene_metrics;

} // namespace cellerator::preprocess
