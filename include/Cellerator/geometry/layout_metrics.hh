#pragma once

#include "Cellerator/geometry/pack.hh"
#include "Cellerator/memory/view.hh"
#include "Cellerator/memory/workspace.hh"

#include <vector>

namespace cellpack {

struct layout_metrics_config {
    u32 value_bytes = 4u;
    u32 index_bytes = 4u;
    u32 output_columns = 1u;
    u32 blocked_ell_width_alignment = 1u;
    u32 sliced_ell_slice_height = 2u;
};

struct row_width_distribution {
    u32 min_width = 0u;
    u32 max_width = 0u;
    u64 width_sum = 0u;
    u64 squared_width_sum = 0u;
};

struct launch_group_key {
    layout_kind layout = layout_kind::unknown;
    u32 block_size = 0u;
    u32 width_class = 0u;
    u32 output_columns = 1u;
};

struct region_layout_metrics {
    u32 region_id = invalid_id;
    layout_kind source_layout = layout_kind::unknown;
    region_role role = region_role::unknown;
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz = 0u;
    row_width_distribution row_widths{};
    u64 blocked_ell_padded_slots = 0u;
    u64 sliced_ell_padded_slots = 0u;
    u64 dense_tile_slots = 0u;
    double blocked_ell_fill_ratio = 0.0;
    double sliced_ell_fill_ratio = 0.0;
    double dense_tile_fill_ratio = 0.0;
    u64 csr_index_bytes = 0u;
    u64 csr_value_bytes = 0u;
    u64 blocked_ell_index_bytes = 0u;
    u64 blocked_ell_value_bytes = 0u;
    u64 sliced_ell_index_bytes = 0u;
    u64 sliced_ell_value_bytes = 0u;
    u64 dense_tile_value_bytes = 0u;
    u64 estimated_output_bytes = 0u;
    double residual_fraction = 0.0;
    launch_group_key launch_key{};
};

struct layout_metrics_plan {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz = 0u;
    std::vector<region_layout_metrics> regions;
};

struct layout_metrics_plan_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz = 0u;
    ::cellerator::memory::const_array_view<region_layout_metrics> regions;
};

struct layout_metrics_storage {
    ::cellerator::memory::array_view<region_layout_metrics> regions;
};

struct layout_plan_summary {
    u32 region_count = 0u;
    u32 launch_group_count = 0u;
    double residual_nnz_fraction = 0.0;
    u64 total_estimated_bytes = 0u;
    double min_blocked_ell_fill = 0.0;
    double max_blocked_ell_fill = 0.0;
    double mean_blocked_ell_fill = 0.0;
    u32 min_sliced_ell_width = 0u;
    u32 max_sliced_ell_width = 0u;
    double dense_tile_candidate_coverage = 0.0;
};

validation_result build_layout_metrics(
    const static_plan &plan,
    const packed_coordinate_plan &packed,
    const layout_metrics_config &config,
    layout_metrics_plan *out);

::cellerator::memory::workspace_requirement layout_metrics_workspace_requirement(
    const static_plan &plan,
    ::cellerator::memory::placement where = {});

validation_result build_layout_metrics_into(
    const static_plan &plan,
    packed_coordinate_plan_view packed,
    const layout_metrics_config &config,
    layout_metrics_storage storage,
    ::cellerator::memory::workspace workspace,
    layout_metrics_plan_view *out);

layout_metrics_plan_view view_layout_metrics(const layout_metrics_plan &metrics);

layout_plan_summary summarize_layout_metrics(const layout_metrics_plan &metrics);
layout_plan_summary summarize_layout_metrics(layout_metrics_plan_view metrics);

u64 csr_estimated_bytes(const region_layout_metrics &metrics);
u64 blocked_ell_estimated_bytes(const region_layout_metrics &metrics);
u64 sliced_ell_estimated_bytes(const region_layout_metrics &metrics);
u64 dense_tile_estimated_bytes(const region_layout_metrics &metrics);

} // namespace cellpack
