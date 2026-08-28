#pragma once

#include "Cellerator/geometry/layout_metrics.hh"

#include <vector>

namespace cellpack {

struct layout_selector_config {
    double min_blocked_ell_fill = 0.55;
    double min_sliced_ell_fill = 0.45;
    double min_dense_tile_fill = 0.72;
    double max_structured_vs_csr_bytes = 0.95;
    u32 min_structured_nnz = 16u;
    u32 min_structured_rows = 2u;
    u32 dense_tile_multiple = 8u;
    bool allow_dense_tile = true;
    bool allow_tensor_core_candidate = true;
};

struct layout_selection_entry {
    u32 region_id = invalid_id;
    layout_kind selected_layout = layout_kind::unknown;
    u64 selected_estimated_bytes = 0u;
    bool tensor_core_candidate = false;
    launch_group_key launch_key{};
};

struct layout_selection_plan {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz = 0u;
    std::vector<layout_selection_entry> entries;
    layout_plan_summary summary{};
};

validation_result select_layouts(
    const layout_metrics_plan &metrics,
    const layout_selector_config &config,
    layout_selection_plan *out);

validation_result apply_layout_selection(
    const static_plan &source,
    const layout_selection_plan &selection,
    static_plan *out);

} // namespace cellpack
