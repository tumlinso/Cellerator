#include "Cellerator/geometry/layout_selector.hh"

#include <algorithm>
#include <utility>

namespace cellpack {
namespace {

layout_selection_entry choose_layout(
    const region_layout_metrics &metrics,
    const layout_selector_config &config) {
    layout_selection_entry entry;
    entry.region_id = metrics.region_id;
    entry.selected_layout = layout_kind::residual_csr;
    entry.selected_estimated_bytes = csr_estimated_bytes(metrics);
    entry.launch_key.layout = layout_kind::residual_csr;
    entry.launch_key.block_size = 0u;
    entry.launch_key.width_class = 0u;
    entry.launch_key.output_columns = metrics.launch_key.output_columns;

    if (metrics.role == region_role::residual
        || metrics.nnz < config.min_structured_nnz
        || metrics.row_count < config.min_structured_rows) {
        return entry;
    }

    const u64 csr_bytes = csr_estimated_bytes(metrics);
    const u64 blocked_bytes = blocked_ell_estimated_bytes(metrics);
    const u64 sliced_bytes = sliced_ell_estimated_bytes(metrics);
    const u64 dense_bytes = dense_tile_estimated_bytes(metrics);
    const double byte_limit = static_cast<double>(csr_bytes) * config.max_structured_vs_csr_bytes;

    if (config.allow_dense_tile
        && metrics.dense_tile_fill_ratio >= config.min_dense_tile_fill
        && dense_bytes <= csr_bytes) {
        entry.selected_layout = layout_kind::dense_tile;
        entry.selected_estimated_bytes = dense_bytes;
        entry.tensor_core_candidate = config.allow_tensor_core_candidate
            && config.dense_tile_multiple != 0u
            && metrics.row_count % config.dense_tile_multiple == 0u
            && metrics.feature_count % config.dense_tile_multiple == 0u;
        entry.launch_key.layout = layout_kind::dense_tile;
        entry.launch_key.width_class = metrics.feature_count;
        return entry;
    }

    const bool blocked_beats_fallback = metrics.blocked_ell_fill_ratio >= config.min_blocked_ell_fill
        && static_cast<double>(blocked_bytes) <= byte_limit
        && blocked_bytes <= sliced_bytes;
    if (blocked_beats_fallback) {
        entry.selected_layout = layout_kind::blocked_ell;
        entry.selected_estimated_bytes = blocked_bytes;
        entry.launch_key.layout = layout_kind::blocked_ell;
        entry.launch_key.block_size = metrics.launch_key.block_size;
        entry.launch_key.width_class = metrics.launch_key.width_class;
        return entry;
    }

    const bool sliced_beats_fallback = metrics.sliced_ell_fill_ratio >= config.min_sliced_ell_fill
        && static_cast<double>(sliced_bytes) <= byte_limit;
    if (sliced_beats_fallback) {
        entry.selected_layout = layout_kind::sliced_ell;
        entry.selected_estimated_bytes = sliced_bytes;
        entry.launch_key.layout = layout_kind::sliced_ell;
        entry.launch_key.block_size = metrics.launch_key.block_size;
        entry.launch_key.width_class = metrics.row_widths.max_width;
        return entry;
    }

    return entry;
}

const layout_selection_entry *find_entry(const layout_selection_plan &selection, u32 region_id) {
    for (const layout_selection_entry &entry : selection.entries) {
        if (entry.region_id == region_id) return &entry;
    }
    return nullptr;
}

bool launch_key_equal(const launch_group_key &lhs, const launch_group_key &rhs) {
    return lhs.layout == rhs.layout
        && lhs.block_size == rhs.block_size
        && lhs.width_class == rhs.width_class
        && lhs.output_columns == rhs.output_columns;
}

} // namespace

validation_result select_layouts(
    const layout_metrics_plan &metrics,
    const layout_selector_config &config,
    layout_selection_plan *out) {
    if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "layout selection output is null");
    layout_selection_plan selection;
    selection.entries.resize(metrics.regions.size());
    layout_selection_plan_view view;
    validation_result result = select_layouts_into(
        view_layout_metrics(metrics), config,
        {{selection.entries.data(), selection.entries.size(), {}}}, &view);
    if (!result) return result;
    selection.row_count = view.row_count;
    selection.feature_count = view.feature_count;
    selection.nnz = view.nnz;
    selection.summary = view.summary;
    *out = std::move(selection);
    return validation_ok();
}

validation_result select_layouts_into(
    layout_metrics_plan_view metrics,
    const layout_selector_config &config,
    layout_selection_storage storage,
    layout_selection_plan_view *out) {
    if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "layout selection view output is null");
    if (config.min_blocked_ell_fill < 0.0 || config.min_blocked_ell_fill > 1.0
        || config.min_sliced_ell_fill < 0.0 || config.min_sliced_ell_fill > 1.0
        || config.min_dense_tile_fill < 0.0 || config.min_dense_tile_fill > 1.0
        || config.max_structured_vs_csr_bytes <= 0.0) {
        return validation_error(validation_code::invalid_layout, invalid_id, "layout selector thresholds are invalid");
    }

    if (storage.entries.count < metrics.regions.count) {
        return validation_error(validation_code::invalid_offsets, invalid_id, "layout selection storage capacity is insufficient");
    }
    if (metrics.regions.count != 0u && (metrics.regions.data == nullptr || storage.entries.data == nullptr)) {
        return validation_error(validation_code::null_pointer, invalid_id, "layout selection input or storage is null");
    }
    layout_plan_summary summary;
    u64 selected_total_bytes = 0u;
    u32 selected_residual_nnz = 0u;
    u32 dense_candidate_nnz = 0u;

    for (std::size_t index = 0; index < metrics.regions.count; ++index) {
        const region_layout_metrics &region = metrics.regions.data[index];
        layout_selection_entry entry = choose_layout(region, config);
        selected_total_bytes += entry.selected_estimated_bytes;
        if (entry.selected_layout == layout_kind::residual_csr) selected_residual_nnz += region.nnz;
        if (entry.selected_layout == layout_kind::dense_tile) dense_candidate_nnz += region.nnz;
        bool new_group = true;
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (launch_key_equal(storage.entries.data[prior].launch_key, entry.launch_key)) {
                new_group = false;
                break;
            }
        }
        summary.launch_group_count += new_group ? 1u : 0u;
        storage.entries.data[index] = entry;
    }

    summary.region_count = static_cast<u32>(metrics.regions.count);
    summary.residual_nnz_fraction = metrics.nnz == 0u
        ? 0.0
        : static_cast<double>(selected_residual_nnz) / static_cast<double>(metrics.nnz);
    summary.total_estimated_bytes = selected_total_bytes;
    summary.dense_tile_candidate_coverage = metrics.nnz == 0u
        ? 0.0
        : static_cast<double>(dense_candidate_nnz) / static_cast<double>(metrics.nnz);
    for (std::size_t index = 0; index < metrics.regions.count; ++index) {
        const region_layout_metrics &region = metrics.regions.data[index];
        summary.min_blocked_ell_fill = summary.min_blocked_ell_fill == 0.0
            ? region.blocked_ell_fill_ratio
            : std::min(summary.min_blocked_ell_fill, region.blocked_ell_fill_ratio);
        summary.max_blocked_ell_fill = std::max(summary.max_blocked_ell_fill, region.blocked_ell_fill_ratio);
        summary.mean_blocked_ell_fill += region.blocked_ell_fill_ratio;
        summary.min_sliced_ell_width = summary.min_sliced_ell_width == 0u
            ? region.row_widths.max_width
            : std::min(summary.min_sliced_ell_width, region.row_widths.max_width);
        summary.max_sliced_ell_width = std::max(summary.max_sliced_ell_width, region.row_widths.max_width);
    }
    if (metrics.regions.count != 0u) {
        summary.mean_blocked_ell_fill /= static_cast<double>(metrics.regions.count);
    }
    out->row_count = metrics.row_count;
    out->feature_count = metrics.feature_count;
    out->nnz = metrics.nnz;
    out->entries = {storage.entries.data, metrics.regions.count, storage.entries.where};
    out->summary = summary;
    return validation_ok();
}

layout_selection_plan_view view_layout_selection(const layout_selection_plan &selection) {
    return {selection.row_count, selection.feature_count, selection.nnz,
        {selection.entries.data(), selection.entries.size(), {}}, selection.summary};
}

validation_result apply_layout_selection(
    const static_plan &source,
    const layout_selection_plan &selection,
    static_plan *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "selected plan output is null");
    }
    if (selection.entries.size() != source.regions.size()) {
        return validation_error(validation_code::invalid_layout, invalid_id, "layout selection entry count does not match plan region count");
    }
    static_plan selected = source;
    for (packed_region_desc &region : selected.regions) {
        const layout_selection_entry *entry = find_entry(selection, region.region_id);
        if (entry == nullptr || !is_valid_layout(entry->selected_layout)) {
            return validation_error(validation_code::invalid_layout, region.region_id, "missing or invalid selected region layout");
        }
        region.layout = to_u32(entry->selected_layout);
        region.width_class = entry->launch_key.width_class;
        region.nnz_count = 0u;
        if (entry->selected_layout == layout_kind::dense_tile) {
            region.flags |= region_flag_dense_rhs_local;
        }
    }
    validation_result region_result = validate_region_sequence(
        selected.regions.data(),
        static_cast<u32>(selected.regions.size()),
        selected.desc.row_count,
        selected.desc.feature_count);
    if (!region_result) return region_result;
    *out = std::move(selected);
    return validation_ok();
}

} // namespace cellpack
