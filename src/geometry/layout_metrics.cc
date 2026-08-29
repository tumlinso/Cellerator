#include "Cellerator/geometry/layout_metrics.hh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

namespace cellpack {
namespace {

u64 align_up_u64(u64 value, u32 alignment) {
    if (alignment <= 1u) return value;
    const u64 align = static_cast<u64>(alignment);
    return ((value + align - 1u) / align) * align;
}

double ratio(u64 numerator, u64 denominator) {
    return denominator == 0u ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator);
}

const packed_region_desc *find_region(const static_plan &plan, u32 region_id) {
    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id == region_id) return &region;
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

u64 csr_estimated_bytes(const region_layout_metrics &metrics) {
    return metrics.csr_index_bytes + metrics.csr_value_bytes + metrics.estimated_output_bytes;
}

u64 blocked_ell_estimated_bytes(const region_layout_metrics &metrics) {
    return metrics.blocked_ell_index_bytes + metrics.blocked_ell_value_bytes + metrics.estimated_output_bytes;
}

u64 sliced_ell_estimated_bytes(const region_layout_metrics &metrics) {
    return metrics.sliced_ell_index_bytes + metrics.sliced_ell_value_bytes + metrics.estimated_output_bytes;
}

u64 dense_tile_estimated_bytes(const region_layout_metrics &metrics) {
    return metrics.dense_tile_value_bytes + metrics.estimated_output_bytes;
}

validation_result build_layout_metrics(
    const static_plan &plan,
    const packed_coordinate_plan &packed,
    const layout_metrics_config &config,
    layout_metrics_plan *out) {
    layout_metrics_plan metrics;
    metrics.regions.resize(plan.regions.size());
    const auto requirement = layout_metrics_workspace_requirement(plan);
    std::vector<unsigned char> workspace_bytes(requirement.bytes + requirement.alignment - 1u);
    ::cellerator::memory::workspace workspace{
        workspace_bytes.data(), workspace_bytes.size(), 0u, {}};
    layout_metrics_plan_view view;
    validation_result result = build_layout_metrics_into(
        plan, view_packed_coordinates(packed), config,
        {{metrics.regions.data(), metrics.regions.size(), {}}}, workspace, &view);
    if (!result) return result;
    metrics.row_count = view.row_count;
    metrics.feature_count = view.feature_count;
    metrics.nnz = view.nnz;
    *out = std::move(metrics);
    return validation_ok();
}

::cellerator::memory::workspace_requirement layout_metrics_workspace_requirement(
    const static_plan &plan,
    ::cellerator::memory::placement where) {
    std::size_t row_count = 0u;
    for (const packed_region_desc &region : plan.regions) row_count += region.row_count;
    return {(plan.regions.size() + 1u + row_count) * sizeof(u32), alignof(u32), where};
}

validation_result build_layout_metrics_into(
    const static_plan &plan,
    packed_coordinate_plan_view packed,
    const layout_metrics_config &config,
    layout_metrics_storage storage,
    ::cellerator::memory::workspace workspace,
    layout_metrics_plan_view *out) {
    if (out == nullptr) return validation_error(validation_code::null_pointer, invalid_id, "layout metrics view output is null");
    if (config.value_bytes == 0u || config.index_bytes == 0u || config.output_columns == 0u
        || config.sliced_ell_slice_height == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "layout metrics configuration contains zero sizes");
    }
    if (packed.row_count != plan.desc.row_count || packed.feature_count != plan.desc.feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "packed coordinate dimensions do not match plan");
    }
    if (packed.coordinates.count != 0u && packed.coordinates.data == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "packed coordinate storage is null");
    }
    validation_result desc_result = validate_plan_desc(plan.desc);
    if (!desc_result) return desc_result;
    validation_result region_result = validate_region_sequence(
        plan.regions.data(), static_cast<u32>(plan.regions.size()),
        plan.desc.row_count, plan.desc.feature_count);
    if (!region_result) return region_result;
    if (storage.regions.count < plan.regions.size()) {
        return validation_error(validation_code::invalid_offsets, invalid_id, "layout metrics storage capacity is insufficient");
    }
    if (!plan.regions.empty() && storage.regions.data == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "layout metrics storage is null");
    }

    u32 *row_begins = nullptr;
    const std::size_t total_rows = [&]() {
        std::size_t count = 0u;
        for (const packed_region_desc &region : plan.regions) count += region.row_count;
        return count;
    }();
    if (::cellerator::memory::take(&workspace, plan.regions.size() + 1u, &row_begins)
            != ::cellerator::memory::status::success) {
        return validation_error(validation_code::invalid_offsets, invalid_id, "layout metrics workspace is insufficient for region offsets");
    }
    u32 *row_widths = nullptr;
    if (::cellerator::memory::take(&workspace, total_rows, &row_widths)
            != ::cellerator::memory::status::success) {
        return validation_error(validation_code::invalid_offsets, invalid_id, "layout metrics workspace is insufficient for row widths");
    }
    row_begins[0] = 0u;
    for (std::size_t i = 0; i < plan.regions.size(); ++i) {
        row_begins[i + 1u] = row_begins[i] + plan.regions[i].row_count;
    }
    std::fill(row_widths, row_widths + total_rows, 0u);
    for (u32 entry = 0; entry < static_cast<u32>(packed.coordinates.count); ++entry) {
        const packed_coordinate &coordinate = packed.coordinates.data[entry];
        const packed_region_desc *region = find_region(plan, coordinate.region_id);
        if (region == nullptr) return validation_error(validation_code::missing_region, entry, "packed coordinate references an unknown region");
        if (region->region_id >= plan.regions.size() || coordinate.permuted_row < region->row_begin
            || coordinate.permuted_row >= region->row_begin + region->row_count) {
            return validation_error(validation_code::invalid_region_bounds, entry, "packed coordinate row is outside its region");
        }
        ++row_widths[row_begins[region->region_id] + coordinate.permuted_row - region->row_begin];
    }

    for (const packed_region_desc &region : plan.regions) {
        const u32 *region_row_widths = row_widths + row_begins[region.region_id];
        region_layout_metrics region_metrics;
        region_metrics.region_id = region.region_id;
        region_metrics.source_layout = static_cast<layout_kind>(region.layout);
        region_metrics.role = static_cast<region_role>(region.role);
        region_metrics.row_count = region.row_count;
        region_metrics.feature_count = region.feature_count;
        region_metrics.row_widths.min_width = region.row_count == 0u ? 0u : std::numeric_limits<u32>::max();
        u64 sliced_slots = 0u;
        for (u32 row = 0; row < region.row_count; ++row) {
            const u32 width = region_row_widths[row];
            region_metrics.nnz += width;
            region_metrics.row_widths.min_width = std::min(region_metrics.row_widths.min_width, width);
            region_metrics.row_widths.max_width = std::max(region_metrics.row_widths.max_width, width);
            region_metrics.row_widths.width_sum += width;
            region_metrics.row_widths.squared_width_sum += static_cast<u64>(width) * static_cast<u64>(width);
        }
        for (u32 row = 0; row < region.row_count; row += config.sliced_ell_slice_height) {
            const u32 slice_end = std::min(region.row_count, row + config.sliced_ell_slice_height);
            u32 slice_width = 0u;
            for (u32 local_row = row; local_row < slice_end; ++local_row) {
                slice_width = std::max(slice_width, region_row_widths[local_row]);
            }
            sliced_slots += static_cast<u64>(slice_width) * static_cast<u64>(slice_end - row);
        }
        if (region.row_count == 0u) region_metrics.row_widths.min_width = 0u;

        const u64 ell_width = align_up_u64(region_metrics.row_widths.max_width, config.blocked_ell_width_alignment);
        region_metrics.blocked_ell_padded_slots = ell_width * static_cast<u64>(region.row_count);
        region_metrics.sliced_ell_padded_slots = sliced_slots;
        region_metrics.dense_tile_slots = static_cast<u64>(region.row_count) * static_cast<u64>(region.feature_count);
        region_metrics.blocked_ell_fill_ratio = ratio(region_metrics.nnz, region_metrics.blocked_ell_padded_slots);
        region_metrics.sliced_ell_fill_ratio = ratio(region_metrics.nnz, region_metrics.sliced_ell_padded_slots);
        region_metrics.dense_tile_fill_ratio = ratio(region_metrics.nnz, region_metrics.dense_tile_slots);
        region_metrics.csr_index_bytes = (static_cast<u64>(region_metrics.nnz) + static_cast<u64>(region.row_count) + 1u)
            * static_cast<u64>(config.index_bytes);
        region_metrics.csr_value_bytes = static_cast<u64>(region_metrics.nnz) * static_cast<u64>(config.value_bytes);
        region_metrics.blocked_ell_index_bytes = region_metrics.blocked_ell_padded_slots * static_cast<u64>(config.index_bytes);
        region_metrics.blocked_ell_value_bytes = region_metrics.blocked_ell_padded_slots * static_cast<u64>(config.value_bytes);
        region_metrics.sliced_ell_index_bytes = region_metrics.sliced_ell_padded_slots * static_cast<u64>(config.index_bytes);
        region_metrics.sliced_ell_value_bytes = region_metrics.sliced_ell_padded_slots * static_cast<u64>(config.value_bytes);
        region_metrics.dense_tile_value_bytes = region_metrics.dense_tile_slots * static_cast<u64>(config.value_bytes);
        region_metrics.estimated_output_bytes = static_cast<u64>(region.row_count)
            * static_cast<u64>(config.output_columns)
            * static_cast<u64>(config.value_bytes);
        region_metrics.residual_fraction = packed.coordinates.count == 0u ? 0.0 : ratio(region_metrics.nnz, packed.coordinates.count);
        region_metrics.launch_key.layout = region_metrics.source_layout;
        region_metrics.launch_key.block_size = region.block_size;
        region_metrics.launch_key.width_class = static_cast<u32>(ell_width);
        region_metrics.launch_key.output_columns = config.output_columns;
        storage.regions.data[region.region_id] = region_metrics;
    }
    out->row_count = plan.desc.row_count;
    out->feature_count = plan.desc.feature_count;
    out->nnz = static_cast<u32>(packed.coordinates.count);
    out->regions = {storage.regions.data, plan.regions.size(), storage.regions.where};
    return validation_ok();
}

layout_metrics_plan_view view_layout_metrics(const layout_metrics_plan &metrics) {
    return {metrics.row_count, metrics.feature_count, metrics.nnz,
        {metrics.regions.data(), metrics.regions.size(), {}}};
}

layout_plan_summary summarize_layout_metrics(const layout_metrics_plan &metrics) {
    return summarize_layout_metrics(view_layout_metrics(metrics));
}

layout_plan_summary summarize_layout_metrics(layout_metrics_plan_view metrics) {
    layout_plan_summary summary;
    summary.region_count = static_cast<u32>(metrics.regions.count);
    double blocked_fill_sum = 0.0;
    u32 blocked_fill_count = 0u;
    u32 dense_candidate_nnz = 0u;
    bool have_blocked = false;

    for (std::size_t index = 0; index < metrics.regions.count; ++index) {
        const region_layout_metrics &region = metrics.regions.data[index];
        summary.total_estimated_bytes += csr_estimated_bytes(region);
        if (region.source_layout == layout_kind::residual_csr || region.role == region_role::residual) {
            summary.residual_nnz_fraction += region.residual_fraction;
        }
        if (region.source_layout == layout_kind::blocked_ell) {
            summary.min_blocked_ell_fill = have_blocked
                ? std::min(summary.min_blocked_ell_fill, region.blocked_ell_fill_ratio)
                : region.blocked_ell_fill_ratio;
            summary.max_blocked_ell_fill = have_blocked
                ? std::max(summary.max_blocked_ell_fill, region.blocked_ell_fill_ratio)
                : region.blocked_ell_fill_ratio;
            blocked_fill_sum += region.blocked_ell_fill_ratio;
            ++blocked_fill_count;
            have_blocked = true;
        }
        summary.min_sliced_ell_width = summary.min_sliced_ell_width == 0u
            ? region.row_widths.max_width
            : std::min(summary.min_sliced_ell_width, region.row_widths.max_width);
        summary.max_sliced_ell_width = std::max(summary.max_sliced_ell_width, region.row_widths.max_width);
        if (region.dense_tile_fill_ratio >= 0.72) dense_candidate_nnz += region.nnz;
        bool new_group = true;
        for (std::size_t prior = 0; prior < index; ++prior) {
            if (launch_key_equal(metrics.regions.data[prior].launch_key, region.launch_key)) {
                new_group = false;
                break;
            }
        }
        summary.launch_group_count += new_group ? 1u : 0u;
    }
    summary.mean_blocked_ell_fill = blocked_fill_count == 0u ? 0.0 : blocked_fill_sum / static_cast<double>(blocked_fill_count);
    summary.dense_tile_candidate_coverage = metrics.nnz == 0u ? 0.0 : ratio(dense_candidate_nnz, metrics.nnz);
    return summary;
}

} // namespace cellpack
