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

bool launch_key_less(const launch_group_key &lhs, const launch_group_key &rhs) {
    if (lhs.layout != rhs.layout) return lhs.layout < rhs.layout;
    if (lhs.block_size != rhs.block_size) return lhs.block_size < rhs.block_size;
    if (lhs.width_class != rhs.width_class) return lhs.width_class < rhs.width_class;
    return lhs.output_columns < rhs.output_columns;
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
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "layout metrics output is null");
    }
    if (config.value_bytes == 0u || config.index_bytes == 0u || config.output_columns == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "layout metrics byte sizes must be nonzero");
    }
    if (config.sliced_ell_slice_height == 0u) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "Sliced-ELL slice height must be nonzero");
    }
    if (packed.row_count != plan.desc.row_count || packed.feature_count != plan.desc.feature_count) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id, "packed coordinate dimensions do not match plan");
    }
    validation_result desc_result = validate_plan_desc(plan.desc);
    if (!desc_result) return desc_result;
    validation_result region_result = validate_region_sequence(
        plan.regions.data(),
        static_cast<u32>(plan.regions.size()),
        plan.desc.row_count,
        plan.desc.feature_count);
    if (!region_result) return region_result;

    layout_metrics_plan metrics;
    metrics.row_count = plan.desc.row_count;
    metrics.feature_count = plan.desc.feature_count;
    metrics.nnz = static_cast<u32>(packed.coordinates.size());
    metrics.regions.reserve(plan.regions.size());

    std::vector<std::vector<u32>> row_widths_by_region(plan.regions.size());
    for (u32 i = 0; i < static_cast<u32>(plan.regions.size()); ++i) {
        row_widths_by_region[i].assign(plan.regions[i].row_count, 0u);
    }

    for (u32 entry = 0; entry < static_cast<u32>(packed.coordinates.size()); ++entry) {
        const packed_coordinate &coordinate = packed.coordinates[entry];
        const packed_region_desc *region = find_region(plan, coordinate.region_id);
        if (region == nullptr) {
            return validation_error(validation_code::missing_region, entry, "packed coordinate references an unknown region");
        }
        if (coordinate.permuted_row < region->row_begin
            || coordinate.permuted_row >= region->row_begin + region->row_count) {
            return validation_error(validation_code::invalid_region_bounds, entry, "packed coordinate row is outside its region");
        }
        const u32 local_row = coordinate.permuted_row - region->row_begin;
        row_widths_by_region[region->region_id][local_row] += 1u;
    }

    for (const packed_region_desc &region : plan.regions) {
        if (region.region_id >= row_widths_by_region.size()) {
            return validation_error(validation_code::invalid_region_bounds, region.region_id, "region id is outside metrics workspace");
        }
        const std::vector<u32> &row_widths = row_widths_by_region[region.region_id];
        region_layout_metrics region_metrics;
        region_metrics.region_id = region.region_id;
        region_metrics.source_layout = static_cast<layout_kind>(region.layout);
        region_metrics.role = static_cast<region_role>(region.role);
        region_metrics.row_count = region.row_count;
        region_metrics.feature_count = region.feature_count;
        region_metrics.row_widths.min_width = region.row_count == 0u ? 0u : std::numeric_limits<u32>::max();
        u64 sliced_slots = 0u;
        for (u32 row = 0; row < region.row_count; ++row) {
            const u32 width = row_widths[row];
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
                slice_width = std::max(slice_width, row_widths[local_row]);
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
        region_metrics.residual_fraction = metrics.nnz == 0u ? 0.0 : ratio(region_metrics.nnz, metrics.nnz);
        region_metrics.launch_key.layout = region_metrics.source_layout;
        region_metrics.launch_key.block_size = region.block_size;
        region_metrics.launch_key.width_class = static_cast<u32>(ell_width);
        region_metrics.launch_key.output_columns = config.output_columns;
        metrics.regions.push_back(region_metrics);
    }

    *out = std::move(metrics);
    return validation_ok();
}

layout_plan_summary summarize_layout_metrics(const layout_metrics_plan &metrics) {
    layout_plan_summary summary;
    summary.region_count = static_cast<u32>(metrics.regions.size());
    std::vector<launch_group_key> groups;
    groups.reserve(metrics.regions.size());
    double blocked_fill_sum = 0.0;
    u32 blocked_fill_count = 0u;
    u32 dense_candidate_nnz = 0u;
    bool have_blocked = false;

    for (const region_layout_metrics &region : metrics.regions) {
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
        groups.push_back(region.launch_key);
    }
    summary.mean_blocked_ell_fill = blocked_fill_count == 0u ? 0.0 : blocked_fill_sum / static_cast<double>(blocked_fill_count);
    summary.dense_tile_candidate_coverage = metrics.nnz == 0u ? 0.0 : ratio(dense_candidate_nnz, metrics.nnz);
    std::sort(groups.begin(), groups.end(), launch_key_less);
    groups.erase(std::unique(groups.begin(), groups.end(), launch_key_equal), groups.end());
    summary.launch_group_count = static_cast<u32>(groups.size());
    return summary;
}

} // namespace cellpack
