#include <CellPack/layout_selector.hh>

#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

cellpack::region_layout_metrics make_region(
    cellpack::u32 region_id,
    cellpack::u32 row_count,
    cellpack::u32 feature_count,
    cellpack::u32 nnz,
    cellpack::u64 blocked_slots,
    cellpack::u64 sliced_slots,
    cellpack::region_role role = cellpack::region_role::primary) {
    cellpack::region_layout_metrics region;
    region.region_id = region_id;
    region.source_layout = cellpack::layout_kind::blocked_ell;
    region.role = role;
    region.row_count = row_count;
    region.feature_count = feature_count;
    region.nnz = nnz;
    region.row_widths.max_width = row_count == 0u ? 0u : static_cast<cellpack::u32>(blocked_slots / row_count);
    region.blocked_ell_padded_slots = blocked_slots;
    region.sliced_ell_padded_slots = sliced_slots;
    region.dense_tile_slots = static_cast<cellpack::u64>(row_count) * static_cast<cellpack::u64>(feature_count);
    region.blocked_ell_fill_ratio = blocked_slots == 0u ? 0.0 : static_cast<double>(nnz) / static_cast<double>(blocked_slots);
    region.sliced_ell_fill_ratio = sliced_slots == 0u ? 0.0 : static_cast<double>(nnz) / static_cast<double>(sliced_slots);
    region.dense_tile_fill_ratio = region.dense_tile_slots == 0u ? 0.0 : static_cast<double>(nnz) / static_cast<double>(region.dense_tile_slots);
    region.csr_index_bytes = (static_cast<cellpack::u64>(nnz) + static_cast<cellpack::u64>(row_count) + 1u) * 4u;
    region.csr_value_bytes = static_cast<cellpack::u64>(nnz) * 4u;
    region.blocked_ell_index_bytes = blocked_slots * 4u;
    region.blocked_ell_value_bytes = blocked_slots * 4u;
    region.sliced_ell_index_bytes = sliced_slots * 4u;
    region.sliced_ell_value_bytes = sliced_slots * 4u;
    region.dense_tile_value_bytes = region.dense_tile_slots * 4u;
    region.estimated_output_bytes = static_cast<cellpack::u64>(row_count) * 4u;
    region.launch_key.layout = cellpack::layout_kind::blocked_ell;
    region.launch_key.width_class = region.row_widths.max_width;
    region.launch_key.output_columns = 1u;
    return region;
}

} // namespace

int main() {
    cellpack::layout_metrics_plan metrics;
    metrics.row_count = 32u;
    metrics.feature_count = 64u;
    metrics.regions.push_back(make_region(0u, 8u, 8u, 4u, 64u, 32u));
    metrics.regions.push_back(make_region(1u, 16u, 8u, 96u, 128u, 160u));
    metrics.regions.push_back(make_region(2u, 8u, 16u, 48u, 128u, 56u));
    metrics.regions.push_back(make_region(3u, 8u, 8u, 56u, 64u, 64u));
    metrics.nnz = 204u;
    for (cellpack::region_layout_metrics &region : metrics.regions) {
        region.residual_fraction = static_cast<double>(region.nnz) / static_cast<double>(metrics.nnz);
    }

    cellpack::layout_selector_config config;
    config.min_structured_nnz = 16u;
    config.min_blocked_ell_fill = 0.65;
    config.min_sliced_ell_fill = 0.60;
    config.min_dense_tile_fill = 0.80;
    config.max_structured_vs_csr_bytes = 1.25;

    cellpack::layout_selection_plan selection;
    cellpack::validation_result result = cellpack::select_layouts(metrics, config, &selection);
    require(static_cast<bool>(result), result.message);
    require(selection.entries.size() == 4u, "selection entry count mismatch");
    require(selection.entries[0].selected_layout == cellpack::layout_kind::residual_csr, "low-fill sparse region should remain residual CSR");
    require(selection.entries[1].selected_layout == cellpack::layout_kind::blocked_ell, "regular high-fill region should choose Blocked-ELL");
    require(selection.entries[2].selected_layout == cellpack::layout_kind::sliced_ell, "variable grouped region should choose Sliced-ELL");
    require(selection.entries[3].selected_layout == cellpack::layout_kind::dense_tile, "dense local tile should choose dense tile");
    require(selection.entries[3].tensor_core_candidate, "dense 8x8 tile should be Tensor Core candidate metadata");
    require(selection.summary.launch_group_count == 4u, "selection launch group count mismatch");

    cellpack::layout_selection_plan second_selection;
    result = cellpack::select_layouts(metrics, config, &second_selection);
    require(static_cast<bool>(result), result.message);
    for (cellpack::u32 i = 0; i < static_cast<cellpack::u32>(selection.entries.size()); ++i) {
        require(selection.entries[i].region_id == second_selection.entries[i].region_id, "selection region order is not deterministic");
        require(selection.entries[i].selected_layout == second_selection.entries[i].selected_layout, "selection layout is not deterministic");
        require(selection.entries[i].selected_estimated_bytes == second_selection.entries[i].selected_estimated_bytes,
                "selection estimate is not deterministic");
    }

    cellpack::static_plan source;
    source.desc.version = cellpack::abi_version;
    source.desc.row_count = 32u;
    source.desc.feature_count = 64u;
    source.desc.region_count = 4u;
    for (cellpack::u32 i = 0u; i < 4u; ++i) {
        cellpack::packed_region_desc region{};
        region.region_id = i;
        region.parent_id = cellpack::invalid_id;
        region.layout = cellpack::to_u32(cellpack::layout_kind::blocked_ell);
        region.role = cellpack::to_u32(cellpack::region_role::primary);
        region.row_begin = i * 8u;
        region.row_count = 8u;
        region.feature_begin = 0u;
        region.feature_count = 8u;
        region.index_offset = cellpack::invalid_id;
        region.value_offset = cellpack::invalid_id;
        region.aux_offset = cellpack::invalid_id;
        region.weight_offset = cellpack::invalid_id;
        region.output_offset = cellpack::invalid_id;
        source.regions.push_back(region);
    }

    cellpack::static_plan selected_plan;
    result = cellpack::apply_layout_selection(source, selection, &selected_plan);
    require(static_cast<bool>(result), result.message);
    require(selected_plan.regions[0].layout == cellpack::to_u32(cellpack::layout_kind::residual_csr), "selected plan residual layout mismatch");
    require(selected_plan.regions[1].layout == cellpack::to_u32(cellpack::layout_kind::blocked_ell), "selected plan Blocked-ELL layout mismatch");
    require(selected_plan.regions[2].layout == cellpack::to_u32(cellpack::layout_kind::sliced_ell), "selected plan Sliced-ELL layout mismatch");
    require(selected_plan.regions[3].layout == cellpack::to_u32(cellpack::layout_kind::dense_tile), "selected plan dense layout mismatch");
    require((selected_plan.regions[3].flags & cellpack::region_flag_dense_rhs_local) != 0u, "dense selected plan flag missing");

    config.min_dense_tile_fill = 2.0;
    result = cellpack::select_layouts(metrics, config, &selection);
    require(result.code == cellpack::validation_code::invalid_layout, "invalid selector bounds were accepted");

    return 0;
}
