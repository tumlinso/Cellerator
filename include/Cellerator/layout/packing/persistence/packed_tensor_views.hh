#pragma once

#include <Cellerator/layout/packing/warp_tiles.hh>
#include <Cellerator/tensor/block_masked/row_view.hh>
#include <Cellerator/tensor/block_masked/view.hh>

#include <stdint.h>

namespace cellerator::geometry {
namespace detail {

template<class T, class Source>
bool packed_values_match(const cellpack::frozen_packing_plan& plan, const Source& source) noexcept {
    return source.value_size_bytes == sizeof(T) && source.layout.features == plan.feature_count()
        && source.layout.feature_blocks == plan.feature_block_count()
        && source.feature_block_geometry_identity == plan.feature_block_geometry_identity()
        && source.feature_axis_fingerprint == plan.identity().feature_axis_fingerprint
        && source.feature_axis_fingerprint_version == plan.identity().feature_axis_fingerprint_version
        && source.row_domain_identity == plan.identity().row_domain_identity
        && (!source.layout.stored_values || (source.values && reinterpret_cast<uintptr_t>(source.values) % alignof(T) == 0));
}

} // namespace detail

// The source has already passed structural validation. Bindings check only
// descriptor fields and may borrow arrays on a device. A supplied block-offset
// replica must match the validated plan and the placement of the source arrays.
// The caller keeps all source storage live through the last consumer.
template<class T>
cellpack::validation_result bind_packed_row_records(
    const cellpack::frozen_packing_plan& plan, const cellpack::cell_block_record_view& source,
    const uint32_t* feature_block_offsets, tensor::block_masked::RowView<const T>* out) noexcept {
    auto layout = source.layout;
    if (feature_block_offsets) layout.feature_block_offsets = feature_block_offsets;
    if (source.record_schema_version != cellpack::cell_block_record_schema_version)
        return cellpack::validation_error(cellpack::validation_code::unsupported_version,
            source.record_schema_version, "unsupported packed row-record version");
    if (!out || !cellpack::packed_extents_fit(source) || !detail::packed_values_match<T>(plan, source)
        || !layout.feature_block_offsets
        || !layout.row_block_offsets || !layout.row_block_value_offsets
        || (layout.row_blocks && (!layout.block_ids || !layout.feature_masks)))
        return cellpack::validation_error(cellpack::validation_code::invalid_matrix_view,
            cellpack::invalid_id, "invalid packed row-record binding");
    *out = {layout, static_cast<const T*>(source.values)};
    return cellpack::validation_ok();
}

template<class T>
cellpack::validation_result bind_row_masked_tiles(
    const cellpack::frozen_packing_plan& plan, const cellpack::warp_tile_view& source,
    const uint32_t* feature_block_offsets, tensor::block_masked::View<const T>* out) noexcept {
    auto layout = source.layout;
    if (feature_block_offsets) layout.feature_block_offsets = feature_block_offsets;
    if (source.tile_schema_version != cellpack::warp_tile_schema_version)
        return cellpack::validation_error(cellpack::validation_code::unsupported_version,
            source.tile_schema_version, "unsupported row-masked tile version");
    if (!out || !cellpack::packed_extents_fit(source) || !detail::packed_values_match<T>(plan, source)
        || !layout.feature_block_offsets
        || !layout.tile_row_width || layout.tile_row_width > cellpack::warp_tile_cell_mask_bits
        || !layout.tile_block_offsets || !layout.block_row_entry_offsets || !layout.row_block_value_offsets
        || (layout.tile_blocks && (!layout.block_ids || !layout.participating_row_masks))
        || (layout.row_block_entries && !layout.feature_masks))
        return cellpack::validation_error(cellpack::validation_code::invalid_matrix_view,
            cellpack::invalid_id, "invalid row-masked tile binding");
    *out = {layout, static_cast<const T*>(source.values)};
    return cellpack::validation_ok();
}

} // namespace cellerator::geometry
