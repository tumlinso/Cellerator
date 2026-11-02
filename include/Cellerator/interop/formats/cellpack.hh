#pragma once

#include <Cellerator/geometry/warp_tiles.hh>
#include <Cellerator/tensor/block_masked/row_view.hh>
#include <Cellerator/tensor/block_masked/view.hh>

#include <cstdint>
#include <stdexcept>

namespace cellerator::interop::formats {
namespace detail {
template<class T, class Source>
void cellpack_value_contract(const cellpack::frozen_packing_plan& plan, const Source& source) {
    if (source.value_size_bytes != sizeof(T) || source.feature_count != plan.feature_count() ||
        source.feature_block_count != plan.feature_block_count() ||
        source.feature_block_geometry_identity != plan.feature_block_geometry_identity() ||
        source.feature_axis_fingerprint != plan.identity().feature_axis_fingerprint ||
        source.feature_axis_fingerprint_version != plan.identity().feature_axis_fingerprint_version ||
        source.row_domain_identity != plan.identity().row_domain_identity)
        throw std::invalid_argument("CellPack numeric view type/geometry/axis mismatch");
    if (source.nnz_count && (!source.values || reinterpret_cast<std::uintptr_t>(source.values) % alignof(T)))
        throw std::invalid_argument("CellPack numeric values are missing/misaligned");
}
}

// Import a source already validated by its CellPack record/order validator.
// Only cold compatibility fields are checked here: pointer arrays may reside on
// a device and are never dereferenced. This aliases values and metadata without
// allocation/decoding. The caller retains plan/source owners and separately
// binds canonical feature/row maps, domain, order and generations to execution.
// feature_block_offsets must be the matching validated metadata replica in the
// same memory placement as source; a host plan pointer is not a device replica.
template<class T>
tensor::block_masked::RowView<const T> borrow_validated_cellpack_records(
    const cellpack::frozen_packing_plan& plan, const cellpack::cell_block_record_view& source,
    const std::uint32_t* feature_block_offsets) {
    detail::cellpack_value_contract<T>(plan, source);
    if (source.record_schema_version != cellpack::cell_block_record_schema_version ||
        !feature_block_offsets || !source.row_record_offsets || !source.record_value_offsets ||
        (source.record_count && (!source.record_block_ids || !source.record_gene_masks)))
        throw std::invalid_argument("invalid CellPack record descriptor");
    tensor::block_masked::RowLayoutView<> layout;
    layout.row_block_offsets = source.row_record_offsets;
    layout.block_ids = source.record_block_ids;
    layout.feature_masks = source.record_gene_masks;
    layout.row_block_value_offsets = source.record_value_offsets;
    layout.feature_block_offsets = feature_block_offsets;
    layout.rows = source.row_count; layout.features = source.feature_count;
    layout.feature_blocks = source.feature_block_count; layout.row_blocks = source.record_count;
    layout.stored_values = source.nnz_count;
    return {layout, static_cast<const T*>(source.values)};
}

template<class T>
tensor::block_masked::View<const T> borrow_validated_cellpack_tiles(
    const cellpack::frozen_packing_plan& plan, const cellpack::warp_tile_view& source,
    const std::uint32_t* feature_block_offsets) {
    detail::cellpack_value_contract<T>(plan, source);
    if (source.tile_schema_version != cellpack::warp_tile_schema_version ||
        source.tile_row_width == 0 || source.tile_row_width > 32 ||
        !feature_block_offsets || !source.tile_block_offsets || !source.block_row_entry_offsets || !source.row_block_value_offsets ||
        (source.tile_block_count && (!source.tile_block_ids || !source.tile_block_cell_masks)) ||
        (source.row_block_entry_count && !source.row_block_gene_masks))
        throw std::invalid_argument("invalid CellPack tile descriptor");
    tensor::block_masked::LayoutView<> layout;
    layout.tile_block_offsets = source.tile_block_offsets;
    layout.block_ids = source.tile_block_ids;
    layout.participating_row_masks = source.tile_block_cell_masks;
    layout.block_row_entry_offsets = source.block_row_entry_offsets;
    layout.feature_masks = source.row_block_gene_masks;
    layout.row_block_value_offsets = source.row_block_value_offsets;
    layout.feature_block_offsets = feature_block_offsets;
    layout.tile_row_width = source.tile_row_width;
    layout.rows = source.row_count; layout.features = source.feature_count;
    layout.tiles = source.tile_count; layout.stored_values = source.nnz_count;
    layout.feature_blocks = source.feature_block_count; layout.tile_blocks = source.tile_block_count;
    layout.row_block_entries = source.row_block_entry_count;
    return {layout, static_cast<const T*>(source.values)};
}

} // namespace cellerator::interop::formats
