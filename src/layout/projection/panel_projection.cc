#include <Cellerator/layout/projection/panel_projection.hh>

namespace cellerator::layout::projection {
namespace {

using Status = PanelPackingStatus;
constexpr uint32_t ambiguous_position = uint32_t(1) << 31;

Status validate_rows(PanelPackingInput input, size_t source_rows,
                     PanelPackingWorkspace workspace) noexcept {
    if (input.row_count > source_rows) return Status::invalid_argument;
    if (!input.row_ids) return Status::ok;
    if (input.row_count <= panel_rows) {
        for (size_t i = 0; i < input.row_count; ++i) {
            const uint32_t id = input.row_ids[i];
            if (id >= source_rows) return Status::invalid_argument;
            for (size_t j = 0; j < i; ++j)
                if (input.row_ids[j] == id) return Status::duplicate_row;
        }
        return Status::ok;
    }

    if (!workspace.row_marks || workspace.row_mark_bytes < source_rows/8 + (source_rows%8 != 0))
        return Status::insufficient_capacity;
    Status status = Status::ok;
    size_t i = 0;
    for (; i < input.row_count; ++i) {
        const uint32_t id = input.row_ids[i];
        if (id >= source_rows) { status = Status::invalid_argument; break; }
        const auto bit = uint8_t(1u << (id%8));
        if (workspace.row_marks[id/8] & bit) { status = Status::duplicate_row; break; }
        workspace.row_marks[id/8] |= bit;
    }
    for (size_t j = 0; j < i; ++j) {
        const uint32_t id = input.row_ids[j];
        workspace.row_marks[id/8] &= uint8_t(~(1u << (id%8)));
    }
    return status;
}

void unbind_blocks(const uint16_t *block_ids, size_t count, uint16_t *slots) noexcept {
    for (size_t i = 0; i < count; ++i) slots[block_ids[i]] = 0;
}

void unbind_columns(const PanelPackingSource &source, PanelPackingInput input,
                    PanelPackingWorkspace workspace, size_t count) noexcept {
    if (!source.ambiguous_features) return;
    for (size_t column = 0; column < count; ++column) {
        const size_t block = input.dense_block_ids[column/panel_tile_columns];
        const uint16_t feature = source.block_features[block*panel_tile_columns + column%panel_tile_columns];
        if (feature == panel_padding_feature) continue;
        const uint32_t position = source.feature_positions[feature];
        if (position >= ambiguous_position) workspace.ambiguous_columns[position - ambiguous_position] = 0;
    }
}

void unbind_selection(const PanelPackingSource &source, PanelPackingInput input,
                      PanelPackingWorkspace workspace) noexcept {
    unbind_columns(source, input, workspace, input.dense_block_count*panel_tile_columns);
    unbind_blocks(input.dense_block_ids, input.dense_block_count, workspace.block_slots);
}

Status bind_blocks(const PanelPackingSource &source, PanelPackingInput input,
                   PanelPackingWorkspace workspace) noexcept {
    if (!input.dense_block_count) return Status::ok;
    if (!workspace.block_slots || workspace.block_capacity < source.block_count)
        return Status::insufficient_capacity;
    if (source.ambiguous_features && (!workspace.ambiguous_columns ||
        workspace.ambiguous_capacity < source.ambiguous_features)) return Status::insufficient_capacity;

    size_t i = 0;
    Status status = Status::ok;
    for (; i < input.dense_block_count; ++i) {
        const uint16_t block = input.dense_block_ids[i];
        if (block >= source.block_count) { status = Status::invalid_argument; break; }
        if (workspace.block_slots[block]) { status = Status::duplicate_feature; break; }
        workspace.block_slots[block] = uint16_t(i + 1);
    }
    if (status != Status::ok) {
        unbind_blocks(input.dense_block_ids, i, workspace.block_slots);
        return status;
    }
    if (!source.ambiguous_features && !source.has_empty_blocks) return Status::ok;

    for (i = 0; i < input.dense_block_count; ++i) {
        const uint16_t *features = source.block_features + size_t(input.dense_block_ids[i])*panel_tile_columns;
        bool has_feature = false;
        size_t lane = 0;
        for (; lane < panel_tile_columns; ++lane) {
            const uint16_t feature = features[lane];
            if (feature == panel_padding_feature) continue;
            has_feature = true;
            const uint32_t position = source.feature_positions[feature];
            if (position < ambiguous_position) continue;
            uint32_t &column = workspace.ambiguous_columns[position - ambiguous_position];
            if (column) { status = Status::duplicate_feature; break; }
            column = uint32_t(i*panel_tile_columns + lane + 1);
        }
        if (status != Status::ok || !has_feature) {
            unbind_columns(source, input, workspace, i*panel_tile_columns + lane);
            unbind_blocks(input.dense_block_ids, input.dense_block_count, workspace.block_slots);
            return has_feature ? status : Status::invalid_argument;
        }
    }
    return status;
}

struct TransientColumns {
    const uint32_t *positions;
    const uint16_t *block_slots;
    const uint32_t *ambiguous_columns;
};

uint32_t packed_column(TransientColumns columns, uint32_t feature) noexcept {
    if (!columns.positions) return UINT32_MAX;
    const uint32_t position = columns.positions[feature];
    if (position >= ambiguous_position)
        return position == UINT32_MAX ? UINT32_MAX :
            columns.ambiguous_columns[position - ambiguous_position] - 1u;
    const uint16_t slot = columns.block_slots[position/panel_tile_columns];
    return slot ? uint32_t((slot - 1)*panel_tile_columns + position%panel_tile_columns) : UINT32_MAX;
}

template<class Column>
uint32_t packed_column(const Column *columns, uint32_t feature) noexcept {
    if (!columns) return UINT32_MAX;
    const Column column = columns[feature];
    return column == Column(-1) ? UINT32_MAX : uint32_t(column);
}

template<class Visit>
Status visit_columns(const PanelPackingSource &source, PanelPackingInput input,
    const PanelPackingLayout *layout, PanelPackingWorkspace workspace, Visit visit) noexcept {
    if (layout) {
        if (layout->wide_routes) visit(layout->wide_columns);
        else visit(layout->columns);
        return Status::ok;
    }
    const auto status = bind_blocks(source, input, workspace);
    if (status != Status::ok) return status;
    visit(TransientColumns{input.dense_block_count ? source.feature_positions : nullptr,
                           workspace.block_slots, workspace.ambiguous_columns});
    unbind_selection(source, input, workspace);
    return Status::ok;
}

template<class Column>
Status build_layout_columns(const PanelPackingSource &source, const uint16_t *block_ids,
    size_t blocks, size_t panels, uint16_t *features, Column *columns) noexcept {
    if (!blocks) return Status::ok;
    __builtin_memset(columns, 0xff, size_t(source.csr.cols)*sizeof(Column));
    for (size_t block = 0; block < blocks; ++block) {
        const uint16_t id = block_ids[block];
        if (id >= source.block_count) return Status::invalid_argument;
        const uint16_t *catalog_block = source.block_features + size_t(id)*panel_tile_columns;
        bool has_feature = false;
        for (size_t lane = 0; lane < panel_tile_columns; ++lane) {
            const size_t pos = block*panel_tile_columns + lane;
            const uint16_t feature = catalog_block[lane];
            features[pos] = feature;
            if (feature == panel_padding_feature) continue;
            if (columns[feature] != Column(-1)) return Status::duplicate_feature;
            columns[feature] = Column(pos);
            has_feature = true;
        }
        if (!has_feature) return Status::invalid_argument;
    }
    for (size_t i = blocks*panel_tile_columns; i < panels*panel_columns; ++i)
        features[i] = panel_padding_feature;
    return Status::ok;
}

size_t panel_element(size_t column, size_t row_base) noexcept {
    return (column/panel_columns)*panel_elements + row_base + column%panel_columns;
}

bool metadata_fits(const PanelPackingRequirements &r, PanelPackingMetadata metadata,
                   bool require_dense_features) noexcept {
    const bool omit_dense = !require_dense_features && !metadata.dense_features &&
        !metadata.dense_feature_capacity;
    const bool dense_fits = metadata.dense_feature_capacity >= r.dense_features &&
        (!r.dense_features || metadata.dense_features);
    return (omit_dense || dense_fits) &&
        metadata.residual_offset_capacity >= r.residual_offsets &&
        metadata.residual_entry_capacity >= r.residual_nnz &&
        metadata.residual_offsets && (!r.residual_nnz || metadata.residual_features);
}

template<class Column, bool write_values, class Lookup>
void emit_packing(const PanelPackingSelection &selection, Lookup lookup,
    PanelPackingMetadata metadata, Column *columns, const uint16_t *src,
    uint16_t *dense, uint16_t *residual_values) noexcept {
    const auto &r = selection.shape;
    const auto &input = selection.input;
    const auto &source = selection.source;
    if (metadata.dense_features) {
        for (size_t i = 0; i < input.dense_block_count; ++i) {
            const uint16_t *features = source.block_features + size_t(input.dense_block_ids[i])*panel_tile_columns;
            __builtin_memcpy(metadata.dense_features + i*panel_tile_columns, features,
                             panel_tile_columns*sizeof(uint16_t));
        }
        for (size_t i = input.dense_block_count*panel_tile_columns; i < r.dense_features; ++i)
            metadata.dense_features[i] = panel_padding_feature;
    }

    uint32_t residual = 0;
    for (size_t row = 0; row < r.rows; ++row) {
        metadata.residual_offsets[row] = residual;
        const size_t id = input.row_ids ? input.row_ids[row] : row;
        const size_t row_base = (row/panel_rows)*r.panels*panel_elements + (row%panel_rows)*panel_columns;
        const uint32_t first = source.csr.major_ptr[id], end = source.csr.major_ptr[id + 1];
        for (uint32_t k = first; k < end; ++k) {
            const uint32_t feature = source.csr.minor_idx[k];
            const uint32_t column = packed_column(lookup, feature);
            if constexpr (!write_values) *columns++ = Column(column);
            if (column == UINT32_MAX) {
                metadata.residual_features[residual] = uint16_t(feature);
                if constexpr (write_values) residual_values[residual] = src[k];
                ++residual;
            } else if constexpr (write_values) {
                dense[panel_element(column, row_base)] = src[k];
            }
        }
    }
    for (size_t row = r.rows; row < r.residual_offsets; ++row) metadata.residual_offsets[row] = residual;
}

template<bool scatter, class Column, class Source, class Packed>
void transfer_values(const PanelPackingPlan &plan, const Column *columns, Source *source,
                     Packed *dense, Packed *residual) noexcept {
    const auto &r = plan.shape;
    constexpr Column residual_column = Column(-1);
    for (size_t row = 0; row < r.rows; ++row) {
        const size_t id = plan.row_ids ? plan.row_ids[row] : row;
        const uint32_t first = plan.source_offsets[id], end = plan.source_offsets[id + 1];
        const size_t row_base = (row/panel_rows)*r.panels*panel_elements + (row%panel_rows)*panel_columns;
        uint32_t j = plan.residual_offsets[row];
        for (uint32_t k = first; k < end; ++k) {
            const Column column = *columns++;
            if (column == residual_column) {
                if constexpr (scatter) source[k] = residual[j++];
                else residual[j++] = source[k];
            } else {
                const size_t offset = panel_element(column, row_base);
                if constexpr (scatter) source[k] = dense[offset];
                else dense[offset] = source[k];
            }
        }
    }
}

template<bool clear_dense>
Status write_panel_values(const PanelPackingPlan &plan, const uint16_t *src, size_t source_count,
    uint16_t *dense, size_t dense_count, uint16_t *residual, size_t residual_count) noexcept {
    if (!plan.source_offsets) return Status::invalid_argument;
    const auto &r = plan.shape;
    if (source_count < r.source_nnz || dense_count < r.dense_values || residual_count < r.residual_nnz ||
        (r.routes && !src) || (r.dense_values && !dense) || (r.residual_nnz && !residual))
        return Status::insufficient_capacity;
    if constexpr (clear_dense)
        if (r.dense_values) __builtin_memset(dense, 0, r.dense_values*sizeof(uint16_t));
    if (r.wide_routes) transfer_values<false>(plan, plan.wide_columns, src, dense, residual);
    else transfer_values<false>(plan, plan.columns, src, dense, residual);
    return Status::ok;
}

} // namespace

PanelPackingStatus prepare_panel_packing_source(
    tensor::base::CompressedStructureView<uint32_t> csr, const uint16_t *block_features,
    size_t block_count, uint32_t *feature_positions, size_t feature_capacity,
    PanelPackingSource *source) noexcept {
    if (!source) return Status::invalid_argument;
    *source = {};
    if (csr.axis != tensor::base::CompressedAxis::row || !csr.major_ptr ||
        (csr.nnz && !csr.minor_idx)) return Status::invalid_csr;
    if (csr.rows > UINT32_MAX || csr.nnz > UINT32_MAX ||
        csr.rows >= SIZE_MAX/sizeof(uint32_t) || csr.nnz > SIZE_MAX/sizeof(uint32_t))
        return Status::overflow;
    if (csr.cols > panel_padding_feature || block_count > size_t(UINT16_MAX) + 1 ||
        (block_count && !block_features)) return Status::invalid_argument;
    if (block_count && (!feature_positions || feature_capacity < csr.cols))
        return Status::insufficient_capacity;
    if (csr.major_ptr[0] || csr.major_ptr[csr.rows] != csr.nnz) return Status::invalid_csr;

    for (size_t row = 0; row < csr.rows; ++row) {
        const uint32_t begin = csr.major_ptr[row], end = csr.major_ptr[row + 1];
        if (begin > end || end > csr.nnz) return Status::invalid_csr;
        for (uint32_t k = begin; k < end; ++k)
            if (csr.minor_idx[k] >= csr.cols || (k != begin && csr.minor_idx[k] <= csr.minor_idx[k - 1]))
                return Status::invalid_csr;
    }

    if (block_count) __builtin_memset(feature_positions, 0xff, size_t(csr.cols)*sizeof(uint32_t));
    uint32_t ambiguous = 0;
    bool has_empty_blocks = false;
    for (size_t block = 0; block < block_count; ++block) {
        bool has_feature = false;
        for (size_t lane = 0; lane < panel_tile_columns; ++lane) {
            const size_t position = block*panel_tile_columns + lane;
            const uint16_t feature = block_features[position];
            if (feature == panel_padding_feature) continue;
            if (feature >= csr.cols) return Status::invalid_argument;
            uint32_t &p = feature_positions[feature];
            if (p == UINT32_MAX) p = uint32_t(position);
            else if (p < ambiguous_position) p = ambiguous_position + ambiguous++;
            has_feature = true;
        }
        has_empty_blocks |= !has_feature;
    }

    *source = {csr, block_features, block_count, block_count ? feature_positions : nullptr,
               ambiguous, has_empty_blocks};
    return Status::ok;
}

PanelPackingStatus prepare_panel_packing_layout(const PanelPackingSource &source,
    const uint16_t *block_ids, size_t block_count, PanelPackingLayoutStorage storage,
    PanelPackingLayout *layout) noexcept {
    if (!layout) return Status::invalid_argument;
    *layout = {};
    if (!source.csr.major_ptr || block_count > source.block_count ||
        block_count > UINT16_MAX || (block_count && !block_ids)) return Status::invalid_argument;

    const size_t features = size_t(source.csr.cols);
    const size_t panels = block_count/panel_tiles + (block_count%panel_tiles != 0);
    const bool wide = block_count*panel_tile_columns > UINT16_MAX;
    if (block_count && (storage.dense_feature_capacity < panels*panel_columns ||
        storage.feature_capacity < features || !storage.dense_features ||
        (wide ? !storage.wide_columns : !storage.columns))) return Status::insufficient_capacity;

    const auto status = wide
        ? build_layout_columns(source, block_ids, block_count, panels, storage.dense_features,
                               storage.wide_columns)
        : build_layout_columns(source, block_ids, block_count, panels, storage.dense_features,
                               storage.columns);
    if (status != Status::ok) return status;
    *layout = {features, block_count, panels, block_count ? block_ids : nullptr,
               block_count ? storage.dense_features : nullptr,
               block_count && !wide ? storage.columns : nullptr,
               wide ? storage.wide_columns : nullptr,
               source.block_features, source.block_count, wide, true};
    return Status::ok;
}

namespace {

Status query_panel_packing_impl(const PanelPackingSource &source, PanelPackingInput input,
    const PanelPackingLayout *layout, PanelPackingWorkspace workspace,
    PanelPackingSelection *selection) noexcept {
    if (!selection) return Status::invalid_argument;
    *selection = {};
    const auto &csr = source.csr;
    if (!csr.major_ptr) return Status::invalid_argument;
    if (layout && (!layout->prepared || layout->features != csr.cols ||
        layout->catalog != source.block_features || layout->catalog_blocks != source.block_count))
        return Status::invalid_argument;
    if (input.dense_block_count > source.block_count ||
        input.dense_block_count > UINT16_MAX ||
        (input.dense_block_count && !input.dense_block_ids)) return Status::invalid_argument;

    PanelPackingRequirements r;
    r.rows = input.row_count;
    if (r.rows > csr.rows) return Status::invalid_argument;
    r.features = size_t(csr.cols);
    r.source_nnz = size_t(csr.nnz);
    r.groups = r.rows/panel_rows + (r.rows%panel_rows != 0);
    r.panels = layout ? layout->panels :
        input.dense_block_count/panel_tiles + (input.dense_block_count%panel_tiles != 0);
    r.dense_features = r.panels*panel_columns;
    r.wide_routes = layout ? layout->wide_routes : input.dense_block_count*panel_tile_columns > UINT16_MAX;
    if (r.groups > (SIZE_MAX/sizeof(uint32_t) - 1)/panel_rows ||
        (r.panels && r.groups > SIZE_MAX/sizeof(float)/panel_elements/r.panels)) return Status::overflow;
    r.dense_values = r.groups*r.panels*panel_elements;
    r.residual_offsets = r.groups*panel_rows + 1;

    auto status = validate_rows(input, size_t(csr.rows), workspace);
    if (status != Status::ok) return status;
    status = visit_columns(source, input, layout, workspace, [&](auto lookup) {
        for (size_t row = 0; row < r.rows; ++row) {
            const size_t id = input.row_ids ? input.row_ids[row] : row;
            const uint32_t first = csr.major_ptr[id], end = csr.major_ptr[id + 1];
            r.routes += end - first;
            if (r.panels)
                for (uint32_t k = first; k < end; ++k)
                    r.residual_nnz += packed_column(lookup, csr.minor_idx[k]) == UINT32_MAX;
        }
    });
    if (status != Status::ok) return status;
    if (!r.panels) r.residual_nnz = r.routes;
    *selection = {r, source, input, layout};
    return Status::ok;
}

} // namespace

PanelPackingStatus query_panel_packing(const PanelPackingSource &source, PanelPackingInput input,
    PanelPackingWorkspace workspace, PanelPackingSelection *selection) noexcept {
    return query_panel_packing_impl(source, input, nullptr, workspace, selection);
}

PanelPackingStatus query_panel_packing(const PanelPackingSource &source,
    const PanelPackingLayout &layout, const uint32_t *row_ids, size_t row_count,
    PanelPackingWorkspace workspace, PanelPackingSelection *selection) noexcept {
    const PanelPackingInput input{row_ids, row_count, layout.block_ids, layout.block_count};
    return query_panel_packing_impl(source, input, &layout, workspace, selection);
}

PanelPackingStatus prepare_panel_packing(const PanelPackingSelection &selection,
    PanelPackingWorkspace workspace, PanelPackingStorage storage, PanelPackingPlan *plan) noexcept {
    if (!plan) return Status::invalid_argument;
    *plan = {};
    if (!selection.source.csr.major_ptr) return Status::invalid_argument;
    const auto &r = selection.shape;
    const auto &metadata = storage.metadata;
    if (selection.layout && (metadata.dense_features || metadata.dense_feature_capacity))
        return Status::invalid_argument;
    if (!metadata_fits(r, metadata, !selection.layout) || storage.route_capacity < r.routes ||
        (r.routes && (r.wide_routes ? !storage.wide_columns : !storage.columns)))
        return Status::insufficient_capacity;
    const auto status = visit_columns(selection.source, selection.input, selection.layout, workspace,
        [&](auto lookup) {
            if (r.wide_routes)
                emit_packing<uint32_t, false>(selection, lookup, metadata,
                                             storage.wide_columns, nullptr, nullptr, nullptr);
            else
                emit_packing<uint16_t, false>(selection, lookup, metadata,
                                             storage.columns, nullptr, nullptr, nullptr);
        });
    if (status != Status::ok) return status;

    const uint16_t *features = selection.layout ? selection.layout->dense_features : metadata.dense_features;
    *plan = {r, selection.input.row_ids, selection.source.csr.major_ptr, features,
             r.wide_routes ? nullptr : storage.columns, r.wide_routes ? storage.wide_columns : nullptr,
             metadata.residual_offsets, metadata.residual_features};
    return Status::ok;
}

PanelPackingStatus pack_panel_block_features(const uint32_t *feature_order,
    size_t feature_count, const uint32_t *offsets, size_t blocks,
    uint16_t *features, size_t capacity) noexcept {
    if (feature_count > panel_padding_feature || blocks > size_t(UINT16_MAX) + 1 ||
        !offsets || (feature_count && !feature_order) || offsets[0] || offsets[blocks] != feature_count)
        return Status::invalid_argument;
    if (capacity < blocks*panel_tile_columns || (blocks && !features))
        return Status::insufficient_capacity;
    for (size_t b = 0; b < blocks; ++b)
        if (offsets[b] >= offsets[b + 1] || offsets[b + 1] - offsets[b] > panel_tile_columns)
            return Status::invalid_argument;
    for (size_t i = 0; i < feature_count; ++i)
        if (feature_order[i] >= feature_count) return Status::invalid_argument;
    for (size_t b = 0; b < blocks; ++b) {
        const size_t count = offsets[b + 1] - offsets[b];
        for (size_t i = 0; i < panel_tile_columns; ++i)
            *features++ = i < count ? uint16_t(feature_order[offsets[b] + i]) : panel_padding_feature;
    }
    return Status::ok;
}

PanelPackingStatus pack_panel_values(const PanelPackingPlan &plan,
    const uint16_t *src, size_t source_count, uint16_t *dense, size_t dense_count,
    uint16_t *residual, size_t residual_count) noexcept {
    return write_panel_values<true>(plan, src, source_count, dense, dense_count, residual, residual_count);
}

PanelPackingStatus refresh_panel_values(const PanelPackingPlan &plan,
    const uint16_t *src, size_t source_count, uint16_t *dense, size_t dense_count,
    uint16_t *residual, size_t residual_count) noexcept {
    return write_panel_values<false>(plan, src, source_count, dense, dense_count, residual, residual_count);
}

PanelPackingStatus pack_panel_values_once(const PanelPackingSelection &selection,
    PanelPackingWorkspace workspace, PanelPackingMetadata metadata,
    const uint16_t *src, size_t source_count, uint16_t *dense, size_t dense_count,
    uint16_t *residual, size_t residual_count) noexcept {
    if (!selection.source.csr.major_ptr) return Status::invalid_argument;
    const auto &r = selection.shape;
    if (selection.layout && (metadata.dense_features || metadata.dense_feature_capacity))
        return Status::invalid_argument;
    if (!metadata_fits(r, metadata, false) || source_count < r.source_nnz ||
        dense_count < r.dense_values || residual_count < r.residual_nnz ||
        (r.routes && !src) || (r.dense_values && !dense) || (r.residual_nnz && !residual))
        return Status::insufficient_capacity;
    return visit_columns(selection.source, selection.input, selection.layout, workspace,
        [&](auto lookup) {
            if (r.dense_values) __builtin_memset(dense, 0, r.dense_values*sizeof(uint16_t));
            emit_packing<uint16_t, true>(selection, lookup, metadata, nullptr, src, dense, residual);
        });
}

namespace {

Status pack_weight_blocks(const uint16_t *block_features, size_t blocks, size_t features,
    const uint16_t *weights, size_t output_tiles, size_t stride,
    uint16_t *packed, size_t capacity) noexcept {
    constexpr size_t tile_elements = panel_rows * panel_tile_columns;
    constexpr size_t max_elements = SIZE_MAX / sizeof(uint16_t);
    if (!output_tiles || !blocks) return Status::ok;
    if (stride < features || !block_features) return Status::invalid_argument;
    if (output_tiles > max_elements/tile_elements/blocks ||
        (stride && output_tiles > max_elements/panel_rows/stride)) return Status::overflow;
    const size_t count = output_tiles*blocks*tile_elements;
    if (!packed || capacity < count || (features && !weights)) return Status::insufficient_capacity;
    if (!features) {
        __builtin_memset(packed, 0, count*sizeof(uint16_t));
        return Status::ok;
    }
    for (size_t out = 0; out < output_tiles; ++out)
        for (size_t b = 0; b < blocks; ++b) {
            const uint16_t *ids = block_features + b*panel_tile_columns;
            for (size_t lane = 0; lane < panel_rows; ++lane) {
                const uint16_t *row = weights + (out*panel_rows + lane)*stride;
                for (size_t k = 0; k < panel_tile_columns; ++k)
                    *packed++ = ids[k] == panel_padding_feature ? 0 : row[ids[k]];
            }
        }
    return Status::ok;
}

} // namespace

PanelPackingStatus pack_panel_weights(const PanelPackingPlan &plan,
    const uint16_t *weights, size_t output_tiles, size_t stride,
    uint16_t *packed, size_t capacity) noexcept {
    return pack_weight_blocks(plan.dense_features, plan.shape.panels*panel_tiles, plan.shape.features,
                              weights, output_tiles, stride, packed, capacity);
}

PanelPackingStatus pack_panel_weights(const PanelPackingLayout &layout,
    const uint16_t *weights, size_t output_tiles, size_t stride,
    uint16_t *packed, size_t capacity) noexcept {
    if (!layout.prepared) return Status::invalid_argument;
    return pack_weight_blocks(layout.dense_features, layout.panels*panel_tiles, layout.features,
                              weights, output_tiles, stride, packed, capacity);
}

PanelPackingStatus pack_panel_source_weights(const PanelPackingSource &source,
    const uint16_t *weights, size_t output_tiles, size_t stride,
    uint16_t *packed, size_t capacity) noexcept {
    return pack_weight_blocks(source.block_features, source.block_count, source.csr.cols,
                              weights, output_tiles, stride, packed, capacity);
}

PanelPackingStatus scatter_panel_value_gradients(const PanelPackingPlan &plan,
    const float *dense, size_t dense_count, const float *residual, size_t residual_count,
    float *source, size_t source_count) noexcept {
    if (!plan.source_offsets) return Status::invalid_argument;
    const auto &r = plan.shape;
    if (source_count < r.source_nnz || dense_count < r.dense_values || residual_count < r.residual_nnz ||
        (r.routes && !source) || (r.routes != r.residual_nnz && !dense) || (r.residual_nnz && !residual))
        return Status::insufficient_capacity;
    if (r.wide_routes) transfer_values<true>(plan, plan.wide_columns, source, dense, residual);
    else transfer_values<true>(plan, plan.columns, source, dense, residual);
    return Status::ok;
}

PanelPackingStatus accumulate_panel_weight_gradients(const PanelPackingPlan &plan,
    const float *packed, size_t packed_count, size_t output_tiles,
    float *canonical, size_t stride) noexcept {
    const auto &r = plan.shape;
    if (!output_tiles || !r.panels) return Status::ok;
    if (stride < r.features || !stride) return Status::invalid_argument;
    if (output_tiles > SIZE_MAX/sizeof(float)/panel_elements/r.panels ||
        output_tiles > SIZE_MAX/sizeof(float)/panel_rows/stride) return Status::overflow;
    if (!packed || !canonical || packed_count < output_tiles*r.panels*panel_elements)
        return Status::insufficient_capacity;
    for (size_t out = 0; out < output_tiles; ++out)
        for (size_t p = 0; p < r.panels; ++p)
            for (size_t b = 0; b < panel_tiles; ++b) {
                const uint16_t *features = plan.dense_features + p*panel_columns + b*panel_tile_columns;
                for (size_t lane = 0; lane < panel_rows; ++lane) {
                    float *row = canonical + (out*panel_rows + lane)*stride;
                    for (size_t k = 0; k < panel_tile_columns; ++k)
                        if (features[k] != panel_padding_feature) row[features[k]] += packed[k];
                    packed += panel_tile_columns;
                }
            }
    return Status::ok;
}

} // namespace cellerator::layout::projection
