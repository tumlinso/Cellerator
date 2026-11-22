#include <Cellerator/layout/projection/feature_major_tiles.hh>
#include "../../geometry/packing/feature_major_row_order.hh"

namespace cellerator::layout::projection {
namespace {

using Status = FeatureMajorTilesStatus;
using Grouping = FeatureMajorTileGrouping;
constexpr uint32_t tile_rows = tensor::feature_major_tile_rows;
constexpr uint32_t grouping_window = geometry::packing::feature_major_row_order_window;
constexpr uint32_t chunk_entries = feature_major_reverse_chunk_entries;
struct Workspace {
    size_t count;
    size_t tile_feature_capacity;
};

bool valid_grouping(Grouping grouping) noexcept {
    return grouping == Grouping::contiguous || grouping == Grouping::local_feature_hash;
}

Status query_workspace(FeatureMajorTilesCsr csr, Grouping grouping,
                       Workspace &workspace) noexcept {
    if (!valid_grouping(grouping)) return Status::invalid_argument;
    if (csr.rows > UINT32_MAX || csr.cols > UINT32_MAX || csr.nnz > UINT32_MAX
        || csr.rows + 1 > SIZE_MAX / sizeof(uint32_t)
        || csr.nnz > SIZE_MAX / sizeof(uint32_t)) return Status::overflow;
    if (csr.axis != tensor::base::CompressedAxis::row || !csr.major_ptr
        || (csr.nnz && (!csr.minor_idx || !csr.cols)) || csr.major_ptr[0] != 0)
        return Status::invalid_csr;

    uint32_t max_row = 0;
    for (size_t row = 0; row < csr.rows; ++row) {
        uint32_t begin = csr.major_ptr[row], end = csr.major_ptr[row + 1];
        if (end < begin || end > csr.nnz) return Status::invalid_csr;
        if (end - begin > max_row) max_row = end - begin;
    }
    if (csr.major_ptr[csr.rows] != csr.nnz) return Status::invalid_csr;
    for (size_t p = 0; p < csr.nnz; ++p)
        if (csr.minor_idx[p] >= csr.cols) return Status::invalid_csr;

    uint64_t tile_capacity = uint64_t(max_row) * tile_rows;
    if (tile_capacity > csr.nnz) tile_capacity = csr.nnz;
    if (tile_capacity > csr.cols) tile_capacity = csr.cols;
    uint64_t words = 4 * csr.cols + 2 * tile_capacity;
    if (words > SIZE_MAX / sizeof(uint32_t)) return Status::overflow;
    workspace = {size_t(words), size_t(tile_capacity)};
    return Status::ok;
}

const uint32_t *sort_features(uint32_t *features, uint32_t *temp, size_t count,
                              uint32_t max_feature) noexcept {
    if (count < 32) {
        for (size_t i = 1; i < count; ++i) {
            uint32_t feature = features[i];
            size_t j = i;
            for (; j && features[j - 1] > feature; --j) features[j] = features[j - 1];
            features[j] = feature;
        }
        return features;
    }
    for (uint32_t shift = 0; max_feature; shift += 8, max_feature >>= 8) {
        uint32_t next[256] = {};
        for (size_t i = 0; i < count; ++i)
            ++next[(features[i] >> shift) & 255u];
        uint32_t offset = 0;
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t n = next[i];
            next[i] = offset;
            offset += n;
        }
        for (size_t i = 0; i < count; ++i)
            temp[next[(features[i] >> shift) & 255u]++] = features[i];
        uint32_t *swap = features;
        features = temp;
        temp = swap;
    }
    return features;
}

template<class VisitTile>
Status visit_tiles(FeatureMajorTilesCsr csr, Grouping grouping, uint32_t *workspace,
                   size_t capacity, VisitTile visit_tile) noexcept {
    size_t features = size_t(csr.cols);
    uint32_t *masks = features ? workspace + 2 * features : nullptr;
    uint32_t *touched = capacity ? workspace + 4 * features : nullptr;
    uint32_t *sort_scratch = capacity ? touched + capacity : nullptr;
    for (size_t f = 0; f < features; ++f) masks[f] = 0;
    uint32_t order[grouping_window];
    for (size_t begin = 0; begin < csr.rows;) {
        uint32_t count = csr.rows - begin < grouping_window ?
            uint32_t(csr.rows - begin) : grouping_window;
        if (grouping == Grouping::contiguous) {
            for (uint32_t i = 0; i < count; ++i) order[i] = uint32_t(begin + i);
        } else {
            geometry::packing::order_rows_by_local_feature_hash(csr.major_ptr, csr.minor_idx,
                                                               begin, count, order);
        }
        for (uint32_t i = 0; i < count; i += tile_rows) {
            uint32_t width = count - i < tile_rows ? count - i : tile_rows;
            size_t used = 0;
            for (uint32_t lane = 0; lane < width; ++lane) {
                uint32_t row = order[i + lane];
                uint32_t bit = uint32_t(1) << lane;
                for (uint32_t p = csr.major_ptr[row]; p < csr.major_ptr[size_t(row) + 1]; ++p) {
                    uint32_t feature = csr.minor_idx[p];
                    uint32_t mask = masks[feature];
                    if (mask & bit) return Status::duplicate_coordinate;
                    if (!mask) touched[used++] = feature;
                    masks[feature] = mask | bit;
                }
            }
            visit_tile(begin + i, order + i, width, masks, touched, sort_scratch, used);
            for (size_t p = 0; p < used; ++p) masks[touched[p]] = 0;
        }
        begin += count;
    }
    return Status::ok;
}

Status index_count(const FeatureMajorTilesRequirements &r, size_t &count) noexcept {
    uint64_t words = uint64_t(r.tile_count) + 1 + uint64_t(r.record_count) * 5 + r.nnz
        + uint64_t(r.features) + 1;
    if (r.chunk_count) words += uint64_t(r.features) + 1;
    if (r.grouping == Grouping::local_feature_hash) words += r.rows;
    uint64_t chunk_bytes = uint64_t(r.chunk_count) * sizeof(FeatureMajorTileRange);
    if (words > SIZE_MAX / sizeof(uint32_t) || chunk_bytes > SIZE_MAX) return Status::overflow;
    count = size_t(words);
    return Status::ok;
}

uint32_t count_tiles(uint32_t rows) noexcept {
    return rows / tile_rows + uint32_t(rows % tile_rows != 0);
}

Status analyze(FeatureMajorTilesCsr csr, Grouping grouping, uint32_t *workspace,
               size_t capacity, FeatureMajorTilesRequirements &r,
               Workspace &scratch) noexcept {
    Status status = query_workspace(csr, grouping, scratch);
    if (status != Status::ok) return status;
    if (capacity < scratch.count || (scratch.count && !workspace))
        return Status::insufficient_capacity;
    uint32_t *used = workspace;
    uint32_t *splits = csr.cols ? workspace + csr.cols : nullptr;
    for (size_t f = 0; f < csr.cols; ++f) used[f] = splits[f] = 0;
    r = {uint32_t(csr.rows), uint32_t(csr.cols), count_tiles(uint32_t(csr.rows)),
         0, uint32_t(csr.nnz), 0, grouping, 0, scratch.count};
    status = visit_tiles(csr, grouping, workspace, scratch.tile_feature_capacity,
        [&](size_t, const uint32_t *, uint32_t, const uint32_t *masks,
            const uint32_t *touched, uint32_t *, size_t records) {
            r.record_count += uint32_t(records);
            for (size_t p = 0; p < records; ++p) {
                uint32_t feature = touched[p];
                uint32_t count = uint32_t(__builtin_popcount(masks[feature]));
                if (used[feature] + count > chunk_entries) {
                    ++splits[feature];
                    used[feature] = 0;
                }
                used[feature] += count;
            }
        });
    if (status != Status::ok) return status;
    for (size_t f = 0; f < csr.cols; ++f)
        if (splits[f]) r.chunk_count += splits[f] + 1;
    return index_count(r, r.index_count);
}

template<class Word>
struct Sections {
    Word *tile_offsets, *feature_ids, *row_masks, *value_offsets, *source_positions;
    Word *feature_offsets, *tile_ids, *record_ids, *chunk_offsets, *logical_rows;
};

template<class Word>
Sections<Word> sections(const FeatureMajorTilesRequirements &r, Word *data) noexcept {
    Word *tile_offsets = data;
    Word *feature_ids = tile_offsets + size_t(r.tile_count) + 1;
    Word *row_masks = feature_ids + r.record_count;
    Word *value_offsets = row_masks + r.record_count;
    Word *source_positions = value_offsets + r.record_count;
    Word *feature_offsets = source_positions + r.nnz;
    Word *tile_ids = feature_offsets + size_t(r.features) + 1;
    Word *record_ids = tile_ids + r.record_count;
    Word *tail = record_ids + r.record_count;
    Word *chunk_offsets = r.chunk_count ? tail : nullptr;
    if (chunk_offsets) tail += size_t(r.features) + 1;
    Word *logical_rows = r.grouping == Grouping::local_feature_hash ? tail : nullptr;
    return {tile_offsets, feature_ids, row_masks, value_offsets, source_positions,
            feature_offsets, tile_ids, record_ids, chunk_offsets, logical_rows};
}

} // namespace

FeatureMajorTilesStatus feature_major_tiles_workspace_count(
    FeatureMajorTilesCsr csr, Grouping grouping, size_t *count) noexcept {
    if (!count) return Status::invalid_argument;
    Workspace workspace{};
    Status status = query_workspace(csr, grouping, workspace);
    if (status == Status::ok) *count = workspace.count;
    return status;
}

FeatureMajorTilesStatus query_feature_major_tiles(
    FeatureMajorTilesCsr csr, Grouping grouping, uint32_t *workspace,
    size_t workspace_count, FeatureMajorTilesRequirements *required) noexcept {
    if (!required) return Status::invalid_argument;
    FeatureMajorTilesRequirements measured{};
    Workspace scratch{};
    Status status = analyze(csr, grouping, workspace, workspace_count, measured, scratch);
    if (status == Status::ok) *required = measured;
    return status;
}

FeatureMajorTilesStatus prepare_feature_major_tiles(
    FeatureMajorTilesCsr csr, const FeatureMajorTilesRequirements &required,
    uint32_t *workspace, size_t workspace_count, FeatureMajorTilesStorage storage) noexcept {
    FeatureMajorTilesRequirements r{};
    Workspace scratch{};
    Status status = analyze(csr, required.grouping, workspace, workspace_count, r, scratch);
    if (status != Status::ok) return status;
    if (r.rows != required.rows || r.features != required.features || r.tile_count != required.tile_count
        || r.record_count != required.record_count || r.nnz != required.nnz
        || r.chunk_count != required.chunk_count || r.index_count != required.index_count
        || r.workspace_count != required.workspace_count) return Status::invalid_argument;
    if (storage.index_capacity < r.index_count || !storage.indices
        || storage.chunk_capacity < r.chunk_count || (r.chunk_count && !storage.chunks))
        return Status::insufficient_capacity;

    auto out = sections(r, storage.indices);
    uint32_t *splits = r.features ? workspace + r.features : nullptr;
    if (out.chunk_offsets) {
        out.chunk_offsets[0] = 0;
        for (size_t f = 0; f < r.features; ++f)
            out.chunk_offsets[f + 1] = out.chunk_offsets[f] + (splits[f] ? splits[f] + 1 : 0);
    }
    for (size_t f = 0; f <= r.features; ++f) out.feature_offsets[f] = 0;

    uint32_t record = 0, value = 0;
    uint32_t *next_slot = r.features ? workspace + 3 * size_t(r.features) : nullptr;
    status = visit_tiles(csr, r.grouping, workspace, scratch.tile_feature_capacity,
        [&](size_t begin, const uint32_t *rows, uint32_t width, const uint32_t *masks,
            uint32_t *touched, uint32_t *temp, size_t records) {
        out.tile_offsets[begin / tile_rows] = record;
        auto emit = [&](uint32_t feature, uint32_t mask) {
            out.feature_ids[record] = feature;
            out.row_masks[record] = mask;
            out.value_offsets[record++] = value;
            ++out.feature_offsets[size_t(feature) + 1];
            next_slot[feature] = value;
            value += uint32_t(__builtin_popcount(mask));
        };
        // Dense tiles can visit their feature masks linearly in ID order.
        // For sparse tiles sort only the occupied feature IDs.
        if (records > r.features / 8) {
            for (uint32_t f = 0; f < r.features; ++f)
                if (masks[f]) emit(f, masks[f]);
        } else if (records) {
            const uint32_t *sorted = sort_features(touched, temp, records, r.features - 1);
            for (size_t i = 0; i < records; ++i) {
                uint32_t feature = sorted[i];
                emit(feature, masks[feature]);
            }
        }
        // Visiting rows in lane order fills each record in increasing bit order.
        for (uint32_t lane = 0; lane < width; ++lane) {
            uint32_t row = rows[lane];
            if (out.logical_rows) out.logical_rows[begin + lane] = row;
            for (uint32_t p = csr.major_ptr[row]; p < csr.major_ptr[size_t(row) + 1]; ++p)
                out.source_positions[next_slot[csr.minor_idx[p]]++] = p;
        }
    });
    if (status != Status::ok) return status;
    out.tile_offsets[r.tile_count] = record;
    for (size_t f = 0; f < r.features; ++f) {
        out.feature_offsets[f + 1] += out.feature_offsets[f];
        workspace[f] = out.feature_offsets[f];
    }
    for (size_t tile = 0; tile < r.tile_count; ++tile) {
        for (uint32_t p = out.tile_offsets[tile]; p < out.tile_offsets[tile + 1]; ++p) {
            uint32_t reverse = workspace[out.feature_ids[p]]++;
            out.tile_ids[reverse] = uint32_t(tile);
            out.record_ids[reverse] = p;
        }
    }
    if (out.chunk_offsets) {
        for (uint32_t f = 0; f < r.features; ++f) {
            uint32_t chunk = out.chunk_offsets[f];
            if (chunk == out.chunk_offsets[size_t(f) + 1]) continue;
            uint32_t begin = out.feature_offsets[f], end = out.feature_offsets[size_t(f) + 1];
            uint32_t used = 0;
            for (uint32_t p = begin; p < end; ++p) {
                uint32_t count = uint32_t(__builtin_popcount(out.row_masks[out.record_ids[p]]));
                if (used + count > chunk_entries) {
                    storage.chunks[chunk++] = {f, begin, p};
                    begin = p;
                    used = 0;
                }
                used += count;
            }
            storage.chunks[chunk] = {f, begin, end};
        }
    }
    return Status::ok;
}

FeatureMajorTilesStatus bind_feature_major_tiles(
    const FeatureMajorTilesRequirements &r, const uint32_t *indices,
    size_t index_capacity, const FeatureMajorTileRange *chunks, size_t chunk_capacity,
    tensor::FeatureMajorTileStructureView *structure, FeatureMajorTilesIndexView *index) noexcept {
    if (!structure || !index || !valid_grouping(r.grouping) || r.tile_count != count_tiles(r.rows)
        || r.record_count > r.nnz || r.chunk_count > r.record_count
        || (r.nnz && (!r.rows || !r.features || !r.record_count)))
        return Status::invalid_argument;
    size_t count = 0;
    Status status = index_count(r, count);
    if (status != Status::ok) return status;
    if (count != r.index_count) return Status::invalid_argument;
    if (!indices || index_capacity < count || (r.chunk_count && !chunks)
        || chunk_capacity < r.chunk_count) return Status::insufficient_capacity;
    auto view = sections(r, indices);
    *structure = {r.rows, r.features, r.tile_count, r.record_count, r.nnz, tile_rows,
                  view.tile_offsets, view.feature_ids, view.row_masks, view.value_offsets};
    *index = {view.logical_rows, view.source_positions, view.feature_offsets, view.tile_ids,
              view.record_ids, view.chunk_offsets, r.chunk_count ? chunks : nullptr, r.chunk_count};
    return Status::ok;
}

} // namespace cellerator::layout::projection
