#pragma once

#include <stdint.h>

namespace cellerator::tensor {

inline constexpr uint32_t feature_major_tile_rows = 32;

// Each tile names a range of feature records in tile_offsets. Each record
// contains one feature, a nonempty mask of tile rows, and the offset of its
// first value. Values are consecutive in order of increasing set row bits.
// The arrays are borrowed and must remain live through their final consumer.
struct FeatureMajorTileStructureView {
    uint32_t rows = 0, features = 0, tile_count = 0, record_count = 0, nnz = 0;
    uint32_t tile_row_width = feature_major_tile_rows;
    const uint32_t *tile_offsets = nullptr;
    const uint32_t *feature_ids = nullptr;
    const uint32_t *row_masks = nullptr;
    const uint32_t *value_offsets = nullptr;
};

template<class Value>
struct FeatureMajorTilesView {
    FeatureMajorTileStructureView structure{};
    Value *values = nullptr;
};

} // namespace cellerator::tensor
