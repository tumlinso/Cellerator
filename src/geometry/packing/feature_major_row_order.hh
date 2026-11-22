#pragma once

#include <stddef.h>
#include <stdint.h>

namespace cellerator::geometry::packing {

inline constexpr uint32_t feature_major_row_order_window = 128;

inline uint32_t feature_hash(uint32_t feature) noexcept {
    feature ^= feature >> 16;
    feature *= 0x7feb352du;
    feature ^= feature >> 15;
    feature *= 0x846ca68bu;
    return feature ^ (feature >> 16);
}

// Use a validated CSR window [begin, begin + count). The caller supplies
// storage for count row IDs, with count at most feature_major_row_order_window.
inline void order_rows_by_local_feature_hash(const uint32_t *offsets, const uint32_t *features,
                                             size_t begin, uint32_t count, uint32_t *rows) noexcept {
    uint32_t signatures[feature_major_row_order_window];
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t row = uint32_t(begin + i);
        uint32_t signature = UINT32_MAX;
        for (uint32_t p = offsets[row]; p < offsets[size_t(row) + 1]; ++p) {
            uint32_t key = feature_hash(features[p]);
            if (key < signature) signature = key;
        }
        // Insertion preserves logical row order for tied signatures.
        uint32_t j = i;
        for (; j && signatures[j - 1] > signature; --j) {
            signatures[j] = signatures[j - 1];
            rows[j] = rows[j - 1];
        }
        signatures[j] = signature;
        rows[j] = row;
    }
}

} // namespace cellerator::geometry::packing
