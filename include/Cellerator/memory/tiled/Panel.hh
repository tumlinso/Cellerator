#pragma once

#include <Cellerator/common/compiler_annotations.hh>
#include <Cellerator/memory/view.hh>
#include <stddef.h>

namespace cellerator::memory {

inline constexpr size_t panel_rows = 16, panel_tile_columns = 16, panel_tiles = 4;
inline constexpr size_t panel_columns = panel_tile_columns*panel_tiles;
inline constexpr size_t panel_elements = panel_rows*panel_columns;

// One 16-cell by 64-feature panel. Tile i borrows 16 columns from each row.
// The caller supplies a whole panel and a tile index in [0, panel_tiles).
template<class T>
CELLERATOR_HOST_DEVICE constexpr matrix_view<T> panel_tile_view(
    T *panel, size_t i, placement where = {}) noexcept {
    return {panel + i*panel_tile_columns, panel_rows, panel_tile_columns, panel_columns, where};
}

// The producer writes H[neighborhood][cell][channel], all axes of extent 16.
// The same storage supplies one cell's neighborhoods without a transpose.
template<class T>
CELLERATOR_HOST_DEVICE constexpr matrix_view<T> cell_neighborhood_view(
    T *h, size_t cell, placement where = {}) noexcept {
    return {h + cell*panel_tile_columns, panel_rows, panel_tile_columns,
            panel_rows*panel_tile_columns, where};
}

} // namespace cellerator::memory
