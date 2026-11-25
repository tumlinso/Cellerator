#pragma once

#include <Cellerator/ops/matrix/tiled_matmul_cuda.cuh>
#include <stdint.h>

namespace cellerator::ops::matrix {

struct alignas(4) IndexedTilePair {
    uint16_t feature_block;
    uint16_t physical_slot;
};

struct IndexedTileWorklist {
    memory::const_array_view<uint32_t> group_offsets;
    memory::const_array_view<IndexedTilePair> pairs;
};

struct PackedPanelWorklist {
    memory::const_array_view<uint32_t> group_offsets;
    memory::const_array_view<uint32_t> group_panel_bases;
    memory::const_array_view<uint16_t> feature_blocks;
};

class IndexedTiledMatmulPlan {
public:
    IndexedTiledMatmulPlan() = default;

private:
    friend cudaError_t prepare_indexed_tiled_matmul(int, TiledMatmulShape,
        IndexedTiledMatmulPlan &) noexcept;
    friend cudaError_t enqueue_indexed_tiled_matmul(const IndexedTiledMatmulPlan &,
        memory::const_array_view<memory::Tile<__half>>, memory::const_array_view<memory::Tile<__half>>,
        IndexedTileWorklist, memory::matrix_view<float>, cudaStream_t) noexcept;
    friend cudaError_t enqueue_indexed_panel_matmul(const IndexedTiledMatmulPlan &,
        memory::const_array_view<memory::Tile<__half>>, memory::const_array_view<__half>,
        IndexedTileWorklist, memory::matrix_view<float>, cudaStream_t) noexcept;
    friend cudaError_t enqueue_indexed_panel_matmul(const IndexedTiledMatmulPlan &,
        memory::const_array_view<memory::Tile<__half>>, memory::const_array_view<__half>,
        PackedPanelWorklist, memory::matrix_view<float>, cudaStream_t) noexcept;

    TiledMatmulShape shape_{};
    int device_ordinal_ = -1;
};

// Validate the host-accessible metadata before upload. The offsets span
// shape.cell_tiles + 1 entries, starting at zero and ending at pairs.count.
cudaError_t validate_indexed_tile_worklist(TiledMatmulShape shape,
    IndexedTileWorklist work, size_t physical_slots) noexcept;

// Validate host metadata against a window of whole panels. A pair's physical
// slot is a slice index, with four consecutive indices per panel.
cudaError_t validate_indexed_panel_worklist(TiledMatmulShape shape,
    IndexedTileWorklist work, size_t panel_count) noexcept;

// group_offsets partitions feature_blocks as above. group_panel_bases gives
// one window-relative panel index per group. Each group's IDs describe its
// consecutive slices in packing order, starting with slice zero of its base
// panel. Each group occupies ceil(group length / 4) panels, independently.
cudaError_t validate_indexed_panel_worklist(TiledMatmulShape shape,
    PackedPanelWorklist work, size_t panel_count) noexcept;

cudaError_t prepare_indexed_tiled_matmul(int device_ordinal, TiledMatmulShape shape,
    IndexedTiledMatmulPlan &plan) noexcept;

// For each cell group, sum W[output tile, pair.feature_block] * X[pair.physical_slot]
// in worklist order. Empty groups produce zeros. All tiles are row-major FP16.
// Y is row-major FP32 [output, cell], with a 32-byte-aligned base and a row
// stride divisible by four. The validated device worklist and operands must
// stay live and immutable through stream completion. The stream and nonempty
// views belong to the current device. Output is disjoint from all input views.
cudaError_t enqueue_indexed_tiled_matmul(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights,
    memory::const_array_view<memory::Tile<__half>> features, IndexedTileWorklist work,
    memory::matrix_view<float> output, cudaStream_t stream) noexcept;

// The same projection and lifetime contract using FP16 activations packed as
// [panel][cell][feature], with 16 cells and 64 features per panel. Slice s of
// a panel occupies its feature columns [16*s, 16*(s+1)). weights is indexed
// [output tile][logical feature block]. Callers supply validated worklists
// matching the resident slices and the feature order of the weight blocks.
cudaError_t enqueue_indexed_panel_matmul(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights,
    memory::const_array_view<__half> panels, IndexedTileWorklist work,
    memory::matrix_view<float> output, cudaStream_t stream) noexcept;

cudaError_t enqueue_indexed_panel_matmul(const IndexedTiledMatmulPlan &plan,
    memory::const_array_view<memory::Tile<__half>> weights,
    memory::const_array_view<__half> panels, PackedPanelWorklist work,
    memory::matrix_view<float> output, cudaStream_t stream) noexcept;

} // namespace cellerator::ops::matrix
