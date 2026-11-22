#pragma once

#include <Cellerator/tensor/base/compressed.hh>
#include <Cellerator/tensor/feature_major_tiles.hh>
#include <stddef.h>

namespace cellerator::layout::projection {

using FeatureMajorTilesCsr = tensor::base::CompressedStructureView<uint32_t>;

enum class FeatureMajorTileGrouping : uint8_t { contiguous, local_feature_hash };
enum class FeatureMajorTilesStatus : uint8_t {
    ok, invalid_argument, invalid_csr, duplicate_coordinate, insufficient_capacity, overflow
};

inline constexpr uint32_t feature_major_reverse_chunk_entries = 256;

struct FeatureMajorTileRange {
    uint32_t feature, begin, end;
};

// The reverse index is feature-major, then in increasing tile order. Each
// heavy feature's ranges address consecutive reverse entries. A null row
// map means physical row equals logical row. Source positions index the
// caller's original CSR entries, in their supplied order.
struct FeatureMajorTilesIndexView {
    const uint32_t *logical_rows = nullptr;
    const uint32_t *source_value_positions = nullptr;
    const uint32_t *feature_offsets = nullptr;
    const uint32_t *tile_ids = nullptr;
    const uint32_t *record_ids = nullptr;
    const uint32_t *feature_chunk_offsets = nullptr;
    const FeatureMajorTileRange *chunks = nullptr;
    uint32_t chunk_count = 0;
};

struct FeatureMajorTilesRequirements {
    uint32_t rows = 0, features = 0, tile_count = 0, record_count = 0, nnz = 0;
    uint32_t chunk_count = 0;
    FeatureMajorTileGrouping grouping = FeatureMajorTileGrouping::contiguous;
    size_t index_count = 0, workspace_count = 0;
};

struct FeatureMajorTilesStorage {
    uint32_t *indices = nullptr;
    size_t index_capacity = 0;
    FeatureMajorTileRange *chunks = nullptr;
    size_t chunk_capacity = 0;
};

// CPU preparation. Capacities are in array elements. All inputs, output
// storage, and workspace must be disjoint. Empty CSR still supplies an
// offsets array of rows + 1 elements. CSR structure is immutable throughout
// sizing and build. The exact query checks coordinate uniqueness.
FeatureMajorTilesStatus feature_major_tiles_workspace_count(
    FeatureMajorTilesCsr csr, FeatureMajorTileGrouping grouping, size_t *count) noexcept;

FeatureMajorTilesStatus query_feature_major_tiles(
    FeatureMajorTilesCsr csr, FeatureMajorTileGrouping grouping, uint32_t *workspace,
    size_t workspace_count, FeatureMajorTilesRequirements *required) noexcept;

FeatureMajorTilesStatus prepare_feature_major_tiles(
    FeatureMajorTilesCsr csr, const FeatureMajorTilesRequirements &required,
    uint32_t *workspace, size_t workspace_count, FeatureMajorTilesStorage storage) noexcept;

// Rebind the exact buffers emitted by prepare, or their copies at a new
// address. This checks extents without reading the arrays, so indices and
// chunks may reside on a CUDA device.
FeatureMajorTilesStatus bind_feature_major_tiles(
    const FeatureMajorTilesRequirements &required, const uint32_t *indices,
    size_t index_capacity, const FeatureMajorTileRange *chunks, size_t chunk_capacity,
    tensor::FeatureMajorTileStructureView *structure, FeatureMajorTilesIndexView *index) noexcept;

} // namespace cellerator::layout::projection
