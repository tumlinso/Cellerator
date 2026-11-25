#pragma once

#include <Cellerator/memory/tiled/Panel.hh>
#include <Cellerator/tensor/base/compressed.hh>
#include <stddef.h>
#include <stdint.h>

namespace cellerator::layout::projection {

using memory::panel_rows;
using memory::panel_tile_columns;
using memory::panel_tiles;
using memory::panel_columns;
using memory::panel_elements;
inline constexpr uint16_t panel_padding_feature = UINT16_MAX;

enum class PanelPackingStatus : uint8_t {
    ok, invalid_argument, invalid_csr, duplicate_row, duplicate_feature,
    insufficient_capacity, overflow
};

// A validated, borrowed CSR and feature order. Each logical block has 16
// slots, with UINT16_MAX for padding. feature_positions is the prepared
// inverse map, including compact indices for features shared by blocks.
struct PanelPackingSource {
    tensor::base::CompressedStructureView<uint32_t> csr{};
    const uint16_t *block_features = nullptr;
    size_t block_count = 0;
    const uint32_t *feature_positions = nullptr;
    size_t ambiguous_features = 0;
    bool has_empty_blocks = false;
};

// The CSR rows have sorted, unique minor indices. Provide csr.cols positions
// when block_count is nonzero. Keep all source arrays and prepared positions
// immutable while making selections from the source.
PanelPackingStatus prepare_panel_packing_source(
    tensor::base::CompressedStructureView<uint32_t> csr, const uint16_t *block_features,
    size_t block_count, uint32_t *feature_positions, size_t feature_capacity,
    PanelPackingSource *source) noexcept;

struct PanelPackingLayoutStorage {
    uint16_t *dense_features = nullptr;
    size_t dense_feature_capacity = 0;
    uint16_t *columns = nullptr;
    uint32_t *wide_columns = nullptr;
    size_t feature_capacity = 0;
};

// A reusable block order, padded feature table, and inverse map. Block IDs,
// catalog, and populated storage stay live and immutable while the layout
// is used. Each column map uses its largest code for genes sent to residual.
struct PanelPackingLayout {
    size_t features = 0, block_count = 0, panels = 0;
    const uint16_t *block_ids = nullptr;
    const uint16_t *dense_features = nullptr;
    const uint16_t *columns = nullptr;
    const uint32_t *wide_columns = nullptr;
    const uint16_t *catalog = nullptr;
    size_t catalog_blocks = 0;
    bool wide_routes = false;
    bool prepared = false;
};

// Selected logical blocks must be nonempty and have disjoint features. Supply
// ceil(block_count/4)*64 dense features and source.csr.cols column entries
// when any block is selected. When block_count*16 exceeds UINT16_MAX use
// wide_columns, otherwise use columns. All written storage is disjoint
// from the inputs and other outputs. Build a new layout when the catalog or
// the ordered block selection changes.
PanelPackingStatus prepare_panel_packing_layout(const PanelPackingSource &source,
    const uint16_t *block_ids, size_t block_count, PanelPackingLayoutStorage storage,
    PanelPackingLayout *layout) noexcept;

struct PanelPackingInput {
    const uint32_t *row_ids = nullptr;
    size_t row_count = 0;
    const uint16_t *dense_block_ids = nullptr;
    size_t dense_block_count = 0;
};

// Zero the supplied arrays before their first use. Each call restores the
// slots and marks it touches, including on failure. block_slots has room for
// source.block_count entries when selecting any blocks. Explicit selections
// longer than 16 rows need ceil(source.csr.rows/8) mark bytes. Selections
// from an overlapping catalog use source.ambiguous_features column entries.
struct PanelPackingWorkspace {
    uint16_t *block_slots = nullptr;
    size_t block_capacity = 0;
    uint8_t *row_marks = nullptr;
    size_t row_mark_bytes = 0;
    uint32_t *ambiguous_columns = nullptr;
    size_t ambiguous_capacity = 0;
};

struct PanelPackingRequirements {
    size_t rows = 0, features = 0, source_nnz = 0;
    size_t groups = 0, panels = 0, dense_values = 0, dense_features = 0;
    size_t routes = 0, residual_offsets = 0, residual_nnz = 0;
    bool wide_routes = false;
};

// Returned by a successful query. Keep the selection, row IDs, block IDs,
// source structure and any shared layout immutable through preparation or
// one-shot packing. Changing the selection requires a fresh query.
struct PanelPackingSelection {
    PanelPackingRequirements shape{};
    PanelPackingSource source{};
    PanelPackingInput input{};
    const PanelPackingLayout *layout = nullptr;
};

struct PanelPackingMetadata {
    uint16_t *dense_features = nullptr;
    size_t dense_feature_capacity = 0;
    uint32_t *residual_offsets = nullptr;
    size_t residual_offset_capacity = 0;
    uint16_t *residual_features = nullptr;
    size_t residual_entry_capacity = 0;
};

struct PanelPackingStorage {
    PanelPackingMetadata metadata{};
    uint16_t *columns = nullptr;
    uint32_t *wide_columns = nullptr;
    size_t route_capacity = 0;
};

// The plan borrows CSR offsets, row IDs and prepared storage, including a
// shared layout's dense feature table when supplied. Keep those arrays
// immutable and values in original CSR entry order. Each route is a packed
// column within its group or the maximum code of its width for a residual
// entry. Exactly one route array is used.
struct PanelPackingPlan {
    PanelPackingRequirements shape{};
    const uint32_t *row_ids = nullptr;
    const uint32_t *source_offsets = nullptr;
    const uint16_t *dense_features = nullptr;
    const uint16_t *columns = nullptr;
    const uint32_t *wide_columns = nullptr;
    const uint32_t *residual_offsets = nullptr;
    const uint16_t *residual_features = nullptr;
};

// Count the selected CSR entries and residuals for exact sizing. The selected
// blocks share one order within this selection and must be nonempty with
// disjoint feature sets. A null row_ids selects
// [0, row_count). Physical rows extend to groups*16 with empty tail cells.
// Callers can query separate cohorts to select different block orders.
PanelPackingStatus query_panel_packing(const PanelPackingSource &source,
    PanelPackingInput input, PanelPackingWorkspace workspace,
    PanelPackingSelection *selection) noexcept;

// A layout may be used with distinct prepared CSR sources that have the
// same feature count, catalog pointer, and catalog block count. Copies of
// the catalog at other addresses need their own layout. Only row_marks is
// used from workspace, for explicit selections longer than 16 rows.
PanelPackingStatus query_panel_packing(const PanelPackingSource &source,
    const PanelPackingLayout &layout, const uint32_t *row_ids, size_t row_count,
    PanelPackingWorkspace workspace, PanelPackingSelection *selection) noexcept;

// For a shared layout, pass a null dense_features pointer and zero
// dense_feature_capacity. The plan borrows the layout's existing table.
// The transient selection path writes the table into caller storage.
// Keep outputs disjoint from the selection and its borrowed arrays.
PanelPackingStatus prepare_panel_packing(const PanelPackingSelection &selection,
    PanelPackingWorkspace workspace, PanelPackingStorage storage, PanelPackingPlan *plan) noexcept;

// Convert a prepared feature permutation and its block boundaries (including
// the terminal offset) into the padded 16-wide block table. Callers keep the
// original canonical IDs and can select any ordered subset for projection.
PanelPackingStatus pack_panel_block_features(const uint32_t *feature_order,
    size_t feature_count, const uint32_t *block_offsets, size_t block_count,
    uint16_t *block_features, size_t block_feature_capacity) noexcept;

// Pack the caller's binary16 bit patterns. Dense storage is
// [group][panel][cell lane][feature lane], with exact zeros in unfilled slots.
// The residual values use residual_offsets/residual_features in physical row
// order. Source, destination, and preparation arrays must be disjoint.
PanelPackingStatus pack_panel_values(const PanelPackingPlan &plan,
    const uint16_t *source, size_t source_count, uint16_t *dense, size_t dense_count,
    uint16_t *residual, size_t residual_count) noexcept;

// Overwrite every structural entry, including zeros. The destination uses
// this exact plan's layout and retains the zero holes from a preceding pack.
PanelPackingStatus refresh_panel_values(const PanelPackingPlan &plan,
    const uint16_t *source, size_t source_count, uint16_t *dense, size_t dense_count,
    uint16_t *residual, size_t residual_count) noexcept;

// Emit metadata and values directly from a queried selection. Pass a null
// dense_features pointer and zero dense_feature_capacity for a shared layout
// or when a transient consumer already has the feature order. Residual
// metadata is still required. Use disjoint input, output and scratch storage.
PanelPackingStatus pack_panel_values_once(const PanelPackingSelection &selection,
    PanelPackingWorkspace workspace, PanelPackingMetadata metadata,
    const uint16_t *source, size_t source_count, uint16_t *dense, size_t dense_count,
    uint16_t *residual, size_t residual_count) noexcept;

// Canonical weights are row-major [output][feature] with 16 outputs per tile.
// Packed weights are [output tile][panel][input tile][output lane][input lane].
// The caller supplies output_tiles*16 canonical rows of weight_row_stride
// elements and disjoint storage for output_tiles*panels*panel_elements elements.
PanelPackingStatus pack_panel_weights(const PanelPackingPlan &plan,
    const uint16_t *weights, size_t output_tiles, size_t weight_row_stride,
    uint16_t *packed, size_t packed_count) noexcept;

// Pack the same panel order before any cohort is selected.
PanelPackingStatus pack_panel_weights(const PanelPackingLayout &layout,
    const uint16_t *weights, size_t output_tiles, size_t weight_row_stride,
    uint16_t *packed, size_t packed_count) noexcept;

// Pack one weight tile per logical block, in source block order, for the
// indexed paths. Reuse across cohorts until the weights or feature order
// change. packed_count is in binary16 elements (256 per weight tile).
PanelPackingStatus pack_panel_source_weights(const PanelPackingSource &source,
    const uint16_t *weights, size_t output_tiles, size_t weight_row_stride,
    uint16_t *packed, size_t packed_count) noexcept;

// Write selected input gradients in original CSR entry order. Entries in source
// rows outside the selection retain their previous value. Padding gradients are
// ignored. Source, destination, and preparation arrays must be disjoint.
PanelPackingStatus scatter_panel_value_gradients(const PanelPackingPlan &plan,
    const float *dense, size_t dense_count, const float *residual, size_t residual_count,
    float *source, size_t source_count) noexcept;

// Add one plan's packed FP32 weight gradients to a canonical row-major
// gradient. Repeated features across separately prepared plans accumulate in
// the same canonical weight. The caller orders updates to overlapping rows.
PanelPackingStatus accumulate_panel_weight_gradients(const PanelPackingPlan &plan,
    const float *packed, size_t packed_count, size_t output_tiles,
    float *canonical, size_t canonical_row_stride) noexcept;

} // namespace cellerator::layout::projection
