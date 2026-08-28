#pragma once

#include "Cellerator/geometry/cell_block_records.hh"

#include <cstddef>

namespace cellpack {

inline constexpr u32 local_cell_order_schema_version = 1u;
inline constexpr u32 local_cell_signature_algorithm_version = 1u;
inline constexpr u32 local_cell_signature_lane_count = 4u;

enum class local_cell_order_kind : u32 {
    inferred_minhash = 1u,
    original = 2u,
    deterministic_random = 3u,
    row_nnz_descending = 4u
};

struct local_cell_order_config {
    local_cell_order_kind kind = local_cell_order_kind::inferred_minhash;
    u32 window_size = 1024u;
    u32 group_width = 32u;
    u64 seed = 0x6a09e667f3bcc909ull;
};

struct local_cell_order_requirements {
    std::size_t row_capacity = 0u;
    std::size_t block_epoch_capacity = 0u;
};

struct local_cell_order_buffers {
    std::size_t row_capacity = 0u;
    u64 *primary_keys = nullptr;
    u32 *secondary_keys = nullptr;
    u32 *active_block_counts = nullptr;
    u32 *row_nnz_counts = nullptr;
    // permutation[execution_row] is the canonical partition-local row.
    u32 *row_permutation = nullptr;
    // inverse_row_permutation[canonical_local_row] is its execution row.
    u32 *inverse_row_permutation = nullptr;
};

// Pointer-first semantic result. Global row identity is recovered as
// global_row_begin + row_permutation[execution_row]. Windows are partition-local
// half-open ranges of window_size rows; no row may move between windows.
struct local_cell_order_view {
    u32 order_schema_version = 0u;
    u32 signature_algorithm_version = 0u;
    local_cell_order_kind kind = local_cell_order_kind::inferred_minhash;
    u32 window_size = 0u;
    u32 group_width = 0u;
    u64 seed = 0u;
    u64 ordering_identity = 0u;
    u64 global_row_begin = 0u;
    u32 full_row_count = 0u;
    u32 row_count = 0u;
    u32 feature_block_count = 0u;
    u64 feature_block_geometry_identity = 0u;
    u64 row_domain_identity = 0u;
    const u64 *primary_keys = nullptr;
    const u32 *secondary_keys = nullptr;
    const u32 *active_block_counts = nullptr;
    const u32 *row_nnz_counts = nullptr;
    const u32 *row_permutation = nullptr;
    const u32 *inverse_row_permutation = nullptr;
};

struct local_cell_order_metric_workspace {
    std::size_t block_epoch_capacity = 0u;
    u32 *block_epochs = nullptr;
};

struct local_cell_order_metrics {
    u32 group_width = 0u;
    u32 group_count = 0u;
    u64 total_active_block_references = 0u;
    u64 total_group_block_union_references = 0u;
    u32 maximum_group_block_union = 0u;
    u64 block_id_metadata_bytes = 0u;
};

// Stable semantic identity derived from record-domain identity and the versioned
// ordering configuration. It does not dereference record payload pointers.
u64 local_cell_order_identity(
    const cell_block_record_view &records,
    const local_cell_order_config &config) noexcept;

validation_result query_local_cell_order_requirements_host(
    const cell_block_record_view &records,
    local_cell_order_requirements *out);

validation_result build_local_cell_order_host(
    const cell_block_record_view &records,
    const local_cell_order_config &config,
    const local_cell_order_buffers &buffers,
    local_cell_order_view *out);

validation_result validate_local_cell_order_view_host(
    const cell_block_record_view &records,
    const local_cell_order_view &order);

validation_result evaluate_local_cell_order_metrics_host(
    const cell_block_record_view &records,
    const local_cell_order_view &order,
    const local_cell_order_metric_workspace &workspace,
    local_cell_order_metrics *out);

} // namespace cellpack
