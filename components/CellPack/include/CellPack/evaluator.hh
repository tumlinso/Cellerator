#pragma once

#include "CellPack/planner.hh"

#include <cstddef>

namespace cellpack {

// Structural CSR input for exact plan evaluation. Values are intentionally
// absent: one stored coordinate contributes one incidence A[row, feature] = 1.
struct csr_support_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 nnz_count = 0u;
    const u32 *row_offsets = nullptr;
    const u32 *feature_ids = nullptr;
};

// Non-owning source-side context. Preparation validates canonical CSR support
// once; callers must keep the referenced arrays immutable and alive.
struct prepared_csr_support {
    csr_support_view support{};
    bool validated = false;
};

// A PackingPlan is two-sided execution geometry. Permutations map execution
// positions to canonical ids; inverse permutations map canonical ids back to
// execution positions. A null permutation/inverse pair means identity.
struct packing_plan_view {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    const u32 *row_permutation = nullptr;
    const u32 *inverse_row_permutation = nullptr;
    const u32 *feature_permutation = nullptr;
    const u32 *inverse_feature_permutation = nullptr;
    u32 row_group_count = 0u;
    const u32 *row_group_offsets = nullptr;
    u32 feature_block_count = 0u;
    const u32 *feature_block_offsets = nullptr;
};

struct packing_evaluation_entry {
    u64 tile_id = 0u;
    u32 execution_row = 0u;
    u32 reserved = 0u;
};

struct packing_evaluation_workspace_view {
    packing_evaluation_entry *entries = nullptr;
    u32 entry_capacity = 0u;
};

struct count_distribution {
    u64 sample_count = 0u;
    u64 minimum = 0u;
    u64 maximum = 0u;
    u64 total = 0u;
    double squared_total = 0.0;
};

struct real_distribution {
    u64 sample_count = 0u;
    double minimum = 0.0;
    double maximum = 0.0;
    double total = 0.0;
    double squared_total = 0.0;
};

struct occupied_tile_occupancy {
    u32 row_group = 0u;
    u32 feature_block = 0u;
    u32 participating_rows = 0u;
    u32 reserved = 0u;
    u64 nnz = 0u;
    u64 logical_slots = 0u;
    u64 dense_padding = 0u;
    double density = 0.0;
    double row_participation = 0.0;
};

struct row_group_occupancy {
    u32 row_group = 0u;
    u32 row_count = 0u;
    u32 active_feature_blocks = 0u;
    u32 reserved = 0u;
    u64 nnz = 0u;
    u64 participating_row_block_references = 0u;
    u64 occupied_dense_slots = 0u;
    u64 dense_padding = 0u;
};

struct packing_occupancy_buffers {
    occupied_tile_occupancy *occupied_tiles = nullptr;
    u32 occupied_tile_capacity = 0u;
    u32 *active_feature_blocks_per_execution_row = nullptr;
    u32 execution_row_capacity = 0u;
    row_group_occupancy *row_groups = nullptr;
    u32 row_group_capacity = 0u;
};

struct packing_evaluation_requirements {
    u32 workspace_entry_capacity = 0u;
    u32 occupied_tile_capacity = 0u;
    u32 execution_row_capacity = 0u;
    u32 row_group_capacity = 0u;
    u64 logical_tile_count = 0u;
    std::size_t temporary_workspace_bytes = 0u;
    std::size_t output_buffer_bytes = 0u;
};

struct packing_occupancy_totals {
    u64 total_nnz = 0u;
    u64 logical_tile_count = 0u;
    u64 occupied_tile_count = 0u;
    u64 empty_tile_count = 0u;
    u64 occupied_dense_slots = 0u;
    u64 dense_padding = 0u;
    u64 row_active_block_references = 0u;
    u64 row_group_active_block_references = 0u;
};

struct packing_occupancy_result {
    u32 row_count = 0u;
    u32 feature_count = 0u;
    u32 row_group_count = 0u;
    u32 feature_block_count = 0u;
    occupied_tile_occupancy *occupied_tiles = nullptr;
    u32 occupied_tile_count = 0u;
    u32 *active_feature_blocks_per_execution_row = nullptr;
    row_group_occupancy *row_groups = nullptr;
    packing_occupancy_totals totals{};
    count_distribution nnz_per_occupied_tile{};
    real_distribution tile_density{};
    count_distribution active_feature_blocks_per_row{};
    count_distribution active_feature_blocks_per_row_group{};
    count_distribution participating_rows_per_occupied_tile{};
    real_distribution feature_block_reuse{};
    count_distribution dense_padding_per_occupied_tile{};
};

// Reference-only hypothetical cost assumptions. This is evaluator policy, not
// a physical format descriptor or durable ABI.
struct packing_cost_model {
    u32 value_bytes = 2u;
    u32 per_nnz_index_bytes = 0u;
    u32 occupied_tile_metadata_bytes = 0u;
    u32 row_active_block_metadata_bytes = 0u;
    u32 row_group_metadata_bytes = 0u;
    bool dense_values_within_occupied_tiles = false;
    double byte_weight = 1.0;
    double occupied_tile_weight = 0.0;
    double row_active_block_weight = 0.0;
};

struct packing_cost_estimate {
    u64 value_slots = 0u;
    u64 value_bytes = 0u;
    u64 per_nnz_index_bytes = 0u;
    u64 occupied_tile_metadata_bytes = 0u;
    u64 row_active_block_metadata_bytes = 0u;
    u64 row_group_metadata_bytes = 0u;
    u64 total_bytes = 0u;
    double score = 0.0;
};

validation_result validate_csr_support_view(const csr_support_view &source);
validation_result validate_packing_plan_view(const packing_plan_view &plan);

validation_result prepare_csr_support(const csr_support_view &source, prepared_csr_support *out);

packing_plan_view make_packing_plan_view(const static_plan &plan);

validation_result query_packing_evaluation_requirements(
    const csr_support_view &source,
    const packing_plan_view &plan,
    packing_evaluation_requirements *out);

validation_result query_packing_evaluation_requirements(
    const prepared_csr_support &source,
    const packing_plan_view &plan,
    packing_evaluation_requirements *out);

validation_result evaluate_packing_plan(
    const csr_support_view &source,
    const packing_plan_view &plan,
    const packing_evaluation_workspace_view &workspace,
    const packing_occupancy_buffers &buffers,
    packing_occupancy_result *out);

validation_result evaluate_packing_plan(
    const prepared_csr_support &source,
    const packing_plan_view &plan,
    const packing_evaluation_workspace_view &workspace,
    const packing_occupancy_buffers &buffers,
    packing_occupancy_result *out);

validation_result estimate_packing_cost(
    const packing_occupancy_result &occupancy,
    const packing_cost_model &model,
    packing_cost_estimate *out);

count_distribution merge_count_distributions(const count_distribution &lhs, const count_distribution &rhs);
real_distribution merge_real_distributions(const real_distribution &lhs, const real_distribution &rhs);

} // namespace cellpack
