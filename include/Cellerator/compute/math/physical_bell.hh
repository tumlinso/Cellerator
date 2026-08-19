#pragma once

#include <CellPack/local_cell_ordering.hh>
#include <CellPack/packing_plan.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellerator::compute::math {

inline constexpr std::uint32_t physical_bell_schema_version = 1u;
inline constexpr std::uint32_t physical_bell_candidate_count = 3u;

enum class bell_candidate_state : std::uint32_t {
    legal = 0u,
    empty_source = 1u,
    value_expansion_exceeded = 2u,
    storage_expansion_exceeded = 3u,
    persistent_bytes_exceeded = 4u,
    overflow = 5u
};

enum class bell_lowering_status_code : std::uint32_t {
    ok = 0u,
    invalid_argument = 1u,
    invalid_source = 2u,
    incompatible_plan = 3u,
    incompatible_order = 4u,
    insufficient_capacity = 5u,
    candidate_rejected = 6u,
    candidate_mismatch = 7u,
    overflow = 8u
};

struct bell_lowering_status {
    bell_lowering_status_code code = bell_lowering_status_code::ok;
    const char *message = "ok";

    constexpr explicit operator bool() const noexcept {
        return code == bell_lowering_status_code::ok;
    }
};

// Canonical partition-local CSR. Feature ids are canonical and strictly
// increasing within each row; values are copied byte-for-byte during lowering.
struct bell_csr_source_view {
    std::uint32_t row_count = 0u;
    std::uint32_t feature_count = 0u;
    std::uint32_t nnz_count = 0u;
    std::uint32_t value_size_bytes = 0u;
    const std::uint32_t *row_offsets = nullptr;
    const std::uint32_t *feature_ids = nullptr;
    const void *values = nullptr;
};

// Read-only projection of the frozen PackingPlan ABI. Offsets remain semantic;
// a BELL candidate pads each block independently but never changes membership
// or execution order.
struct bell_semantic_plan_view {
    std::uint32_t semantic_schema_version = 0u;
    std::uint32_t full_row_count = 0u;
    std::uint32_t feature_count = 0u;
    std::uint32_t feature_block_count = 0u;
    std::uint64_t feature_block_geometry_identity = 0u;
    std::uint64_t row_domain_identity = 0u;
    const std::uint32_t *feature_block_offsets = nullptr;
    const std::uint32_t *feature_permutation = nullptr;
    const std::uint32_t *inverse_feature_permutation = nullptr;
};

inline bell_semantic_plan_view make_bell_semantic_plan_view(
    const cellpack::frozen_packing_plan &plan) noexcept {
    bell_semantic_plan_view result;
    result.semantic_schema_version = plan.semantic_schema_version();
    result.full_row_count = plan.row_count();
    result.feature_count = plan.feature_count();
    result.feature_block_count = plan.feature_block_count();
    result.feature_block_geometry_identity = plan.feature_block_geometry_identity();
    result.row_domain_identity = plan.identity().row_domain_identity;
    result.feature_block_offsets = plan.feature_block_offsets();
    result.feature_permutation = plan.feature_permutation();
    result.inverse_feature_permutation = plan.inverse_feature_permutation();
    return result;
}

struct bell_lowering_policy {
    double maximum_value_slot_expansion = 64.0;
    double maximum_storage_expansion = 64.0;
    std::size_t maximum_persistent_bytes = std::numeric_limits<std::size_t>::max();
};

struct bell_candidate_metrics {
    std::uint64_t occupied_blocks = 0u;
    std::uint64_t stored_blocks = 0u;
    std::uint64_t dense_value_slots = 0u;
    std::uint64_t persistent_bytes = 0u;
    std::uint64_t source_bytes = 0u;
    double scalar_occupancy = 0.0;
    double ell_slot_utilization = 0.0;
    double value_slot_expansion = 0.0;
    double storage_expansion = 0.0;
};

struct bell_candidate_requirements {
    bell_candidate_state state = bell_candidate_state::overflow;
    std::uint32_t block_size = 0u;
    std::uint32_t row_count = 0u;
    std::uint32_t feature_count = 0u;
    std::uint32_t padded_row_count = 0u;
    std::uint32_t padded_feature_count = 0u;
    std::uint32_t block_row_count = 0u;
    std::uint32_t ell_blocks_per_row = 0u;
    std::uint32_t ell_columns = 0u;
    std::size_t feature_block_offset_count = 0u;
    std::size_t column_index_count = 0u;
    std::size_t value_bytes = 0u;
    std::uint64_t candidate_identity = 0u;
    bell_candidate_metrics metrics{};
};

struct bell_candidate_set {
    bell_candidate_requirements candidates[physical_bell_candidate_count]{};
};

struct bell_lowering_workspace_requirements {
    std::size_t marker_count = 0u;
    std::size_t feature_block_offset_count = 0u;
};

struct bell_lowering_workspace {
    std::size_t marker_capacity = 0u;
    std::uint32_t *markers = nullptr;
    std::size_t feature_block_offset_capacity = 0u;
    std::uint32_t *feature_block_block_offsets = nullptr;
};

struct bell_candidate_buffers {
    std::size_t feature_block_offset_capacity = 0u;
    std::uint32_t *padded_feature_block_offsets = nullptr;
    std::size_t column_index_capacity = 0u;
    std::int32_t *column_indices = nullptr;
    std::size_t value_capacity_bytes = 0u;
    void *values = nullptr;
};

// Pointers alias caller-owned candidate buffers. padded_row_count and
// padded_feature_count are the cuSPARSE descriptor dimensions; row_count and
// feature_count retain the logical partition dimensions for independent decode.
struct physical_bell_view {
    std::uint32_t schema_version = physical_bell_schema_version;
    std::uint32_t block_size = 0u;
    std::uint32_t row_count = 0u;
    std::uint32_t feature_count = 0u;
    std::uint32_t padded_row_count = 0u;
    std::uint32_t padded_feature_count = 0u;
    std::uint32_t ell_columns = 0u;
    std::uint32_t value_size_bytes = 0u;
    std::uint64_t feature_block_geometry_identity = 0u;
    std::uint64_t ordering_identity = 0u;
    std::uint64_t row_domain_identity = 0u;
    std::uint64_t candidate_identity = 0u;
    const std::uint32_t *padded_feature_block_offsets = nullptr;
    const std::int32_t *column_indices = nullptr;
    const void *values = nullptr;
    bell_candidate_metrics metrics{};
};

bell_lowering_status query_bell_lowering_workspace_requirements(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    bell_lowering_workspace_requirements *out) noexcept;

bell_lowering_status query_bell_candidates_host(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const bell_lowering_policy &policy,
    const bell_lowering_workspace &workspace,
    bell_candidate_set *out) noexcept;

bell_lowering_status materialize_bell_candidate_host(
    const bell_csr_source_view &source,
    const bell_semantic_plan_view &plan,
    const cellpack::local_cell_order_view &order,
    const bell_lowering_policy &policy,
    const bell_candidate_requirements &candidate,
    const bell_lowering_workspace &workspace,
    const bell_candidate_buffers &buffers,
    physical_bell_view *out) noexcept;

static_assert(std::is_trivially_copyable<physical_bell_view>::value,
    "physical BELL view must remain pointer-copyable");

} // namespace cellerator::compute::math
