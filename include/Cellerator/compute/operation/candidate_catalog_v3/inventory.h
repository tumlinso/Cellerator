#pragma once

#include <cstdint>

namespace cellerator::compute::operation::catalog_v3 {

enum class relation_operation_v3 : std::uint32_t {
    forward_apply = 1,
    transpose_apply,
    contract_on_support,
    segment_reduce,
    segment_normalize,
    edge_map,
    edge_gate,
    sparse_axis_update,
    relation_bundle,
    relation_chain,
    relation_moments,
    relation_exchange,
};

enum operation_capability_v3 : std::uint64_t {
    operation_forward = 1ULL << 0U,
    operation_backward = 1ULL << 1U,
    operation_graph_capture = 1ULL << 2U,
    operation_logical_values = 1ULL << 3U,
    operation_projection_values = 1ULL << 4U,
    operation_packed_output = 1ULL << 5U,
    operation_canonical_output = 1ULL << 6U,
    operation_dynamic_values = 1ULL << 7U,
};

struct relation_operation_record_v3 {
    relation_operation_v3 operation = relation_operation_v3::forward_apply;
    std::uint64_t stable_operation_id = 0;
    std::uint64_t capabilities = 0;
    char stable_name[40]{};
};

enum provider_capability_v3 : std::uint64_t {
    provider_pure_sparse = 1ULL << 0U,
    provider_semantic_geometry = 1ULL << 1U,
    provider_physical_projection = 1ULL << 2U,
    provider_residual = 1ULL << 3U,
    provider_mma = 1ULL << 4U,
    provider_vendor = 1ULL << 5U,
    provider_experimental = 1ULL << 6U,
};

struct provider_inventory_record_v3 {
    std::uint64_t stable_provider_id = 0;
    std::uint64_t capabilities = 0;
    std::uint32_t minimum_compute_major = 0;
    std::uint32_t minimum_compute_minor = 0;
    bool compiled = false;
    std::uint8_t reserved[7]{};
    char stable_name[40]{};
};

struct provider_operation_inventory_v3 {
    const provider_inventory_record_v3* providers = nullptr;
    std::uint64_t provider_count = 0;
    const relation_operation_record_v3* operations = nullptr;
    std::uint64_t operation_count = 0;
};

provider_operation_inventory_v3 built_in_provider_operation_inventory_v3() noexcept;
bool validate_provider_operation_inventory_v3(
        const provider_operation_inventory_v3& inventory) noexcept;

}  // namespace cellerator::compute::operation::catalog_v3
