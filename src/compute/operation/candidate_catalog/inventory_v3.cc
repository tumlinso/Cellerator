#include "Cellerator/compute/operation/candidate_catalog_v3/inventory.h"

#include <cstdint>

namespace cellerator::compute::operation::catalog_v3 {
namespace {

constexpr std::uint64_t fnv(const char* text) noexcept {
    std::uint64_t hash = 14695981039346656037ULL;
    while (*text != 0) {
        hash ^= static_cast<std::uint8_t>(*text++);
        hash *= 1099511628211ULL;
    }
    return hash;
}

constexpr std::uint64_t common = operation_forward | operation_graph_capture |
        operation_logical_values | operation_projection_values |
        operation_packed_output | operation_canonical_output;

constexpr relation_operation_record_v3 operations[] = {
    {relation_operation_v3::forward_apply, fnv("relation.forward.v3"), common | operation_backward | operation_dynamic_values, "forward-apply"},
    {relation_operation_v3::transpose_apply, fnv("relation.transpose.v3"), common | operation_backward | operation_dynamic_values, "transpose-apply"},
    {relation_operation_v3::contract_on_support, fnv("relation.contract.v3"), common | operation_backward, "contract-on-support"},
    {relation_operation_v3::segment_reduce, fnv("relation.segment-reduce.v3"), common | operation_backward, "segment-reduce"},
    {relation_operation_v3::segment_normalize, fnv("relation.segment-normalize.v3"), common | operation_backward, "segment-normalize"},
    {relation_operation_v3::edge_map, fnv("relation.edge-map.v3"), common | operation_backward | operation_dynamic_values, "edge-map"},
    {relation_operation_v3::edge_gate, fnv("relation.edge-gate.v3"), common | operation_backward | operation_dynamic_values, "edge-gate"},
    {relation_operation_v3::sparse_axis_update, fnv("relation.sparse-update.v3"), common | operation_dynamic_values, "sparse-axis-update"},
    {relation_operation_v3::relation_bundle, fnv("relation.bundle.v3"), common | operation_backward, "relation-bundle"},
    {relation_operation_v3::relation_chain, fnv("relation.chain.v3"), common | operation_backward, "relation-chain"},
    {relation_operation_v3::relation_moments, fnv("relation.moments.v3"), common | operation_backward, "relation-moments"},
    {relation_operation_v3::relation_exchange, fnv("relation.exchange.v3"), common, "relation-exchange"},
};

constexpr provider_inventory_record_v3 providers[] = {
    {fnv("provider.scalar-sparse.v3"), provider_pure_sparse, 0, 0, true, {}, "scalar-sparse"},
    {fnv("provider.warp-sparse.v3"), provider_pure_sparse, 7, 0, true, {}, "warp-sparse"},
    {fnv("provider.csg-projection.v3"), provider_semantic_geometry | provider_physical_projection, 7, 0, true, {}, "csg-projection"},
    {fnv("provider.sm70-mma-residual.v3"), provider_physical_projection | provider_residual | provider_mma, 7, 0, true, {}, "sm70-mma-residual"},
    {fnv("provider.vendor-sparse.v3"), provider_pure_sparse | provider_vendor, 7, 0, true, {}, "vendor-sparse"},
    {fnv("provider.experimental.v3"), provider_physical_projection | provider_experimental, 7, 0, false, {}, "experimental"},
};

}  // namespace

provider_operation_inventory_v3 built_in_provider_operation_inventory_v3() noexcept {
    return {providers, sizeof(providers) / sizeof(providers[0]),
            operations, sizeof(operations) / sizeof(operations[0])};
}

bool validate_provider_operation_inventory_v3(
        const provider_operation_inventory_v3& inventory) noexcept {
    if ((inventory.provider_count != 0 && inventory.providers == nullptr) ||
        (inventory.operation_count != 0 && inventory.operations == nullptr)) return false;
    for (std::uint64_t i = 0; i < inventory.provider_count; ++i) {
        if (inventory.providers[i].stable_provider_id == 0 || inventory.providers[i].stable_name[0] == 0) return false;
        for (std::uint64_t j = 0; j < i; ++j) if (inventory.providers[j].stable_provider_id == inventory.providers[i].stable_provider_id) return false;
    }
    for (std::uint64_t i = 0; i < inventory.operation_count; ++i) {
        if (inventory.operations[i].stable_operation_id == 0 || inventory.operations[i].stable_name[0] == 0) return false;
        for (std::uint64_t j = 0; j < i; ++j) if (inventory.operations[j].stable_operation_id == inventory.operations[i].stable_operation_id) return false;
    }
    return inventory.provider_count >= 5 && inventory.operation_count >= 12;
}

}  // namespace cellerator::compute::operation::catalog_v3
