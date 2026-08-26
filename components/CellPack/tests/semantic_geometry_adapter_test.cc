#include "CellPack/semantic_geometry.hh"

#include <Cellerator/types.cuh>

#include <cstdint>
#include <cstdlib>
#include <iostream>

namespace {

namespace ce = cellerator::execution;
namespace cp = cellpack;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "cellPackSemanticGeometryAdapterTest: " << message << '\n';
        std::exit(1);
    }
}

ce::axis_identity axis(std::uint32_t base) {
    return {{base, 1u}, {base + 1u, 1u}, {base + 2u, 1u}, {base + 3u, 1u}};
}

cp::persistent_packing_payload_view fixture() {
    alignas(64) static unsigned char image[512]{};
    static const cp::u32 feature_permutation[] = {2u, 0u, 3u, 1u};
    static const cp::u32 inverse_feature_permutation[] = {1u, 3u, 0u, 2u};
    static const cp::u32 feature_block_offsets[] = {0u, 2u, 4u};
    static const cp::u32 feature_to_block[] = {0u, 1u, 0u, 1u};
    static const cp::u32 feature_to_local[] = {1u, 1u, 0u, 0u};
    static const cp::u32 row_group_offsets[] = {0u, 2u};
    static const cp::u32 row_permutation[] = {1u, 0u};
    static const cp::u32 inverse_row_permutation[] = {1u, 0u};
    static const cp::u32 row_nnz[] = {3u, 1u};
    static const cp::u32 tile_block_offsets[] = {0u, 2u};
    static const cp::u32 tile_block_ids[] = {0u, 1u};
    static const cp::u32 tile_block_cell_masks[] = {3u, 1u};
    static const cp::u32 block_row_entry_offsets[] = {0u, 2u, 3u};
    static const cp::u32 row_block_gene_masks[] = {1u, 2u, 3u};
    static const cp::u32 row_block_value_offsets[] = {0u, 1u, 2u, 4u};
    alignas(cellerator::real::storage_t) static const unsigned char values[
        4u * sizeof(cellerator::real::storage_t)]{};

    cp::persistent_packing_payload_view payload;
    payload.payload_schema_version = cp::persistent_packing_payload_schema_version;
    payload.payload_kind = cp::persistent_packing_payload_kind;
    payload.payload_identity = 0xabcdu;
    payload.image_base = image;
    payload.image_bytes = sizeof(image);
    payload.objective_kind = cp::packing_exact_objective_kind::row_active_block_references;
    payload.cost_policy_identity = 0x55u;
    payload.maximum_feature_block_width = 2u;
    payload.row_group_width = 2u;
    payload.inverse_feature_permutation = inverse_feature_permutation;
    payload.feature_to_block = feature_to_block;
    payload.feature_to_local = feature_to_local;
    payload.row_group_count = 1u;
    payload.row_group_offsets = row_group_offsets;
    payload.plan.semantic_plan_schema_version = cp::packing_plan_semantic_schema_version;
    payload.plan.geometry_identity_version = cp::feature_block_geometry_identity_version;
    payload.plan.feature_count = 4u;
    payload.plan.feature_block_count = 2u;
    payload.plan.feature_block_geometry_identity = 0x111u;
    payload.plan.feature_block_offsets = feature_block_offsets;
    payload.plan.feature_permutation = feature_permutation;
    payload.order.order_schema_version = cp::local_cell_order_schema_version;
    payload.order.signature_algorithm_version = cp::local_cell_signature_algorithm_version;
    payload.order.ordering_identity = 0x222u;
    payload.order.row_count = 2u;
    payload.order.row_nnz_counts = row_nnz;
    payload.order.row_permutation = row_permutation;
    payload.order.inverse_row_permutation = inverse_row_permutation;
    payload.tiles.tile_schema_version = cp::warp_tile_schema_version;
    payload.tiles.tile_identity = 0x333u;
    payload.tiles.feature_block_geometry_identity = 0x111u;
    payload.tiles.ordering_identity = 0x222u;
    payload.tiles.row_count = 2u;
    payload.tiles.feature_count = 4u;
    payload.tiles.feature_block_count = 2u;
    payload.tiles.tile_count = 1u;
    payload.tiles.tile_block_count = 2u;
    payload.tiles.row_block_entry_count = 3u;
    payload.tiles.nnz_count = 4u;
    payload.tiles.value_size_bytes = sizeof(cellerator::real::storage_t);
    payload.tiles.tile_block_offsets = tile_block_offsets;
    payload.tiles.tile_block_ids = tile_block_ids;
    payload.tiles.tile_block_cell_masks = tile_block_cell_masks;
    payload.tiles.block_row_entry_offsets = block_row_entry_offsets;
    payload.tiles.row_block_gene_masks = row_block_gene_masks;
    payload.tiles.row_block_value_offsets = row_block_value_offsets;
    payload.tiles.values = values;
    return payload;
}

} // namespace

int main() {
    const auto payload = fixture();
    cp::cp_bp_v1_adapter_request request;
    request.row_axis = axis(1u);
    request.feature_axis = axis(11u);
    request.structure = {21u, 1u};
    request.structure_epoch_value = {7u};
    request.projection_catalog = {22u, 1u};
    request.projection = {23u, 1u};
    request.value_generation_value = {9u};
    request.value_location = {ce::residency_kind::host, {}, -1, 0u};
    request.numeric = {ce::numeric_type::f32, ce::numeric_type::f32,
        ce::numeric_type::f32, 0u};

    cp::cp_bp_v1_compatibility_adapter adapter;
    require(static_cast<bool>(
                cp::build_cp_bp_v1_compatibility_adapter_host(payload, request, &adapter)),
        "build adapter");
    require(adapter.payload.image_base == payload.image_base
            && adapter.direct_tiles.values == payload.tiles.values
            && adapter.values.values == payload.tiles.values,
        "adapter must alias CPK1 without rebuilding");
    require(adapter.geometry.execution_to_canonical_feature
                == payload.plan.feature_permutation
            && adapter.geometry.canonical_to_execution_feature
                == payload.inverse_feature_permutation
            && adapter.geometry.execution_to_canonical_row == payload.order.row_permutation
            && adapter.geometry.canonical_to_execution_row
                == payload.order.inverse_row_permutation,
        "canonical recovery maps changed");
    require(adapter.geometry.v1_objective_kind
            == cp::packing_exact_objective_kind::row_active_block_references,
        "v1 objective meaning changed");
    require(ce::same_axis_identity(adapter.structure.source_axis,
                request.feature_axis)
            && ce::same_axis_identity(adapter.structure.destination_axis,
                request.row_axis),
        "forward relation must be feature source to row destination");
    require(ce::validate_relation_structure(adapter.structure)
                == ce::lifetime_validation_code::ok
            && ce::validate_value_plane(adapter.structure, adapter.values)
                == ce::lifetime_validation_code::ok
            && static_cast<bool>(
                cp::validate_cp_bp_v1_compatibility_adapter_host(adapter)),
        "structure/value separation is invalid");

    cp::semantic_geometry_hot_summary hot;
    cp::semantic_geometry_cold_sidecar cold;
    require(static_cast<bool>(
                cp::evaluate_cp_bp_v1_semantic_statistics_host(adapter, &hot, &cold)),
        "evaluate semantic statistics");
    require(hot.nnz_count == 4u && hot.projection_value_bytes
                == 4u * sizeof(cellerator::real::storage_t)
            && hot.maximum_row_nnz == 3u && hot.tile_block_count == 2u,
        "hot summary is incorrect");
    require(cold.row_nnz_histogram[1] == 1u && cold.row_nnz_histogram[2] == 1u
            && cold.row_mask_popcount_histogram[1] == 1u
            && cold.row_mask_popcount_histogram[2] == 1u,
        "cold sidecar histograms are incorrect");
    const auto manifest = cp::cp_bp_semantic_statistics_manifest();
    require((manifest.requires_external_semantics_mask
                & cp::statistic_mask(cp::semantic_statistic::module_occupancy)) != 0u
            && (cold.available_mask
                & cp::statistic_mask(cp::semantic_statistic::module_occupancy)) == 0u,
        "unavailable biological statistics were guessed");

    auto stale = request;
    stale.structure_epoch_value = {0u};
    require(!cp::build_cp_bp_v1_compatibility_adapter_host(payload, stale, &adapter),
        "zero structure epoch was accepted");
    auto malformed = payload;
    malformed.inverse_feature_permutation = nullptr;
    require(!cp::build_cp_bp_v1_compatibility_adapter_host(malformed, request, &adapter),
        "missing canonical recovery map was accepted");
    cp::cp_bp_v1_compatibility_adapter swapped;
    require(static_cast<bool>(
                cp::build_cp_bp_v1_compatibility_adapter_host(payload, request, &swapped)),
        "build adapter for swapped-axis rejection");
    const auto source_axis = swapped.structure.source_axis;
    swapped.structure.source_axis = swapped.structure.destination_axis;
    swapped.structure.destination_axis = source_axis;
    require(!cp::validate_cp_bp_v1_compatibility_adapter_host(swapped),
        "swapped row/feature axes were not rejected");

    std::cout << "cellPackSemanticGeometryAdapterTest passed"
              << " adapter_bytes=" << sizeof(adapter)
              << " metadata_bytes=" << hot.projection_metadata_bytes
              << " value_bytes=" << hot.projection_value_bytes << '\n';
    return 0;
}
