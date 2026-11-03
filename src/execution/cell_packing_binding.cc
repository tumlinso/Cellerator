#include "Cellerator/execution/cell_packing_binding.hh"

namespace cellerator::execution {
using namespace cellpack;
namespace {
bool valid_payload_metadata(const persistent_packing_payload_view &payload) noexcept {
    const auto &plan = payload.plan;
    const auto &order = payload.order;
    const auto &tiles = payload.tiles;
    if (payload.payload_schema_version != persistent_packing_payload_schema_version
        || payload.payload_kind != persistent_packing_payload_kind
        || payload.payload_identity == 0u || payload.image_base == nullptr
        || payload.image_bytes == 0u || payload.cost_policy_identity == 0u)
        return false;
    if (plan.semantic_plan_schema_version != packing_plan_semantic_schema_version
        || plan.geometry_identity_version != feature_block_geometry_identity_version
        || plan.feature_count == 0u || plan.feature_block_count == 0u
        || plan.feature_block_geometry_identity == 0u
        || plan.feature_block_offsets == nullptr || plan.feature_permutation == nullptr
        || payload.inverse_feature_permutation == nullptr
        || payload.feature_to_block == nullptr || payload.feature_to_local == nullptr)
        return false;
    if (order.order_schema_version != local_cell_order_schema_version
        || order.ordering_identity == 0u || order.row_count == 0u
        || order.row_permutation == nullptr || order.inverse_row_permutation == nullptr
        || order.row_nnz_counts == nullptr || payload.row_group_count == 0u
        || payload.row_group_offsets == nullptr)
        return false;
    if (tiles.tile_schema_version != warp_tile_schema_version
        || tiles.tile_identity == 0u || tiles.ordering_identity != order.ordering_identity
        || tiles.feature_block_geometry_identity != plan.feature_block_geometry_identity
        || tiles.row_count != order.row_count || tiles.feature_count != plan.feature_count
        || tiles.feature_block_count != plan.feature_block_count
        || tiles.value_size_bytes == 0u)
        return false;
    if ((tiles.tile_count != 0u && tiles.tile_block_offsets == nullptr)
        || (tiles.tile_block_count != 0u
            && (tiles.tile_block_ids == nullptr || tiles.tile_block_cell_masks == nullptr
                || tiles.block_row_entry_offsets == nullptr))
        || (tiles.row_block_entry_count != 0u && tiles.row_block_gene_masks == nullptr)
        || (tiles.nnz_count != 0u
            && (tiles.row_block_value_offsets == nullptr || tiles.values == nullptr)))
        return false;
    return true;
}
} // namespace

validation_result validate_cell_packing_binding_host(
    const CellPackingBinding &adapter) noexcept {
    if (adapter.schema_version != cell_packing_binding_schema_version
        || adapter.geometry.schema_version
            != layout::cell_packing_geometry_schema_version
        || !valid_payload_metadata(adapter.payload)
        || !same_axis_identity(adapter.structure.source_axis,
            adapter.feature_axis)
        || !same_axis_identity(adapter.structure.destination_axis,
            adapter.row_axis)) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CP-BP adapter requires feature-source to row-destination orientation");
    }
    if (validate_relation_structure(adapter.structure)
            != lifetime_validation_code::ok
        || validate_value_plane(adapter.structure, adapter.values)
            != lifetime_validation_code::ok
        || adapter.structure.logical_edge_count != adapter.payload.tiles.nnz_count
        || adapter.values.element_count != adapter.payload.tiles.nnz_count
        || adapter.values.values != adapter.payload.tiles.values
        || adapter.direct_tiles.values != adapter.payload.tiles.values
        || adapter.direct_plan.feature_permutation
            != adapter.payload.plan.feature_permutation
        || adapter.direct_order.row_permutation
            != adapter.payload.order.row_permutation) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CP-BP adapter lifetime, edge identity, or aliasing contract is invalid");
    }
    return validation_ok();
}

validation_result build_cell_packing_binding_host(
    const persistent_packing_payload_view &payload,
    const CellPackingBindingRequest &request,
    CellPackingBinding *out) noexcept {
    if (out == nullptr) return validation_error(
        validation_code::null_pointer, invalid_id, "CP-BP adapter output is null");
    if (!valid_payload_metadata(payload)) return validation_error(
        validation_code::invalid_matrix_view, invalid_id, "CPK1 metadata is invalid");
    if (!valid_axis_identity(request.row_axis)
        || !valid_axis_identity(request.feature_axis)
        || !valid_handle(request.structure) || !valid_handle(request.projection)
        || !valid_projection_catalog(request.projection_catalog)
        || request.structure_epoch_value.value == 0u
        || request.value_generation_value.value == 0u
        || !valid_location(request.value_location)) {
        return validation_error(validation_code::invalid_signature, invalid_id,
            "CP-BP adapter identity or residency is invalid");
    }
    if (request.numeric.storage == numeric_type::invalid
        || request.numeric.dequantized == numeric_type::invalid
        || request.numeric.accumulation == numeric_type::invalid) {
        return validation_error(validation_code::invalid_matrix_view, invalid_id,
            "CP-BP adapter numeric policy is invalid");
    }
    const u64 value_bytes = static_cast<u64>(payload.tiles.nnz_count)
        * payload.tiles.value_size_bytes;
    if (payload.tiles.nnz_count != 0u
        && value_bytes / payload.tiles.nnz_count != payload.tiles.value_size_bytes) {
        return validation_error(validation_code::integer_overflow, invalid_id,
            "CP-BP v1 value byte count overflows");
    }

    CellPackingBinding result;
    result.row_axis = request.row_axis;
    result.feature_axis = request.feature_axis;
    result.geometry.v1_feature_block_geometry_identity =
        payload.plan.feature_block_geometry_identity;
    result.geometry.v1_ordering_identity = payload.order.ordering_identity;
    result.geometry.v1_payload_identity = payload.payload_identity;
    result.geometry.cost_policy_identity = payload.cost_policy_identity;
    result.geometry.v1_objective_kind = payload.objective_kind;
    result.geometry.row_count = payload.order.row_count;
    result.geometry.feature_count = payload.plan.feature_count;
    result.geometry.feature_block_count = payload.plan.feature_block_count;
    result.geometry.row_group_count = payload.row_group_count;
    result.geometry.feature_block_offsets = payload.plan.feature_block_offsets;
    result.geometry.feature_to_block = payload.feature_to_block;
    result.geometry.feature_to_local = payload.feature_to_local;
    result.geometry.row_group_offsets = payload.row_group_offsets;
    result.geometry.execution_to_canonical_feature = payload.plan.feature_permutation;
    result.geometry.canonical_to_execution_feature = payload.inverse_feature_permutation;
    result.geometry.execution_to_canonical_row = payload.order.row_permutation;
    result.geometry.canonical_to_execution_row = payload.order.inverse_row_permutation;

    result.structure.identity = request.structure;
    result.structure.epoch = request.structure_epoch_value;
    result.structure.source_axis = request.feature_axis;
    result.structure.destination_axis = request.row_axis;
    result.structure.projections = request.projection_catalog;
    result.structure.logical_edge_count = payload.tiles.nnz_count;
    result.values.structure = request.structure;
    result.values.structure_epoch_value = request.structure_epoch_value;
    result.values.values = const_cast<void *>(payload.tiles.values);
    result.values.location = request.value_location;
    result.values.numeric = request.numeric;
    result.values.quantization.kind = quantization_kind::none;
    result.values.quantization.scale_type = numeric_type::invalid;
    result.values.quantization.offset_type = numeric_type::invalid;
    result.values.layout = value_layout_kind::projection_local_order;
    result.values.generation = request.value_generation_value;
    result.values.element_count = payload.tiles.nnz_count;
    result.values.value_bytes = value_bytes;
    result.projection = request.projection;
    result.payload = payload;
    result.direct_plan = payload.plan;
    result.direct_order = payload.order;
    result.direct_tiles = payload.tiles;
    const validation_result validated =
        validate_cell_packing_binding_host(result);
    if (!validated) return validated;
    *out = result;
    return validation_ok();
}
} // namespace cellerator::execution
