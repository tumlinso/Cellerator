#pragma once

#include <Cellerator/execution/execution_contract.hh>
#include <Cellerator/layout/cell_packing_geometry.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution {

inline constexpr std::uint32_t cell_packing_binding_schema_version = 1u;

struct CellPackingBindingRequest {
    axis_identity row_axis{}, feature_axis{};
    structure_handle structure{};
    ::cellerator::structure::structure_epoch structure_epoch_value{};
    projection_catalog_handle projection_catalog{};
    projection_handle projection{};
    value_generation value_generation_value{};
    device_location value_location{};
    value_numeric_policy numeric{};
};

// Borrowed association to an existing validated CPK1 payload. It retains no
// bytes, does not change packing order, and does not own runtime resources.
struct CellPackingBinding {
    std::uint32_t schema_version = cell_packing_binding_schema_version;
    layout::CellPackingGeometryView geometry{};
    axis_identity row_axis{}, feature_axis{};
    relation_structure structure{};
    value_plane values{};
    projection_handle projection{};
    cellpack::persistent_packing_payload_view payload{};
    cellpack::feature_weighted_row_reduction_plan_view direct_plan{};
    cellpack::local_cell_order_view direct_order{};
    cellpack::warp_tile_view direct_tiles{};
};

cellpack::validation_result build_cell_packing_binding_host(
    const cellpack::persistent_packing_payload_view&, const CellPackingBindingRequest&,
    CellPackingBinding*) noexcept;
cellpack::validation_result validate_cell_packing_binding_host(const CellPackingBinding&) noexcept;

static_assert(std::is_trivially_copyable<CellPackingBinding>::value,
              "packing bindings must remain borrowed views");
} // namespace cellerator::execution
