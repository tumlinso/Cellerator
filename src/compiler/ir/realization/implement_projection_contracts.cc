#include <Cellerator/compiler/ir/realization/implement_projection_contracts_v1.hh>

#include <set>

namespace cellerator::compiler::ir::realization::v1 {
namespace {

projection_status_v1 fail(projection_status_v1 status, std::string* error,
    const char* message) noexcept {
    if (error != nullptr) *error = message;
    return status;
}

bool power_of_two(std::uint32_t value) noexcept {
    return value != 0u && (value & (value - 1u)) == 0u;
}

bool valid_map(const std::vector<value_position_v1>& map) {
    std::set<std::uint64_t> logical, physical;
    for (const auto& value : map) {
        if (!logical.insert(value.logical_value).second ||
            !physical.insert(value.physical_position).second) return false;
    }
    return true;
}

projection_kind_v1 map_kind(cellpack::persistence::execution_projection_kind kind) {
    using source = cellpack::persistence::execution_projection_kind;
    switch (kind) {
        case source::csr: return projection_kind_v1::csr;
        case source::native_feature_major: return projection_kind_v1::feature_major;
        case source::native_row_masked: return projection_kind_v1::row_masked;
        case source::dense_fragment: return projection_kind_v1::dense_fragment;
        case source::cta_macrotile: return projection_kind_v1::mma_hybrid;
        case source::transpose_backward: return projection_kind_v1::transpose;
        case source::vendor_specific: return projection_kind_v1::vendor_specific;
        default: return projection_kind_v1::extension;
    }
}

} // namespace

projection_status_v1 validate_projection_contract_v1(
    const projection_contract_v1& projection,
    const target_capability_v1* capability,
    std::string* error) noexcept {
    if (!valid(projection.identity) || !valid(projection.structure_plane) ||
        !valid(projection.value_plane)) {
        return fail(projection_status_v1::invalid_identity, error,
            "projection and plane identities are required");
    }
    if (!valid(projection.payload.identity) || projection.payload.schema_version == 0u ||
        projection.payload.bytes == 0u || !power_of_two(projection.payload.alignment)) {
        return fail(projection_status_v1::invalid_payload, error,
            "payload identity, schema, bytes, and alignment are required");
    }
    if (projection.forward_capable && projection.forward_value_map.empty())
        return fail(projection_status_v1::missing_forward_map, error, "forward map required");
    if (projection.transpose_capable && projection.transpose_value_map.empty())
        return fail(projection_status_v1::missing_transpose_map, error, "transpose map required");
    if (!valid_map(projection.forward_value_map) || !valid_map(projection.transpose_value_map))
        return fail(projection_status_v1::duplicate_logical_value, error, "value maps must be bijective");
    if (valid(projection.capability_identity) &&
        (capability == nullptr || !(capability->identity == projection.capability_identity)))
        return fail(projection_status_v1::capability_mismatch, error, "capability identity mismatch");
    if (error != nullptr) error->clear();
    return projection_status_v1::valid;
}

projection_contract_v1 import_cpe2_projection_v1(
    const cellpack::persistence::execution_projection_entry_v1& entry,
    stable_identity_v1 structure_plane,
    stable_identity_v1 value_plane) {
    projection_contract_v1 result;
    result.identity = {entry.identity_high, entry.identity_low};
    result.structure_plane = structure_plane;
    result.value_plane = value_plane;
    result.kind = map_kind(entry.kind);
    result.payload = {{entry.identity_high, entry.identity_low}, entry.schema_version,
        cellpack::persistence::execution_image_v2_alignment, 1u};
    result.forward_capable =
        (entry.flags & cellpack::persistence::projection_forward_capable) != 0u;
    result.transpose_capable =
        (entry.flags & cellpack::persistence::projection_transpose_capable) != 0u;
    return result;
}

} // namespace cellerator::compiler::ir::realization::v1
