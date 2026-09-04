#pragma once

#include <Cellerator/compiler/ir/realization/implement_physical_plane_representation_v1.hh>
#include <Cellerator/geometry/persistence/execution_image_v2.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class projection_kind_v1 : std::uint8_t {
    csr = 1u, feature_major, row_masked, dense_fragment, mma_hybrid,
    transpose, vendor_specific, extension,
};

struct projection_payload_abi_v1 {
    stable_identity_v1 identity{};
    std::uint32_t schema_version = 0u;
    std::uint32_t alignment = 1u;
    std::uint64_t bytes = 0u;
};

struct value_position_v1 {
    std::uint64_t logical_value = 0u;
    std::uint64_t physical_position = 0u;
};

struct projection_contract_v1 {
    stable_identity_v1 identity{};
    stable_identity_v1 structure_plane{};
    stable_identity_v1 value_plane{};
    stable_identity_v1 capability_identity{};
    projection_kind_v1 kind = projection_kind_v1::csr;
    projection_payload_abi_v1 payload{};
    std::vector<value_position_v1> forward_value_map;
    std::vector<value_position_v1> transpose_value_map;
    bool forward_capable = false;
    bool transpose_capable = false;
};

enum class projection_status_v1 : std::uint8_t {
    valid = 0u, invalid_identity, invalid_payload, duplicate_logical_value,
    duplicate_physical_position, missing_forward_map, missing_transpose_map,
    capability_mismatch,
};

[[nodiscard]] projection_status_v1 validate_projection_contract_v1(
    const projection_contract_v1& projection,
    const target_capability_v1* capability = nullptr,
    std::string* error = nullptr) noexcept;

[[nodiscard]] projection_contract_v1 import_cpe2_projection_v1(
    const cellpack::persistence::execution_projection_entry_v1& entry,
    stable_identity_v1 structure_plane,
    stable_identity_v1 value_plane);

} // namespace cellerator::compiler::ir::realization::v1
