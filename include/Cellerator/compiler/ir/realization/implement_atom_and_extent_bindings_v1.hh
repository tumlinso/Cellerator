#pragma once

#include <Cellerator/compiler/ir/realization/freeze_realization_ir_module_and_target_scopes_v1.hh>

#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compiler::ir::realization::v1 {

enum class physical_instance_role_v1 : std::uint8_t {
    owner = 1u,
    halo,
    replica,
    partial_contributor,
};

enum class address_space_class_v1 : std::uint8_t {
    host = 1u,
    device_global,
    device_constant,
    device_shared,
    peer_device,
};

enum class local_index_width_v1 : std::uint8_t {
    bits_16 = 16u,
    bits_32 = 32u,
    bits_64 = 64u,
};

struct extent_slice_v1 {
    std::uint64_t global_begin = 0u;
    std::uint64_t element_count = 0u;
    std::uint64_t local_begin = 0u;
    std::uint64_t stride = 1u;
};

// This is an address-free compiler object. Runtime acquisition binds the
// symbolic artifact identity to live memory after validation.
struct atom_extent_binding_v1 {
    stable_identity_v1 identity{};
    stable_identity_v1 atom_identity{};
    stable_identity_v1 parent_atom_identity{};
    stable_identity_v1 artifact_identity{};
    physical_instance_role_v1 role = physical_instance_role_v1::owner;
    address_space_class_v1 address_space = address_space_class_v1::host;
    local_index_width_v1 local_index_width = local_index_width_v1::bits_32;
    std::uint32_t alignment = 1u;
    std::uint64_t global_element_count = 0u;
    std::vector<extent_slice_v1> extents;
    std::vector<std::uint64_t> canonical_recovery;
};

enum class atom_extent_status_v1 : std::uint8_t {
    valid = 0u,
    invalid_identity,
    invalid_extent,
    overlapping_global_extent,
    overlapping_local_extent,
    invalid_alignment,
    index_width_overflow,
    invalid_recovery,
};

[[nodiscard]] atom_extent_status_v1 validate_atom_extent_binding_v1(
    const atom_extent_binding_v1& binding,
    std::string* error = nullptr) noexcept;

[[nodiscard]] bool equivalent_atom_extent_binding_v1(
    const atom_extent_binding_v1& lhs,
    const atom_extent_binding_v1& rhs) noexcept;

} // namespace cellerator::compiler::ir::realization::v1
