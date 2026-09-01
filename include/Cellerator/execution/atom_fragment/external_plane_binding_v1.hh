#pragma once

#include <Cellerator/execution/atom_fragment/prepared_atom_fragment_v1.hh>
#include <Cellerator/execution/joint_compiler/external_binding_v1.hh>

namespace cellerator::execution::atom_fragment {

struct bound_atom_extent_v1 {
    joint_compiler::persistent_identity_v1 plane_identity{};
    const void *address = nullptr;
    device_location location{};
    std::uint64_t plane_byte_offset = 0u;
    std::uint64_t bytes = 0u;
    joint_compiler::opaque_runtime_token_v1 readiness{};
    joint_compiler::opaque_runtime_token_v1 lease{};
};

enum class external_plane_binding_status_code_v1 : std::uint8_t {
    success = 0u,
    invalid_prepared_fragment,
    invalid_requirement,
    invalid_affordance,
    mismatched_contract,
    missing_bindings,
    invalid_binding,
    missing_plane,
    ambiguous_plane,
    incompatible_order,
    incompatible_generation,
    insufficient_capacity,
};

struct external_plane_binding_status_v1 {
    external_plane_binding_status_code_v1 code =
        external_plane_binding_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t required_capacity = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == external_plane_binding_status_code_v1::success;
    }
};

external_plane_binding_status_v1 bind_external_atom_planes_v1(
    const prepared_atom_fragment_v1 &prepared,
    const joint_compiler::atom_requirement_v1 &requirement,
    const joint_compiler::atom_affordance_v1 &affordance,
    const joint_compiler::external_binding_v1 *bindings,
    std::uint64_t binding_count,
    bound_atom_extent_v1 *output,
    std::uint64_t output_capacity,
    std::uint64_t *written) noexcept;

} // namespace cellerator::execution::atom_fragment
