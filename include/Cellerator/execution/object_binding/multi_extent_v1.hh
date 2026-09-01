#pragma once

#include <Cellerator/execution/geometry_acquisition_v2/schema.hh>

#include <cstdint>
#include <type_traits>

namespace cellerator::execution::object_binding {

using stable_identity_v1 = acquisition_v2::stable_identity;

enum class binding_status_code_v1 : std::uint8_t {
    success = 0,
    invalid_argument,
    invalid_identity,
    duplicate_port,
    duplicate_atom,
    invalid_extent,
    insufficient_capacity,
    incompatible_requirement,
};

struct binding_status_v1 {
    binding_status_code_v1 code = binding_status_code_v1::success;
    std::uint64_t index = 0u;
    std::uint64_t required_capacity = 0u;

    constexpr explicit operator bool() const noexcept {
        return code == binding_status_code_v1::success;
    }
};

enum class port_access_v1 : std::uint8_t {
    read_only = 1u,
    write_only = 2u,
    read_write = 3u,
};

struct atom_port_binding_v1 {
    stable_identity_v1 atom_identity{};
    std::uint64_t logical_begin = 0u;
    std::uint64_t logical_extent = 0u;
};

struct multi_atom_port_binding_v1 {
    stable_identity_v1 port_identity{};
    stable_identity_v1 domain_identity{};
    stable_identity_v1 order_identity{};
    const atom_port_binding_v1 *atoms = nullptr;
    std::uint64_t atom_count = 0u;
    port_access_v1 access = port_access_v1::read_only;
    std::uint8_t reserved[7]{};
};

struct multi_atom_port_binding_list_v1 {
    const multi_atom_port_binding_v1 *ports = nullptr;
    std::uint64_t port_count = 0u;
};

constexpr bool valid_identity_v1(stable_identity_v1 identity) noexcept {
    return identity.low != 0u || identity.high != 0u;
}

inline binding_status_v1 validate_multi_atom_port_bindings_v1(
    const multi_atom_port_binding_list_v1 &list) noexcept {
    if (list.port_count != 0u && list.ports == nullptr) {
        return {binding_status_code_v1::invalid_argument};
    }
    for (std::uint64_t port_index = 0u; port_index < list.port_count;
         ++port_index) {
        const auto &port = list.ports[port_index];
        if (!valid_identity_v1(port.port_identity) ||
            !valid_identity_v1(port.domain_identity) ||
            !valid_identity_v1(port.order_identity)) {
            return {binding_status_code_v1::invalid_identity, port_index};
        }
        if (port.atom_count == 0u || port.atoms == nullptr) {
            return {binding_status_code_v1::invalid_argument, port_index};
        }
        for (std::uint64_t other = 0u; other < port_index; ++other) {
            const auto &identity = list.ports[other].port_identity;
            if (identity.low == port.port_identity.low &&
                identity.high == port.port_identity.high) {
                return {binding_status_code_v1::duplicate_port, port_index};
            }
        }
        for (std::uint64_t atom_index = 0u; atom_index < port.atom_count;
             ++atom_index) {
            const auto &atom = port.atoms[atom_index];
            if (!valid_identity_v1(atom.atom_identity)) {
                return {binding_status_code_v1::invalid_identity, atom_index};
            }
            if (atom.logical_extent == 0u ||
                atom.logical_begin > UINT64_MAX - atom.logical_extent) {
                return {binding_status_code_v1::invalid_extent, atom_index};
            }
            for (std::uint64_t other = 0u; other < atom_index; ++other) {
                const auto &identity = port.atoms[other].atom_identity;
                if (identity.low == atom.atom_identity.low &&
                    identity.high == atom.atom_identity.high) {
                    return {binding_status_code_v1::duplicate_atom, atom_index};
                }
            }
        }
    }
    return {};
}

static_assert(std::is_trivially_copyable_v<atom_port_binding_v1>);
static_assert(std::is_trivially_copyable_v<multi_atom_port_binding_v1>);
static_assert(std::is_trivially_copyable_v<multi_atom_port_binding_list_v1>);

}  // namespace cellerator::execution::object_binding
