#pragma once

#include <Cellerator/compiler/discovery/import_atom_envelope_and_typed_ports_v1.hh>

#include <cstdint>
#include <string_view>

namespace Cellerator::compiler::discovery {

inline constexpr std::uint32_t cellshard_compiler_compatibility_schema_v1 = 1;

struct cellshard_compatibility_retirement_gate_v1 {
    std::uint64_t preserved_consumer_count = 0;
    std::uint32_t minimum_replacement_schema = 0;
    bool replacement_interface_frozen = false;
};

struct cellshard_compatibility_manifest_v1 {
    std::string_view deprecated_surface;
    std::string_view replacement_surface;
    std::string_view retirement_todo;
};

[[nodiscard]] constexpr bool cellshard_compatibility_retirement_ready_v1(
    cellshard_compatibility_retirement_gate_v1 gate) noexcept {
    return gate.preserved_consumer_count == 0 &&
        gate.minimum_replacement_schema >= 1 &&
        gate.replacement_interface_frozen;
}

[[nodiscard]] const cellshard_compatibility_manifest_v1&
cellshard_compiler_compatibility_manifest_v1() noexcept;

}  // namespace Cellerator::compiler::discovery

namespace cellshard::compiler::compatibility_v1 {

using atom_persistent_identity_v1 =
    Cellerator::compiler::discovery::persistent_atom_identity_v1;
using atom_identity_contract_v1 =
    Cellerator::compiler::discovery::atom_identity_contract_v1;
using atom_typed_port_v1 =
    Cellerator::compiler::discovery::atom_typed_port_v1;
using common_atom_v1 =
    Cellerator::compiler::discovery::planning_atom_envelope_v1;

[[deprecated("use Cellerator::compiler::discovery::persistent_atom_identity_v1")]]
[[nodiscard]] constexpr atom_persistent_identity_v1 make_atom_identity_v1(
    std::uint64_t producer_namespace,
    std::uint64_t local_identity) noexcept {
    return {producer_namespace, local_identity};
}

[[deprecated("use Cellerator compiler discovery contracts directly")]]
[[nodiscard]] inline bool valid_common_atom_v1(const common_atom_v1& atom) noexcept {
    return Cellerator::compiler::discovery::validate_atom_envelope_v1(atom) ==
        Cellerator::compiler::discovery::atom_envelope_status_v1::success;
}

}  // namespace cellshard::compiler::compatibility_v1
