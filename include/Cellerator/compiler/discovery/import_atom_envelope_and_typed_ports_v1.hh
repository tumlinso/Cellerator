#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

enum class atom_certification_state_v1 : std::uint8_t {
    candidate = 1,
    certified,
};

enum class atom_port_direction_v1 : std::uint8_t {
    input = 1,
    output,
    inout,
};

enum class atom_dependency_effect_v1 : std::uint8_t {
    cost_only = 1,
    values,
    structure,
    correctness,
};

struct atom_typed_port_v1 {
    persistent_atom_identity_v1 port_identity{};
    persistent_atom_identity_v1 domain_identity{};
    persistent_atom_identity_v1 axis_identity{};
    persistent_atom_identity_v1 order_identity{};
    persistent_atom_identity_v1 plane_kind_identity{};
    persistent_atom_identity_v1 storage_type_identity{};
    persistent_atom_identity_v1 logical_type_identity{};
    persistent_atom_identity_v1 accumulation_type_identity{};
    std::uint64_t generation = 0;
    atom_port_direction_v1 direction = atom_port_direction_v1::input;
};

struct atom_plane_binding_v1 {
    persistent_atom_identity_v1 plane_kind_identity{};
    persistent_atom_identity_v1 plane_identity{};
    std::uint64_t generation = 0;
};

struct atom_dependency_v1 {
    persistent_atom_identity_v1 atom_identity{};
    std::uint64_t required_generation = 0;
    atom_dependency_effect_v1 effect = atom_dependency_effect_v1::correctness;
};

struct atom_exact_coverage_binding_v1 {
    persistent_atom_identity_v1 coverage_identity{};
    std::uint64_t logical_member_count = 0;
    bool certified_exact = false;
};

struct planning_atom_envelope_v1 {
    atom_identity_contract_v1 identities{};
    atom_certification_state_v1 certification =
        atom_certification_state_v1::candidate;
    atom_exact_coverage_binding_v1 exact_coverage{};
    std::vector<atom_typed_port_v1> ports;
    std::vector<atom_plane_binding_v1> planes;
    std::vector<atom_dependency_v1> dependencies;
    persistent_atom_identity_v1 lineage_identity{};
    std::uint64_t lineage_generation = 0;
};

enum class atom_envelope_status_v1 : std::uint8_t {
    success = 0,
    invalid_identity,
    invalid_certification,
    invalid_coverage,
    empty_ports,
    invalid_port,
    unordered_ports,
    invalid_plane,
    unordered_planes,
    invalid_dependency,
    unordered_dependencies,
    invalid_lineage,
    allocation_failure,
};

[[nodiscard]] atom_envelope_status_v1 validate_atom_envelope_v1(
    const planning_atom_envelope_v1& envelope) noexcept;

[[nodiscard]] atom_envelope_status_v1 clone_atom_envelope_v1(
    const planning_atom_envelope_v1& source,
    planning_atom_envelope_v1* output) noexcept;

}  // namespace Cellerator::compiler::discovery
