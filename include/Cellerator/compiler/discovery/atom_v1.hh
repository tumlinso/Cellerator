#pragma once

#include <Cellerator/compiler/discovery/create_temporary_cellshard_compiler_compatibility_adapte_v1.hh>
#include <Cellerator/compiler/discovery/import_atom_envelope_and_typed_ports_v1.hh>
#include <Cellerator/compiler/discovery/import_atom_plane_separation_v1.hh>
#include <Cellerator/compiler/discovery/import_atom_requirement_affordance_matching_v1.hh>
#include <Cellerator/compiler/discovery/import_scalable_certification_indexes_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

inline constexpr std::uint32_t atom_compiler_contract_version_v1 = 1;

struct certified_atom_request_v1 {
    atom_identity_contract_v1 identities{};
    persistent_atom_identity_v1 coverage_identity{};
    std::vector<atom_typed_port_v1> ports;
    std::vector<atom_plane_binding_v1> planes;
    std::vector<atom_dependency_v1> dependencies;
    persistent_atom_identity_v1 lineage_identity{};
    std::uint64_t lineage_generation = 0;
};

enum class certified_atom_status_v1 : std::uint8_t {
    success = 0,
    invalid_certificate,
    invalid_request,
};

// Exact certification is a compiler fact, not runtime authorization. This
// adapter only produces certified Planning IR; later planning/lowering stages
// retain responsibility for selecting an executable realization.
[[nodiscard]] certified_atom_status_v1 build_certified_atom_v1(
    const exact_proposal_certificate_v1& certificate,
    const certified_atom_request_v1& request,
    planning_atom_envelope_v1* output) noexcept;

}  // namespace Cellerator::compiler::discovery
