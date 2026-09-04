#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace Cellerator::compiler::migration {

enum class certification_responsibility_v1 : std::uint32_t {
    canonical_identity = 1,
    exact_coverage,
    ownership,
    recovery,
    dependency_closure,
};

struct certification_migration_v1 {
    std::string_view source_header;
    std::string_view planning_ir_contract;
    certification_responsibility_v1 responsibility;
};

inline constexpr std::array<certification_migration_v1, 16>
    exact_certification_map_v1{{
        {"atom_certification_v1.hh", "certified_atom_v1", certification_responsibility_v1::exact_coverage},
        {"canonical_domain_v1.hh", "canonical_domain_v1", certification_responsibility_v1::canonical_identity},
        {"contribution_owner_v1.hh", "contribution_owner_v1", certification_responsibility_v1::ownership},
        {"dependency_closure_v1.hh", "dependency_closure_v1", certification_responsibility_v1::dependency_closure},
        {"duplicate_detection_v1.hh", "duplicate_omission_proof_v1", certification_responsibility_v1::exact_coverage},
        {"entity_coverage_v1.hh", "entity_coverage_v1", certification_responsibility_v1::exact_coverage},
        {"exact_atom_certificate_v1.hh", "exact_atom_certificate_v1", certification_responsibility_v1::exact_coverage},
        {"independent_verifier_v1.hh", "independent_verifier_v1", certification_responsibility_v1::exact_coverage},
        {"local_identity_map_v1.hh", "canonical_recovery_map_v1", certification_responsibility_v1::recovery},
        {"multimodal_mapping_v1.hh", "multimodal_identity_map_v1", certification_responsibility_v1::recovery},
        {"partial_result_compatibility_v1.hh", "partial_algebra_proof_v1", certification_responsibility_v1::exact_coverage},
        {"physical_replica_v1.hh", "physical_replica_owner_v1", certification_responsibility_v1::ownership},
        {"read_only_halo_v1.hh", "read_only_halo_v1", certification_responsibility_v1::ownership},
        {"relation_edge_coverage_v1.hh", "relation_edge_coverage_v1", certification_responsibility_v1::exact_coverage},
        {"residual_coverage_v1.hh", "residual_coverage_v1", certification_responsibility_v1::exact_coverage},
        {"trajectory_lineage_v1.hh", "trajectory_lineage_map_v1", certification_responsibility_v1::recovery},
    }};

struct exact_certification_prerequisites_v1 {
    bool canonical_identities = false;
    bool sorted_unique_members = false;
    bool complete_relation_edges = false;
    bool contribution_owners = false;
    bool residual_accounted = false;
    bool inverse_recovery = false;
    bool dependency_closure = false;
};

[[nodiscard]] constexpr bool may_certify_execution_v1(
    exact_certification_prerequisites_v1 value) noexcept {
    return value.canonical_identities && value.sorted_unique_members
        && value.complete_relation_edges && value.contribution_owners
        && value.residual_accounted && value.inverse_recovery
        && value.dependency_closure;
}

} // namespace Cellerator::compiler::migration
