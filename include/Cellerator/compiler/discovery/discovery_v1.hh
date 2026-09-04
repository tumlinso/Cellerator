#pragma once

#include <Cellerator/compiler/discovery/import_co_support_and_overlap_discovery_v1.hh>
#include <Cellerator/compiler/discovery/import_exact_rescan_and_proposal_certification_v1.hh>
#include <Cellerator/compiler/discovery/import_factor_bicluster_and_signature_proposal_strategie_v1.hh>
#include <Cellerator/compiler/discovery/import_multimodal_and_identity_spine_discovery_v1.hh>
#include <Cellerator/compiler/discovery/import_relation_motif_and_operation_trace_discovery_v1.hh>
#include <Cellerator/compiler/discovery/import_support_signature_discovery_v1.hh>
#include <Cellerator/compiler/discovery/import_trajectory_and_lineage_pattern_discovery_v1.hh>
#include <Cellerator/compiler/profile/profile_environment_v1.hh>

#include <cstdint>

namespace Cellerator::compiler::discovery {

inline constexpr std::uint32_t discovery_contract_version_v1 = 1;

enum class profile_discovery_status_v1 : std::uint8_t {
    ready = 0,
    unsupported_profile_contract,
    unsupported_structure_contract,
    missing_profile_identity,
    missing_relation_identity,
    missing_structure_epoch,
    unavailable_support,
};

[[nodiscard]] constexpr persistent_atom_identity_v1 profile_identity_to_atom_v1(
    cellerator::compiler::profile::v1::profile_identity_v1 identity) noexcept {
    return {identity.high, identity.low};
}

[[nodiscard]] constexpr profile_discovery_status_v1 validate_profile_for_discovery_v1(
    const cellerator::compiler::profile::v1::profile_compile_state_v1& profile) noexcept {
    using namespace cellerator::compiler::profile::v1;
    if (profile.contract_version != profile_environment_contract_version_v1) {
        return profile_discovery_status_v1::unsupported_profile_contract;
    }
    if (profile.structure.schema_version != structural_profile_evidence_schema_version_v1) {
        return profile_discovery_status_v1::unsupported_structure_contract;
    }
    if (profile.state.low == 0 || profile.state.high == 0) {
        return profile_discovery_status_v1::missing_profile_identity;
    }
    if (profile.structure.relation.low == 0 || profile.structure.relation.high == 0) {
        return profile_discovery_status_v1::missing_relation_identity;
    }
    if (profile.structure.structure_epoch == 0) {
        return profile_discovery_status_v1::missing_structure_epoch;
    }
    if (profile.structure.support_count == 0) {
        return profile_discovery_status_v1::unavailable_support;
    }
    return profile_discovery_status_v1::ready;
}

}  // namespace Cellerator::compiler::discovery
