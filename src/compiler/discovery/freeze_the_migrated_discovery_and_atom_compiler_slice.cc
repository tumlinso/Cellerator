#include <Cellerator/compiler/discovery/freeze_the_migrated_discovery_and_atom_compiler_slice_v1.hh>

#include <Cellerator/compiler/discovery/port_discovery_tests_and_evidence_fixtures_v1.hh>
#include <Cellerator/compiler/discovery/preserve_migration_provenance_in_source_and_artifacts_v1.hh>
#include <Cellerator/compiler/discovery/validate_no_compiler_discovery_remains_authoritative_in_v1.hh>

#include <utility>

namespace Cellerator::compiler::discovery {
namespace {

constexpr std::size_t k_provider_family_count = 7;

const discovery_atom_slice_receipt_v1 k_receipt{
    discovery_contract_version_v1,
    "CE-CCP1-I02-JBC-MIGRATION-MANIFEST@1",
    "CE-CCP1-I18-PROFILE-ENVIRONMENT@1",
    "CE-CCP1-I19-PLANNING-IR@1",
    "CE-CCP1-I20-DISCOVERY-ATOM@1",
    "tumlinso/CellShard",
    "b9749ad3e5146a04f847533d8c6f1a54146aed20",
    0,
    0,
    k_provider_family_count,
    true,
    true,
    true,
};

}  // namespace

certified_atom_status_v1 build_certified_atom_v1(
    const exact_proposal_certificate_v1& certificate,
    const certified_atom_request_v1& request,
    planning_atom_envelope_v1* output) noexcept {
    if (output == nullptr || !certificate.exact_cover ||
        certificate.canonical_edge_count == 0 ||
        certificate.covered_edge_count != certificate.canonical_edge_count ||
        !certificate.omitted_edge_identities.empty() ||
        !certificate.duplicate_receipts.empty()) {
        return certified_atom_status_v1::invalid_certificate;
    }
    try {
        planning_atom_envelope_v1 candidate{
            request.identities,
            atom_certification_state_v1::certified,
            {request.coverage_identity, certificate.canonical_edge_count, true},
            request.ports,
            request.planes,
            request.dependencies,
            request.lineage_identity,
            request.lineage_generation,
        };
        if (validate_atom_envelope_v1(candidate) !=
            atom_envelope_status_v1::success) {
            return certified_atom_status_v1::invalid_request;
        }
        *output = std::move(candidate);
    } catch (...) {
        return certified_atom_status_v1::invalid_request;
    }
    return certified_atom_status_v1::success;
}

const discovery_atom_slice_receipt_v1& get_discovery_atom_slice_receipt_v1() noexcept {
    static const discovery_atom_slice_receipt_v1 receipt = [] {
        auto value = k_receipt;
        std::size_t provenance_count = 0;
        (void)jbc_discovery_migration_manifest_v1(&provenance_count);
        value.migrated_source_record_count = provenance_count;
        value.migrated_fixture_source_file_count =
            migrated_fixture_source_file_count_v1();
        return value;
    }();
    return receipt;
}

bool valid_discovery_atom_slice_receipt_v1() noexcept {
    const auto& receipt = get_discovery_atom_slice_receipt_v1();
    std::size_t provenance_count = 0;
    const auto* provenance = jbc_discovery_migration_manifest_v1(&provenance_count);
    bool valid_provenance = provenance != nullptr && provenance_count != 0;
    for (std::size_t index = 0; index < provenance_count; ++index) {
        valid_provenance = valid_provenance &&
            valid_jbc_migration_provenance_v1(provenance[index]);
    }
    const auto& audit = cellshard_compiler_authority_audit_receipt_v1();
    const bool retirement_ready = cellshard_compatibility_retirement_ready_v1(
        {audit.production_authority_consumer_count,
         cellshard_compiler_compatibility_schema_v1, true});
    return receipt.contract_version == 1 &&
        receipt.migration_manifest_interface ==
            "CE-CCP1-I02-JBC-MIGRATION-MANIFEST@1" &&
        receipt.profile_environment_interface ==
            "CE-CCP1-I18-PROFILE-ENVIRONMENT@1" &&
        receipt.planning_ir_interface == "CE-CCP1-I19-PLANNING-IR@1" &&
        receipt.published_interface == "CE-CCP1-I20-DISCOVERY-ATOM@1" &&
        receipt.migrated_from_repository == "tumlinso/CellShard" &&
        receipt.migrated_from_commit.size() == 40 &&
        receipt.migrated_source_record_count == provenance_count &&
        receipt.migrated_fixture_source_file_count ==
            migrated_fixture_source_file_count_v1() &&
        receipt.provider_family_count == k_provider_family_count &&
        receipt.exact_certification_required &&
        receipt.execution_authorization_separate &&
        receipt.compatibility_retirement_ready == retirement_ready &&
        valid_provenance && valid_cellshard_compiler_authority_audit_v1();
}

}  // namespace Cellerator::compiler::discovery
