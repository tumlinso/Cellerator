#include <Cellerator/compiler/discovery/preserve_migration_provenance_in_source_and_artifacts_v1.hh>

#include <array>

namespace Cellerator::compiler::discovery {
namespace {

constexpr std::string_view repository = "tumlinso/CellShard";
constexpr std::string_view commit =
    "b9749ad3e5146a04f847533d8c6f1a54146aed20";

constexpr std::array<jbc_migration_provenance_v1, 13> manifest{{
    {"import_the_common_jbc_atom_identity_adapters", repository, commit,
     "include/CellShard/compiler/atom", "CS-JBC-A01;CS-JBC-A05;CS-JBC-A13",
     "CE-CCP1-E02-001"},
    {"import_the_overlapping_evidence_atlas_core", repository, commit,
     "include/CellShard/compiler/evidence", "CS-JBC-E01..CS-JBC-E16",
     "CE-CCP1-E02-002"},
    {"import_support_signature_discovery", repository, commit,
     "include/CellShard/compiler/discovery/support_signature",
     "CS-JBC-SS01..CS-JBC-SS10", "CE-CCP1-E02-003"},
    {"import_co_support_and_overlap_discovery", repository, commit,
     "include/CellShard/compiler/discovery/co_support;include/CellShard/compiler/discovery/overlap",
     "CS-JBC-CO01..CS-JBC-CO10;CS-JBC-OC01..CS-JBC-OC06",
     "CE-CCP1-E02-004"},
    {"import_relation_motif_and_operation_trace_discovery", repository, commit,
     "include/CellShard/compiler/discovery/motif;include/CellShard/compiler/discovery/operation_trace",
     "CS-JBC-MF01..CS-JBC-MF08;CS-JBC-OT01..CS-JBC-OT08",
     "CE-CCP1-E02-005"},
    {"import_trajectory_and_lineage_pattern_discovery", repository, commit,
     "include/CellShard/compiler/discovery/trajectory",
     "CS-JBC-TR01..CS-JBC-TR12", "CE-CCP1-E02-006"},
    {"import_multimodal_and_identity_spine_discovery", repository, commit,
     "include/CellShard/compiler/discovery/multimodal",
     "CS-JBC-MM01..CS-JBC-MM10", "CE-CCP1-E02-007"},
    {"import_factor_bicluster_and_signature_proposal_strategie", repository,
     commit,
     "include/CellShard/compiler/discovery/factor_topic;include/CellShard/compiler/discovery/bicluster;include/CellShard/compiler/discovery/support_signature",
     "CS-JBC-FT01..CS-JBC-FT06;CS-JBC-BC01..CS-JBC-BC08;CS-JBC-SS01..CS-JBC-SS10",
     "CE-CCP1-E02-008"},
    {"import_exact_rescan_and_proposal_certification", repository, commit,
     "include/CellShard/compiler/discovery;include/CellShard/compiler/certification",
     "CS-JBC-SS06;CS-JBC-FT04;CS-JBC-MM08", "CE-CCP1-E02-009"},
    {"import_atom_envelope_and_typed_ports", repository, commit,
     "include/CellShard/compiler/atom", "CS-JBC-A01..CS-JBC-A16",
     "CE-CCP1-E02-010"},
    {"import_atom_plane_separation", repository, commit,
     "include/CellShard/compiler/atom", "CS-JBC-A06..CS-JBC-A13",
     "CE-CCP1-E02-011"},
    {"import_atom_requirement_affordance_matching", repository, commit,
     "include/CellShard/compiler/atom;include/Cellerator/execution/joint_compiler",
     "CS-JBC-A14;CE-JBC-F01..CE-JBC-F08", "CE-CCP1-E02-012"},
    {"import_scalable_certification_indexes", repository, commit,
     "include/CellShard/compiler/discovery;include/CellShard/compiler/certification",
     "CS-JBC-SS06;CS-JBC-CO09;CS-JBC-FT04", "CE-CCP1-E02-013"},
}};

}  // namespace

const jbc_migration_provenance_v1* jbc_discovery_migration_manifest_v1(
    std::size_t* count) noexcept {
    if (count != nullptr) {
        *count = manifest.size();
    }
    return manifest.data();
}

const jbc_migration_provenance_v1* find_jbc_migration_provenance_v1(
    std::string_view destination_path) noexcept {
    for (const auto& record : manifest) {
        if (destination_path.find(record.destination_stem) !=
            std::string_view::npos) {
            return &record;
        }
    }
    return nullptr;
}

bool valid_jbc_migration_provenance_v1(
    const jbc_migration_provenance_v1& record) noexcept {
    return !record.destination_stem.empty() &&
        record.migrated_from_repository == repository &&
        record.migrated_from_commit.size() == 40 &&
        !record.migrated_from_path.empty() &&
        !record.migrated_from_todos.empty() &&
        record.cellerator_todo.substr(0, 12) == "CE-CCP1-E02-";
}

}  // namespace Cellerator::compiler::discovery
