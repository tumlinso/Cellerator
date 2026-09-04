#pragma once

#include <array>
#include <cstdint>
#include <string_view>
#include <type_traits>

namespace Cellerator::compiler::migration {

enum class evidence_migration_disposition_v1 : std::uint32_t {
    rehome_semantics = 1,
    split_storage_adapter,
    compatibility_adapter,
};

struct evidence_ownership_mapping_v1 {
    std::string_view source_prefix;
    std::string_view destination_namespace;
    evidence_migration_disposition_v1 disposition;
};

// These rows cover every evidence-producing branch below CellShard's
// compiler/evidence and compiler/discovery trees at the pinned source commit.
// They define target ownership, not forwarding aliases to CellShard types.
inline constexpr std::array<evidence_ownership_mapping_v1, 13>
    evidence_ownership_map_v1{{
        {"include/CellShard/compiler/evidence/",
         "Cellerator::compiler::profile::evidence",
         evidence_migration_disposition_v1::split_storage_adapter},
        {"include/CellShard/compiler/discovery/bicluster/",
         "Cellerator::compiler::profile::discovery::bicluster",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/co_support/",
         "Cellerator::compiler::profile::discovery::co_support",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/factor_topic/",
         "Cellerator::compiler::profile::discovery::factor_topic",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/motif/",
         "Cellerator::compiler::profile::discovery::motif",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/multimodal/",
         "Cellerator::compiler::profile::discovery::multimodal",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/operation_trace/",
         "Cellerator::compiler::profile::discovery::operation_trace",
         evidence_migration_disposition_v1::compatibility_adapter},
        {"include/CellShard/compiler/discovery/overlap/",
         "Cellerator::compiler::profile::discovery::overlap",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/sequence_compat/",
         "Cellerator::compiler::profile::discovery::sequence",
         evidence_migration_disposition_v1::compatibility_adapter},
        {"include/CellShard/compiler/discovery/support_signature/",
         "Cellerator::compiler::profile::discovery::support_signature",
         evidence_migration_disposition_v1::rehome_semantics},
        {"include/CellShard/compiler/discovery/trajectory/",
         "Cellerator::compiler::profile::discovery::trajectory",
         evidence_migration_disposition_v1::rehome_semantics},
        {"src/compiler/evidence/",
         "Cellerator::compiler::profile::evidence",
         evidence_migration_disposition_v1::split_storage_adapter},
        {"src/compiler/discovery/",
         "Cellerator::compiler::profile::discovery",
         evidence_migration_disposition_v1::rehome_semantics},
    }};

// Proposal identity is composed only from explicit semantic identities and
// generations. Concrete pointers, byte offsets, file paths, chunk addresses,
// allocation handles and payload digests are deliberately absent.
struct proposal_evidence_identity_v1 {
    std::uint64_t producer_namespace = 0;
    std::uint64_t local_identity = 0;
    std::uint64_t observation_generation = 0;
    std::uint64_t dataset_identity = 0;
    std::uint64_t relation_identity = 0;
};

[[nodiscard]] constexpr bool valid_proposal_evidence_identity_v1(
    proposal_evidence_identity_v1 identity) noexcept {
    return identity.producer_namespace != 0 && identity.local_identity != 0
        && identity.observation_generation != 0
        && identity.dataset_identity != 0 && identity.relation_identity != 0;
}

// Discovery evidence can rank or reject a proposal. Only the independent
// exact-certification subsystem may turn one into executable coverage.
[[nodiscard]] constexpr bool authorizes_execution(
    proposal_evidence_identity_v1) noexcept {
    return false;
}

static_assert(std::is_standard_layout_v<proposal_evidence_identity_v1>);
static_assert(std::is_trivially_copyable_v<proposal_evidence_identity_v1>);
static_assert(sizeof(proposal_evidence_identity_v1) == 5 * sizeof(std::uint64_t));

} // namespace Cellerator::compiler::migration
