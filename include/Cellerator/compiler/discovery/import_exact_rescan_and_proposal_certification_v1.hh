#pragma once

#include <Cellerator/compiler/discovery/import_the_common_jbc_atom_identity_adapters_v1.hh>

#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

struct canonical_relation_edge_v1 {
    persistent_atom_identity_v1 edge_identity{};
    persistent_atom_identity_v1 source_identity{};
    persistent_atom_identity_v1 destination_identity{};
};

struct proposal_logical_coverage_v1 {
    persistent_atom_identity_v1 proposal_identity{};
    std::vector<persistent_atom_identity_v1> edge_identities;
};

struct canonical_edge_owner_v1 {
    persistent_atom_identity_v1 edge_identity{};
    persistent_atom_identity_v1 proposal_identity{};
};

struct duplicate_edge_receipt_v1 {
    persistent_atom_identity_v1 edge_identity{};
    persistent_atom_identity_v1 first_proposal_identity{};
    persistent_atom_identity_v1 duplicate_proposal_identity{};
};

struct exact_proposal_certificate_v1 {
    persistent_atom_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
    std::uint64_t canonical_edge_count = 0;
    std::uint64_t covered_edge_count = 0;
    std::vector<canonical_edge_owner_v1> canonical_owners;
    std::vector<persistent_atom_identity_v1> omitted_edge_identities;
    std::vector<duplicate_edge_receipt_v1> duplicate_receipts;
    bool exact_cover = false;
};

enum class exact_rescan_status_v1 : std::uint8_t {
    success = 0,
    invalid_relation,
    invalid_canonical_edge,
    unordered_canonical_edges,
    invalid_proposal,
    unordered_proposals,
    unordered_proposal_edges,
    unknown_edge,
    work_bound_exceeded,
    allocation_failure,
};

[[nodiscard]] exact_rescan_status_v1 certify_proposal_logical_coverage_v1(
    persistent_atom_identity_v1 relation_identity,
    std::uint64_t relation_generation,
    const std::vector<canonical_relation_edge_v1>& canonical_edges,
    const std::vector<proposal_logical_coverage_v1>& proposals,
    std::uint64_t maximum_scan_items,
    exact_proposal_certificate_v1* output) noexcept;

[[nodiscard]] constexpr bool authorizes_execution(
    const exact_proposal_certificate_v1&) noexcept {
    return false;
}

}  // namespace Cellerator::compiler::discovery
