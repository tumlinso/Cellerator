#include <Cellerator/compiler/discovery/import_exact_rescan_and_proposal_certification_v1.hh>

#include <algorithm>
#include <map>

namespace Cellerator::compiler::discovery {
namespace {

struct identity_less_v1 {
    bool operator()(persistent_atom_identity_v1 left,
                    persistent_atom_identity_v1 right) const noexcept {
        return persistent_atom_identity_less_v1(left, right);
    }
};

}  // namespace

exact_rescan_status_v1 certify_proposal_logical_coverage_v1(
    persistent_atom_identity_v1 relation_identity,
    std::uint64_t relation_generation,
    const std::vector<canonical_relation_edge_v1>& canonical_edges,
    const std::vector<proposal_logical_coverage_v1>& proposals,
    std::uint64_t maximum_scan_items,
    exact_proposal_certificate_v1* output) noexcept {
    if (output == nullptr || !valid_persistent_atom_identity_v1(relation_identity) ||
        relation_generation == 0 || maximum_scan_items == 0) {
        return exact_rescan_status_v1::invalid_relation;
    }
    if (canonical_edges.size() > maximum_scan_items) {
        return exact_rescan_status_v1::work_bound_exceeded;
    }
    try {
        std::map<persistent_atom_identity_v1, const canonical_relation_edge_v1*,
                 identity_less_v1>
            canonical;
        for (std::size_t index = 0; index < canonical_edges.size(); ++index) {
            const auto& edge = canonical_edges[index];
            if (!valid_persistent_atom_identity_v1(edge.edge_identity) ||
                !valid_persistent_atom_identity_v1(edge.source_identity) ||
                !valid_persistent_atom_identity_v1(edge.destination_identity)) {
                return exact_rescan_status_v1::invalid_canonical_edge;
            }
            if (index != 0 && !persistent_atom_identity_less_v1(
                                  canonical_edges[index - 1].edge_identity,
                                  edge.edge_identity)) {
                return exact_rescan_status_v1::unordered_canonical_edges;
            }
            canonical.emplace(edge.edge_identity, &edge);
        }
        std::uint64_t work_items = canonical_edges.size();
        std::map<persistent_atom_identity_v1, persistent_atom_identity_v1,
                 identity_less_v1>
            owners;
        std::vector<duplicate_edge_receipt_v1> duplicates;
        for (std::size_t proposal_index = 0; proposal_index < proposals.size();
             ++proposal_index) {
            const auto& proposal = proposals[proposal_index];
            if (!valid_persistent_atom_identity_v1(proposal.proposal_identity)) {
                return exact_rescan_status_v1::invalid_proposal;
            }
            if (proposal_index != 0 && !persistent_atom_identity_less_v1(
                                           proposals[proposal_index - 1]
                                               .proposal_identity,
                                           proposal.proposal_identity)) {
                return exact_rescan_status_v1::unordered_proposals;
            }
            for (std::size_t edge_index = 0;
                 edge_index < proposal.edge_identities.size(); ++edge_index) {
                const auto edge_identity = proposal.edge_identities[edge_index];
                if (edge_index != 0 && !persistent_atom_identity_less_v1(
                                           proposal.edge_identities[edge_index - 1],
                                           edge_identity)) {
                    return exact_rescan_status_v1::unordered_proposal_edges;
                }
                if (work_items == maximum_scan_items) {
                    return exact_rescan_status_v1::work_bound_exceeded;
                }
                ++work_items;
                if (canonical.find(edge_identity) == canonical.end()) {
                    return exact_rescan_status_v1::unknown_edge;
                }
                const auto inserted = owners.emplace(
                    edge_identity, proposal.proposal_identity);
                if (!inserted.second) {
                    duplicates.push_back({edge_identity, inserted.first->second,
                                          proposal.proposal_identity});
                }
            }
        }

        exact_proposal_certificate_v1 certificate;
        certificate.relation_identity = relation_identity;
        certificate.relation_generation = relation_generation;
        certificate.canonical_edge_count = canonical_edges.size();
        certificate.covered_edge_count = owners.size();
        for (const auto& edge : canonical_edges) {
            const auto owner = owners.find(edge.edge_identity);
            if (owner == owners.end()) {
                certificate.omitted_edge_identities.push_back(edge.edge_identity);
            } else {
                certificate.canonical_owners.push_back(
                    {edge.edge_identity, owner->second});
            }
        }
        certificate.duplicate_receipts = std::move(duplicates);
        certificate.exact_cover =
            certificate.omitted_edge_identities.empty() &&
            certificate.duplicate_receipts.empty();
        *output = std::move(certificate);
        return exact_rescan_status_v1::success;
    } catch (...) {
        return exact_rescan_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
