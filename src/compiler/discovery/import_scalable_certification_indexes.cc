#include <Cellerator/compiler/discovery/import_scalable_certification_indexes_v1.hh>

#include <algorithm>
#include <limits>

namespace Cellerator::compiler::discovery {
namespace {

bool membership_less_v1(const indexed_proposal_membership_v1& left,
                        const indexed_proposal_membership_v1& right) noexcept {
    if (left.edge_identity != right.edge_identity) {
        return persistent_atom_identity_less_v1(left.edge_identity,
                                                right.edge_identity);
    }
    return persistent_atom_identity_less_v1(left.proposal_identity,
                                            right.proposal_identity);
}

}  // namespace

scalable_certification_status_v1 build_scalable_certification_index_v1(
    persistent_atom_identity_v1 relation_identity,
    std::uint64_t relation_generation,
    const std::vector<canonical_relation_edge_v1>& canonical_edges,
    const std::vector<indexed_proposal_membership_v1>& memberships,
    std::uint64_t maximum_work_items,
    std::uint64_t maximum_workspace_bytes,
    scalable_certification_result_v1* output) noexcept {
    if (output == nullptr || !valid_persistent_atom_identity_v1(relation_identity) ||
        relation_generation == 0 || maximum_work_items == 0 ||
        maximum_workspace_bytes == 0) {
        return scalable_certification_status_v1::invalid_relation;
    }
    if (canonical_edges.size() > maximum_work_items ||
        memberships.size() > maximum_work_items - canonical_edges.size()) {
        return scalable_certification_status_v1::work_bound_exceeded;
    }
    if (memberships.size() > std::numeric_limits<std::uint64_t>::max() /
                                 sizeof(indexed_proposal_membership_v1)) {
        return scalable_certification_status_v1::workspace_bound_exceeded;
    }
    const auto workspace = memberships.size() * sizeof(indexed_proposal_membership_v1);
    if (workspace > maximum_workspace_bytes) {
        return scalable_certification_status_v1::workspace_bound_exceeded;
    }
    for (std::size_t index = 0; index < canonical_edges.size(); ++index) {
        const auto& edge = canonical_edges[index];
        if (!valid_persistent_atom_identity_v1(edge.edge_identity) ||
            !valid_persistent_atom_identity_v1(edge.source_identity) ||
            !valid_persistent_atom_identity_v1(edge.destination_identity) ||
            (index != 0 && !persistent_atom_identity_less_v1(
                               canonical_edges[index - 1].edge_identity,
                               edge.edge_identity))) {
            return scalable_certification_status_v1::invalid_canonical;
        }
    }
    for (const auto& membership : memberships) {
        if (!valid_persistent_atom_identity_v1(membership.edge_identity) ||
            !valid_persistent_atom_identity_v1(membership.proposal_identity)) {
            return scalable_certification_status_v1::invalid_membership;
        }
    }

    try {
        auto ordered = memberships;
        std::sort(ordered.begin(), ordered.end(), membership_less_v1);
        scalable_certification_result_v1 result;
        result.certificate.relation_identity = relation_identity;
        result.certificate.relation_generation = relation_generation;
        result.certificate.canonical_edge_count = canonical_edges.size();
        result.metrics = {canonical_edges.size(), memberships.size(), 0, workspace};

        std::size_t membership_index = 0;
        for (const auto& edge : canonical_edges) {
            while (membership_index < ordered.size() &&
                   persistent_atom_identity_less_v1(
                       ordered[membership_index].edge_identity,
                       edge.edge_identity)) {
                return scalable_certification_status_v1::unknown_edge;
            }
            ++result.metrics.exact_lookups;
            if (membership_index == ordered.size() ||
                ordered[membership_index].edge_identity != edge.edge_identity) {
                result.certificate.omitted_edge_identities.push_back(
                    edge.edge_identity);
                continue;
            }
            const auto owner = ordered[membership_index].proposal_identity;
            result.certificate.canonical_owners.push_back(
                {edge.edge_identity, owner});
            ++result.certificate.covered_edge_count;
            ++membership_index;
            while (membership_index < ordered.size() &&
                   ordered[membership_index].edge_identity == edge.edge_identity) {
                result.certificate.duplicate_receipts.push_back(
                    {edge.edge_identity, owner,
                     ordered[membership_index].proposal_identity});
                ++membership_index;
            }
        }
        if (membership_index != ordered.size()) {
            return scalable_certification_status_v1::unknown_edge;
        }
        result.certificate.exact_cover =
            result.certificate.omitted_edge_identities.empty() &&
            result.certificate.duplicate_receipts.empty();
        *output = std::move(result);
        return scalable_certification_status_v1::success;
    } catch (...) {
        return scalable_certification_status_v1::allocation_failure;
    }
}

}  // namespace Cellerator::compiler::discovery
