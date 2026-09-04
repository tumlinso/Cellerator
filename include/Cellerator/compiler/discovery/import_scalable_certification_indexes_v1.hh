#pragma once

#include <Cellerator/compiler/discovery/import_exact_rescan_and_proposal_certification_v1.hh>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace Cellerator::compiler::discovery {

struct indexed_proposal_membership_v1 {
    persistent_atom_identity_v1 edge_identity{};
    persistent_atom_identity_v1 proposal_identity{};
};

struct certification_index_metrics_v1 {
    std::uint64_t canonical_items = 0;
    std::uint64_t membership_items = 0;
    std::uint64_t exact_lookups = 0;
    std::uint64_t peak_workspace_bytes = 0;
};

struct scalable_certification_result_v1 {
    exact_proposal_certificate_v1 certificate;
    certification_index_metrics_v1 metrics;
};

enum class scalable_certification_status_v1 : std::uint8_t {
    success = 0,
    invalid_relation,
    invalid_canonical,
    invalid_membership,
    unknown_edge,
    work_bound_exceeded,
    workspace_bound_exceeded,
    allocation_failure,
};

[[nodiscard]] scalable_certification_status_v1
build_scalable_certification_index_v1(
    persistent_atom_identity_v1 relation_identity,
    std::uint64_t relation_generation,
    const std::vector<canonical_relation_edge_v1>& canonical_edges,
    const std::vector<indexed_proposal_membership_v1>& memberships,
    std::uint64_t maximum_work_items,
    std::uint64_t maximum_workspace_bytes,
    scalable_certification_result_v1* output) noexcept;

}  // namespace Cellerator::compiler::discovery
