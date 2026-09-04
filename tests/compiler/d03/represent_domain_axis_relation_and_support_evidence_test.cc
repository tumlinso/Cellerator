#include <Cellerator/compiler/profile/represent_domain_axis_relation_and_support_evidence_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator;
    const std::uint64_t offsets[] = {0u, 1u, 3u, 3u, 6u};
    const std::uint32_t sources[] = {0u, 1u, 2u, 0u, 2u, 3u};
    geometry::support_relation_view_v1 relation{};
    relation.relation_identity = 10u;
    relation.structure_identity = 11u;
    relation.structure_epoch = 2u;
    relation.source_axis_identity = 12u;
    relation.destination_axis_identity = 13u;
    relation.source_count = 4u;
    relation.destination_count = 4u;
    relation.edge_count = 6u;
    relation.destination_offsets = offsets;
    relation.source_ids = sources;
    compiler::profile::v1::structural_profile_evidence_v1 evidence{};
    assert(compiler::profile::v1::derive_exact_structural_profile_evidence_v1(
               relation, {20u, 21u}, 1.0, &evidence)
           == compiler::profile::v1::structural_profile_evidence_status_v1::ok);
    assert(evidence.support_count == 6u);
    assert(evidence.nonempty_destination_count == 3u);
    assert(evidence.degree.minimum == 0.0 && evidence.degree.maximum == 3.0);
    assert(evidence.degree.mean == 1.5 && evidence.occupancy.mean == 0.375);
    assert(compiler::profile::v1::validate_structural_profile_evidence_v1(evidence)
           == compiler::profile::v1::structural_profile_evidence_status_v1::ok);
}
