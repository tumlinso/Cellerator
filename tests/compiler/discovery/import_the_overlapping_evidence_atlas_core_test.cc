#include <Cellerator/compiler/discovery/import_the_overlapping_evidence_atlas_core_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::discovery;

int main() {
    proposal_evidence_record_v1 first;
    first.evidence_identity = {1, 1};
    first.subject_atom_identity = {1, 10};
    first.provenance_identity = {2, 20};
    first.observation_generation = 3;
    first.approximate_members = {{4, 100}, {4, 200}};
    first.confidence_numerator = 8;
    first.confidence_denominator = 10;
    first.stable_resamples = 7;
    first.total_resamples = 10;
    first.exact_visited = 1000;
    first.exact_assigned = 800;
    first.exact_rescan = exact_rescan_status_v1::complete;

    auto second = first;
    second.evidence_identity = {1, 2};
    second.subject_atom_identity = {1, 11};
    second.approximate_members = {{4, 200}, {4, 300}};
    second.confidence_numerator = 2;
    second.negative_reason = negative_evidence_reason_v1::unstable;
    second.exact_rescan = exact_rescan_status_v1::incomplete;

    const overlapping_evidence_atlas_v1 atlas{{9, 99}, 7, {first, second}};
    assert(validate_overlapping_evidence_atlas_v1(atlas) ==
           evidence_atlas_status_v1::success);

    const auto image = serialize_overlapping_evidence_atlas_v1(atlas);
    assert(image);
    const auto restored = deserialize_overlapping_evidence_atlas_v1(*image);
    assert(restored && equivalent_evidence_atlas_v1(atlas, *restored));
    assert(serialize_overlapping_evidence_atlas_v1(*restored) == image);
    assert(restored->proposals[0].approximate_members[1] ==
           restored->proposals[1].approximate_members[0]);

    auto corrupt = *image;
    corrupt[20] ^= 1;
    evidence_atlas_status_v1 status{};
    assert(!deserialize_overlapping_evidence_atlas_v1(corrupt, &status));
    assert(status == evidence_atlas_status_v1::checksum_mismatch);

    auto invalid = atlas;
    invalid.proposals[0].approximate_members.push_back({4, 100});
    assert(validate_overlapping_evidence_atlas_v1(invalid) ==
           evidence_atlas_status_v1::unordered_or_duplicate_member);
}
