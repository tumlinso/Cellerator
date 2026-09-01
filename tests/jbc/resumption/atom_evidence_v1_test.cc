#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const unsigned char payload[] = {1u, 2u};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 0u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::atom_evidence;
    artifact.artifact_identity = {6u, 1u};
    artifact.context = expected;
    artifact.payload = payload;
    artifact.payload_bytes = sizeof(payload);
    artifact.content_hash[0] = 7u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_atom_evidence_v1(
        artifact, expected, &cursor));
    assert(cursor.stage == resume::lowering_stage_v1::atom_evidence);

    artifact.context.structure_epoch = 3u;
    const auto stale = resume::resume_from_atom_evidence_v1(
        artifact, expected, &cursor);
    assert(stale.code ==
        resume::compatibility_code_v1::structure_epoch_mismatch);
    assert(stale.earliest_compatible_stage ==
        resume::lowering_stage_v1::atom_evidence);
    assert(cursor.payload == nullptr);
}
