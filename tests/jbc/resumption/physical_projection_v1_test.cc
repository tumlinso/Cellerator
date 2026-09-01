#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const unsigned char cpe2[] = {'C', 'P', 'E', '2'};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 0u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::physical_projection;
    artifact.artifact_identity = {6u, 1u};
    artifact.context = expected;
    artifact.payload = cpe2;
    artifact.payload_bytes = sizeof(cpe2);
    artifact.content_hash[0] = 7u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_physical_projection_v1(
        artifact, expected, &cursor));

    artifact.context.target_identity = {8u, 1u};
    auto mismatch = resume::resume_from_physical_projection_v1(
        artifact, expected, &cursor);
    assert(mismatch.code == resume::compatibility_code_v1::target_mismatch);
    assert(mismatch.earliest_compatible_stage ==
        resume::lowering_stage_v1::physical_projection);

    artifact.context.target_identity = expected.target_identity;
    artifact.context.toolchain_identity = {9u, 1u};
    mismatch = resume::resume_from_physical_projection_v1(
        artifact, expected, &cursor);
    assert(mismatch.code == resume::compatibility_code_v1::toolchain_mismatch);
}
