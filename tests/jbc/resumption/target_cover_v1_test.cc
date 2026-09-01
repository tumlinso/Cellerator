#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const unsigned char cover[] = {1u, 1u, 0u};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 0u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::target_cover;
    artifact.artifact_identity = {6u, 1u};
    artifact.cover_identity = {7u, 1u};
    artifact.context = expected;
    artifact.payload = cover;
    artifact.payload_bytes = sizeof(cover);
    artifact.content_hash[0] = 8u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_target_cover_v1(
        artifact, expected, {7u, 1u}, &cursor));
    assert(cursor.stage == resume::lowering_stage_v1::target_cover);

    const auto wrong_cover = resume::resume_from_target_cover_v1(
        artifact, expected, {9u, 1u}, &cursor);
    assert(wrong_cover.code ==
        resume::compatibility_code_v1::identity_mismatch);
    assert(wrong_cover.earliest_compatible_stage ==
        resume::lowering_stage_v1::canonical_source);
}
