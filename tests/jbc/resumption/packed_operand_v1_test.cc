#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const float values[] = {1.0f, 2.0f};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 9u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::packed_operand;
    artifact.artifact_identity = {6u, 1u};
    artifact.context = expected;
    artifact.payload = values;
    artifact.payload_bytes = sizeof(values);
    artifact.content_hash[0] = 7u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_packed_operand_v1(
        artifact, expected, &cursor));
    assert(cursor.context.value_generation == 9u);

    artifact.context.value_generation = 8u;
    const auto stale = resume::resume_from_packed_operand_v1(
        artifact, expected, &cursor);
    assert(stale.code ==
        resume::compatibility_code_v1::value_generation_stale);
    assert(stale.earliest_compatible_stage ==
        resume::lowering_stage_v1::packed_operand);
    assert(stale.detail == 8u);
}
