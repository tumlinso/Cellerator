#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const unsigned char basis[] = {4u, 5u, 6u};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 0u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::semantic_atom;
    artifact.artifact_identity = {6u, 1u};
    artifact.context = expected;
    artifact.payload = basis;
    artifact.payload_bytes = sizeof(basis);
    artifact.content_hash[0] = 7u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_semantic_atom_v1(
        artifact, expected, &cursor));
    assert(cursor.stage == resume::lowering_stage_v1::semantic_atom);

    artifact.context.order_identity = {8u, 1u};
    const auto wrong_order = resume::resume_from_semantic_atom_v1(
        artifact, expected, &cursor);
    assert(wrong_order.code == resume::compatibility_code_v1::order_mismatch);
    assert(wrong_order.earliest_compatible_stage ==
        resume::lowering_stage_v1::semantic_atom);
}
