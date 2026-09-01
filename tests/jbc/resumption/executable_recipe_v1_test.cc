#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const std::uint64_t recipe[] = {17u, 18u};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 9u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::executable_recipe;
    artifact.artifact_identity = {6u, 1u};
    artifact.executable_identity = {7u, 1u};
    artifact.context = expected;
    artifact.payload = recipe;
    artifact.payload_bytes = sizeof(recipe);
    artifact.content_hash[0] = 8u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_executable_recipe_v1(
        artifact, expected, {7u, 1u}, &cursor));
    assert(cursor.stage == resume::lowering_stage_v1::executable_recipe);

    const auto wrong_recipe = resume::resume_from_executable_recipe_v1(
        artifact, expected, {10u, 1u}, &cursor);
    assert(wrong_recipe.code ==
        resume::compatibility_code_v1::identity_mismatch);
    assert(cursor.payload == nullptr);
}
