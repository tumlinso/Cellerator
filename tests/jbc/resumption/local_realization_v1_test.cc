#include <Cellerator/execution/lowering_resumption/resumption_v1.hh>

#include <cassert>

namespace resume = cellerator::execution::lowering_resumption;

int main() {
    const std::uint32_t local_map[] = {2u, 0u, 1u};
    const resume::lowering_identity_context_v1 expected{
        {1u, 1u}, 2u, {3u, 1u}, {4u, 1u}, {5u, 1u}, 9u};
    resume::lowering_artifact_v1 artifact{};
    artifact.stage = resume::lowering_stage_v1::local_realization;
    artifact.artifact_identity = {6u, 1u};
    artifact.topology_identity = {7u, 1u};
    artifact.topology_epoch = 11u;
    artifact.partition_identity = {8u, 1u};
    artifact.local_extent = 3u;
    artifact.context = expected;
    artifact.payload = local_map;
    artifact.payload_bytes = sizeof(local_map);
    artifact.content_hash[0] = 10u;
    resume::resumption_cursor_v1 cursor{};
    assert(resume::resume_from_local_realization_v1(artifact, expected,
        {7u, 1u}, 11u, {8u, 1u}, &cursor));

    artifact.topology_epoch = 12u;
    const auto stale = resume::resume_from_local_realization_v1(artifact,
        expected, {7u, 1u}, 11u, {8u, 1u}, &cursor);
    assert(stale.code ==
        resume::compatibility_code_v1::structure_epoch_mismatch);
    assert(stale.detail == 12u);
    assert(cursor.payload == nullptr);
}
