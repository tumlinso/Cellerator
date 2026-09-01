#include <Cellerator/execution/atom_fragment/prepared_atom_fragment_v1.hh>

#include <cassert>

namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;
namespace program = execution::program;

program::program_status launch(const void *,
    const program::launch_binding_v2 &, void *) noexcept {
    return program::program_status::success;
}

int main() {
    fragment::atom_bound_candidate_v1 candidate{};
    candidate.candidate_id = 7u;
    candidate.atom_identity = {1u, 1u};
    candidate.requirement_identity = {2u, 1u};
    candidate.affordance_identity = {3u, 1u};
    const std::uint64_t dependencies[] = {0u};
    program::prepared_stage_v2 stages[2]{};
    stages[0].stable_stage_id = 1u;
    stages[0].candidate_id = candidate.candidate_id;
    stages[0].launch = launch;
    stages[0].binding_index = 0u;
    stages[0].required_workspace_bytes = 64u;
    stages[1].stable_stage_id = 2u;
    stages[1].candidate_id = candidate.candidate_id;
    stages[1].launch = launch;
    stages[1].first_dependency = 0u;
    stages[1].dependency_count = 1u;
    stages[1].binding_index = 2u;
    stages[1].required_workspace_bytes = 128u;
    program::prepared_program_v2 source{};
    source.stages = stages;
    source.stage_count = 2u;
    source.dependencies = dependencies;
    source.dependency_count = 1u;

    fragment::prepared_atom_fragment_v1 prepared{};
    assert(fragment::prepare_atom_fragment_v1(
        candidate, source, {10u, 1u}, {11u, 1u}, &prepared));
    assert(prepared.program == &source);
    assert(prepared.binding_count == 3u);
    assert(prepared.maximum_binding_workspace_bytes == 128u);

    stages[1].candidate_id = 8u;
    const auto foreign = fragment::prepare_atom_fragment_v1(
        candidate, source, {10u, 1u}, {11u, 1u}, &prepared);
    assert(foreign.code == fragment::prepared_atom_fragment_status_code_v1::
        foreign_candidate_stage);
    assert(foreign.index == 1u);
    assert(prepared.program == nullptr);

    stages[1].candidate_id = candidate.candidate_id;
    const auto invalid_order = fragment::prepare_atom_fragment_v1(
        candidate, source, {}, {11u, 1u}, &prepared);
    assert(invalid_order.code == fragment::
        prepared_atom_fragment_status_code_v1::invalid_order);
}
