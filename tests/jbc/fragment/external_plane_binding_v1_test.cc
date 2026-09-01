#include <Cellerator/execution/atom_fragment/external_plane_binding_v1.hh>

#include <cassert>

namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;
namespace joint = execution::joint_compiler;
namespace program = execution::program;

program::program_status launch(const void *,
    const program::launch_binding_v2 &, void *) noexcept {
    return program::program_status::success;
}

int main() {
    const joint::persistent_identity_v1 atom{1u, 1u};
    const joint::persistent_identity_v1 plane{2u, 1u};
    program::prepared_stage_v2 stage{};
    stage.stable_stage_id = 1u;
    stage.candidate_id = 7u;
    stage.launch = launch;
    program::prepared_program_v2 source{};
    source.stages = &stage;
    source.stage_count = 1u;
    fragment::atom_bound_candidate_v1 candidate{};
    candidate.candidate_id = 7u;
    candidate.atom_identity = atom;
    candidate.requirement_identity = {3u, 1u};
    candidate.affordance_identity = {4u, 1u};
    fragment::prepared_atom_fragment_v1 prepared{};
    assert(fragment::prepare_atom_fragment_v1(
        candidate, source, {10u, 1u}, {11u, 1u}, &prepared));

    const joint::persistent_identity_v1 species[] = {{5u, 1u}};
    const joint::persistent_identity_v1 planes[] = {plane};
    joint::atom_requirement_v1 requirement{};
    requirement.requirement_identity = candidate.requirement_identity;
    requirement.exact_coverage_identity = {6u, 1u};
    requirement.accepted_atom_species = species;
    requirement.accepted_atom_species_count = 1u;
    requirement.required_planes = planes;
    requirement.required_plane_count = 1u;
    requirement.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    requirement.required_order = {12u, 1u};
    requirement.generation_policy = joint::generation_requirement_v1::exact;
    requirement.required_generation = {7u};

    joint::atom_plane_affordance_v1 plane_affordance{};
    plane_affordance.plane_identity = plane;
    plane_affordance.order = requirement.required_order;
    plane_affordance.storage = execution::numeric_type::f32;
    plane_affordance.logical = execution::numeric_type::f32;
    plane_affordance.generation = requirement.required_generation;
    joint::atom_affordance_v1 affordance{};
    affordance.affordance_identity = candidate.affordance_identity;
    affordance.atom_species = species[0];
    affordance.exact_coverage_identity = requirement.exact_coverage_identity;
    affordance.physical_encoding = {7u, 1u};
    affordance.local_projection_abi = {8u, 1u};
    affordance.planes = &plane_affordance;
    affordance.plane_count = 1u;

    alignas(16) unsigned char bytes[32]{};
    joint::external_extent_v1 extent{};
    extent.address = bytes;
    extent.location = {execution::residency_kind::host, {0u, 0u, 0u}, -1, 1u};
    extent.bytes = sizeof(bytes);
    extent.alignment = 16u;
    extent.order = requirement.required_order;
    extent.generation = requirement.required_generation;
    extent.readiness = {1u, 1u};
    extent.lease = {2u, 3u};
    joint::external_binding_v1 binding{};
    binding.binding_identity = {9u, 1u};
    binding.atom_identity = atom;
    binding.plane_identity = plane;
    binding.extents = &extent;
    binding.extent_count = 1u;
    binding.total_bytes = sizeof(bytes);

    fragment::bound_atom_extent_v1 output{};
    std::uint64_t written = 0u;
    assert(fragment::bind_external_atom_planes_v1(prepared, requirement,
        affordance, &binding, 1u, &output, 1u, &written));
    assert(written == 1u && output.address == bytes);
    assert(output.lease.slot == 2u && output.lease.generation == 3u);

    const auto capacity = fragment::bind_external_atom_planes_v1(prepared,
        requirement, affordance, &binding, 1u, nullptr, 0u, &written);
    assert(capacity.code == fragment::external_plane_binding_status_code_v1::
        insufficient_capacity);
    assert(capacity.required_capacity == 1u && written == 0u);

    extent.generation.value = 8u;
    const auto stale = fragment::bind_external_atom_planes_v1(prepared,
        requirement, affordance, &binding, 1u, &output, 1u, &written);
    assert(stale.code == fragment::external_plane_binding_status_code_v1::
        incompatible_generation);
}
