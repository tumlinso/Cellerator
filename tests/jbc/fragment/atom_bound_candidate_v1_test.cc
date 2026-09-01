#include <Cellerator/execution/atom_fragment/atom_bound_candidate_v1.hh>

#include <cassert>
#include <cstring>

namespace catalog = cellerator::compute::operation::catalog_v3;
namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;
namespace joint = execution::joint_compiler;

int main() {
    catalog::candidate_stage_v3 stage{};
    stage.stage_id = 1u;
    stage.kernel_id = 2u;
    std::strcpy(stage.stable_name, "atom-local");
    catalog::candidate_descriptor_v3 candidate{};
    candidate.identity.candidate_id = 7u;
    candidate.identity.provider_id = 8u;
    candidate.identity.operation_id = 9u;
    candidate.identity.width_min = 1u;
    candidate.identity.width_max = 16u;
    candidate.stages = &stage;
    candidate.stage_count = 1u;
    const catalog::candidate_catalog_view_v3 catalog_view{&candidate, 1u};

    const joint::persistent_identity_v1 species[] = {{1u, 1u}};
    const joint::persistent_identity_v1 planes[] = {{2u, 1u}};
    joint::atom_requirement_v1 requirement{};
    requirement.requirement_identity = {10u, 1u};
    requirement.exact_coverage_identity = {11u, 1u};
    requirement.accepted_atom_species = species;
    requirement.accepted_atom_species_count = 1u;
    requirement.required_planes = planes;
    requirement.required_plane_count = 1u;
    requirement.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    requirement.required_order = {12u, 1u};

    joint::atom_binding_request_v1 binding{};
    binding.atom_identity = {20u, 1u};
    binding.requirement_identity = requirement.requirement_identity;
    binding.affordance_identity = {21u, 1u};
    fragment::atom_bound_candidate_v1 output{};
    std::uint64_t written = 0u;
    assert(fragment::discover_atom_bound_candidates_v1(catalog_view,
        &requirement, 1u, &binding, 1u, &output, 1u, &written));
    assert(written == 1u && output.candidate_id == 7u);
    assert(output.atom_identity.local_identity == 1u);

    joint::atom_binding_request_v1 duplicate[] = {binding, binding};
    const auto ambiguous = fragment::discover_atom_bound_candidates_v1(
        catalog_view, &requirement, 1u, duplicate, 2u, &output, 1u, &written);
    assert(ambiguous.code == fragment::atom_bound_candidate_status_code_v1::
        ambiguous_requirement_binding);
    assert(written == 0u);

    binding.requirement_identity = {99u, 1u};
    const auto missing = fragment::discover_atom_bound_candidates_v1(
        catalog_view, &requirement, 1u, &binding, 1u, &output, 1u, &written);
    assert(missing.code == fragment::atom_bound_candidate_status_code_v1::
        missing_requirement_binding);

    binding.requirement_identity = requirement.requirement_identity;
    const auto capacity = fragment::discover_atom_bound_candidates_v1(
        catalog_view, &requirement, 1u, &binding, 1u, nullptr, 0u, &written);
    assert(capacity.code == fragment::atom_bound_candidate_status_code_v1::
        insufficient_capacity);
    assert(capacity.required_capacity == 1u);
}
