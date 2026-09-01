#include <Cellerator/execution/atom_fragment/local_candidate_requirements_v1.hh>

#include <cassert>
#include <cstring>

namespace catalog = cellerator::compute::operation::catalog_v3;
namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;
namespace joint = execution::joint_compiler;

joint::atom_requirement_v1 requirement(std::uint64_t identity,
    const joint::persistent_identity_v1 *species,
    const joint::persistent_identity_v1 *planes) {
    joint::atom_requirement_v1 result{};
    result.requirement_identity = {10u, identity};
    result.exact_coverage_identity = {20u, identity};
    result.accepted_atom_species = species;
    result.accepted_atom_species_count = 1u;
    result.required_planes = planes;
    result.required_plane_count = 1u;
    result.numeric = {execution::numeric_type::f32,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    result.required_order = {identity, 100u};
    return result;
}

catalog::candidate_descriptor_v3 candidate(std::uint64_t id,
    catalog::candidate_stage_v3 *stage) {
    catalog::candidate_descriptor_v3 result{};
    result.identity.candidate_id = id;
    result.identity.provider_id = 1u;
    result.identity.operation_id = 2u;
    result.identity.width_min = 1u;
    result.identity.width_max = 8u;
    result.stages = stage;
    result.stage_count = 1u;
    return result;
}

int main() {
    joint::persistent_identity_v1 species[] = {{1u, 1u}};
    joint::persistent_identity_v1 planes[] = {{2u, 1u}};
    catalog::candidate_stage_v3 stages[2]{};
    stages[0].stage_id = 1u;
    stages[0].kernel_id = 10u;
    std::strcpy(stages[0].stable_name, "candidate-a");
    stages[1].stage_id = 2u;
    stages[1].kernel_id = 11u;
    std::strcpy(stages[1].stable_name, "candidate-b");
    catalog::candidate_descriptor_v3 candidates[] = {
        candidate(7u, &stages[0]), candidate(9u, &stages[1])};
    const catalog::candidate_catalog_view_v3 catalog_view{candidates, 2u};
    fragment::local_candidate_atom_contract_v1 contracts[] = {
        {7u, requirement(1u, species, planes)},
        {9u, requirement(2u, species, planes)},
    };
    joint::atom_requirement_v1 output[2]{};
    std::uint64_t written = 99u;
    assert(fragment::extract_local_candidate_requirements_v1(catalog_view,
        contracts, 2u, output, 2u, &written));
    assert(written == 2u);
    assert(output[0].requirement_identity.local_identity == 1u);
    assert(output[1].requirement_identity.local_identity == 2u);

    const auto capacity = fragment::extract_local_candidate_requirements_v1(
        catalog_view, contracts, 2u, nullptr, 0u, &written);
    assert(capacity.code == fragment::
        local_candidate_requirement_status_code_v1::insufficient_capacity);
    assert(capacity.required_capacity == 2u && written == 0u);

    const auto missing = fragment::extract_local_candidate_requirements_v1(
        catalog_view, contracts, 1u, output, 2u, &written);
    assert(missing.code == fragment::
        local_candidate_requirement_status_code_v1::
            missing_candidate_contract);
    assert(missing.index == 1u && written == 0u);

    contracts[1].candidate_id = 7u;
    const auto duplicate = fragment::extract_local_candidate_requirements_v1(
        catalog_view, contracts, 2u, output, 2u, &written);
    assert(duplicate.code == fragment::
        local_candidate_requirement_status_code_v1::
            duplicate_or_unordered_contract);
}
