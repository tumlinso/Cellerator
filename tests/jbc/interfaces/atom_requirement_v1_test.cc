#include <Cellerator/execution/joint_compiler/atom_requirement_v1.hh>

#include <cassert>
#include <cstdint>

namespace joint_compiler = cellerator::execution::joint_compiler;
namespace execution = cellerator::execution;

int main() {
    const joint_compiler::persistent_identity_v1 species[] = {
        {1u, 1u}, {1u, 2u}};
    const joint_compiler::persistent_identity_v1 planes[] = {
        {2u, 1u}, {2u, 2u}};
    const joint_compiler::persistent_identity_v1 transforms[] = {
        {3u, 1u}, {3u, 2u}};

    joint_compiler::atom_requirement_v1 requirement{};
    requirement.requirement_identity = {4u, 1u};
    requirement.exact_coverage_identity = {4u, 2u};
    requirement.accepted_atom_species = species;
    requirement.accepted_atom_species_count = 2u;
    requirement.required_planes = planes;
    requirement.required_plane_count = 2u;
    requirement.numeric = {execution::numeric_type::f16,
        execution::numeric_type::f32, execution::numeric_type::f32, 0u};
    requirement.index_width = execution::local_index_width_v1::u32;
    requirement.required_order = {5u, 1u};
    requirement.minimum_alignment = 128u;
    requirement.contiguity =
        joint_compiler::contiguity_requirement_v1::contiguous;
    requirement.mutability =
        joint_compiler::mutability_requirement_v1::mutable_value_generation;
    requirement.generation_policy =
        joint_compiler::generation_requirement_v1::at_least;
    requirement.required_generation = {7u};
    requirement.graph_stable_address = true;
    requirement.minimum_extent_count = 1u;
    requirement.maximum_extent_count = 8u;
    requirement.transform_paths = transforms;
    requirement.transform_path_count = 2u;
    assert(joint_compiler::validate_atom_requirement_v1(requirement));

    auto malformed = requirement;
    malformed.schema_version += 1u;
    assert(joint_compiler::validate_atom_requirement_v1(malformed).code
        == joint_compiler::atom_requirement_validation_code_v1::
            unsupported_schema);
    malformed = requirement;
    malformed.accepted_atom_species = nullptr;
    assert(joint_compiler::validate_atom_requirement_v1(malformed).code
        == joint_compiler::atom_requirement_validation_code_v1::
            missing_atom_species);
    const joint_compiler::persistent_identity_v1 duplicate_species[] = {
        {1u, 1u}, {1u, 1u}};
    malformed = requirement;
    malformed.accepted_atom_species = duplicate_species;
    assert(joint_compiler::validate_atom_requirement_v1(malformed).code
        == joint_compiler::atom_requirement_validation_code_v1::
            duplicate_or_unordered_atom_species);
    malformed = requirement;
    malformed.minimum_alignment = 96u;
    assert(joint_compiler::validate_atom_requirement_v1(malformed).code
        == joint_compiler::atom_requirement_validation_code_v1::
            invalid_alignment);
    malformed = requirement;
    malformed.minimum_extent_count = 9u;
    assert(joint_compiler::validate_atom_requirement_v1(malformed).code
        == joint_compiler::atom_requirement_validation_code_v1::
            invalid_extent_count);
    malformed = requirement;
    malformed.generation_policy =
        joint_compiler::generation_requirement_v1::any_current;
    assert(joint_compiler::validate_atom_requirement_v1(malformed).code
        == joint_compiler::atom_requirement_validation_code_v1::
            invalid_generation);

    // Transform-free requirements are valid and carry no hidden route.
    requirement.transform_paths = nullptr;
    requirement.transform_path_count = 0u;
    assert(joint_compiler::validate_atom_requirement_v1(requirement));

    return 0;
}
