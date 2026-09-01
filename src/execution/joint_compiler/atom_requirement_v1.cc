#include <Cellerator/execution/joint_compiler/atom_requirement_v1.hh>

namespace cellerator::execution::joint_compiler {
namespace {

atom_requirement_validation_result_v1 failure(
    atom_requirement_validation_code_v1 code,
    std::uint64_t index = 0u) noexcept {
    return {code, index};
}

bool identity_less(
    persistent_identity_v1 lhs, persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

atom_requirement_validation_result_v1 validate_id_array(
    const persistent_identity_v1 *values,
    std::uint64_t count,
    atom_requirement_validation_code_v1 missing,
    atom_requirement_validation_code_v1 invalid,
    atom_requirement_validation_code_v1 unordered) noexcept {
    if (count == 0u || values == nullptr) return failure(missing);
    for (std::uint64_t index = 0u; index < count; ++index) {
        if (!validate_persistent_identity_v1(values[index]))
            return failure(invalid, index);
        if (index != 0u && !identity_less(values[index - 1u], values[index]))
            return failure(unordered, index);
    }
    return {};
}

bool valid_numeric(numeric_type type) noexcept {
    return type >= numeric_type::bit && type <= numeric_type::f64;
}

}  // namespace

atom_requirement_validation_result_v1 validate_atom_requirement_v1(
    const atom_requirement_v1 &requirement) noexcept {
    if (requirement.schema_version != atom_requirement_schema_version_v1)
        return failure(atom_requirement_validation_code_v1::unsupported_schema);
    if (requirement.record_bytes != sizeof(atom_requirement_v1))
        return failure(
            atom_requirement_validation_code_v1::invalid_record_bytes);
    if (requirement.reserved0[0] != 0u || requirement.reserved0[1] != 0u
        || requirement.reserved0[2] != 0u
        || requirement.numeric.reserved != 0u)
        return failure(atom_requirement_validation_code_v1::nonzero_reserved);
    if (!validate_persistent_identity_v1(requirement.requirement_identity))
        return failure(atom_requirement_validation_code_v1::
            invalid_requirement_identity);
    if (!validate_persistent_identity_v1(requirement.exact_coverage_identity))
        return failure(
            atom_requirement_validation_code_v1::invalid_coverage_identity);

    auto array_result = validate_id_array(requirement.accepted_atom_species,
        requirement.accepted_atom_species_count,
        atom_requirement_validation_code_v1::missing_atom_species,
        atom_requirement_validation_code_v1::invalid_atom_species,
        atom_requirement_validation_code_v1::
            duplicate_or_unordered_atom_species);
    if (!array_result) return array_result;
    array_result = validate_id_array(requirement.required_planes,
        requirement.required_plane_count,
        atom_requirement_validation_code_v1::missing_planes,
        atom_requirement_validation_code_v1::invalid_plane,
        atom_requirement_validation_code_v1::duplicate_or_unordered_plane);
    if (!array_result) return array_result;

    if (!valid_numeric(requirement.numeric.storage)
        || !valid_numeric(requirement.numeric.logical)
        || !valid_numeric(requirement.numeric.accumulation))
        return failure(atom_requirement_validation_code_v1::invalid_numeric);
    if (requirement.index_width != local_index_width_v1::u16
        && requirement.index_width != local_index_width_v1::u32
        && requirement.index_width != local_index_width_v1::u64)
        return failure(atom_requirement_validation_code_v1::invalid_index_width);
    if (!valid_identity(requirement.required_order))
        return failure(atom_requirement_validation_code_v1::invalid_order);
    if (requirement.minimum_alignment == 0u
        || (requirement.minimum_alignment
            & (requirement.minimum_alignment - 1u)) != 0u)
        return failure(atom_requirement_validation_code_v1::invalid_alignment);
    if (requirement.contiguity < contiguity_requirement_v1::any
        || requirement.contiguity > contiguity_requirement_v1::regular_stride)
        return failure(atom_requirement_validation_code_v1::invalid_contiguity);
    if (requirement.mutability < mutability_requirement_v1::immutable
        || requirement.mutability
            > mutability_requirement_v1::mutable_value_generation)
        return failure(atom_requirement_validation_code_v1::invalid_mutability);
    if (requirement.generation_policy < generation_requirement_v1::any_current
        || requirement.generation_policy > generation_requirement_v1::at_least)
        return failure(
            atom_requirement_validation_code_v1::invalid_generation_policy);
    if ((requirement.generation_policy
            == generation_requirement_v1::any_current)
            != (requirement.required_generation.value == 0u))
        return failure(atom_requirement_validation_code_v1::invalid_generation);
    if (requirement.minimum_extent_count == 0u
        || requirement.maximum_extent_count < requirement.minimum_extent_count)
        return failure(
            atom_requirement_validation_code_v1::invalid_extent_count);

    if (requirement.transform_path_count == 0u) {
        if (requirement.transform_paths != nullptr)
            return failure(atom_requirement_validation_code_v1::
                missing_transform_paths);
        return {};
    }
    return validate_id_array(requirement.transform_paths,
        requirement.transform_path_count,
        atom_requirement_validation_code_v1::missing_transform_paths,
        atom_requirement_validation_code_v1::invalid_transform_path,
        atom_requirement_validation_code_v1::
            duplicate_or_unordered_transform_path);
}

}  // namespace cellerator::execution::joint_compiler
