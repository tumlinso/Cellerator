#include <Cellerator/execution/joint_compiler/atom_affordance_v1.hh>

namespace cellerator::execution::joint_compiler {
namespace {

atom_affordance_validation_result_v1 failure(
    atom_affordance_validation_code_v1 code,
    std::uint64_t index = 0u) noexcept {
    return {code, index};
}

bool identity_less(
    persistent_identity_v1 lhs, persistent_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

bool valid_numeric(numeric_type type) noexcept {
    return type >= numeric_type::bit && type <= numeric_type::f64;
}

}  // namespace

atom_affordance_validation_result_v1 validate_atom_affordance_v1(
    const atom_affordance_v1 &affordance) noexcept {
    if (affordance.schema_version != atom_affordance_schema_version_v1)
        return failure(atom_affordance_validation_code_v1::unsupported_schema);
    if (affordance.record_bytes != sizeof(atom_affordance_v1))
        return failure(
            atom_affordance_validation_code_v1::invalid_record_bytes);
    if (!validate_persistent_identity_v1(affordance.affordance_identity))
        return failure(atom_affordance_validation_code_v1::
            invalid_affordance_identity);
    if (!validate_persistent_identity_v1(affordance.atom_species))
        return failure(
            atom_affordance_validation_code_v1::invalid_atom_species);
    if (!validate_persistent_identity_v1(affordance.exact_coverage_identity))
        return failure(
            atom_affordance_validation_code_v1::invalid_coverage_identity);
    if (!validate_persistent_identity_v1(affordance.physical_encoding))
        return failure(
            atom_affordance_validation_code_v1::invalid_physical_encoding);
    if (!validate_persistent_identity_v1(affordance.local_projection_abi))
        return failure(
            atom_affordance_validation_code_v1::invalid_projection_abi);
    if (affordance.plane_count == 0u || affordance.planes == nullptr)
        return failure(atom_affordance_validation_code_v1::missing_planes);

    for (std::uint64_t index = 0u; index < affordance.plane_count; ++index) {
        const atom_plane_affordance_v1 &plane = affordance.planes[index];
        if (!validate_persistent_identity_v1(plane.plane_identity))
            return failure(
                atom_affordance_validation_code_v1::invalid_plane_identity,
                index);
        if (index != 0u && !identity_less(
                affordance.planes[index - 1u].plane_identity,
                plane.plane_identity))
            return failure(atom_affordance_validation_code_v1::
                duplicate_or_unordered_plane, index);
        if (!valid_identity(plane.order))
            return failure(
                atom_affordance_validation_code_v1::invalid_plane_order,
                index);
        if (!valid_numeric(plane.storage) || !valid_numeric(plane.logical))
            return failure(
                atom_affordance_validation_code_v1::invalid_plane_numeric,
                index);
        if (plane.mutability < mutability_requirement_v1::immutable
            || plane.mutability
                > mutability_requirement_v1::mutable_value_generation)
            return failure(
                atom_affordance_validation_code_v1::invalid_plane_mutability,
                index);
        if (plane.reserved != 0u)
            return failure(atom_affordance_validation_code_v1::
                nonzero_reserved, index);
        if (plane.generation.value == 0u)
            return failure(atom_affordance_validation_code_v1::
                invalid_plane_generation, index);
    }

    if (affordance.extent_count == 0u)
        return failure(
            atom_affordance_validation_code_v1::invalid_extent_count);
    if (affordance.extent_count > 1u
        && (affordance.flags & multi_extent_legal_v1) == 0u)
        return failure(
            atom_affordance_validation_code_v1::multi_extent_flag_missing);
    if ((affordance.flags & ~known_atom_affordance_flags_v1) != 0u)
        return failure(atom_affordance_validation_code_v1::unknown_flag);

    if (affordance.fused_transform_count == 0u) {
        if (affordance.fused_transforms != nullptr)
            return failure(atom_affordance_validation_code_v1::
                inconsistent_fused_transform_pointer);
        return {};
    }
    if (affordance.fused_transforms == nullptr)
        return failure(atom_affordance_validation_code_v1::
            inconsistent_fused_transform_pointer);
    for (std::uint64_t index = 0u;
         index < affordance.fused_transform_count; ++index) {
        if (!validate_persistent_identity_v1(
                affordance.fused_transforms[index]))
            return failure(atom_affordance_validation_code_v1::
                invalid_fused_transform, index);
        if (index != 0u && !identity_less(
                affordance.fused_transforms[index - 1u],
                affordance.fused_transforms[index]))
            return failure(atom_affordance_validation_code_v1::
                duplicate_or_unordered_fused_transform, index);
    }
    return {};
}

}  // namespace cellerator::execution::joint_compiler
