#include <Cellerator/execution/preparation/atom_affordance_v1.hh>

#include <cassert>

namespace preparation = cellerator::execution::preparation;
namespace execution = cellerator::execution;

int main() {
    const preparation::atom_plane_affordance_v1 planes[] = {
        {{1u, 1u}, {2u, 1u}, execution::numeric_type::f16,
            execution::numeric_type::f32,
            preparation::mutability_requirement_v1::immutable, 0u, {1u}},
        {{1u, 2u}, {2u, 2u}, execution::numeric_type::f32,
            execution::numeric_type::f32,
            preparation::mutability_requirement_v1::mutable_value_generation,
            0u, {9u}}};
    const preparation::persistent_identity_v1 transforms[] = {
        {3u, 1u}, {3u, 2u}};

    preparation::atom_affordance_v1 affordance{};
    affordance.affordance_identity = {4u, 1u};
    affordance.atom_species = {4u, 2u};
    affordance.exact_coverage_identity = {4u, 3u};
    affordance.physical_encoding = {4u, 4u};
    affordance.local_projection_abi = {4u, 5u};
    affordance.planes = planes;
    affordance.plane_count = 2u;
    affordance.extent_count = 3u;
    affordance.flags = preparation::multi_extent_legal_v1
        | preparation::direct_output_support_v1
        | preparation::persistence_eligible_v1;
    affordance.fused_transforms = transforms;
    affordance.fused_transform_count = 2u;
    assert(preparation::validate_atom_affordance_v1(affordance));

    auto malformed = affordance;
    malformed.record_bytes -= 1u;
    assert(preparation::validate_atom_affordance_v1(malformed).code
        == preparation::atom_affordance_validation_code_v1::
            invalid_record_bytes);
    malformed = affordance;
    malformed.extent_count = 3u;
    malformed.flags &= ~preparation::multi_extent_legal_v1;
    assert(preparation::validate_atom_affordance_v1(malformed).code
        == preparation::atom_affordance_validation_code_v1::
            multi_extent_flag_missing);
    const preparation::atom_plane_affordance_v1 duplicate_planes[] = {
        planes[0], planes[0]};
    malformed = affordance;
    malformed.planes = duplicate_planes;
    assert(preparation::validate_atom_affordance_v1(malformed).code
        == preparation::atom_affordance_validation_code_v1::
            duplicate_or_unordered_plane);
    malformed = affordance;
    malformed.flags |= 1u << 31u;
    assert(preparation::validate_atom_affordance_v1(malformed).code
        == preparation::atom_affordance_validation_code_v1::unknown_flag);
    malformed = affordance;
    malformed.fused_transforms = nullptr;
    assert(preparation::validate_atom_affordance_v1(malformed).code
        == preparation::atom_affordance_validation_code_v1::
            inconsistent_fused_transform_pointer);

    affordance.fused_transforms = nullptr;
    affordance.fused_transform_count = 0u;
    assert(preparation::validate_atom_affordance_v1(affordance));
    return 0;
}
