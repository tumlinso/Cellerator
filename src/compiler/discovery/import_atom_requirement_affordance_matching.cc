#include <Cellerator/compiler/discovery/import_atom_requirement_affordance_matching_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::discovery {
namespace {

bool sorted_unique_ids_v1(
    const std::vector<persistent_atom_identity_v1>& identities) noexcept {
    if (identities.empty()) {
        return false;
    }
    for (std::size_t index = 0; index < identities.size(); ++index) {
        if (!valid_persistent_atom_identity_v1(identities[index]) ||
            (index != 0 && !persistent_atom_identity_less_v1(
                               identities[index - 1], identities[index]))) {
            return false;
        }
    }
    return true;
}

bool generation_satisfies_v1(const migrated_atom_requirement_v1& requirement,
                             std::uint64_t generation) noexcept {
    if (requirement.generation_policy == atom_generation_policy_v1::any_current) {
        return generation != 0;
    }
    if (requirement.generation_policy == atom_generation_policy_v1::exact) {
        return generation == requirement.required_generation;
    }
    return generation >= requirement.required_generation;
}

}  // namespace

atom_match_result_v1 match_migrated_atom_v1(
    const migrated_atom_requirement_v1& requirement,
    const migrated_atom_affordance_v1& affordance) noexcept {
    if (!valid_persistent_atom_identity_v1(requirement.requirement_identity) ||
        !valid_persistent_atom_identity_v1(requirement.exact_coverage_identity) ||
        !valid_persistent_atom_identity_v1(requirement.required_order_identity) ||
        !valid_persistent_atom_identity_v1(requirement.required_projection_abi) ||
        !sorted_unique_ids_v1(requirement.accepted_species) ||
        !sorted_unique_ids_v1(requirement.required_planes) ||
        requirement.minimum_extent_count == 0 ||
        requirement.maximum_extent_count < requirement.minimum_extent_count ||
        (requirement.generation_policy != atom_generation_policy_v1::any_current &&
         requirement.required_generation == 0)) {
        return {atom_match_status_v1::invalid_requirement};
    }
    if (!valid_persistent_atom_identity_v1(affordance.affordance_identity) ||
        !valid_persistent_atom_identity_v1(affordance.atom_identity) ||
        !valid_persistent_atom_identity_v1(affordance.species_identity) ||
        !valid_persistent_atom_identity_v1(affordance.exact_coverage_identity) ||
        !valid_persistent_atom_identity_v1(affordance.projection_abi) ||
        affordance.extent_count == 0) {
        return {atom_match_status_v1::invalid_affordance};
    }
    for (std::size_t index = 0; index < affordance.planes.size(); ++index) {
        const auto& plane = affordance.planes[index];
        if (!valid_persistent_atom_identity_v1(plane.plane_identity) ||
            !valid_persistent_atom_identity_v1(plane.order_identity) ||
            plane.generation == 0 ||
            (index != 0 && !persistent_atom_identity_less_v1(
                               affordance.planes[index - 1].plane_identity,
                               plane.plane_identity))) {
            return {atom_match_status_v1::invalid_affordance, index};
        }
    }
    if (!std::binary_search(requirement.accepted_species.begin(),
                            requirement.accepted_species.end(),
                            affordance.species_identity,
                            persistent_atom_identity_less_v1)) {
        return {atom_match_status_v1::species_mismatch};
    }
    if (requirement.exact_coverage_identity != affordance.exact_coverage_identity) {
        return {atom_match_status_v1::coverage_mismatch};
    }
    if (requirement.required_projection_abi != affordance.projection_abi) {
        return {atom_match_status_v1::projection_mismatch};
    }
    if ((affordance.target_capabilities & requirement.required_target_capabilities) !=
        requirement.required_target_capabilities) {
        return {atom_match_status_v1::target_capability_mismatch};
    }
    if (affordance.extent_count < requirement.minimum_extent_count ||
        affordance.extent_count > requirement.maximum_extent_count) {
        return {atom_match_status_v1::extent_mismatch};
    }
    for (std::size_t index = 0; index < requirement.required_planes.size(); ++index) {
        const auto plane = std::lower_bound(
            affordance.planes.begin(), affordance.planes.end(),
            requirement.required_planes[index], [](const auto& candidate, auto identity) {
                return persistent_atom_identity_less_v1(
                    candidate.plane_identity, identity);
            });
        if (plane == affordance.planes.end() ||
            plane->plane_identity != requirement.required_planes[index]) {
            return {atom_match_status_v1::missing_plane, index};
        }
        if (plane->order_identity != requirement.required_order_identity) {
            return {atom_match_status_v1::order_mismatch, index};
        }
        if (!generation_satisfies_v1(requirement, plane->generation)) {
            return {atom_match_status_v1::generation_mismatch, index};
        }
    }
    return {atom_match_status_v1::matched, requirement.required_planes.size()};
}

}  // namespace Cellerator::compiler::discovery
