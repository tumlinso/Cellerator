#include <Cellerator/compiler/planning/represent_split_axes_fragments_halos_and_replicas_v1.hh>

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace Cellerator::compiler::planning {

planning_decomposition_validation_code_v1 validate_planning_decomposition_v1(
    const planning_decomposition_v1& decomposition) noexcept {
    if (decomposition.decomposition_identity == 0u ||
        decomposition.exact_logical_extent == 0u || decomposition.fragments.empty()) {
        return planning_decomposition_validation_code_v1::invalid_decomposition;
    }

    std::unordered_set<std::uint64_t> contributors;
    std::vector<const planning_fragment_v1*> owners;
    owners.reserve(decomposition.fragments.size());
    constexpr std::uint32_t known_roles = exact_input_read_v1 |
        exact_output_owner_v1 | exact_contribution_owner_v1 |
        read_only_halo_v1 | physical_replica_v1;
    for (const auto& fragment : decomposition.fragments) {
        if (fragment.fragment_identity == 0u || fragment.contributor_identity == 0u ||
            fragment.logical_count == 0u || fragment.input_order_identity == 0u ||
            fragment.output_order_identity == 0u ||
            fragment.logical_begin > decomposition.exact_logical_extent ||
            fragment.logical_count >
                decomposition.exact_logical_extent - fragment.logical_begin) {
            return planning_decomposition_validation_code_v1::invalid_fragment;
        }
        if (fragment.extent_lower_bound > fragment.extent_upper_bound ||
            fragment.logical_count < fragment.extent_lower_bound ||
            fragment.logical_count > fragment.extent_upper_bound) {
            return planning_decomposition_validation_code_v1::invalid_extent_bounds;
        }
        if (fragment.roles == 0u || (fragment.roles & ~known_roles) != 0u) {
            return planning_decomposition_validation_code_v1::invalid_role;
        }
        if (!contributors.insert(fragment.contributor_identity).second) {
            return planning_decomposition_validation_code_v1::duplicate_contributor;
        }
        if ((fragment.roles & read_only_halo_v1) != 0u &&
            (fragment.roles & (exact_output_owner_v1 | exact_contribution_owner_v1)) != 0u) {
            return planning_decomposition_validation_code_v1::invalid_halo;
        }
        if ((fragment.roles & physical_replica_v1) != 0u &&
            fragment.replica_group_identity == 0u) {
            return planning_decomposition_validation_code_v1::invalid_replica;
        }
        if ((fragment.roles & exact_contribution_owner_v1) != 0u) owners.push_back(&fragment);
    }

    std::sort(owners.begin(), owners.end(), [](const auto* lhs, const auto* rhs) {
        return lhs->logical_begin < rhs->logical_begin;
    });
    std::uint64_t covered = 0u;
    for (const auto* owner : owners) {
        if (owner->logical_begin < covered) {
            return planning_decomposition_validation_code_v1::overlapping_exact_coverage;
        }
        if (owner->logical_begin != covered) {
            return planning_decomposition_validation_code_v1::incomplete_exact_coverage;
        }
        covered += owner->logical_count;
    }
    return covered == decomposition.exact_logical_extent
        ? planning_decomposition_validation_code_v1::ok
        : planning_decomposition_validation_code_v1::incomplete_exact_coverage;
}

}  // namespace Cellerator::compiler::planning
