#include <Cellerator/compute/architecture/target_refinement.hh>

#include <cassert>
#include <cstdint>
#include <type_traits>

namespace architecture = cellerator::compute::architecture;
namespace execution = cellerator::execution;
namespace geometry = cellerator::geometry;

int main() {
    static_assert(std::is_standard_layout<
        architecture::target_refinement_problem_v1>::value);
    static_assert(std::is_standard_layout<
        architecture::target_refinement_solution_v1>::value);

    geometry::semantic_component_v1 component{};
    component.component_id = 7u;
    component.kind = geometry::semantic_component_kind::rectangular;
    component.logical_edge_count = 2u;
    const std::uint64_t logical_edges[] = {1u, 0u};

    geometry::relation_cover_view_v1 cover{};
    cover.structure = {41u, 1u};
    cover.structure_epoch.value = 3u;
    cover.source_axis = {{11u, 1u}, {12u, 1u}, {13u, 1u}, {14u, 1u}};
    cover.destination_axis = {{21u, 1u}, {22u, 1u}, {23u, 1u}, {24u, 1u}};
    cover.logical_edge_count = 2u;
    cover.component_count = 1u;
    cover.components = &component;
    cover.logical_edge_ids = logical_edges;

    architecture::matrix_engine_capability_v1 capability{};
    capability.identity = {101u, 102u};
    capability.provider_identity = {201u, 202u};

    architecture::target_refinement_problem_v1 problem{};
    problem.semantic_cover = cover;
    problem.provider_identity = capability.provider_identity;
    problem.capabilities = &capability;
    problem.capability_count = 1u;
    problem.dense_width = 64u;
    problem.policy.tier = architecture::target_refinement_tier_v1::bounded;
    problem.policy.maximum_iterations = 8u;
    problem.policy.expected_reuse = 16u;
    problem.sparse_baseline.execution_nanoseconds = 12.5;

    assert(problem.semantic_cover.components == &component);
    assert(problem.capabilities == &capability);
    assert(problem.semantic_cover.structure.slot == 41u);
    assert(architecture::valid_target_refinement_tier_v1(problem.policy.tier));

    architecture::target_refinement_region_v1 region{};
    region.semantic_component_id = component.component_id;
    region.region_id = 0u;
    region.role = architecture::target_region_role_v1::matrix_engine;
    region.capability_identity = capability.identity;
    region.logical_edge_count = 2u;
    region.estimated_cost.execution_nanoseconds = 4.0;
    const std::uint32_t edge_owners[] = {0u, 0u};

    architecture::target_refinement_solution_v1 solution{};
    solution.provider_identity = problem.provider_identity;
    solution.structure = cover.structure;
    solution.structure_epoch = cover.structure_epoch;
    solution.conservative_hybrid.kind =
        architecture::target_cover_kind_v1::conservative_hybrid;
    solution.conservative_hybrid.regions = &region;
    solution.conservative_hybrid.region_count = 1u;
    solution.conservative_hybrid.logical_edge_to_region = edge_owners;
    solution.conservative_hybrid.logical_edge_count = 2u;

    assert(solution.conservative_hybrid.regions == &region);
    assert(solution.conservative_hybrid.logical_edge_to_region[1] == 0u);
    assert(architecture::valid_target_region_role_v1(region.role));
    assert(architecture::valid_target_cover_kind_v1(
        solution.conservative_hybrid.kind));
    assert(!architecture::valid_target_cover_kind_v1(
        static_cast<architecture::target_cover_kind_v1>(0u)));

    return 0;
}
