#include <Cellerator/compiler/backend/nvptx/validate_backend_fallback_and_mixed_routes_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;

namespace {

mixed_backend_candidate_v1 candidate(std::uint64_t identity,
                                     mixed_backend_route_v1 route,
                                     bool available,
                                     std::uint64_t input_order,
                                     std::uint64_t output_order,
                                     std::uint64_t input_generation,
                                     std::uint64_t output_generation) {
    return {identity, route, available, true, 11u, 21u, input_order, output_order,
            input_generation, output_generation};
}

}  // namespace

int main() {
    mixed_backend_graph_v1 graph;
    graph.abi_identity = 11u;
    graph.structure_identity = 21u;
    graph.initial_order_identity = 31u;
    graph.initial_generation = 1u;
    graph.stages = {
        {101u, {candidate(201u, mixed_backend_route_v1::prelinked_provider, true,
                          31u, 32u, 1u, 1u)}},
        {102u, {candidate(202u, mixed_backend_route_v1::generated_nvcc, true,
                          32u, 33u, 1u, 2u)}},
        {103u, {candidate(203u, mixed_backend_route_v1::direct_ptx, false,
                          33u, 34u, 2u, 2u),
                candidate(204u, mixed_backend_route_v1::generated_nvcc, true,
                          33u, 34u, 2u, 2u)}}};
    const auto fallback = validate_and_select_mixed_backend_graph_v1(graph);
    assert(fallback && fallback.selections.size() == 3u &&
           fallback.selections[0].route == mixed_backend_route_v1::prelinked_provider &&
           fallback.selections[1].route == mixed_backend_route_v1::generated_nvcc &&
           fallback.selections[2].route == mixed_backend_route_v1::generated_nvcc &&
           fallback.selections[2].fallback_used && fallback.final_order_identity == 34u &&
           fallback.final_generation == 2u);

    graph.stages[2].prioritized_candidates[0].available = true;
    const auto direct = validate_and_select_mixed_backend_graph_v1(graph);
    assert(direct && direct.selections[2].route == mixed_backend_route_v1::direct_ptx &&
           !direct.selections[2].fallback_used);

    graph.stages[2].prioritized_candidates[0].available = false;
    graph.stages[2].prioritized_candidates[1].input_order_identity = 999u;
    const auto incompatible = validate_and_select_mixed_backend_graph_v1(graph);
    assert(!incompatible && incompatible.status ==
           mixed_backend_graph_status_v1::no_available_exact_route);
    std::cout << "prelinked + NVCC + direct PTX graph and explicit compatible fallback validated\n";
}
