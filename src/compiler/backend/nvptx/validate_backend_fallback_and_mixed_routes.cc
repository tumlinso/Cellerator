#include <Cellerator/compiler/backend/nvptx/validate_backend_fallback_and_mixed_routes_v1.hh>

#include <sstream>
#include <unordered_set>

namespace Cellerator::compiler::backend::nvptx {

mixed_backend_graph_result_v1 validate_and_select_mixed_backend_graph_v1(
    const mixed_backend_graph_v1& graph) {
    mixed_backend_graph_result_v1 result;
    if (graph.abi_identity == 0u || graph.structure_identity == 0u ||
        graph.initial_order_identity == 0u || graph.initial_generation == 0u ||
        graph.stages.empty()) {
        result.diagnostics.emplace_back("graph ABI, structure, initial order/generation, and stages are required");
        return result;
    }
    std::unordered_set<std::uint64_t> stage_identities;
    std::uint64_t order = graph.initial_order_identity;
    std::uint64_t generation = graph.initial_generation;
    for (const auto& stage : graph.stages) {
        if (stage.stage_identity == 0u || !stage_identities.insert(stage.stage_identity).second ||
            stage.prioritized_candidates.empty()) {
            result.diagnostics.emplace_back("stage identity or candidate list is invalid");
            return result;
        }
        const mixed_backend_candidate_v1* selected = nullptr;
        std::size_t selected_index = 0u;
        for (std::size_t index = 0u; index < stage.prioritized_candidates.size(); ++index) {
            const auto& candidate = stage.prioritized_candidates[index];
            std::ostringstream reason;
            reason << "stage " << stage.stage_identity << " candidate " << candidate.candidate_identity;
            if (!candidate.available) {
                result.diagnostics.push_back(reason.str() + " unavailable; trying explicit fallback");
                continue;
            }
            if (candidate.candidate_identity == 0u || !candidate.exact) {
                result.diagnostics.push_back(reason.str() + " rejected: route is not exact");
                continue;
            }
            if (candidate.abi_identity != graph.abi_identity) {
                result.diagnostics.push_back(reason.str() + " rejected: ABI identity mismatch");
                continue;
            }
            if (candidate.structure_identity != graph.structure_identity) {
                result.diagnostics.push_back(reason.str() + " rejected: structure identity mismatch");
                continue;
            }
            if (candidate.input_order_identity != order) {
                result.diagnostics.push_back(reason.str() + " rejected: input order mismatch");
                continue;
            }
            if (candidate.input_generation != generation || candidate.output_generation < generation) {
                result.diagnostics.push_back(reason.str() + " rejected: generation contract mismatch");
                continue;
            }
            if (candidate.output_order_identity == 0u || candidate.output_generation == 0u) {
                result.diagnostics.push_back(reason.str() + " rejected: output contract is incomplete");
                continue;
            }
            selected = &candidate;
            selected_index = index;
            break;
        }
        if (selected == nullptr) {
            result.status = mixed_backend_graph_status_v1::no_available_exact_route;
            result.final_order_identity = order;
            result.final_generation = generation;
            return result;
        }
        result.selections.push_back({stage.stage_identity, selected->candidate_identity,
                                     selected->route, selected_index != 0u});
        order = selected->output_order_identity;
        generation = selected->output_generation;
    }
    result.status = mixed_backend_graph_status_v1::success;
    result.final_order_identity = order;
    result.final_generation = generation;
    return result;
}

}  // namespace Cellerator::compiler::backend::nvptx
