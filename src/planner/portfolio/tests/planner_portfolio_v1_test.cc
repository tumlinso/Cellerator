#include <Cellerator/planner/portfolio/planner_portfolio_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using namespace cellerator;
using namespace cellerator::planner;
using namespace cellerator::planner::portfolio;
using namespace cellerator::planner::resource;

operation_core::stable_id stable(std::uint64_t low) {
    return {low, 1u};
}

execution::order_id order(std::uint64_t low) {
    return {low, 1u};
}

candidate_resource_manifest_v1 manifest(
    operation_core::stable_id candidate,
    const planning_stage_v1 *stages,
    const stage_resource_receipt_v1 *receipts,
    std::uint64_t memory) {
    candidate_resource_manifest_v1 result{};
    result.candidate = candidate;
    result.provider = stable(1000u);
    result.capability = stable(1001u);
    result.projection = {1002u, 1u};
    result.geometry = {1003u, 1u};
    result.mechanism = {100u, 112u, 100u, 12u, 400u, 800u,
        400u, 200u, 7u, 4u, 8u, 100u};
    result.stages = stages;
    result.resources = receipts;
    result.stage_count = 2u;
    result.resource_count = 2u;
    result.persistent_bytes = memory;
    result.transient_bytes = 10u;
    result.cold_resource_query_complete = true;
    return result;
}

}  // namespace

int main() {
    const std::array<planning_stage_v1, 2> stages{{
        {stable(2000u), 1u, "ce_exop_pack", planning_stage_kind_v1::value_pack,
            0u, 0u, 1u, 3.0, 1u, 4u},
        {stable(2001u), 2u, "ce_exop_kernel", planning_stage_kind_v1::kernel,
            planning_stage_requires_measurement_v1, 0u, 1u, 7.0, 1u, 8u},
    }};
    const std::array<stage_resource_receipt_v1, 2> receipts{{
        {stable(2000u), resource_evidence_kind_v1::declared, {},
            0u, 0u, 0u, 0u, 0u, 0u},
        {stable(2001u), resource_evidence_kind_v1::compiled_attribute_query, {},
            128u, 4u, 32u, 1024u, 0u, 0u},
    }};
    auto base_manifest = manifest(stable(1u), stages.data(), receipts.data(), 4u);
    assert(validate_candidate_resource_manifest_v1(base_manifest));
    phase_costs manifest_costs{};
    assert(compute_manifest_phase_costs_v1(base_manifest, &manifest_costs));
    assert(manifest_costs.static_value_pack_ns == 3.0);
    assert(manifest_costs.kernel_ns == 7.0);

    constexpr std::uint64_t count = 96u;
    std::vector<candidate_resource_manifest_v1> manifests(count);
    std::vector<portfolio_candidate_v1> candidates(count);
    for (std::uint64_t index = 0u; index < count; ++index) {
        const auto identity = stable(index + 1u);
        manifests[index] = manifest(identity, stages.data(), receipts.data(),
            200u - index);
        candidates[index].identity = identity;
        candidates[index].manifest = &manifests[index];
        candidates[index].predicted_end_to_end_ns =
            static_cast<double>(index + 1u);
        candidates[index].predicted_preparation_ns = 1.0;
        candidates[index].predicted_value_update_ns = 1.0;
        candidates[index].predicted_layout_ns = 1.0;
        candidates[index].forward_quality = 1.0;
        candidates[index].transpose_quality = 1.0;
        candidates[index].contraction_quality = 1.0;
        candidates[index].flags = portfolio_candidate_compatible_v1
            | portfolio_candidate_correct_v1;
        if ((index + 1u) % 10u == 0u) {
            candidates[index].flags |= portfolio_candidate_experimental_v1;
        }
    }

    std::vector<candidate_workspace_state_v1> states(count);
    std::vector<std::uint64_t> ordering(count);
    std::vector<std::uint64_t> pareto(count);
    std::vector<double> scalar_costs(count);
    candidate_workspace_v1 workspace{states.data(), states.size(),
        ordering.data(), ordering.size(), pareto.data(), pareto.size(),
        scalar_costs.data(), scalar_costs.size(), 0u};
    assert(initialize_candidate_workspace_v1(count, &workspace));
    pareto_result_v1 result{};
    pareto_policy_v1 policy{};
    assert(build_pareto_portfolio_v1(candidates.data(), candidates.size(),
        policy, &workspace, &result));
    assert(result.compatible_count == 87u);
    assert(result.frontier_count == 87u);
    assert(pareto[0] == 0u && pareto[result.frontier_count - 1u] == 95u);

    policy.forced_candidate = stable(10u);
    assert(build_pareto_portfolio_v1(candidates.data(), candidates.size(),
        policy, &workspace, &result).code
        == workspace_status_code_v1::invalid_argument);
    policy.allow_forced_experimental = true;
    assert(build_pareto_portfolio_v1(candidates.data(), candidates.size(),
        policy, &workspace, &result));
    assert(result.forced_candidate_index == 9u);
    assert(result.compatible_count == 88u);

    const auto saved_identity = candidates[11].identity;
    candidates[11].identity = candidates[10].identity;
    assert(build_pareto_portfolio_v1(candidates.data(), candidates.size(),
        policy, &workspace, &result).code
        == workspace_status_code_v1::invalid_argument);
    candidates[11].identity = saved_identity;

    const auto saved_bytes = manifests[0].transient_bytes;
    manifests[0].persistent_bytes = std::numeric_limits<std::uint64_t>::max();
    manifests[0].transient_bytes = 1u;
    assert(build_pareto_portfolio_v1(candidates.data(), candidates.size(),
        policy, &workspace, &result).code
        == workspace_status_code_v1::invalid_argument);
    manifests[0].persistent_bytes = 4u;
    manifests[0].transient_bytes = saved_bytes;

    phase_costs phases{};
    phases.backend_prepare_ns = 4.0;
    phases.static_value_pack_ns = 2.0;
    phases.kernel_ns = 10.0;
    phases.persistent_bytes = 20u;
    phases.transient_bytes = 30u;
    std::array<operation_economics_v1, 2> operations{{
        {stable(1u), order(1u), order(2u), phases,
            planner_value_mode_v1::logical_primary, {}, 2u, 3u, 2u, 2u, 3u,
            false, true, {}},
        {stable(2u), order(2u), order(3u), phases,
            planner_value_mode_v1::projection_primary, {}, 1u, 1u, 2u, 2u, 3u,
            true, true, {}},
    }};
    std::array<layout_transition_economics_v1, 1> transitions{{
        {order(2u), order(2u), 0.0, 1.0, 0u, true, {}},
    }};
    connected_program_economics_v1 program{operations.data(),
        operations.size(), transitions.data(), transitions.size(), order(4u),
        5.0, 64u};
    connected_economics_result_v1 economics{};
    assert(compute_connected_economics_v1(program, &economics));
    assert(economics.complete_cost_ns > 0.0);
    assert(economics.layout_cost_ns == 5.0);
    assert(economics.persistent_bytes == 40u);
    assert(economics.peak_transient_bytes == 64u);

    policy = {};
    frozen_planner_portfolio_v1 frozen{candidates.data(), candidates.size(),
        policy, workspace, &program};
    frozen_planner_result_v1 frozen_result{};
    assert(validate_frozen_planner_portfolio_v1(&frozen, &frozen_result));
    assert(frozen_result.pareto.compatible_count == 87u);
    assert(frozen_result.has_connected_economics);
    assert(frozen_result.connected.complete_cost_ns
        == economics.complete_cost_ns);

    transitions[0].destination_order = order(9u);
    assert(compute_connected_economics_v1(program, &economics).code
        == economics_status_code_v1::invalid_order);
}
