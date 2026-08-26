#include <bench/ce_live/planner/live_planner_inputs.hh>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>

namespace inputs = cellerator::ce_live::planner_inputs;
namespace planner = cellerator::planner;

namespace {

constexpr std::uint64_t offsets[]{0u, 2u, 4u, 7u};
constexpr std::uint32_t sources[]{0u, 2u, 1u, 3u, 0u, 2u, 3u};
float generation_1[]{2.0f, -1.0f, 3.0f, 4.0f, 5.0f, 6.0f, -2.0f};
float generation_2[]{-0.5f, 1.25f, 2.0f, -3.0f, 0.75f, -1.5f, 4.0f};

void require(bool condition, const char *message) {
    if (condition) return;
    std::cerr << "ce_live_planner_features_test: " << message << '\n';
    std::exit(1);
}

inputs::relation_identity_input identities() {
    return {
        {11u, 12u}, {21u, 22u}, {31u, 32u}, {41u, 42u},
        {51u, 52u}, {61u, 62u}, {71u, 72u}, {3u}};
}

inputs::quantitative_relation_input relation(float *values,
    std::uint64_t generation) {
    return {identities(), offsets, sources, values, 4u, 3u, 7u, {generation}};
}

inputs::live_planner_input derive(float *values, std::uint64_t generation,
    inputs::reuse_horizons reuse = {32u, 16u, 4u}) {
    inputs::live_planner_input result{};
    require(inputs::derive_live_planner_input(relation(values, generation),
                {81u, 82u}, {1u, 7u, 0u, 700u},
                {100u, 200u, 300u, 400u}, reuse, 1u, 2u, 3u, 4u,
                &result)
            == inputs::live_input_status::ok,
        "valid live planner input was rejected");
    return result;
}

void test_statistics_and_exact_keys() {
    const inputs::live_planner_input live = derive(generation_1, 1u);
    require(live.structure.source_count == 4u
            && live.structure.destination_count == 3u
            && live.structure.logical_edge_count == 7u
            && live.structure.minimum_destination_degree == 2u
            && live.structure.maximum_destination_degree == 3u
            && std::fabs(live.structure.mean_destination_degree - 7.0 / 3.0)
                < 1.0e-12
            && std::fabs(live.structure.density - 7.0 / 12.0) < 1.0e-12,
        "structural statistics are incomplete");
    require(live.values.observed_generation.value == 1u
            && live.values.nonzero_count == 7u
            && live.values.minimum == -2.0
            && live.values.maximum == 6.0
            && live.values.l1_norm == 23.0,
        "quantitative statistics are incomplete");
    require(live.keys.structures.count == 1u
            && live.keys.structures.structures[0].identity.low == 71u
            && live.keys.structures.structures[0].epoch.value == 3u
            && live.keys.geometry.source_domain.low == 11u
            && live.keys.geometry.destination_domain.low == 21u
            && live.keys.policy.structure_reuse == 32u
            && live.keys.policy.projection_reuse == 16u
            && live.keys.policy.value_reuse == 4u,
        "persistent planning key lost identities or reuse horizons");
}

void test_generation_and_pointer_are_runtime_state() {
    const inputs::live_planner_input first = derive(generation_1, 1u);
    const inputs::live_planner_input second = derive(generation_2, 2u);
    require(planner::same_planning_keys(first.keys, second.keys),
        "value generation or pointer entered persistent planning identity");
    require(first.values.observed_generation.value
            != second.values.observed_generation.value
            && first.values.l1_norm != second.values.l1_norm,
        "quantitative generation did not remain explicit runtime evidence");

    inputs::quantitative_relation_input changed = relation(generation_1, 1u);
    changed.identities.structure_epoch.value += 1u;
    inputs::live_planner_input stale_epoch{};
    require(inputs::derive_live_planner_input(changed, {81u, 82u},
                {1u, 7u, 0u, 700u}, {100u, 200u, 300u, 400u},
                {32u, 16u, 4u}, 1u, 2u, 3u, 4u, &stale_epoch)
            == inputs::live_input_status::ok
            && !planner::same_planning_keys(first.keys, stale_epoch.keys),
        "stale structure epoch failed to invalidate the planning key");

    const inputs::live_planner_input changed_reuse =
        derive(generation_1, 1u, {32u, 16u, 5u});
    require(!planner::same_planning_keys(first.keys, changed_reuse.keys),
        "distinct value reuse horizon failed to invalidate policy key");
}

void test_complete_phase_accounting_and_empirical_authority() {
    inputs::candidate_phase_input candidate{};
    candidate.phases.host_preparation_ns = 13.0;
    candidate.phases.semantic_packing_ns = 320.0;
    candidate.phases.projection_construction_ns = 160.0;
    candidate.phases.backend_prepare_ns = 80.0;
    candidate.phases.static_value_pack_ns = 40.0;
    candidate.phases.h2d_ns = 11.0;
    candidate.phases.dynamic_input_pack_ns = 12.0;
    candidate.phases.kernel_ns = 17.0;
    candidate.phases.epilogue_ns = 7.0;
    candidate.phases.order_transform_ns = 19.0;
    candidate.phases.synchronization_ns = 23.0;
    candidate.phases.communication_ns = 29.0;
    candidate.phases.d2h_ns = 31.0;
    candidate.phases.h2d_bytes = 101u;
    candidate.phases.communication_bytes = 102u;
    candidate.phases.d2h_bytes = 103u;
    candidate.phases.persistent_bytes = 104u;
    candidate.phases.transient_bytes = 105u;
    candidate.reuse = {32u, 16u, 4u};
    planner::total_cost total{};
    require(inputs::account_candidate_phases(candidate, &total)
            == inputs::live_input_status::ok,
        "complete candidate phases were rejected");
    const double expected = 13.0 + 320.0 / 32.0
        + (160.0 + 80.0) / 16.0 + 40.0 / 4.0
        + 11.0 + 12.0 + 17.0 + 7.0 + 19.0 + 23.0 + 29.0 + 31.0;
    require(std::fabs(total.amortized_total_ns - expected) < 1.0e-12
            && total.phases.persistent_bytes == 104u
            && total.phases.transient_bytes == 105u,
        "complete phase cost or memory accounting was lost");
    require(!inputs::authoritative_for_promotion(candidate),
        "analytical shortlist input became authoritative evidence");
    candidate.measured = true;
    require(inputs::authoritative_for_promotion(candidate),
        "measured candidate phases were not marked authoritative");
    candidate.phases.kernel_ns =
        std::numeric_limits<double>::quiet_NaN();
    require(inputs::account_candidate_phases(candidate, &total)
            == inputs::live_input_status::invalid_cost,
        "invalid phase cost was accepted");
}

void test_invalid_inputs() {
    inputs::quantitative_relation_input invalid = relation(generation_1, 1u);
    invalid.source_indices = nullptr;
    inputs::live_planner_input output{};
    require(inputs::derive_live_planner_input(invalid, {81u, 82u},
                {1u, 7u, 0u, 700u}, {100u, 200u, 300u, 400u},
                {32u, 16u, 4u}, 1u, 2u, 3u, 4u, &output)
            == inputs::live_input_status::invalid_support,
        "invalid support was accepted");
    invalid = relation(generation_1, 0u);
    require(inputs::derive_live_planner_input(invalid, {81u, 82u},
                {1u, 7u, 0u, 700u}, {100u, 200u, 300u, 400u},
                {32u, 16u, 4u}, 1u, 2u, 3u, 4u, &output)
            == inputs::live_input_status::invalid_values,
        "invalid value generation was accepted");
}

} // namespace

int main() {
    test_statistics_and_exact_keys();
    test_generation_and_pointer_are_runtime_state();
    test_complete_phase_accounting_and_empirical_authority();
    test_invalid_inputs();
    std::cout << "ce_live_planner_features_test passed\n";
    return 0;
}
