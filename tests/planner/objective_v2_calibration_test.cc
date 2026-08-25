#include <Cellerator/planner/objective_v2_calibration.hh>

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace planner = cellerator::planner;
namespace execution = cellerator::execution;

namespace {

enum class mechanism { row_masked, csr, feature_major };

constexpr std::uint64_t rows = 65536u;
constexpr std::uint64_t features = 32768u;
constexpr std::uint64_t edges = 2097152u;
constexpr std::uint64_t value_bytes = 4194304u;

planner::planning_keys keys(
    const planner::objective_v2_calibration &model,
    std::uint32_t sharing,
    std::uint64_t reuse) {
    planner::planning_keys result{};
    result.problem.identity = {0x7701u, 0x7702u};
    result.structures.count = 1u;
    result.structures.structures[0] = {
        {0x7710u, sharing}, execution::structure_epoch{1u}};
    result.geometry.source_domain = {0x7720u, 1u};
    result.geometry.destination_domain = {0x7720u, 2u};
    result.geometry.geometry = {0x7720u, 3u};
    result.geometry.source_order = {0x7720u, 4u};
    result.geometry.destination_order = {0x7720u, 5u};
    result.geometry.partition = {0x7720u, 6u};
    result.device = model.device;
    result.build = model.build;
    result.policy.structure_reuse = reuse;
    result.policy.projection_reuse = reuse;
    result.policy.value_reuse = reuse;
    result.policy.numeric_policy = 1u;
    result.policy.output_order_policy = 1u;
    return result;
}

std::uint64_t row_metadata(std::uint32_t sharing) {
    if (sharing == 1u) return 1507344u;
    if (sharing == 8u) return 1851408u;
    return 3031056u;
}

std::uint64_t feature_metadata(std::uint32_t sharing) {
    if (sharing == 1u) return 9183552u;
    if (sharing == 8u) return 14688576u;
    return 33562944u;
}

planner::objective_v2_calibration_query query(
    const planner::objective_v2_calibration &model,
    mechanism kind,
    std::uint32_t sharing,
    std::uint32_t dense_width,
    std::uint64_t reuse,
    bool include_preparation) {
    planner::objective_v2_calibration_query result{};
    result.keys = keys(model, sharing, reuse);
    auto &work = result.statistics;
    work.active_rows = rows;
    work.active_features = features;
    work.logical_edges = edges;
    work.useful_interactions = edges * dense_width;
    work.dense_width = dense_width;
    work.value_bytes = value_bytes;
    if (kind == mechanism::row_masked) {
        work.masked_row_lane_slots = (rows / 32u) * 2u * sharing
            * 32u * 16u * dense_width;
        work.launch_count = dense_width;
        work.dynamic_input_pack_bytes = features * dense_width * 4u;
        work.output_order_bytes = rows * dense_width * 8u;
        work.projection_bytes = row_metadata(sharing);
        if (include_preparation) work.backend_prepare_ns = 2200.0;
    } else if (kind == mechanism::csr) {
        work.linear_edge_visits = edges * dense_width;
        work.launch_count = dense_width;
        work.dynamic_input_pack_bytes = features * dense_width * 4u;
        work.output_order_bytes = rows * dense_width * 8u;
        work.projection_bytes = 8650756u;
        if (include_preparation) {
            const double projection_ms = sharing == 1u ? 20.133008
                : (sharing == 8u ? 21.626235 : 28.697232);
            work.projection_construction_ns = projection_ms * 1.0e6;
            work.backend_prepare_ns = 1400.0;
        }
    } else {
        work.masked_feature_lane_slots = rows * sharing * 32u;
        work.dense_rhs_vector_elements = rows * sharing * dense_width;
        work.feature_value_loads = edges;
        work.launch_count = 1u;
        work.projection_bytes = feature_metadata(sharing);
        if (include_preparation) {
            const double projection_ms = sharing == 1u ? 16.576505
                : (sharing == 8u ? 47.328850 : 174.711510);
            const double value_pack_ms = sharing == 1u ? 5.870170
                : (sharing == 8u ? 5.905285 : 5.858388);
            work.projection_construction_ns = projection_ms * 1.0e6;
            work.static_value_pack_ns = value_pack_ms * 1.0e6;
            work.backend_prepare_ns = 1800.0;
        }
    }
    return result;
}

planner::objective_v2_prediction predict(
    const planner::objective_v2_calibration &model,
    const planner::objective_v2_calibration_query &request) {
    planner::objective_v2_prediction result{};
    assert(planner::evaluate_calibrated_objective_v2(model, request, &result));
    return result;
}

mechanism winner(
    const planner::objective_v2_calibration &model,
    std::uint32_t sharing,
    std::uint32_t dense_width,
    std::uint64_t reuse,
    bool include_preparation) {
    const mechanism candidates[] = {mechanism::row_masked,
        mechanism::csr, mechanism::feature_major};
    mechanism best = candidates[0];
    double best_cost = INFINITY;
    for (mechanism candidate : candidates) {
        const auto result = predict(model, query(model, candidate, sharing,
            dense_width, reuse, include_preparation));
        assert(result.state == planner::objective_v2_prediction_state::calibrated);
        assert(result.empirical_measurement_required);
        if (result.predicted.amortized_total_ns < best_cost) {
            best = candidate;
            best_cost = result.predicted.amortized_total_ns;
        }
    }
    return best;
}

void test_measured_regime_predictions() {
    const auto model = planner::ce_arch_76_v100_objective_v2_calibration();
    const std::uint32_t widths[] = {1u, 2u, 4u, 8u, 16u};
    const mechanism high[] = {mechanism::row_masked,
        mechanism::feature_major, mechanism::feature_major,
        mechanism::feature_major, mechanism::feature_major};
    const mechanism medium[] = {mechanism::row_masked,
        mechanism::row_masked, mechanism::row_masked,
        mechanism::feature_major, mechanism::feature_major};
    const mechanism low[] = {mechanism::csr, mechanism::csr, mechanism::csr,
        mechanism::feature_major, mechanism::feature_major};
    bool observed_row = false, observed_csr = false, observed_feature = false;
    for (std::size_t index = 0u; index < 5u; ++index) {
        const mechanism winners[] = {
            winner(model, 1u, widths[index], 1u, false),
            winner(model, 8u, widths[index], 1u, false),
            winner(model, 32u, widths[index], 1u, false)};
        assert(winners[0] == high[index]);
        assert(winners[1] == medium[index]);
        assert(winners[2] == low[index]);
        for (mechanism value : winners) {
            observed_row |= value == mechanism::row_masked;
            observed_csr |= value == mechanism::csr;
            observed_feature |= value == mechanism::feature_major;
        }
        assert(winner(model, 1u, widths[index], 8u, true)
            == mechanism::row_masked);
        assert(winner(model, 8u, widths[index], 8u, true)
            == mechanism::row_masked);
        assert(winner(model, 32u, widths[index], 8u, true)
            == mechanism::row_masked);
    }
    assert(observed_row && observed_csr && observed_feature);
}

void test_determinism_identity_invalidation_and_fallback() {
    const auto model = planner::ce_arch_76_v100_objective_v2_calibration();
    auto request = query(model, mechanism::feature_major, 8u, 8u, 8u, true);
    const auto first = predict(model, request);
    const auto second = predict(model, request);
    assert(first.model_identity == second.model_identity
        && first.predicted.amortized_total_ns
            == second.predicted.amortized_total_ns
        && first.confidence == second.confidence);

    planner::planner_candidate candidate{};
    candidate.flags = 0xa5u;
    assert(planner::apply_objective_v2_prediction(
        first, request.keys, &candidate));
    assert(candidate.analytical.kernel_ns > 0.0
        && candidate.analytical.projection_construction_ns > 0.0
        && (candidate.flags & ~planner::planner_candidate_empirical_required)
            == 0xa5u
        && (candidate.flags & planner::planner_candidate_empirical_required)
            != 0u);

    planner::planning_keys changed_structure = first.keys;
    changed_structure.structures.structures[0].identity.low ^= 1u;
    assert(!planner::same_planning_keys(first.keys, changed_structure));
    assert(!planner::apply_objective_v2_prediction(
        first, changed_structure, &candidate));

    request.keys.build.kernel_build ^= 1u;
    const auto stale = predict(model, request);
    assert(stale.state == planner::objective_v2_prediction_state::stale_identity
        && stale.empirical_measurement_required);

    request = query(model, mechanism::feature_major, 8u, 3u, 8u, false);
    const auto novel = predict(model, request);
    assert(novel.state == planner::objective_v2_prediction_state::novel_regime
        && novel.empirical_measurement_required);
    assert(!planner::apply_objective_v2_prediction(
        novel, request.keys, &candidate));
}

void test_refinement_guidance_uses_measured_total_cost_units() {
    const auto model = planner::ce_arch_76_v100_objective_v2_calibration();
    cellpack::alternating_refinement_objective_weights weights{};
    assert(planner::make_objective_v2_refinement_weights(
        model, 8u, &weights));
    assert(weights.encoded_bytes == 0.0
        && weights.metadata_bytes == 0.0
        && weights.active_block_references == 0.0
        && weights.runtime_mean_nanoseconds == 1.0
        && weights.preprocessing_mean_nanoseconds == 0.125);
}

} // namespace

int main() {
    test_measured_regime_predictions();
    test_determinism_identity_invalidation_and_fallback();
    test_refinement_guidance_uses_measured_total_cost_units();
    return 0;
}
