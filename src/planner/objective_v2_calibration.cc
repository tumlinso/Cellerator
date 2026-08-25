#include <Cellerator/planner/objective_v2_calibration.hh>

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::planner {
namespace {

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool same_device(
    const device_performance_key &lhs,
    const device_performance_key &rhs) noexcept {
    return lhs.vendor == rhs.vendor
        && lhs.architecture_major == rhs.architecture_major
        && lhs.architecture_minor == rhs.architecture_minor
        && lhs.performance_class == rhs.performance_class;
}

bool same_build(
    const runtime_build_key &lhs,
    const runtime_build_key &rhs) noexcept {
    return lhs.runtime == rhs.runtime
        && lhs.kernel_build == rhs.kernel_build
        && lhs.driver == rhs.driver
        && lhs.library == rhs.library;
}

bool valid_structure_dependencies(
    const persistent_structure_set_key &structures) noexcept {
    if (structures.count == 0u
        || structures.count > execution::maximum_operation_structures)
        return false;
    for (std::uint32_t index = 0u; index < structures.count; ++index)
        if (!execution::valid_identity(structures.structures[index].identity)
            || structures.structures[index].epoch.value == 0u)
            return false;
    return true;
}

bool valid_calibration(const objective_v2_calibration &value) noexcept {
    const auto &c = value.coefficients;
    const double numbers[] = {value.median_relative_error_percent,
        value.maximum_relative_error_percent,
        value.maximum_training_spread_percent, c.intercept_ns,
        c.useful_interaction_ns, c.masked_row_lane_slot_ns,
        c.linear_edge_visit_ns, c.masked_feature_lane_slot_ns,
        c.dense_rhs_vector_element_ns, c.feature_value_load_ns,
        c.launch_ns, c.input_pack_byte_ns, c.output_order_byte_ns};
    if (value.schema_version != objective_v2_calibration_schema_version
        || value.model_identity == 0u || value.evidence_revision == 0u
        || value.device.vendor == 0u || value.device.performance_class == 0u
        || value.build.runtime == 0u || value.build.kernel_build == 0u
        || value.build.driver == 0u || value.build.library == 0u
        || value.trained_rows == 0u || value.trained_features == 0u
        || value.trained_edges == 0u
        || value.supported_dense_width_mask == 0u
        || value.sample_count == 0u || value.maximum_launch_count == 0u)
        return false;
    for (double number : numbers)
        if (!finite_nonnegative(number)) return false;
    return true;
}

bool valid_statistics(
    const objective_v2_mechanism_statistics &value) noexcept {
    if (value.active_rows == 0u || value.active_features == 0u
        || value.logical_edges == 0u || value.dense_width == 0u
        || value.launch_count == 0u
        || value.logical_edges > std::numeric_limits<std::uint64_t>::max()
            / value.dense_width
        || value.useful_interactions
            != value.logical_edges * value.dense_width)
        return false;
    return finite_nonnegative(value.projection_construction_ns)
        && finite_nonnegative(value.backend_prepare_ns)
        && finite_nonnegative(value.static_value_pack_ns);
}

bool within_support(
    const objective_v2_calibration &model,
    const objective_v2_mechanism_statistics &work) noexcept {
    if (work.dense_width >= 32u) return false;
    const std::uint32_t width_bit = 1u << work.dense_width;
    return work.active_rows == model.trained_rows
        && work.active_features == model.trained_features
        && work.logical_edges == model.trained_edges
        && (model.supported_dense_width_mask & width_bit) != 0u
        && work.masked_row_lane_slots
            <= model.maximum_masked_row_lane_slots
        && work.linear_edge_visits <= model.maximum_linear_edge_visits
        && work.masked_feature_lane_slots
            <= model.maximum_masked_feature_lane_slots
        && work.dense_rhs_vector_elements
            <= model.maximum_dense_rhs_vector_elements
        && work.feature_value_loads <= model.maximum_feature_value_loads
        && work.launch_count <= model.maximum_launch_count;
}

double term(double coefficient, std::uint64_t count) noexcept {
    return coefficient * static_cast<double>(count);
}

} // namespace

objective_v2_calibration ce_arch_76_v100_objective_v2_calibration() noexcept {
    objective_v2_calibration result;
    result.model_identity = 0x4345373656314e4eull; // CE76V1NN
    result.evidence_revision = 0x08202bdau;
    result.device = {0x10deu, 7u, 0u, 0x5631303053584d32ull};
    result.build = {12090u, 0xdf9d168203125667ull, 13000u,
        0x43454c4c45524154ull};
    result.trained_rows = 65536u;
    result.trained_features = 32768u;
    result.trained_edges = 2097152u;
    result.supported_dense_width_mask = 0x10116u; // N=1,2,4,8,16
    result.sample_count = 45u;
    result.maximum_masked_row_lane_slots = 1073741824ull;
    result.maximum_linear_edge_visits = 33554432u;
    result.maximum_masked_feature_lane_slots = 67108864u;
    result.maximum_dense_rhs_vector_elements = 33554432u;
    result.maximum_feature_value_loads = 2097152u;
    result.maximum_launch_count = 16u;
    result.median_relative_error_percent = 4.804931058684;
    result.maximum_relative_error_percent = 35.924268908424;
    result.maximum_training_spread_percent = 1.3157804;
    // Nonnegative least squares over the 45 CE-ARCH-76 median totals. Counts
    // are mechanism work, never stable candidate identity. Units are ns/item.
    result.coefficients = {13333.33176944439, 0.0014295764809420951,
        0.0013570532524714409, 0.03178201444844455,
        0.006832512665012897, 0.0048355396139240165,
        0.00036790767729590431, 10867.939465551794,
        0.011718750419616699, 0.003417968797683716};
    return result;
}

planner_status evaluate_calibrated_objective_v2(
    const objective_v2_calibration &calibration,
    const objective_v2_calibration_query &query,
    objective_v2_prediction *out) noexcept {
    if (out == nullptr || !valid_calibration(calibration)
        || !valid_statistics(query.statistics)
        || !valid_structure_dependencies(query.keys.structures)
        || query.keys.policy.structure_reuse == 0u
        || query.keys.policy.projection_reuse == 0u
        || query.keys.policy.value_reuse == 0u
        || !finite_nonnegative(query.practical_tolerance_percent))
        return {planner_status_code::invalid_argument,
            "objective v2 calibration query is invalid"};
    *out = objective_v2_prediction{};
    out->keys = query.keys;
    out->model_identity = calibration.model_identity;
    out->evidence_revision = calibration.evidence_revision;
    out->relative_error_bound_percent =
        calibration.maximum_relative_error_percent;
    if (!same_device(query.keys.device, calibration.device)
        || !same_build(query.keys.build, calibration.build)) {
        out->state = objective_v2_prediction_state::stale_identity;
        out->reason = "objective v2 device or build identity is outside calibration";
        return {};
    }
    if (!within_support(calibration, query.statistics)) {
        out->state = objective_v2_prediction_state::novel_regime;
        out->reason = "objective v2 structural or N regime needs measurement";
        return {};
    }

    const auto &work = query.statistics;
    const auto &c = calibration.coefficients;
    const double predicted_steady = c.intercept_ns
        + term(c.useful_interaction_ns, work.useful_interactions)
        + term(c.masked_row_lane_slot_ns, work.masked_row_lane_slots)
        + term(c.linear_edge_visit_ns, work.linear_edge_visits)
        + term(c.masked_feature_lane_slot_ns,
            work.masked_feature_lane_slots)
        + term(c.dense_rhs_vector_element_ns,
            work.dense_rhs_vector_elements)
        + term(c.feature_value_load_ns, work.feature_value_loads)
        + c.launch_ns * static_cast<double>(work.launch_count);
    const double input_pack = term(c.input_pack_byte_ns,
        work.dynamic_input_pack_bytes);
    const double output_order = term(c.output_order_byte_ns,
        work.output_order_bytes);
    if (!finite_nonnegative(predicted_steady)
        || !finite_nonnegative(input_pack)
        || !finite_nonnegative(output_order))
        return {planner_status_code::invalid_cost,
            "objective v2 calibrated estimate overflowed"};
    phase_costs phases{};
    phases.projection_construction_ns = work.projection_construction_ns;
    phases.backend_prepare_ns = work.backend_prepare_ns;
    phases.static_value_pack_ns = work.static_value_pack_ns;
    phases.dynamic_input_pack_ns = input_pack;
    phases.order_transform_ns = output_order;
    phases.kernel_ns = std::max(0.0,
        predicted_steady - input_pack - output_order);
    if (work.projection_bytes
            > std::numeric_limits<std::uint64_t>::max() - work.value_bytes)
        return {planner_status_code::invalid_cost,
            "objective v2 persistent byte count overflowed"};
    phases.persistent_bytes = work.projection_bytes + work.value_bytes;
    phases.transient_bytes = work.transient_bytes;
    const planner_status status = compute_total_cost(phases,
        query.keys.policy.structure_reuse,
        query.keys.policy.projection_reuse,
        query.keys.policy.value_reuse, &out->predicted);
    if (!status) return status;
    out->state = objective_v2_prediction_state::calibrated;
    out->confidence = std::max(0.0,
        1.0 - calibration.maximum_relative_error_percent / 100.0);
    out->empirical_measurement_required =
        calibration.maximum_relative_error_percent
            > query.practical_tolerance_percent
        || calibration.maximum_training_spread_percent
            > query.practical_tolerance_percent;
    out->reason = out->empirical_measurement_required
        ? "calibrated estimate ranks candidates; empirical measurement remains final"
        : "calibrated estimate is within declared evidence tolerance";
    return {};
}

planner_status apply_objective_v2_prediction(
    const objective_v2_prediction &prediction,
    const planning_keys &current_keys,
    planner_candidate *candidate) noexcept {
    if (candidate == nullptr
        || prediction.schema_version != objective_v2_calibration_schema_version
        || prediction.state != objective_v2_prediction_state::calibrated
        || prediction.model_identity == 0u
        || prediction.evidence_revision == 0u
        || !same_planning_keys(prediction.keys, current_keys))
        return {planner_status_code::invalid_argument,
            "planner candidate requires an applicable current-key objective v2 prediction"};
    candidate->analytical = prediction.predicted.phases;
    candidate->flags &= ~planner_candidate_empirical_required;
    if (prediction.empirical_measurement_required)
        candidate->flags |= planner_candidate_empirical_required;
    return {};
}

planner_status make_objective_v2_refinement_weights(
    const objective_v2_calibration &calibration,
    std::uint64_t expected_reuse,
    cellpack::alternating_refinement_objective_weights *out) noexcept {
    if (out == nullptr || expected_reuse == 0u
        || !valid_calibration(calibration))
        return {planner_status_code::invalid_argument,
            "objective v2 refinement guidance requires calibration and reuse"};
    *out = {};
    out->encoded_bytes = 0.0;
    out->runtime_mean_nanoseconds = 1.0;
    out->preprocessing_mean_nanoseconds =
        1.0 / static_cast<double>(expected_reuse);
    return {};
}

} // namespace cellerator::planner
