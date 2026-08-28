#include <Cellerator/planner/end_to_end_planner.hh>

#include <cmath>

namespace cellerator::planner {
namespace {

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

double numeric_width(execution::numeric_type type) noexcept {
    switch (type) {
    case execution::numeric_type::bit: return 0.125;
    case execution::numeric_type::u8: return 1.0;
    case execution::numeric_type::u16:
    case execution::numeric_type::f16:
    case execution::numeric_type::bf16: return 2.0;
    case execution::numeric_type::u32:
    case execution::numeric_type::i32:
    case execution::numeric_type::f32: return 4.0;
    case execution::numeric_type::f64: return 8.0;
    case execution::numeric_type::invalid: return 0.0;
    }
    return 0.0;
}

} // namespace

planner_status evaluate_objective_v2(
    const objective_v2_statistics &statistics,
    const objective_v2_context &context,
    const objective_v2_weights &weights,
    objective_v2_result *out) noexcept {
    if (out == nullptr || statistics.useful_edges == 0u
        || context.dense_width == 0u || context.expected_reuse == 0u
        || context.value_type == execution::numeric_type::invalid)
        return {planner_status_code::invalid_argument,
            "objective v2 requires work, context, and output"};
    *out = objective_v2_result{};
    const double values[] = {statistics.feature_reuse,
        statistics.row_imbalance,
        statistics.module_overlap,
        statistics.module_activation_frequency,
        statistics.transpose_locality,
        statistics.quantization_outlier_fraction,
        weights.byte_cost,
        weights.partial_block_cost,
        weights.imbalance_cost,
        weights.register_pressure_cost,
        weights.shared_pressure_cost,
        weights.order_transform_cost,
        weights.transpose_cost,
        weights.communication_cost,
        weights.quantization_cost,
        weights.reuse_credit,
        weights.epilogue_credit,
        weights.module_credit};
    for (double value : values)
        if (!finite_nonnegative(value))
            return {planner_status_code::invalid_argument,
                "objective v2 inputs must be finite and nonnegative"};

    const double useful_edges = static_cast<double>(statistics.useful_edges);
    const double reuse = static_cast<double>(context.expected_reuse);
    out->storage = weights.byte_cost
        * static_cast<double>(statistics.metadata_bytes + statistics.value_bytes)
        / reuse;
    out->execution = weights.partial_block_cost
            * static_cast<double>(statistics.partial_block_slots)
            * static_cast<double>(context.dense_width)
            * numeric_width(context.value_type)
        + weights.imbalance_cost * statistics.row_imbalance * useful_edges
        + weights.register_pressure_cost
            * static_cast<double>(context.registers_per_thread)
        + weights.shared_pressure_cost
            * static_cast<double>(context.shared_bytes_per_block);
    out->order_and_transpose =
        (context.canonical_output_required
                ? weights.order_transform_cost * useful_edges : 0.0)
        + (context.transpose_required
                ? weights.transpose_cost
                    * (1.0 + (1.0 - statistics.transpose_locality))
                    * useful_edges
                : 0.0);
    out->communication = weights.communication_cost
        * static_cast<double>(statistics.cross_partition_edges);
    out->quantization = context.quantized
        ? weights.quantization_cost
            * statistics.quantization_outlier_fraction * useful_edges
        : 0.0;
    out->credits = weights.reuse_credit * statistics.feature_reuse * useful_edges
        + weights.module_credit * statistics.module_overlap
            * statistics.module_activation_frequency * useful_edges
        + (context.epilogue_fused
                ? weights.epilogue_credit * useful_edges : 0.0);
    out->score = out->storage + out->execution + out->order_and_transpose
        + out->communication + out->quantization - out->credits;
    if (!std::isfinite(out->score))
        return {planner_status_code::invalid_cost,
            "objective v2 score overflowed"};
    return {};
}

} // namespace cellerator::planner
