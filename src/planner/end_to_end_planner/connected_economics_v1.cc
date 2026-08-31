#include "Cellerator/planner/portfolio/connected_economics_v1.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cellerator::planner::portfolio {
namespace {

economics_status_v1 failure(
    economics_status_code_v1 code,
    std::uint64_t subject) noexcept {
    return {code, subject};
}

bool valid_id(operation_core::stable_id id) noexcept {
    return id.low != 0u || id.high != 0u;
}

bool finite_nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0.0;
}

bool add_bytes(std::uint64_t value, std::uint64_t *total) noexcept {
    if (value > std::numeric_limits<std::uint64_t>::max() - *total) {
        return false;
    }
    *total += value;
    return true;
}

std::uint64_t phase_launches(const phase_costs &phases) noexcept {
    std::uint64_t launches = phases.kernel_ns > 0.0 ? 1u : 0u;
    launches += phases.dynamic_input_pack_ns > 0.0 ? 1u : 0u;
    launches += phases.static_value_pack_ns > 0.0 ? 1u : 0u;
    launches += phases.epilogue_ns > 0.0 ? 1u : 0u;
    launches += phases.order_transform_ns > 0.0 ? 1u : 0u;
    return launches;
}

}  // namespace

economics_status_v1 compute_connected_economics_v1(
    const connected_program_economics_v1 &program,
    connected_economics_result_v1 *result) noexcept {
    if (result == nullptr || program.operations == nullptr
        || program.operation_count == 0u
        || program.transition_count == std::numeric_limits<std::uint64_t>::max()
        || program.transition_count + 1u != program.operation_count
        || (program.transition_count != 0u && program.transitions == nullptr)
        || !execution::valid_identity(program.canonical_output_order)
        || !finite_nonnegative(program.final_canonicalization_ns)) {
        return failure(economics_status_code_v1::invalid_argument, 0u);
    }
    connected_economics_result_v1 accumulated{};
    for (std::uint64_t index = 0u; index < program.operation_count; ++index) {
        const operation_economics_v1 &operation = program.operations[index];
        if (!valid_id(operation.candidate)
            || !execution::valid_identity(operation.input_order)
            || !execution::valid_identity(operation.output_order)
            || (operation.value_mode != planner_value_mode_v1::logical_primary
                && operation.value_mode
                    != planner_value_mode_v1::projection_primary)
            || operation.frequency == 0u || operation.repetitions == 0u
            || operation.frequency
                > std::numeric_limits<std::uint64_t>::max()
                    / operation.repetitions) {
            return failure(economics_status_code_v1::invalid_argument, index);
        }
        total_cost amortized{};
        const planner_status cost_status = compute_total_cost(operation.phases,
            operation.structure_reuse, operation.projection_reuse,
            operation.value_reuse, &amortized);
        if (!cost_status) {
            return failure(economics_status_code_v1::invalid_cost, index);
        }
        const double executions = static_cast<double>(operation.frequency)
            * static_cast<double>(operation.repetitions);
        const double operation_cost = amortized.amortized_total_ns * executions;
        if (!finite_nonnegative(operation_cost)
            || !finite_nonnegative(accumulated.operation_cost_ns
                + operation_cost)) {
            return failure(economics_status_code_v1::invalid_cost, index);
        }
        accumulated.operation_cost_ns += operation_cost;
        accumulated.value_pack_cost_ns +=
            operation.phases.static_value_pack_ns
            * executions / static_cast<double>(operation.value_reuse);
        if (!finite_nonnegative(accumulated.value_pack_cost_ns)) {
            return failure(economics_status_code_v1::invalid_cost, index);
        }
        if (!add_bytes(operation.phases.persistent_bytes,
                &accumulated.persistent_bytes)) {
            return failure(economics_status_code_v1::arithmetic_overflow, index);
        }
        accumulated.peak_transient_bytes = std::max(
            accumulated.peak_transient_bytes,
            operation.phases.transient_bytes);
        const std::uint64_t launches = phase_launches(operation.phases);
        const std::uint64_t execution_count =
            operation.frequency * operation.repetitions;
        if (launches != 0u
            && execution_count > std::numeric_limits<std::uint64_t>::max()
                / launches) {
            return failure(economics_status_code_v1::arithmetic_overflow, index);
        }
        if (!add_bytes(launches * execution_count,
                &accumulated.launch_count)) {
            return failure(economics_status_code_v1::arithmetic_overflow, index);
        }
        if (index + 1u < program.operation_count) {
            const layout_transition_economics_v1 &transition =
                program.transitions[index];
            const operation_economics_v1 &next = program.operations[index + 1u];
            if (!execution::same_identity(transition.source_order,
                    operation.output_order)
                || !execution::same_identity(transition.destination_order,
                    next.input_order)
                || !finite_nonnegative(transition.transform_ns)
                || !finite_nonnegative(transition.fusion_savings_ns)
                || (execution::same_identity(operation.output_order,
                        next.input_order)
                    && transition.transform_ns != 0.0)
                || (!execution::same_identity(operation.output_order,
                        next.input_order)
                    && transition.transform_ns == 0.0)
                || (!transition.fused && transition.fusion_savings_ns != 0.0)) {
                return failure(economics_status_code_v1::invalid_order, index);
            }
            accumulated.layout_cost_ns += transition.transform_ns;
            accumulated.fusion_savings_ns += transition.fusion_savings_ns;
            accumulated.peak_transient_bytes = std::max(
                accumulated.peak_transient_bytes, transition.transient_bytes);
        }
    }
    const operation_economics_v1 &last =
        program.operations[program.operation_count - 1u];
    if (last.canonical_output_required
        && !execution::same_identity(last.output_order,
            program.canonical_output_order)) {
        if (program.final_canonicalization_ns == 0.0) {
            return failure(economics_status_code_v1::invalid_order,
                program.operation_count - 1u);
        }
        accumulated.layout_cost_ns += program.final_canonicalization_ns;
        accumulated.peak_transient_bytes = std::max(
            accumulated.peak_transient_bytes,
            program.final_canonicalization_bytes);
    } else if (program.final_canonicalization_ns != 0.0) {
        return failure(economics_status_code_v1::invalid_order,
            program.operation_count);
    }
    const double before_savings = accumulated.operation_cost_ns
        + accumulated.layout_cost_ns;
    if (!finite_nonnegative(before_savings)
        || accumulated.fusion_savings_ns > before_savings) {
        return failure(economics_status_code_v1::invalid_cost,
            program.operation_count);
    }
    accumulated.complete_cost_ns = before_savings
        - accumulated.fusion_savings_ns;
    *result = accumulated;
    return {};
}

}  // namespace cellerator::planner::portfolio
