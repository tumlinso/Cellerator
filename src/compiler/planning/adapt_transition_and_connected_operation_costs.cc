#include <Cellerator/compiler/planning/adapt_transition_and_connected_operation_costs_v1.hh>

#include <cmath>

namespace Cellerator::compiler::planning {
namespace {

bool stable(cellerator::execution::order_id value) noexcept {
    return value.high != 0u || value.low != 0u;
}

}  // namespace

connected_operation_cost_result_v1 adapt_transition_and_connected_operation_costs_v1(
    const connected_operation_transition_cost_v1& cost) noexcept {
    connected_operation_cost_result_v1 result{};
    constexpr std::uint32_t all_effects = connected_order_transform_v1 |
        connected_materialization_v1 | connected_shared_traversal_v1 |
        connected_fusion_v1 | connected_common_output_ownership_v1 |
        connected_canonicalization_v1 | connected_field_boundary_v1;
    if (cost.effects == 0u || (cost.effects & ~all_effects) != 0u) return result;
    if ((cost.effects & connected_order_transform_v1) != 0u &&
        (!stable(cost.source_order) || !stable(cost.destination_order))) {
        result.code = connected_operation_cost_code_v1::invalid_order;
        return result;
    }
    const double values[] = {cost.order_transform_nanoseconds,
        cost.materialization_nanoseconds, cost.shared_traversal_savings_nanoseconds,
        cost.fusion_savings_nanoseconds,
        cost.common_output_ownership_savings_nanoseconds,
        cost.canonicalization_nanoseconds, cost.field_boundary_nanoseconds};
    for (const auto value : values) {
        if (!std::isfinite(value) || value < 0.0) {
            result.code = connected_operation_cost_code_v1::invalid_cost;
            return result;
        }
    }
    auto included = [&](std::uint32_t effect, double value) {
        return (cost.effects & effect) != 0u ? value : 0.0;
    };
    result.gross_nanoseconds =
        included(connected_order_transform_v1, cost.order_transform_nanoseconds) +
        included(connected_materialization_v1, cost.materialization_nanoseconds) +
        included(connected_canonicalization_v1, cost.canonicalization_nanoseconds) +
        included(connected_field_boundary_v1, cost.field_boundary_nanoseconds);
    result.savings_nanoseconds =
        included(connected_shared_traversal_v1, cost.shared_traversal_savings_nanoseconds) +
        included(connected_fusion_v1, cost.fusion_savings_nanoseconds) +
        included(connected_common_output_ownership_v1,
                 cost.common_output_ownership_savings_nanoseconds);
    if (!std::isfinite(result.gross_nanoseconds) ||
        !std::isfinite(result.savings_nanoseconds)) {
        result.code = connected_operation_cost_code_v1::invalid_cost;
        return result;
    }
    if (result.savings_nanoseconds > result.gross_nanoseconds) {
        result.code = connected_operation_cost_code_v1::excessive_savings;
        return result;
    }
    result.complete_nanoseconds = result.gross_nanoseconds - result.savings_nanoseconds;
    result.transition.source_order = cost.source_order;
    result.transition.destination_order = cost.destination_order;
    result.transition.transform_ns = result.gross_nanoseconds;
    result.transition.fusion_savings_ns = result.savings_nanoseconds;
    result.transition.transient_bytes = cost.transient_bytes;
    result.transition.fused = (cost.effects & connected_fusion_v1) != 0u;
    result.code = connected_operation_cost_code_v1::ok;
    return result;
}

}  // namespace Cellerator::compiler::planning
