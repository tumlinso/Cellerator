#include <Cellerator/compiler/planning/adapt_transition_and_connected_operation_costs_v1.hh>

#include <cassert>
#include <cmath>

namespace planning = Cellerator::compiler::planning;

int main() {
    planning::connected_operation_transition_cost_v1 cost{};
    cost.source_order = {1u, 2u};
    cost.destination_order = {3u, 4u};
    cost.effects = planning::connected_order_transform_v1 |
        planning::connected_materialization_v1 |
        planning::connected_shared_traversal_v1 |
        planning::connected_fusion_v1 |
        planning::connected_common_output_ownership_v1 |
        planning::connected_canonicalization_v1 |
        planning::connected_field_boundary_v1;
    cost.order_transform_nanoseconds = 100.0;
    cost.materialization_nanoseconds = 80.0;
    cost.shared_traversal_savings_nanoseconds = 20.0;
    cost.fusion_savings_nanoseconds = 30.0;
    cost.common_output_ownership_savings_nanoseconds = 10.0;
    cost.canonicalization_nanoseconds = 40.0;
    cost.field_boundary_nanoseconds = 20.0;
    cost.transient_bytes = 4096u;
    const auto result = planning::adapt_transition_and_connected_operation_costs_v1(cost);
    assert(result);
    assert(std::abs(result.gross_nanoseconds - 240.0) < 1.0e-12);
    assert(std::abs(result.savings_nanoseconds - 60.0) < 1.0e-12);
    assert(std::abs(result.complete_nanoseconds - 180.0) < 1.0e-12);
    assert(result.transition.fused && result.transition.transient_bytes == 4096u);

    cost.fusion_savings_nanoseconds = 300.0;
    assert(planning::adapt_transition_and_connected_operation_costs_v1(cost).code ==
        planning::connected_operation_cost_code_v1::excessive_savings);
}
