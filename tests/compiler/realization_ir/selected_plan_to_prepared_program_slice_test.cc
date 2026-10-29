#include <Cellerator/compiler/ir/realization/realization_ir_v1.hh>

#include <cassert>
#include <string>

using namespace cellerator::compiler::ir::planning::v1;
using namespace cellerator::compiler::ir::realization::v1;

int main() {
    decision_record_v1 decisions[]{
        {{1u, 1u}, {10u, 1u}, {20u, 1u}, decision_state_v1::rejected},
        {{1u, 2u}, {11u, 1u}, {20u, 1u}, decision_state_v1::selected, 0u, 0u,
            decision_flag_correct_v1, 7u}};
    planning_ir_module_v1 planning{1u, 0u, {30u, 1u}, decisions, 2u, 0u};
    selected_relation_plan_v1 input{&planning, {2u, 1u}, {2u, 2u}, {2u, 3u},
        {2u, 4u}, 9u, 5u, {0u, 2u, 3u}, {0u, 2u, 1u}};
    selected_plan_delivery_status_v1 status{};
    std::string error;
    auto slice = lower_selected_relation_plan_v1(input, &status, &error);
    assert(slice && status == selected_plan_delivery_status_v1::success && error.empty());
    assert(slice->trace.source_operation == stable_identity_v1({1u, 20u}));
    assert(slice->trace.selected_candidate == stable_identity_v1({1u, 11u}));
    assert(slice->serialized_ir.find("selected-candidate") != std::string::npos);
    assert(slice->stage_graph.stages.front().candidate == slice->trace.selected_candidate);

    const double values[]{2.0, 3.0, 4.0};
    const double first_input[]{5.0, 7.0, 11.0};
    double output[2]{};
    assert(execute_prepared_relation_slice_v1(*slice, first_input, 3u, values, 3u,
        output, 2u) == selected_plan_delivery_status_v1::success);
    assert(output[0] == 43.0 && output[1] == 28.0);

    const double rebound_input[]{1.0, 2.0, 3.0};
    assert(execute_prepared_relation_slice_v1(*slice, rebound_input, 3u, values, 3u,
        output, 2u) == selected_plan_delivery_status_v1::success);
    assert(output[0] == 11.0 && output[1] == 8.0);

    decisions[0].state = decision_state_v1::forced;
    assert(!lower_selected_relation_plan_v1(input, &status, &error));
    assert(status == selected_plan_delivery_status_v1::ambiguous_selected_candidate);
}
