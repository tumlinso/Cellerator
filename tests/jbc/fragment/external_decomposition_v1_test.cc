#include <Cellerator/execution/atom_fragment/external_decomposition_v1.hh>

#include <cassert>

namespace fragment = cellerator::execution::atom_fragment;
namespace joint = cellerator::execution::joint_compiler;
namespace operation = cellerator::compute::operation::v2;
namespace decomposition = cellerator::compute::decomposition;
namespace execution = cellerator::execution;

void numeric(operation::numerical_policy *value) {
    value->relation_storage = execution::numeric_type::f32;
    value->state_storage = execution::numeric_type::f32;
    value->multiply = execution::numeric_type::f32;
    value->accumulation = execution::numeric_type::f32;
    value->output_storage = execution::numeric_type::f32;
    value->scalar = execution::numeric_type::f32;
}

int main() {
    operation::operation_problem problem{};
    problem.persistent_problem_identity = {1u, 2u};
    problem.operation_identity = {3u, 4u};
    problem.expected_value_generation = {1u};
    problem.logical_work_items = 1u;
    problem.dense_width = 1u;
    numeric(&problem.numeric);
    // Use the operation validator's sparse-axis-update relation-free form.
    problem.kind = operation::operation_kind::sparse_axis_update;
    problem.values_axis = {{1u, execution::serialized_record_kind::persistent_axis_identity,
        sizeof(execution::persistent_axis_identity)}, {1u,1u},{2u,1u},{3u,1u},{4u,1u}};
    problem.result_axis = problem.values_axis;
    problem.logical_edge_order = {5u, 1u};
    problem.output.produced_axis = problem.result_axis;
    problem.output.canonical_axis = problem.result_axis;

    joint::logical_coverage_view_v1 coverages[2]{};
    coverages[0].coverage_identity = {10u, 1u};
    coverages[1].coverage_identity = {10u, 2u};
    // Portfolio validation does not consume coverage bodies.
    const joint::persistent_identity_v1 input = coverages[0].coverage_identity;
    decomposition::decomposition_alternative_v1 alternative{};
    alternative.alternative_identity = {20u, 1u};
    alternative.candidate_family = {21u, 1u};
    alternative.flags = decomposition::legal_alternative_v1
        | decomposition::complete_unsplit_fallback_v1;
    alternative.required_input_coverages = &input;
    alternative.required_input_coverage_count = 1u;
    alternative.output_coverage = coverages[1].coverage_identity;
    alternative.input_order = {30u, 1u};
    alternative.output_order = {30u, 2u};
    numeric(&alternative.numerical);
    decomposition::decomposition_portfolio_v1 portfolio{};
    portfolio.portfolio_identity = {22u, 1u};
    portfolio.alternatives = &alternative;
    portfolio.alternative_count = 1u;
    const execution::order_id orders[] = {{30u, 1u}, {30u, 2u}};
    assert(fragment::validate_external_decomposition_v1(
        problem, coverages, 2u, orders, 2u, portfolio));

    alternative.output_coverage = {99u, 1u};
    assert(fragment::validate_external_decomposition_v1(
               problem, coverages, 2u, orders, 2u, portfolio)
               .code
        == fragment::external_decomposition_validation_code_v1::missing_coverage);
    alternative.output_coverage = coverages[1].coverage_identity;
    alternative.numerical.accumulation = execution::numeric_type::f64;
    assert(fragment::validate_external_decomposition_v1(
               problem, coverages, 2u, orders, 2u, portfolio)
               .code
        == fragment::external_decomposition_validation_code_v1::
            incompatible_numerical_policy);
    return 0;
}
