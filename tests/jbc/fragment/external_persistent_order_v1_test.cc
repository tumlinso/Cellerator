#include <Cellerator/execution/atom_fragment/external_persistent_order_v1.hh>

#include <algorithm>
#include <cassert>
#include <vector>

namespace execution = cellerator::execution;
namespace atom_fragment = execution::atom_fragment;
namespace decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;

template<typename Tag>
execution::persistent_identity<Tag> id(std::uint64_t value) {
    return {value, value + 100u};
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        id<execution::domain_tag>(seed), id<execution::order_tag>(seed + 1u),
        id<execution::geometry_tag>(seed + 2u),
        id<execution::partition_tag>(seed + 3u)};
}

void numeric(operation::numerical_policy *value) {
    value->relation_storage = execution::numeric_type::f32;
    value->state_storage = execution::numeric_type::f32;
    value->multiply = execution::numeric_type::f32;
    value->accumulation = execution::numeric_type::f32;
    value->output_storage = execution::numeric_type::f32;
    value->scalar = execution::numeric_type::f32;
}

int main() {
    operation::typed_relation relation{};
    relation.structure = id<execution::structure_tag>(1u);
    relation.epoch = {2u};
    relation.source_axis = axis(10u);
    relation.destination_axis = axis(20u);
    relation.logical_edge_order = id<execution::order_tag>(30u);
    relation.logical_edge_count = 8u;
    operation::operation_problem problem{};
    problem.persistent_problem_identity = {1u, 2u};
    problem.operation_identity = {3u, 4u};
    problem.relations = {&relation, 1u};
    problem.values_axis = axis(40u);
    problem.result_axis = axis(50u);
    problem.logical_edge_order = relation.logical_edge_order;
    problem.expected_value_generation = {7u};
    problem.logical_work_items = 8u;
    problem.dense_width = 1u;
    numeric(&problem.numeric);
    problem.output.produced_axis = problem.result_axis;
    problem.output.canonical_axis = problem.result_axis;

    execution::joint_compiler::persistent_identity_v1 coverage{10u, 1u};
    decomposition::decomposition_alternative_v1 alternative{};
    alternative.alternative_identity = {20u, 1u};
    alternative.candidate_family = {21u, 1u};
    alternative.flags = decomposition::legal_alternative_v1
        | decomposition::complete_unsplit_fallback_v1;
    alternative.required_input_coverages = &coverage;
    alternative.required_input_coverage_count = 1u;
    alternative.output_coverage = coverage;
    alternative.input_order = {80u, 1u};
    alternative.output_order = {81u, 1u};
    alternative.numerical = problem.numeric;
    decomposition::decomposition_portfolio_v1 portfolio{};
    portfolio.portfolio_identity = {30u, 1u};
    portfolio.alternatives = &alternative;
    portfolio.alternative_count = 1u;

    std::vector<execution::order_id> orders = {
        problem.values_axis.order,
        problem.result_axis.order,
        problem.logical_edge_order,
        problem.output.produced_axis.order,
        problem.output.canonical_axis.order,
        problem.relations.relations[0].source_axis.order,
        problem.relations.relations[0].destination_axis.order,
        problem.relations.relations[0].logical_edge_order,
        alternative.input_order,
        alternative.output_order,
    };
    std::sort(orders.begin(), orders.end(), [](auto lhs, auto rhs) {
        return lhs.high < rhs.high
            || (lhs.high == rhs.high && lhs.low < rhs.low);
    });
    orders.erase(std::unique(orders.begin(), orders.end(), [](auto lhs,
        auto rhs) { return execution::same_identity(lhs, rhs); }), orders.end());

    assert(atom_fragment::validate_external_persistent_orders_v1(
        problem, portfolio, orders.data(), orders.size()));

    auto missing = orders;
    missing.erase(std::find_if(missing.begin(), missing.end(), [&](auto value) {
        return execution::same_identity(value, alternative.output_order);
    }));
    const auto missing_result =
        atom_fragment::validate_external_persistent_orders_v1(
            problem, portfolio, missing.data(), missing.size());
    assert(missing_result.code == atom_fragment::
        external_persistent_order_validation_code_v1::
            missing_decomposition_order);
    assert(missing_result.index == 0u && missing_result.nested_index == 1u);

    auto unordered = orders;
    std::swap(unordered[0], unordered[1]);
    const auto unordered_result =
        atom_fragment::validate_external_persistent_orders_v1(
            problem, portfolio, unordered.data(), unordered.size());
    assert(unordered_result.code == atom_fragment::
        external_persistent_order_validation_code_v1::
            duplicate_or_unordered_order);

    auto invalid = orders;
    invalid[0] = {};
    const auto invalid_result =
        atom_fragment::validate_external_persistent_orders_v1(
            problem, portfolio, invalid.data(), invalid.size());
    assert(invalid_result.code == atom_fragment::
        external_persistent_order_validation_code_v1::invalid_order);
}
