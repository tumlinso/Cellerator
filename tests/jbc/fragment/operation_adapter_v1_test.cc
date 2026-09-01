#include <Cellerator/execution/atom_fragment/operation_adapter_v1.hh>

#include <cassert>

namespace fragment = cellerator::execution::atom_fragment;
namespace joint = cellerator::execution::joint_compiler;
namespace execution = cellerator::execution;
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
    operation::operation_problem source{};
    source.persistent_problem_identity = {1u, 2u};
    source.operation_identity = {3u, 4u};
    source.relations = {&relation, 1u};
    source.values_axis = axis(40u);
    source.result_axis = axis(50u);
    source.logical_edge_order = relation.logical_edge_order;
    source.expected_value_generation = {7u};
    source.logical_work_items = 8u;
    source.dense_width = 1u;
    numeric(&source.numeric);
    source.output.produced_axis = source.result_axis;
    source.output.canonical_axis = source.result_axis;

    const joint::canonical_interval_v1 interval{2u, 3u};
    joint::logical_coverage_view_v1 coverage{};
    coverage.coverage_identity = {60u, 1u};
    coverage.structure = relation.structure;
    coverage.epoch = relation.epoch;
    coverage.source_axis = relation.source_axis;
    coverage.destination_axis = relation.destination_axis;
    coverage.logical_count = 3u;
    coverage.members = &interval;
    coverage.member_count = 1u;
    coverage.member_bytes = sizeof(interval);

    fragment::operation_fragment_restriction_v1 restriction{};
    restriction.source = &source;
    restriction.exact_coverage = &coverage;
    restriction.expected_values_axis = source.values_axis;
    restriction.expected_result_axis = source.result_axis;
    restriction.expected_value_generation = source.expected_value_generation;
    restriction.logical_work_items = 3u;
    operation::operation_problem adapted{};
    assert(fragment::adapt_operation_problem_to_fragment_v1(
        restriction, &adapted));
    assert(adapted.logical_work_items == 3u);
    assert(adapted.numeric.accumulation == source.numeric.accumulation);
    assert(adapted.determinism.deterministic_required
        == source.determinism.deterministic_required);

    restriction.expected_value_generation.value += 1u;
    assert(fragment::adapt_operation_problem_to_fragment_v1(
               restriction, &adapted)
               .code
        == fragment::operation_fragment_adaptation_code_v1::
            incompatible_generation);
    restriction.expected_value_generation = source.expected_value_generation;
    restriction.expected_result_axis = axis(70u);
    assert(fragment::adapt_operation_problem_to_fragment_v1(
               restriction, &adapted)
               .code
        == fragment::operation_fragment_adaptation_code_v1::incompatible_axis);
    restriction.expected_result_axis = source.result_axis;
    coverage.epoch.value += 1u;
    assert(fragment::adapt_operation_problem_to_fragment_v1(
               restriction, &adapted)
               .code
        == fragment::operation_fragment_adaptation_code_v1::unmatched_relation);
    return 0;
}
