#include <Cellerator/execution/joint_compiler/atom_fragment_request_v1.hh>

#include <cassert>

namespace joint_compiler = cellerator::execution::joint_compiler;
namespace execution = cellerator::execution;
namespace operation = cellerator::compute::operation::v2;
namespace decomposition = cellerator::compute::decomposition;

template<typename Tag>
execution::persistent_identity<Tag> persistent(std::uint64_t value) {
    return {value, value + 100u};
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        persistent<execution::domain_tag>(seed),
        persistent<execution::order_tag>(seed + 1u),
        persistent<execution::geometry_tag>(seed + 2u),
        persistent<execution::partition_tag>(seed + 3u)};
}

void set_numeric(operation::numerical_policy *numeric) {
    numeric->relation_storage = execution::numeric_type::f32;
    numeric->state_storage = execution::numeric_type::f32;
    numeric->multiply = execution::numeric_type::f32;
    numeric->accumulation = execution::numeric_type::f32;
    numeric->output_storage = execution::numeric_type::f32;
    numeric->scalar = execution::numeric_type::f32;
}

int main() {
    operation::typed_relation relation{};
    relation.structure = persistent<execution::structure_tag>(1u);
    relation.epoch = {1u};
    relation.source_axis = axis(10u);
    relation.destination_axis = axis(20u);
    relation.logical_edge_order = persistent<execution::order_tag>(30u);
    relation.logical_edge_count = 2u;
    operation::operation_problem problem{};
    problem.persistent_problem_identity = {1u, 2u};
    problem.operation_identity = {3u, 4u};
    problem.relations = {&relation, 1u};
    problem.values_axis = axis(40u);
    problem.result_axis = axis(50u);
    problem.logical_edge_order = relation.logical_edge_order;
    problem.expected_value_generation = {1u};
    problem.logical_work_items = 2u;
    problem.dense_width = 1u;
    set_numeric(&problem.numeric);
    problem.output.produced_axis = problem.result_axis;
    problem.output.canonical_axis = problem.result_axis;

    const joint_compiler::canonical_interval_v1 interval{0u, 2u};
    joint_compiler::logical_coverage_view_v1 coverage{};
    coverage.coverage_identity = {10u, 1u};
    coverage.structure = relation.structure;
    coverage.epoch = relation.epoch;
    coverage.source_axis = relation.source_axis;
    coverage.destination_axis = relation.destination_axis;
    coverage.logical_count = 2u;
    coverage.members = &interval;
    coverage.member_count = 1u;
    coverage.member_bytes = sizeof(interval);

    const std::uint64_t map[] = {0u, 1u};
    execution::hierarchical_index_component_v1 component{};
    component.component_identity = 1u;
    component.index_space.global_extent = 2u;
    component.index_space.partition_identity = 1u;
    component.index_space.local_extent = 2u;
    component.index_space.local_to_global = map;
    execution::hierarchical_index_space_view_v1 index_space{};
    index_space.relation_identity = 1u;
    index_space.aggregate_extent = 2u;
    index_space.components = &component;
    index_space.component_count = 1u;

    const auto input_coverage = coverage.coverage_identity;
    decomposition::decomposition_alternative_v1 alternative{};
    alternative.alternative_identity = {20u, 1u};
    alternative.candidate_family = {21u, 1u};
    alternative.flags = decomposition::legal_alternative_v1
        | decomposition::complete_unsplit_fallback_v1;
    alternative.required_input_coverages = &input_coverage;
    alternative.required_input_coverage_count = 1u;
    alternative.output_coverage = {22u, 1u};
    alternative.input_order = persistent<execution::order_tag>(60u);
    alternative.output_order = persistent<execution::order_tag>(61u);
    set_numeric(&alternative.numerical);
    decomposition::decomposition_portfolio_v1 portfolio{};
    portfolio.portfolio_identity = {23u, 1u};
    portfolio.alternatives = &alternative;
    portfolio.alternative_count = 1u;

    const execution::order_id external_orders[] = {
        {1u, 1u}, {2u, 1u}};
    const joint_compiler::atom_binding_request_v1 binding{
        {30u, 1u}, {30u, 2u}, {30u, 3u}};
    joint_compiler::atom_fragment_request_v1 request{};
    request.request_identity = {40u, 1u};
    request.operation = &problem;
    request.exact_coverages = &coverage;
    request.exact_coverage_count = 1u;
    request.local_index_spaces = &index_space;
    request.local_index_space_count = 1u;
    request.external_orders = external_orders;
    request.external_order_count = 2u;
    request.decomposition = &portfolio;
    request.atom_bindings = &binding;
    request.atom_binding_count = 1u;
    request.global_cost_contract = {50u, 1u};
    request.target_profile = {50u, 2u};
    request.desired_output_affordance = {50u, 3u};
    request.lowering_resumption_stage = {50u, 4u};
    assert(joint_compiler::validate_atom_fragment_request_v1(request));

    auto malformed = request;
    malformed.exact_coverages = nullptr;
    assert(joint_compiler::validate_atom_fragment_request_v1(malformed).code
        == joint_compiler::atom_fragment_request_validation_code_v1::
            missing_coverages);
    malformed = request;
    malformed.target_profile = {};
    assert(joint_compiler::validate_atom_fragment_request_v1(malformed).code
        == joint_compiler::atom_fragment_request_validation_code_v1::
            invalid_target_profile);
    const std::uint64_t saved_extent = component.index_space.local_extent;
    component.index_space.local_extent = 3u;
    assert(joint_compiler::validate_atom_fragment_request_v1(request).code
        == joint_compiler::atom_fragment_request_validation_code_v1::
            invalid_index_component);
    component.index_space.local_extent = saved_extent;
    assert(joint_compiler::validate_atom_fragment_request_v1(request));
    return 0;
}
