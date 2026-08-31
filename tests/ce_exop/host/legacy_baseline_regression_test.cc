#include <Cellerator/compute/operation/operation_core_v2.hh>

#include <cstdint>
#include <cstdlib>

namespace op = cellerator::compute::operation;
namespace v2 = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

void require(bool value) {
    if (!value) {
        std::abort();
    }
}

template <class Tag>
execution::persistent_identity<Tag> identity(std::uint64_t seed) {
    return {seed, seed + 1u};
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header.schema_version = execution::biological_abi_version;
    result.header.kind = execution::serialized_record_kind::persistent_axis_identity;
    result.header.byte_count = sizeof(result);
    result.domain = identity<execution::domain_tag>(seed);
    result.order = identity<execution::order_tag>(seed + 2u);
    result.geometry = identity<execution::geometry_tag>(seed + 4u);
    result.partition = identity<execution::partition_tag>(seed + 6u);
    return result;
}

op::relation_algebra_problem_v1 baseline_problem() {
    op::relation_algebra_problem_v1 result{};
    result.kind = op::relation_algebra_kind_v1::relation_apply;
    result.operation_identity = {11u, 12u};
    result.relation.structure = identity<execution::structure_tag>(20u);
    result.relation.epoch.value = 3u;
    result.relation.source_axis = axis(30u);
    result.relation.destination_axis = axis(50u);
    result.relation.logical_edge_count = (std::uint64_t{1} << 32u) + 17u;
    result.logical_edge_order = identity<execution::order_tag>(70u);
    result.dense_width = 16u;
    result.numeric.relation_storage = execution::numeric_type::f32;
    result.numeric.state_storage = execution::numeric_type::f32;
    result.numeric.multiply = execution::numeric_type::f32;
    result.numeric.accumulation = execution::numeric_type::f32;
    result.numeric.output_storage = execution::numeric_type::f32;
    result.numeric.scalar = execution::numeric_type::f32;
    result.semantic_flags = op::alpha_applied_once | op::beta_applied_once;
    return result;
}

void adapter_preserves_baseline_semantics() {
    const op::relation_algebra_problem_v1 source = baseline_problem();
    v2::typed_relation relations[1]{};
    v2::relation_binding_contract bindings[1]{};
    v2::relation_value_binding_contract values[1]{};
    v2::v1_adapter_request request{};
    request.persistent_problem_identity = {101u, 102u};
    request.value_generation.value = 9u;
    request.storage = {relations, bindings, values, 1u};
    v2::v1_adapter_result result{};

    require(static_cast<bool>(v2::adapt_relation_algebra_v1(source, request, &result)));
    require(result.authority == v2::compatibility_execution_authority::operation_core_v2);
    require(result.source_only_compatibility);
    require(v2::same_stable_id(result.problem.core.persistent_problem_identity,
                               request.persistent_problem_identity));
    require(result.problem.core.logical_work_items == source.relation.logical_edge_count);
    require(result.problem.core.relations.relations[0].logical_edge_count
            == source.relation.logical_edge_count);
    require(result.problem.core.output.order
            == v2::output_order_requirement::preserve_persistent);

    v2::typed_relation relocated_relations[1]{};
    v2::relation_binding_contract relocated_bindings[1]{};
    v2::relation_value_binding_contract relocated_values[1]{};
    request.storage = {relocated_relations, relocated_bindings, relocated_values, 1u};
    require(static_cast<bool>(v2::adapt_relation_algebra_v1(source, request, &result)));
    require(v2::same_stable_id(result.problem.core.persistent_problem_identity,
                               {101u, 102u}));
    require(result.problem.core.relations.relations == relocated_relations);
}

void baseline_rejections_remain_fail_closed() {
    const op::relation_algebra_problem_v1 source = baseline_problem();
    v2::v1_adapter_request request{};
    v2::v1_adapter_result result{};
    require(v2::adapt_relation_algebra_v1(source, request, &result).code
            == v2::schema_status_code::invalid_argument);

    v2::typed_relation relation{};
    relation.structure = identity<execution::structure_tag>(200u);
    relation.epoch.value = 1u;
    relation.source_axis = axis(210u);
    relation.destination_axis = axis(230u);
    relation.logical_edge_order = identity<execution::order_tag>(250u);
    relation.logical_edge_count = 4u;
    v2::operation_problem problem{};
    problem.persistent_problem_identity = {1u, 2u};
    problem.operation_identity = {3u, 4u};
    problem.relations = {&relation, 1u};
    problem.logical_edge_order = relation.logical_edge_order;
    problem.expected_value_generation.value = 0u;
    problem.logical_work_items = 4u;
    problem.dense_width = 1u;
    problem.values_axis = relation.source_axis;
    problem.result_axis = relation.destination_axis;
    problem.output.produced_axis = relation.destination_axis;
    problem.output.canonical_axis = relation.destination_axis;
    problem.numeric.relation_storage = execution::numeric_type::f32;
    problem.numeric.state_storage = execution::numeric_type::f32;
    problem.numeric.multiply = execution::numeric_type::f32;
    problem.numeric.accumulation = execution::numeric_type::f32;
    problem.numeric.output_storage = execution::numeric_type::f32;
    problem.numeric.scalar = execution::numeric_type::f32;
    require(v2::validate_operation_problem(problem).code
            == v2::schema_status_code::invalid_generation);
}

}  // namespace

int main() {
    adapter_preserves_baseline_semantics();
    baseline_rejections_remain_fail_closed();
    return 0;
}
