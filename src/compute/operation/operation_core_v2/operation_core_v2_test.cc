#include <Cellerator/compute/operation/operation_core_v2.hh>

#include <cstdlib>

using namespace cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

void require(bool condition) {
    if (!condition) {
        std::abort();
    }
}

template <class Tag>
execution::persistent_identity<Tag> persistent(std::uint64_t low) {
    execution::persistent_identity<Tag> result{};
    result.low = low;
    result.high = low + 1;
    return result;
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    execution::persistent_axis_identity result{};
    result.header.schema_version = execution::biological_abi_version;
    result.header.kind = execution::serialized_record_kind::persistent_axis_identity;
    result.header.byte_count = sizeof(result);
    result.domain = persistent<execution::domain_tag>(seed);
    result.order = persistent<execution::order_tag>(seed + 2);
    result.geometry = persistent<execution::geometry_tag>(seed + 4);
    result.partition = persistent<execution::partition_tag>(seed + 6);
    return result;
}

void test_schema_v2_problem() {
    typed_relation relation{};
    relation.structure = persistent<execution::structure_tag>(10);
    relation.epoch.value = 3;
    relation.source_axis = axis(20);
    relation.destination_axis = axis(40);
    relation.logical_edge_order = persistent<execution::order_tag>(60);
    relation.logical_edge_count = (std::uint64_t{1} << 32) + 17;

    operation_problem problem{};
    problem.persistent_problem_identity = {1, 2};
    problem.operation_identity = {3, 4};
    problem.relations = {&relation, 1};
    problem.expected_value_generation.value = 7;
    problem.logical_work_items = relation.logical_edge_count;
    problem.dense_width = 32;
    require(static_cast<bool>(validate_operation_problem(problem)));

    problem.expected_value_generation.value = 0;
    require(validate_operation_problem(problem).code == schema_status_code::invalid_generation);
}

operation_problem relation_problem(
    operation_kind kind,
    relation_orientation orientation,
    const typed_relation *relation) {
    operation_problem problem{};
    problem.kind = kind;
    problem.orientation = orientation;
    problem.persistent_problem_identity = {11, 12};
    problem.operation_identity = {13, 14};
    problem.relations = {relation, 1};
    problem.logical_edge_order = relation->logical_edge_order;
    problem.expected_value_generation.value = 9;
    problem.logical_work_items = relation->logical_edge_count;
    problem.dense_width = 16;
    return problem;
}

void test_complete_relation_algebra_bindings() {
    typed_relation relation{};
    relation.structure = persistent<execution::structure_tag>(100);
    relation.epoch.value = 5;
    relation.source_axis = axis(120);
    relation.destination_axis = axis(140);
    relation.logical_edge_order = persistent<execution::order_tag>(160);
    relation.logical_edge_count = 19;

    relation_value_binding_contract values{};
    values.structure = relation.structure;
    values.epoch = relation.epoch;
    values.generation.value = 9;
    relation_binding_contract binding{};
    binding.source_state_operand = 0;
    binding.destination_state_operand = 1;
    binding.relation_values = 0;

    relation_algebra_problem problem{};
    problem.core = relation_problem(
        operation_kind::relation_apply, relation_orientation::forward, &relation);
    problem.bindings = {&binding, 1};
    problem.value_bindings = &values;
    problem.value_binding_count = 1;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    problem.core.kind = operation_kind::relation_apply_transpose;
    problem.core.orientation = relation_orientation::transpose;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    problem.core.kind = operation_kind::edge_map_or_gate;
    problem.edge = edge_operation::multiplicative_gate;
    problem.gate = gate_indexing::per_source;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));

    values.ownership = value_ownership_mode::projection_primary;
    values.layout = execution::value_layout_kind::projection_local_order;
    values.required_components = mma_physical_value_plane
        | residual_physical_value_plane | physical_to_logical_map;
    require(static_cast<bool>(validate_relation_algebra_problem(problem)));
}

}  // namespace

int main() {
    test_schema_v2_problem();
    test_complete_relation_algebra_bindings();
    return 0;
}
