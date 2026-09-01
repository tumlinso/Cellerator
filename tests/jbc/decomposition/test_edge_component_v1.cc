#include <Cellerator/compute/decomposition/edge_component_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;
namespace geometry = cellerator::geometry;

namespace {

template<typename Identity>
Identity identity(std::uint64_t value) {
    return {value, value + 1u};
}

execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
                execution::serialized_record_kind::persistent_axis_identity,
                sizeof(execution::persistent_axis_identity)},
        identity<execution::domain_id>(seed),
        identity<execution::order_id>(seed + 2u),
        identity<execution::geometry_id>(seed + 4u),
        identity<execution::partition_id>(seed + 6u)};
}

operation::operation_problem problem(operation::typed_relation &relation) {
    const auto source = axis(10u);
    const auto destination = axis(20u);
    relation = {identity<execution::structure_id>(30u),
        {1u}, source, destination, identity<execution::order_id>(40u), 4u};
    operation::operation_problem result{};
    result.persistent_problem_identity = {50u, 51u};
    result.operation_identity = {52u, 53u};
    result.relations = {&relation, 1u};
    result.values_axis = source;
    result.result_axis = destination;
    result.logical_edge_order = relation.logical_edge_order;
    result.expected_value_generation = {1u};
    result.numeric.relation_storage = execution::numeric_type::f32;
    result.numeric.state_storage = execution::numeric_type::f32;
    result.numeric.multiply = execution::numeric_type::f32;
    result.numeric.accumulation = execution::numeric_type::f32;
    result.numeric.output_storage = execution::numeric_type::f32;
    result.numeric.scalar = execution::numeric_type::f32;
    result.output.produced_axis = destination;
    result.output.canonical_axis = destination;
    result.logical_work_items = 4u;
    result.dense_width = 1u;
    return result;
}

}  // namespace

int main() {
    operation::typed_relation relation{};
    auto operation_problem = problem(relation);
    const geometry::semantic_component_v1 components[] = {
        {1u, geometry::semantic_component_kind::rectangular, {}, 0u, 2u},
        {2u, geometry::semantic_component_kind::unstructured, {}, 2u, 2u}};
    const std::uint64_t edge_ids[] = {0u, 2u, 1u, 3u};
    geometry::relation_cover_view_v1 cover{};
    cover.structure = {1u, 1u};
    cover.structure_epoch = {1u};
    cover.source_axis = {{1u, 1u}, {2u, 1u}, {3u, 1u}, {4u, 1u}};
    cover.destination_axis = {{5u, 1u}, {6u, 1u}, {7u, 1u}, {8u, 1u}};
    cover.logical_edge_count = 4u;
    cover.component_count = 2u;
    cover.components = components;
    cover.logical_edge_ids = edge_ids;
    std::uint8_t marks[4]{};
    const geometry::relation_cover_validation_workspace workspace{marks, 4u};
    assert(geometry::validate_relation_cover(cover, workspace));

    const decomposition::edge_component_fragment_v1 fragments[] = {
        {0u, 1u, 0u, 2u}, {1u, 1u, 2u, 2u}};
    decomposition::edge_component_relation_apply_v1 value{};
    value.decomposition_identity = {60u, 61u};
    value.problem = &operation_problem;
    value.cover = &cover;
    value.fragments = fragments;
    value.fragment_count = 2u;
    assert(decomposition::validate_edge_component_relation_apply_v1(
        value, workspace));

    auto invalid = value;
    invalid.requires_partial_algebra = false;
    auto status = decomposition::validate_edge_component_relation_apply_v1(
        invalid, workspace);
    assert(status.code == decomposition::
        edge_component_validation_code_v1::invalid_partial_result_contract);

    const decomposition::edge_component_fragment_v1 bad_edges[] = {
        {0u, 1u, 0u, 1u}, {1u, 1u, 2u, 2u}};
    invalid = value;
    invalid.fragments = bad_edges;
    status = decomposition::validate_edge_component_relation_apply_v1(
        invalid, workspace);
    assert(status.code == decomposition::
        edge_component_validation_code_v1::logical_edge_count_mismatch);

    relation.logical_edge_count = 5u;
    invalid = value;
    status = decomposition::validate_edge_component_relation_apply_v1(
        invalid, workspace);
    assert(status.code == decomposition::
        edge_component_validation_code_v1::relation_edge_count_mismatch);
    return 0;
}
