#include <Cellerator/execution/atom_fragment/canonical_fallback_v1.hh>

#include <cassert>

namespace compute = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;
namespace fragment = execution::atom_fragment;

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

int main() {
    compute::typed_relation relation{};
    relation.structure = id<execution::structure_tag>(1u);
    relation.epoch = {1u};
    relation.source_axis = axis(10u);
    relation.destination_axis = axis(20u);
    relation.logical_edge_order = id<execution::order_tag>(30u);
    relation.logical_edge_count = 8u;
    compute::operation_problem operation{};
    operation.persistent_problem_identity = {1u, 1u};
    operation.operation_identity = {2u, 1u};
    operation.relations = {&relation, 1u};
    operation.values_axis = axis(40u);
    operation.result_axis = axis(50u);
    operation.logical_edge_order = relation.logical_edge_order;
    operation.expected_value_generation = {1u};
    operation.logical_work_items = 8u;
    operation.dense_width = 1u;
    operation.numeric.relation_storage = execution::numeric_type::f32;
    operation.numeric.state_storage = execution::numeric_type::f32;
    operation.numeric.multiply = execution::numeric_type::f32;
    operation.numeric.accumulation = execution::numeric_type::f32;
    operation.numeric.output_storage = execution::numeric_type::f32;
    operation.numeric.scalar = execution::numeric_type::f32;
    operation.output.produced_axis = operation.result_axis;
    operation.output.canonical_axis = axis(60u);
    operation.output.order = compute::output_order_requirement::canonical_required;
    operation.output.explicit_order_transform = true;

    fragment::atom_bound_candidate_v1 candidate{};
    candidate.candidate_id = 7u;
    fragment::canonical_fallback_request_v1 request{};
    request.candidate_id = 7u;
    request.reason = fragment::canonical_fallback_reason_v1::
        bounded_frontier_empty;
    request.requires_order_transform = true;
    request.visible_conversion_bytes = 4096u;
    fragment::canonical_fallback_v1 fallback{};
    fragment::canonical_fallback_diagnostic_v1 diagnostic{};
    assert(fragment::make_canonical_fallback_v1(operation, &candidate, 1u,
        request, &fallback, &diagnostic));
    assert(fallback.candidate.candidate_id == 7u);
    assert(execution::same_identity(fallback.output_order,
        operation.output.canonical_axis.order));
    assert(fallback.visible_conversion_bytes == 4096u);
    assert(diagnostic.code == fragment::
        canonical_fallback_diagnostic_code_v1::selected);

    request.visible_conversion_bytes = 0u;
    assert(!fragment::make_canonical_fallback_v1(operation, &candidate, 1u,
        request, &fallback, &diagnostic));
    assert(diagnostic.code == fragment::
        canonical_fallback_diagnostic_code_v1::hidden_order_transform);

    request.visible_conversion_bytes = 4096u;
    request.candidate_id = 8u;
    assert(!fragment::make_canonical_fallback_v1(operation, &candidate, 1u,
        request, &fallback, &diagnostic));
    assert(diagnostic.code == fragment::
        canonical_fallback_diagnostic_code_v1::candidate_missing);
}
