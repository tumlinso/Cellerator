#include <Cellerator/compute/decomposition/destination_disjoint_v1.hh>
#include <Cellerator/compute/decomposition/provider_registry_v1.hh>

#include <cassert>
#include <cstdint>

namespace decomposition = cellerator::compute::decomposition;
namespace operation = cellerator::compute::operation::v2;
namespace execution = cellerator::execution;

namespace {

template<typename Identity>
Identity identity(std::uint64_t value) { return {value, value + 1u}; }

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
    relation = {identity<execution::structure_id>(30u), {1u}, source,
        destination, identity<execution::order_id>(40u), 8u};
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
    result.logical_work_items = 8u;
    result.dense_width = 1u;
    return result;
}

}  // namespace

int main() {
    const auto registry = decomposition::builtin_decomposition_providers_v1();
    assert(registry.provider_count == decomposition::builtin_provider_count_v1);
    assert(decomposition::validate_provider_registry_v1(registry));

    const auto found = decomposition::find_decomposition_provider_v1(registry,
        decomposition::decomposition_provider_kind_v1::destination_disjoint);
    assert(found);
    const auto missing = decomposition::find_decomposition_provider_v1(registry,
        static_cast<decomposition::decomposition_provider_kind_v1>(255u));
    assert(missing.code == decomposition::provider_lookup_code_v1::no_candidate);

    operation::typed_relation relation{};
    auto operation_problem = problem(relation);
    const decomposition::destination_interval_v1 intervals[] = {
        {0u, 3u}, {3u, 5u}};
    decomposition::destination_disjoint_relation_apply_v1 instance{};
    instance.decomposition_identity = {60u, 61u};
    instance.problem = &operation_problem;
    instance.destination_extent = 8u;
    instance.intervals = intervals;
    instance.interval_count = 2u;
    assert(found.provider->validate_instance(&instance, {})
        == decomposition::provider_instance_validation_code_v1::ok);

    decomposition::decomposition_provider_v1 providers[
        decomposition::builtin_provider_count_v1]{};
    for (std::uint64_t index = 0u; index < registry.provider_count; ++index)
        providers[index] = registry.providers[index];
    providers[1].provider_identity = providers[0].provider_identity;
    const decomposition::decomposition_provider_registry_v1 invalid{
        decomposition::provider_registry_schema_version_v1, 0u, providers,
        decomposition::builtin_provider_count_v1};
    const auto status = decomposition::validate_provider_registry_v1(invalid);
    assert(status.code == decomposition::
        provider_registry_validation_code_v1::provider_order_mismatch);
    return 0;
}
