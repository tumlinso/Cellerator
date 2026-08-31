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

}  // namespace

int main() {
    test_schema_v2_problem();
    return 0;
}
