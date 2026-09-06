#include <Cellerator/compute/decomposition/support_embedding_v1.hh>

#include <Cellerator/compiler/ir/semantic/implement_contraction_segment_and_normalization_operatio_v1.hh>

#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>

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
        destination, identity<execution::order_id>(40u), 12u};
    operation::operation_problem result{};
    result.kind = operation::operation_kind::contract_on_support;
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
    result.logical_work_items = 12u;
    result.dense_width = 16u;
    return result;
}

}  // namespace

int main() {
    using result = cellerator::compute::operation::support_product_result;
    using assembly = cellerator::compute::operation::support_product_assembly;
    for (const unsigned k : {2u, 17u}) {
        operation::typed_relation relation{};
        auto op = problem(relation);
        op.dense_width = k;
        const decomposition::dense_width_interval_v1 panels[] = {
            {0u, 1u}, {1u, k - 1u}};
        decomposition::support_embedding_decomposition_v1 value{};
        value.decomposition_identity = {60u, 61u};
        value.problem = &op;
        value.embedding_intervals = panels;
        value.embedding_interval_count = 2u;
        assert(!decomposition::validate_support_embedding_decomposition_v1(value));
        value.result = result::scalar_dot;
        value.assembly = assembly::sum_partials;
        value.produces_partial_results = true;
        value.requires_partial_algebra = true;
        assert(decomposition::validate_support_embedding_decomposition_v1(value));
        namespace ir = Cellerator::compiler::ir::semantic;
        ir::aggregate_operation_definition_ir_v1 definition{};
        definition.identity = {1, 2};
        definition.support_identity = {3, 4};
        definition.operation = ir::aggregate_operation_ir_v1::support_contraction;
        std::vector<double> left(k), right(k);
        std::vector<std::uint8_t> active(k, 1);
        for (unsigned j = 0; j < k; ++j) {
            left[j] = double(j) - 3;
            right[j] = 2 * double(j) + 1;
        }
        double interpreted = 0;
        assert(ir::interpret_support_contraction_ir_v1(definition, left, right,
            active, &interpreted) == ir::aggregate_operation_status_ir_v1::success);
        double whole = 0, split = 0;
        std::vector<double> expected(k), assembled;
        for (unsigned j = 0; j < k; ++j) {
            expected[j] = (double(j) - 3) * (2 * double(j) + 1);
            whole += expected[j];
        }
        for (const auto panel : panels) {
            double partial = 0;
            for (unsigned j = panel.begin; j < panel.begin + panel.count; ++j) {
                partial += expected[j];
                assembled.push_back(left[j] * right[j]);
            }
            split += partial;
        }
        assert(whole == split && whole == interpreted);
        double interpreted_split = 0;
        for (const auto panel : panels) {
            double partial = 0;
            assert(ir::interpret_support_contraction_ir_v1(definition,
                {left.begin() + panel.begin, left.begin() + panel.begin + panel.count},
                {right.begin() + panel.begin, right.begin() + panel.begin + panel.count},
                std::vector<std::uint8_t>(panel.count, 1), &partial)
                == ir::aggregate_operation_status_ir_v1::success);
            interpreted_split += partial;
        }
        assert(interpreted_split == interpreted);
        std::cout << "K=" << k << " whole_dot=" << whole
            << " split_dot=" << interpreted_split
            << " edge_channels=" << assembled.size() << "\n";
        definition.neutral_element = 1;
        double untouched = 42;
        assert(ir::interpret_support_contraction_ir_v1(definition, left, right,
            active, &untouched) == ir::aggregate_operation_status_ir_v1::invalid_neutral_element);
        assert(untouched == 42);
        assert(cellerator::compute::operation::support_product_output_width(value.result, k) == 1);
        value.assembly = assembly::concatenate_channels;
        assert(decomposition::validate_support_embedding_decomposition_v1(value).code
            == decomposition::support_embedding_validation_code_v1::invalid_partial_result_contract);
        value.result = result::edge_channel_product;
        value.produces_partial_results = false;
        value.requires_partial_algebra = false;
        assert(decomposition::validate_support_embedding_decomposition_v1(value));
        assert(assembled == expected && assembled.size() == k);
        assert(cellerator::compute::operation::support_product_output_width(value.result, k) == k);
        value.assembly = assembly::sum_partials;
        assert(!decomposition::validate_support_embedding_decomposition_v1(value));
        value.assembly = assembly::concatenate_channels;
        const decomposition::dense_width_interval_v1 overlap[] = {{0, 1}, {0, k - 1}};
        value.embedding_intervals = overlap;
        assert(!decomposition::validate_support_embedding_decomposition_v1(value));
    }
}
