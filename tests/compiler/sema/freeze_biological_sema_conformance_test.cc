#include <Cellerator/compiler/sema/operation_resolution_v1.hh>
#include <Cellerator/compiler/sema/semantic_types_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

using namespace cellerator;

namespace {
template<typename Tag>
execution::persistent_identity<Tag> identity(std::uint64_t seed) {
    return {seed, seed + 1};
}
execution::persistent_axis_identity axis(std::uint64_t seed) {
    return {{execution::biological_abi_version,
             execution::serialized_record_kind::persistent_axis_identity,
             sizeof(execution::persistent_axis_identity)},
            identity<execution::domain_tag>(seed), identity<execution::order_tag>(seed + 2),
            identity<execution::geometry_tag>(seed + 4),
            identity<execution::partition_tag>(seed + 6)};
}
}  // namespace

int main() {
    using namespace compiler::sema::v1;
    namespace op = compute::operation::v2;
    op::typed_relation relation{identity<execution::structure_tag>(10), {2}, axis(20),
                                axis(40), identity<execution::order_tag>(60), 17};
    op::relation_binding_contract binding{0, 0, 1, 0};
    op::relation_value_binding_contract values{};
    values.structure = relation.structure;
    values.epoch = relation.epoch;
    values.generation = {3};

    constexpr std::array<op::operation_kind, 8> kinds{{
        op::operation_kind::relation_apply, op::operation_kind::relation_apply_transpose,
        op::operation_kind::contract_on_support, op::operation_kind::segment_reduce,
        op::operation_kind::segment_normalize, op::operation_kind::edge_map_or_gate,
        op::operation_kind::relation_bundle_apply, op::operation_kind::sparse_axis_update}};
    constexpr std::array<execution::numeric_type, 4> numerics{{
        execution::numeric_type::f16, execution::numeric_type::bf16,
        execution::numeric_type::f32, execution::numeric_type::f64}};

    std::uint64_t specialization_checksum = 0;
    for (const auto kind : kinds) {
        for (const auto numeric : numerics) {
            op::relation_algebra_problem fixture{};
            fixture.core.kind = kind;
            fixture.core.orientation = kind == op::operation_kind::relation_apply_transpose
                ? op::relation_orientation::transpose : op::relation_orientation::forward;
            fixture.core.persistent_problem_identity = {1, 2};
            fixture.core.operation_identity = {3, 4};
            fixture.core.relations = {&relation, 1};
            fixture.core.values_axis = axis(70);
            fixture.core.result_axis = axis(80);
            fixture.core.logical_edge_order = relation.logical_edge_order;
            fixture.core.expected_value_generation = {3};
            fixture.core.numeric.relation_storage = numeric;
            fixture.core.numeric.state_storage = numeric;
            fixture.core.numeric.multiply = numeric;
            fixture.core.numeric.accumulation = numeric;
            fixture.core.numeric.output_storage = numeric;
            fixture.core.numeric.scalar = numeric;
            fixture.core.output.produced_axis = fixture.core.result_axis;
            fixture.core.output.canonical_axis = fixture.core.result_axis;
            fixture.core.logical_work_items = 17;
            fixture.core.dense_width = 8;
            fixture.bindings = {&binding, 1};
            fixture.value_bindings = &values;
            fixture.value_binding_count = 1;
            fixture.segment = op::segment_operation::sum;
            fixture.edge = op::edge_operation::multiplicative_gate;
            fixture.gate = op::gate_indexing::per_edge;
            fixture.semantic_flags = op::alpha_applied_once | op::beta_applied_once;

            const auto lowered = lower_through_biological_sema(fixture);
            assert(planning_information_preserved(fixture, lowered));
            assert(recover_operation_problem(lowered).kind == kind);
            assert(lowered.numeric.compute == numeric);
            specialization_checksum += static_cast<std::uint8_t>(lowered.operation)
                + static_cast<std::uint8_t>(numeric);
        }
    }
    assert(specialization_checksum != 0);
}
