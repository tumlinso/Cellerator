#include <Cellerator/compiler/ir/semantic/implement_relation_apply_and_transpose_operations_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;
using cellerator::execution::numeric_type;

namespace {

axis_ir_type_v1 axis(semantic_identity_v1 identity, semantic_identity_v1 domain,
                     semantic_identity_v1 order, std::uint64_t seed, const char* tag) {
    axis_ir_type_v1 result;
    result.identity = identity;
    result.domain = {domain, tag};
    result.order = {order, domain, false};
    result.geometry = {{seed, seed + 1}, domain};
    result.partition = {{seed + 2, seed + 3}, domain, {seed + 4, seed + 5}};
    result.extent = {extent_knowledge_kind_v1::exact, 64, 64};
    return result;
}

state_ir_type_v1 state(std::uint64_t seed, semantic_identity_v1 axis_identity) {
    state_ir_type_v1 result;
    result.identity = {seed, seed + 1};
    result.axes = {axis_identity};
    result.dense_width = 16;
    result.numeric = {numeric_type::f32, numeric_type::f32,
                      numeric_type::f32, numeric_type::f32};
    result.order = {seed + 2, seed + 3};
    result.generation = {1, true};
    return result;
}

relation_ir_type_v1 relation() {
    relation_ir_type_v1 result;
    result.source_axis = axis({10, 11}, {1, 2}, {3, 4}, 20, "gene");
    result.destination_axis = axis({12, 13}, {5, 6}, {7, 8}, 30, "cell");
    result.structure_identity = {40, 41};
    result.structure_epoch = 2;
    result.logical_edge_identity = {42, 43};
    result.logical_edge_order = {44, 45};
    result.logical_edge_count = 1024;
    result.support_identity = {46, 47};
    result.value_plane_identity = {48, 49};
    result.value_generation = 3;
    result.active_support_generation = 4;
    return result;
}

}  // namespace

namespace canonical = cellerator::compute::relation;
namespace legacy = cellerator::compute::operation::v2;

void check_available(const lowered_relation_apply_v1& lowered) {
    assert(lowered.transport_status == relation_transport_status_v1::available);
    assert(canonical::validate(lowered.semantic));
    const auto& expected = lowered.semantic.arithmetic;
    for (const auto* transport : {&lowered.operation, &lowered.algebra.core}) {
        assert(legacy::validate_operation_problem(*transport));
        const auto& actual = transport->numeric;
        assert(actual.relation_storage == expected.relation_storage);
        assert(actual.state_storage == expected.input_storage);
        assert(actual.multiply == expected.multiply);
        assert(actual.accumulation == expected.accumulation);
        assert(actual.output_storage == expected.output_storage);
        assert(actual.scalar == expected.multiply);
        assert(actual.rounding == legacy::rounding_policy::nearest_even);
        assert(actual.saturation == legacy::saturation_policy::none);
        assert(actual.nan == (expected.nonfinite == canonical::nonfinite_policy::reject
            ? legacy::nan_policy::reject : legacy::nan_policy::propagate));
        assert(actual.infinity == (expected.nonfinite == canonical::nonfinite_policy::reject
            ? legacy::infinity_policy::reject : legacy::infinity_policy::propagate));
        assert(transport->relations.relations == &lowered.relation);
    }
    assert(lowered.algebra.bindings.bindings == &lowered.binding);
    assert(lowered.algebra.value_bindings == &lowered.value_binding);
}

void check_unavailable(const lowered_relation_apply_v1& lowered) {
    assert(lowered.transport_status == relation_transport_status_v1::unsupported_arithmetic_policy);
    assert(canonical::validate(lowered.semantic));
    for (const auto* transport : {&lowered.operation, &lowered.algebra.core}) {
        assert(!legacy::validate_operation_problem(*transport));
        assert(!legacy::valid_operation_kind(transport->kind));
        assert(transport->relations.relations == nullptr);
        assert(transport->relations.relation_count == 0);
    }
    assert(lowered.algebra.bindings.bindings == nullptr);
    assert(lowered.algebra.bindings.binding_count == 0);
    assert(lowered.algebra.value_bindings == nullptr);
    assert(lowered.algebra.value_binding_count == 0);
}

int main() {
    relation_apply_operation_ir_v1 input;
    input.identity = {100, 101};
    input.relation = relation();
    input.source = state(110, input.relation.source_axis.identity);
    input.result = state(120, input.relation.destination_axis.identity);
    input.source.order = input.relation.source_axis.order.identity;
    input.result.order = input.relation.destination_axis.order.identity;
    lowered_relation_apply_v1 lowered;
    assert(lowered.transport_status == relation_transport_status_v1::not_lowered);
    assert(!legacy::validate_operation_problem(lowered.operation));
    for (const auto output_type : {numeric_type::f32, numeric_type::f16}) {
        input.source.numeric.output = output_type;
        input.result.numeric.storage = input.result.numeric.output = output_type;
        for (const auto nonfinite : {canonical::nonfinite_policy::propagate,
                                    canonical::nonfinite_policy::reject}) {
            input.nonfinite = nonfinite;
            for (const bool transpose : {false, true}) {
                auto operation = input;
                if (transpose) {
                    operation.relation.orientation = relation_orientation_ir_v1::transpose;
                    operation.source.axes = {operation.relation.destination_axis.identity};
                    operation.result.axes = {operation.relation.source_axis.identity};
                    operation.source.order = operation.relation.destination_axis.order.identity;
                    operation.result.order = operation.relation.source_axis.order.identity;
                }
                assert(lower_relation_apply_operation_v1(operation, &lowered) ==
                    relation_apply_ir_validation_code_v1::success);
                assert(lowered.semantic.arithmetic.relation_storage == numeric_type::f16);
                assert(lowered.semantic.arithmetic.input_storage == numeric_type::f32);
                assert(lowered.semantic.arithmetic.output_storage == output_type);
                check_available(lowered);
                lowered_relation_apply_v1 copied(lowered), moved(std::move(copied));
                check_available(moved);
                copied = lowered;
                check_available(copied);
                moved = std::move(copied);
                check_available(moved);
                const auto original = lowered.semantic;
                for (const unsigned restriction : {1u, 2u, 3u}) {
                    auto restricted = operation;
                    restricted.permit_fma = !(restriction & 1u);
                    restricted.permit_reassociation = !(restriction & 2u);
                    // Reuse a populated output to ensure no prior valid transport survives.
                    assert(lower_relation_apply_operation_v1(restricted, &lowered) ==
                        relation_apply_ir_validation_code_v1::success);
                    auto expected = original;
                    expected.arithmetic.permit_fma = restricted.permit_fma;
                    expected.arithmetic.permit_reassociation = restricted.permit_reassociation;
                    assert(canonical::equivalent(lowered.semantic, expected));
                    check_unavailable(lowered);
                    lowered_relation_apply_v1 unavailable_copy(lowered);
                    lowered_relation_apply_v1 unavailable_move(std::move(unavailable_copy));
                    check_unavailable(unavailable_move);
                    copied = lowered;
                    moved = std::move(copied);
                    check_unavailable(moved);
                }
                assert(lower_relation_apply_operation_v1(operation, &lowered) ==
                    relation_apply_ir_validation_code_v1::success);
                check_available(lowered);
            }
        }
    }
    std::cout << "numeric transport: canonical precision/nonfinite parity, independent output, "
        "restricted arithmetic unavailable, copy/move and reuse passed\n";
}
