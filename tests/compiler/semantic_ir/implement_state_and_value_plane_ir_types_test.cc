#include <Cellerator/compiler/ir/semantic/implement_state_and_value_plane_ir_types_v1.hh>

#include <array>
#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::ir::semantic;
using cellerator::execution::numeric_type;

int main() {
    const std::array<numeric_tuple_ir_v1, 9> current_tuples{{
        {numeric_type::bit, numeric_type::u32, numeric_type::u32, numeric_type::bit},
        {numeric_type::u8, numeric_type::u8, numeric_type::u32, numeric_type::u8},
        {numeric_type::u16, numeric_type::u16, numeric_type::u32, numeric_type::u16},
        {numeric_type::u32, numeric_type::u32, numeric_type::u32, numeric_type::u32},
        {numeric_type::i32, numeric_type::i32, numeric_type::i32, numeric_type::i32},
        {numeric_type::f16, numeric_type::f16, numeric_type::f32, numeric_type::f16},
        {numeric_type::bf16, numeric_type::bf16, numeric_type::f32, numeric_type::bf16},
        {numeric_type::f32, numeric_type::f32, numeric_type::f32, numeric_type::f32},
        {numeric_type::f64, numeric_type::f64, numeric_type::f64, numeric_type::f64},
    }};
    for (const auto& tuple : current_tuples) {
        assert(validate_numeric_tuple_ir_v1(tuple) == state_value_ir_validation_code_v1::success);
        const auto round_trip = from_operation_numeric_policy_v1(
            to_operation_numeric_policy_v1(tuple));
        assert(round_trip.storage == tuple.storage);
        assert(round_trip.compute == tuple.compute);
        assert(round_trip.accumulation == tuple.accumulation);
        assert(round_trip.output == tuple.output);
    }

    state_ir_type_v1 state;
    state.identity = {1, 2};
    state.axes = {{3, 4}, {5, 6}};
    state.dense_width = 32;
    state.numeric = current_tuples[7];
    state.order = {7, 8};
    state.generation = {9, true};
    state.mutability = value_mutability_v1::mutable_values;
    state.address_intent = address_intent_v1::device;
    state.alias = {10, true};
    assert(validate_state_ir_type_v1(state) == state_value_ir_validation_code_v1::success);

    value_plane_ir_type_v1 plane;
    plane.identity = {11, 12};
    plane.structure = {13, 14};
    plane.structure_epoch = 1;
    plane.element_count = 1024;
    plane.numeric = current_tuples[5];
    plane.order = {15, 16};
    plane.generation = {2, true};
    plane.address_intent = address_intent_v1::managed;
    assert(validate_value_plane_ir_type_v1(plane) ==
           state_value_ir_validation_code_v1::success);

    state.alias = {0, true};
    assert(validate_state_ir_type_v1(state) ==
           state_value_ir_validation_code_v1::invalid_alias_contract);

    std::cout << "numeric_tuples=" << current_tuples.size()
              << " state_rank=2 value_plane_generation=2\n";
}
