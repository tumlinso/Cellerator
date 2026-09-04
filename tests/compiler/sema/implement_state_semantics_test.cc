#include <Cellerator/compiler/sema/implement_state_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    float values[8]{};
    axis_type axis{{{1, 2}, 0, false}, 8, {3, 4}, {5, 6}, {7, 8}, 8, {9, 10}};
    state_type type{&axis, 1, cellerator::execution::numeric_type::f32, 1,
                    cellerator::execution::residency_kind::host,
                    state_mutability::read_write, generation_class::evolving};
    const auto state = bind_pointer(values, type);
    assert(validate_state_view(state) == state_validation::ok);

    cellerator::execution::dense_tensor_view operand{};
    operand.data = values;
    operand.location = {cellerator::execution::residency_kind::host, {}, -1, 0};
    operand.value_type = cellerator::execution::numeric_type::f32;
    operand.rank = 1;
    operand.shape[0] = 8;
    assert(validate_against_dense_operand(state, operand) == state_validation::ok);
    operand.shape[0] = 7;
    assert(validate_against_dense_operand(state, operand) == state_validation::shape_mismatch);
}
