#include <Cellerator/compiler/sema/implement_numerical_tuple_semantics_v1.hh>

#include <cassert>

int main() {
    using namespace cellerator::compiler::sema::v1;
    using cellerator::execution::numeric_type;
    const numerical_tuple half{numeric_type::f16, numeric_type::f16, numeric_type::f16,
        numeric_type::f32, numeric_type::f16};
    const numerical_tuple bf16{numeric_type::bf16, numeric_type::bf16, numeric_type::bf16,
        numeric_type::f32, numeric_type::bf16};
    const numerical_tuple fp32{numeric_type::f32, numeric_type::f32, numeric_type::f32,
        numeric_type::f32, numeric_type::f32};
    const numerical_tuple fp64{numeric_type::f64, numeric_type::f64, numeric_type::f64,
        numeric_type::f64, numeric_type::f64};
    const numerical_candidate_capability half_candidate{numeric_type::f16,
        numeric_type::f16, numeric_type::f32, true, false};
    assert(numerical_candidate_legal(half, half_candidate));
    assert(!numerical_candidate_legal(bf16, half_candidate));
    assert(!numerical_candidate_legal(fp32, half_candidate));
    assert(!numerical_candidate_legal(fp64, half_candidate));
}
