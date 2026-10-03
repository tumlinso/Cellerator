#include <Cellerator/compiler/substrate/guarded_scaled_tanh.hh>
#include <cassert>
using namespace cellerator::compiler::substrate;
int main() {
    static_assert(sm70_supports(instruction::scalar_f32));
    static_assert(sm70_supports(instruction::popcount));
    static_assert(sm70_supports(instruction::mma_f16_f32));
    static_assert(!sm70_supports(instruction::binary_mma));
    static_assert(!sm70_supports(instruction::tf32_mma));
    static_assert(!sm70_supports(instruction::cp_async));
    scaled_tanh_plan p{{4,7}};
    assert(p.choose({4,7}) == realization::fused_scaled_tanh);
    assert(p.choose({5,7}) == realization::direct);
    assert(p.choose({4,8}) == realization::direct);
    assert(p.choose({4,7,numerical_policy::approximate}) == realization::direct);
    std::uint64_t rows[]{2,5,8};
    assert(compose_support_row(3,rows,3).row == 7);
    assert(compose_support_row(0,rows,3).row == 0);
    assert(compose_support_row(4,rows,3).row == 8);
    assert(!compose_support_row(8,rows,3).valid);
    assert(!compose_support_row(1,nullptr,1).valid);
    assert(!compose_support_row(0,rows,65).valid);
    assert(compose_support_row(0,nullptr,0).valid);
}
