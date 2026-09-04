#include <Cellerator/compiler/ir/realization/implement_packed_operands_and_invalidation_v1.hh>
#include <cassert>
using namespace cellerator::compiler::ir::realization::v1;
int main() {
    packed_operand_v1 operand{{1,1},{1,2},{1,3},packed_operand_role_v1::values,
        persistence_horizon_v1::value_generation,7,3,4,64,
        {{0,2},{1,0},{2,3}},{{1,1}}};
    assert(packed_operand_readiness_v1(operand,7)==packed_operand_status_v1::ready);
    assert(packed_operand_readiness_v1(operand,8)==packed_operand_status_v1::stale_generation);
    operand.persistence=persistence_horizon_v1::structure_epoch;
    assert(packed_operand_readiness_v1(operand,8)==packed_operand_status_v1::ready);
    operand.padding_holes={{2,1}};
    assert(validate_packed_operand_v1(operand)==packed_operand_status_v1::invalid_padding);
}
