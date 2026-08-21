#include <Cellerator/execution/biological_abi.hh>

#include <type_traits>

namespace ce = cellerator::execution;

__global__ void validate_biological_operand_kernel(
    ce::biological_operand_view operand,
    ce::biological_validation_code *result) {
    if (threadIdx.x == 0u && blockIdx.x == 0u)
        *result = ce::validate_operand(operand);
}

int main() {
    static_assert(std::is_trivially_copyable<ce::biological_operand_view>::value,
        "CUDA launch records must be trivially copyable");
    static_assert(std::is_standard_layout<ce::biological_operand_view>::value,
        "CUDA launch records must have standard layout");
    return 0;
}
