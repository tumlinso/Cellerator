#include <Cellerator/execution/execution_contract.hh>

#include <type_traits>

namespace ce = cellerator::execution;

__global__ void validate_execution_binding_kernel(
    ce::prepared_binding_contract prepared,
    ce::launch_bindings launch,
    ce::binding_validation_code *result) {
    if (threadIdx.x == 0u && blockIdx.x == 0u)
        *result = ce::validate_launch_bindings(prepared, launch);
}

int main() {
    static_assert(std::is_trivially_copyable<ce::relation_structure>::value,
        "relation structure must remain CUDA-copyable");
    static_assert(std::is_trivially_copyable<ce::launch_bindings>::value,
        "launch bindings must remain CUDA-copyable");
    return 0;
}
