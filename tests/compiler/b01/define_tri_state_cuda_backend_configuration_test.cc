#include <Cellerator/compiler/build/define_tri_state_cuda_backend_configuration_v1.hh>

#include <cassert>

int main() {
    using namespace Cellerator::compiler::build;
    assert(resolve_cuda_configuration_v1(cuda_enablement_v1::automatic, true) ==
           cuda_configuration_v1::accelerator_enabled);
    assert(resolve_cuda_configuration_v1(cuda_enablement_v1::automatic, false) ==
           cuda_configuration_v1::host_only);
    assert(resolve_cuda_configuration_v1(cuda_enablement_v1::enabled, true) ==
           cuda_configuration_v1::accelerator_enabled);
    assert(resolve_cuda_configuration_v1(cuda_enablement_v1::enabled, false) ==
           cuda_configuration_v1::missing_required_toolchain);
    assert(resolve_cuda_configuration_v1(cuda_enablement_v1::disabled, true) ==
           cuda_configuration_v1::host_only);
    assert(resolve_cuda_configuration_v1(cuda_enablement_v1::disabled, false) ==
           cuda_configuration_v1::host_only);
}
