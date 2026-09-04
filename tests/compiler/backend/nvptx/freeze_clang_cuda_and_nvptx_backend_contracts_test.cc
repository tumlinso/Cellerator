#include <Cellerator/compiler/backend/nvptx/freeze_clang_cuda_and_nvptx_backend_contracts_v1.hh>

#include <cassert>
#include <type_traits>

using namespace Cellerator::compiler::backend::nvptx;

int main() {
    static_assert(std::is_trivially_copyable_v<nvidia_route_request_v1>);
    static_assert(std::is_trivially_copyable_v<nvidia_route_capability_v1>);

    nvidia_route_request_v1 request;
    request.route = nvidia_optional_route_v1::clang_cuda;
    request.compute_major = 7u;
    request.realization_module_identity = 11u;
    request.source_map = {12u, 13u};

    nvidia_route_capability_v1 capability;
    capability.route = request.route;
    capability.minimum_compute_major = 7u;
    capability.maximum_compute_major = 12u;
    assert(probe_nvidia_optional_route_v1(request, capability) ==
           nvidia_route_probe_status_v1::unavailable);

    capability.frontend_available = true;
    capability.device_library_available = true;
    capability.assembler_available = true;
    capability.linker_available = true;
    assert(probe_nvidia_optional_route_v1(request, capability) ==
           nvidia_route_probe_status_v1::ready);

    request.compute_major = 6u;
    assert(probe_nvidia_optional_route_v1(request, capability) ==
           nvidia_route_probe_status_v1::unsupported_target);
}
