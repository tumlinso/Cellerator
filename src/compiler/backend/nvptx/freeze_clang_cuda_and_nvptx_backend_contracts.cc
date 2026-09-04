#include <Cellerator/compiler/backend/nvptx/freeze_clang_cuda_and_nvptx_backend_contracts_v1.hh>

namespace Cellerator::compiler::backend::nvptx {
namespace {

bool target_in_range(const nvidia_route_request_v1& request,
                     const nvidia_route_capability_v1& capability) noexcept {
    const auto requested = request.compute_major * 100u + request.compute_minor;
    const auto minimum = capability.minimum_compute_major * 100u +
        capability.minimum_compute_minor;
    const auto maximum = capability.maximum_compute_major * 100u +
        capability.maximum_compute_minor;
    return minimum <= requested && requested <= maximum;
}

}  // namespace

nvidia_route_probe_status_v1 probe_nvidia_optional_route_v1(
    const nvidia_route_request_v1& request,
    const nvidia_route_capability_v1& capability) noexcept {
    if (request.compute_major == 0u || request.compute_minor > 99u ||
        request.realization_module_identity == 0u ||
        request.source_map.high == 0u || request.source_map.low == 0u ||
        request.route != capability.route) {
        return nvidia_route_probe_status_v1::invalid_request;
    }
    if (capability.contract_version != nvidia_optional_backend_contract_version_v1) {
        return nvidia_route_probe_status_v1::unsupported_contract;
    }
    if (capability.minimum_compute_major == 0u ||
        capability.maximum_compute_major == 0u ||
        !target_in_range(request, capability)) {
        return nvidia_route_probe_status_v1::unsupported_target;
    }
    if (!capability.frontend_available || !capability.device_library_available ||
        !capability.assembler_available || !capability.linker_available) {
        return nvidia_route_probe_status_v1::unavailable;
    }
    return nvidia_route_probe_status_v1::ready;
}

}  // namespace Cellerator::compiler::backend::nvptx
