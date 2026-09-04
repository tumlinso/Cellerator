#include <Cellerator/compiler/backend/nvptx/freeze_optional_nvidia_backend_routes_v1.hh>

#include <algorithm>

namespace Cellerator::compiler::backend::nvptx {
namespace {

frozen_nvidia_route_record_v1 freeze_frontend_route(
    const frozen_nvidia_route_v1 route,
    const std::uint32_t subsets,
    const nvidia_route_request_v1& request,
    const nvidia_route_capability_v1& capability) {
    frozen_nvidia_route_record_v1 record;
    record.route = route;
    record.supported_subsets = subsets;
    record.minimum_compute_major = capability.minimum_compute_major;
    record.minimum_compute_minor = capability.minimum_compute_minor;
    record.maximum_compute_major = capability.maximum_compute_major;
    record.maximum_compute_minor = capability.maximum_compute_minor;
    record.implementation_identity = request.realization_module_identity;
    record.source_map = request.source_map;
    if (probe_nvidia_optional_route_v1(request, capability) ==
        nvidia_route_probe_status_v1::ready) {
        record.status = frozen_nvidia_route_status_v1::available;
    }
    return record;
}

bool has_valid_direct_measurement(const nvptx_route_comparison_v1& evidence) {
    return std::any_of(evidence.measurements.begin(), evidence.measurements.end(),
        [](const nvptx_route_measurement_v1& measurement) {
            return measurement.route == nvptx_route_v1::direct_ptx &&
                measurement.correctness_passed && measurement.benchmark_mutex_held &&
                !measurement.contaminated;
        });
}

}  // namespace

optional_nvidia_backend_receipt_v1 freeze_optional_nvidia_backend_routes_v1(
    const optional_nvidia_backend_freeze_request_v1& request) {
    optional_nvidia_backend_receipt_v1 receipt;
    receipt.host_build_supported = request.host_backend_available;
    receipt.nvcc_build_supported = request.nvcc_backend_available;
    receipt.realization_ir_interface_identity = request.realization_ir_interface_identity;
    receipt.backend_abi_interface_identity = request.backend_abi_interface_identity;

    if (!request.host_backend_available ||
        request.realization_ir_interface_identity == 0u ||
        request.backend_abi_interface_identity == 0u) {
        receipt.diagnostics.emplace_back(
            "host backend and frozen realization/backend interface identities are required");
        return receipt;
    }

    receipt.optional_routes[0] = freeze_frontend_route(
        frozen_nvidia_route_v1::clang_cuda, subset_cuda_source_action,
        request.clang_cuda_request, request.clang_cuda_capability);
    receipt.optional_routes[1] = freeze_frontend_route(
        frozen_nvidia_route_v1::llvm_nvptx, subset_llvm_module_boundary,
        request.llvm_nvptx_request, request.llvm_nvptx_capability);

    auto& direct = receipt.optional_routes[2];
    direct.route = frozen_nvidia_route_v1::direct_ptx;
    direct.supported_subsets = subset_typed_ptx_operation |
        subset_inline_ptx_binding | subset_deterministic_ptx_assembly |
        subset_object_embedding;
    direct.implementation_identity = request.direct_ptx_implementation_identity;
    direct.source_map = request.direct_ptx_source_map;
    const auto* manifest = request.installed_provider_manifest;
    const bool valid_manifest = manifest != nullptr &&
        static_cast<bool>(cellpack::persistence::validate_execution_capability_manifest_v1(
            *manifest));
    const bool valid_provenance = direct.implementation_identity != 0u &&
        direct.source_map.high != 0u && direct.source_map.low != 0u;
    const bool valid_evidence = has_valid_direct_measurement(request.direct_ptx_evidence) &&
        request.direct_ptx_evidence.disposition !=
            nvptx_route_promotion_v1::invalid_evidence;
    if (valid_manifest) {
        direct.minimum_compute_major = manifest->minimum_compute_capability_major;
        direct.minimum_compute_minor = manifest->minimum_compute_capability_minor;
        direct.maximum_compute_major = manifest->maximum_compute_capability_major;
        direct.maximum_compute_minor = manifest->maximum_compute_capability_minor;
    }
    if (valid_manifest && valid_provenance && valid_evidence) {
        direct.promoted = request.direct_ptx_evidence.disposition ==
            nvptx_route_promotion_v1::promoted;
        direct.status = direct.promoted ? frozen_nvidia_route_status_v1::available :
            frozen_nvidia_route_status_v1::evaluated_not_promoted;
    }

    if (receipt.optional_routes[0].status == frozen_nvidia_route_status_v1::unavailable)
        receipt.diagnostics.emplace_back("Clang CUDA route is optional and unavailable");
    if (receipt.optional_routes[1].status == frozen_nvidia_route_status_v1::unavailable)
        receipt.diagnostics.emplace_back("LLVM/NVPTX route is optional and unavailable");
    if (direct.status == frozen_nvidia_route_status_v1::unavailable)
        receipt.diagnostics.emplace_back(
            "direct PTX route lacks valid provider, provenance, or measurement evidence");
    else if (!direct.promoted)
        receipt.diagnostics.emplace_back(
            "direct PTX was evaluated but did not clear its promotion threshold");

    receipt.frozen = true;
    return receipt;
}

}  // namespace Cellerator::compiler::backend::nvptx
