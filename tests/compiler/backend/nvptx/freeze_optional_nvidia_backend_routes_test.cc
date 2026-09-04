#include <Cellerator/compiler/backend/nvptx/freeze_optional_nvidia_backend_routes_v1.hh>

#include <cassert>
#include <iostream>

using namespace Cellerator::compiler::backend::nvptx;
using namespace cellpack::persistence;

namespace {

nvidia_route_capability_v1 ready_capability(nvidia_optional_route_v1 route) {
    nvidia_route_capability_v1 capability;
    capability.route = route;
    capability.minimum_compute_major = 7u;
    capability.maximum_compute_major = 12u;
    capability.frontend_available = true;
    capability.device_library_available = true;
    capability.assembler_available = true;
    capability.linker_available = true;
    return capability;
}

nvidia_route_request_v1 route_request(nvidia_optional_route_v1 route,
                                      std::uint64_t identity) {
    nvidia_route_request_v1 request;
    request.route = route;
    request.compute_major = 7u;
    request.realization_module_identity = identity;
    request.source_map = {identity + 1u, identity + 2u};
    return request;
}

execution_capability_manifest_v1 installed_sm70_provider() {
    execution_capability_manifest_v1 manifest{};
    manifest.schema_version = execution_capability_manifest_v1_schema_version;
    manifest.record_bytes = sizeof(manifest);
    manifest.endian = execution_capability_manifest_v1_endian_marker;
    manifest.flags = capability_source_linked_implementation |
        capability_fragment_layout_opaque | capability_requires_converged_collective |
        capability_memory_interface_present;
    manifest.provider_identity_low = 1u;
    manifest.provider_abi_identity_low = 2u;
    manifest.capability_identity_low = 3u;
    manifest.hardware_compatibility_identity_low = 4u;
    manifest.runtime_build_identity_low = 5u;
    manifest.kernel_build_identity_low = 6u;
    manifest.memory_interface_identity_low = 7u;
    manifest.vendor = execution_capability_vendor_v1::nvidia;
    manifest.architecture_class = 70u;
    manifest.minimum_compute_capability_major = 7u;
    manifest.maximum_compute_capability_major = 7u;
    manifest.instruction_family = execution_instruction_family_v1::nvidia_wmma;
    manifest.collective_scope = execution_collective_scope_v1::warp;
    manifest.collective_threads = 32u;
    manifest.instruction_m = 16u;
    manifest.instruction_n = 16u;
    manifest.instruction_k = 16u;
    manifest.relation_storage_type = execution_capability_numeric_type_v1::f16;
    manifest.dense_input_type = execution_capability_numeric_type_v1::f16;
    manifest.accumulation_type = execution_capability_numeric_type_v1::f32;
    manifest.output_type = execution_capability_numeric_type_v1::f32;
    manifest.operand_a_layout = execution_matrix_layout_v1::row_major;
    manifest.operand_b_layout = execution_matrix_layout_v1::row_major;
    manifest.accumulation_layout = execution_matrix_layout_v1::opaque;
    manifest.output_layout = execution_matrix_layout_v1::row_major;
    manifest.instruction_sparsity = execution_instruction_sparsity_v1::dense;
    manifest.required_engine_capability = 1u;
    manifest.memory_interface_flags = 1u;
    return manifest;
}

nvptx_route_comparison_v1 negative_direct_evidence() {
    nvptx_route_comparison_v1 evidence;
    evidence.disposition = nvptx_route_promotion_v1::evaluated_not_promoted;
    evidence.selected_route = nvptx_route_v1::nvcc;
    evidence.regime = "V100 sm_70 matched relation apply";
    evidence.reason = "direct PTX did not clear the promotion threshold";
    evidence.measurements.push_back({nvptx_route_v1::direct_ptx, "ptxas-12.9",
        12714679u, 3112u, 14u, 0u, 0u, 8414u, 2u, 2u, true, true, false});
    return evidence;
}

}  // namespace

int main() {
    optional_nvidia_backend_freeze_request_v1 host_only;
    host_only.realization_ir_interface_identity = 24u;
    host_only.backend_abi_interface_identity = 25u;
    host_only.host_backend_available = true;
    const auto host_receipt = freeze_optional_nvidia_backend_routes_v1(host_only);
    assert(host_receipt && host_receipt.host_build_supported &&
           !host_receipt.nvcc_build_supported);
    for (const auto& route : host_receipt.optional_routes) {
        assert(!route.mandatory && route.status ==
               frozen_nvidia_route_status_v1::unavailable);
    }

    auto provider = installed_sm70_provider();
    assert(validate_execution_capability_manifest_v1(provider));
    auto full = host_only;
    full.nvcc_backend_available = true;
    full.clang_cuda_request = route_request(nvidia_optional_route_v1::clang_cuda, 31u);
    full.clang_cuda_capability = ready_capability(nvidia_optional_route_v1::clang_cuda);
    full.llvm_nvptx_request = route_request(nvidia_optional_route_v1::llvm_nvptx, 41u);
    full.llvm_nvptx_capability = ready_capability(nvidia_optional_route_v1::llvm_nvptx);
    full.installed_provider_manifest = &provider;
    full.direct_ptx_evidence = negative_direct_evidence();
    full.direct_ptx_implementation_identity = 51u;
    full.direct_ptx_source_map = {52u, 53u};
    const auto full_receipt = freeze_optional_nvidia_backend_routes_v1(full);
    assert(full_receipt && full_receipt.host_build_supported &&
           full_receipt.nvcc_build_supported);
    assert(full_receipt.optional_routes[0].status ==
           frozen_nvidia_route_status_v1::available);
    assert(full_receipt.optional_routes[1].status ==
           frozen_nvidia_route_status_v1::available);
    assert(full_receipt.optional_routes[2].status ==
           frozen_nvidia_route_status_v1::evaluated_not_promoted);
    assert(!full_receipt.optional_routes[2].promoted &&
           !full_receipt.optional_routes[2].mandatory);

    full.backend_abi_interface_identity = 0u;
    const auto invalid = freeze_optional_nvidia_backend_routes_v1(full);
    assert(!invalid && invalid.host_build_supported && invalid.diagnostics.size() == 1u);

    std::cout << "host/NVCC independence, optional routes, installed provider, provenance, "
                 "and negative direct-PTX promotion frozen\n";
}
