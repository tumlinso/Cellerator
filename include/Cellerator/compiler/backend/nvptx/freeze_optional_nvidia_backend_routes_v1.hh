#pragma once

#include <Cellerator/compiler/backend/nvptx/compare_nvcc_clang_cuda_and_direct_ptx_routes_v1.hh>
#include <Cellerator/compiler/backend/nvptx/freeze_clang_cuda_and_nvptx_backend_contracts_v1.hh>
#include <Cellerator/geometry/persistence/execution_capability_manifest_v1.hh>

#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Cellerator::compiler::backend::nvptx {

inline constexpr std::uint32_t optional_nvidia_backend_routes_version_v1 = 1u;

enum class frozen_nvidia_route_v1 : std::uint8_t {
    clang_cuda = 1u,
    llvm_nvptx,
    direct_ptx,
};

enum optional_nvidia_subset_v1 : std::uint32_t {
    subset_cuda_source_action = 1u << 0u,
    subset_llvm_module_boundary = 1u << 1u,
    subset_typed_ptx_operation = 1u << 2u,
    subset_inline_ptx_binding = 1u << 3u,
    subset_deterministic_ptx_assembly = 1u << 4u,
    subset_object_embedding = 1u << 5u,
};

enum class frozen_nvidia_route_status_v1 : std::uint8_t {
    available = 0u,
    unavailable,
    evaluated_not_promoted,
};

struct frozen_nvidia_route_record_v1 {
    frozen_nvidia_route_v1 route = frozen_nvidia_route_v1::clang_cuda;
    std::uint32_t supported_subsets = 0u;
    frozen_nvidia_route_status_v1 status = frozen_nvidia_route_status_v1::unavailable;
    std::uint32_t minimum_compute_major = 0u;
    std::uint32_t minimum_compute_minor = 0u;
    std::uint32_t maximum_compute_major = 0u;
    std::uint32_t maximum_compute_minor = 0u;
    std::uint64_t implementation_identity = 0u;
    source_map_identity_v1 source_map{};
    bool mandatory = false;
    bool promoted = false;
};

struct optional_nvidia_backend_freeze_request_v1 {
    std::uint64_t realization_ir_interface_identity = 0u;
    std::uint64_t backend_abi_interface_identity = 0u;
    bool host_backend_available = false;
    bool nvcc_backend_available = false;
    nvidia_route_request_v1 clang_cuda_request{};
    nvidia_route_capability_v1 clang_cuda_capability{};
    nvidia_route_request_v1 llvm_nvptx_request{};
    nvidia_route_capability_v1 llvm_nvptx_capability{};
    const cellpack::persistence::execution_capability_manifest_v1*
        installed_provider_manifest = nullptr;
    nvptx_route_comparison_v1 direct_ptx_evidence{};
    std::uint64_t direct_ptx_implementation_identity = 0u;
    source_map_identity_v1 direct_ptx_source_map{};
};

struct optional_nvidia_backend_receipt_v1 {
    std::uint32_t contract_version = optional_nvidia_backend_routes_version_v1;
    bool frozen = false;
    bool host_build_supported = false;
    bool nvcc_build_supported = false;
    std::uint64_t realization_ir_interface_identity = 0u;
    std::uint64_t backend_abi_interface_identity = 0u;
    std::array<frozen_nvidia_route_record_v1, 3u> optional_routes{};
    std::vector<std::string> diagnostics;

    explicit operator bool() const noexcept { return frozen; }
};

[[nodiscard]] optional_nvidia_backend_receipt_v1
freeze_optional_nvidia_backend_routes_v1(
    const optional_nvidia_backend_freeze_request_v1& request);

}  // namespace Cellerator::compiler::backend::nvptx
