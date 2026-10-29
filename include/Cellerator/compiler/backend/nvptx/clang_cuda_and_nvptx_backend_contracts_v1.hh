#pragma once

#include <cstdint>

namespace Cellerator::compiler::backend::nvptx {

inline constexpr std::uint32_t nvidia_optional_backend_contract_version_v1 = 1u;

enum class nvidia_optional_route_v1 : std::uint8_t {
    clang_cuda = 1u,
    llvm_nvptx,
};

struct source_map_identity_v1 {
    std::uint64_t high = 0u;
    std::uint64_t low = 0u;
};

struct nvidia_route_request_v1 {
    nvidia_optional_route_v1 route = nvidia_optional_route_v1::clang_cuda;
    std::uint32_t compute_major = 0u;
    std::uint32_t compute_minor = 0u;
    std::uint64_t realization_module_identity = 0u;
    source_map_identity_v1 source_map{};
};

struct nvidia_route_capability_v1 {
    nvidia_optional_route_v1 route = nvidia_optional_route_v1::clang_cuda;
    std::uint32_t contract_version = nvidia_optional_backend_contract_version_v1;
    std::uint32_t minimum_compute_major = 0u;
    std::uint32_t minimum_compute_minor = 0u;
    std::uint32_t maximum_compute_major = 0u;
    std::uint32_t maximum_compute_minor = 0u;
    bool frontend_available = false;
    bool device_library_available = false;
    bool assembler_available = false;
    bool linker_available = false;
};

enum class nvidia_route_probe_status_v1 : std::uint8_t {
    ready = 0u,
    unavailable,
    invalid_request,
    unsupported_contract,
    unsupported_target,
};

[[nodiscard]] nvidia_route_probe_status_v1 probe_nvidia_optional_route_v1(
    const nvidia_route_request_v1& request,
    const nvidia_route_capability_v1& capability) noexcept;

}  // namespace Cellerator::compiler::backend::nvptx
