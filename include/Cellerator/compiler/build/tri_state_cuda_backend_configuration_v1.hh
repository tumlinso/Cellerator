#pragma once

#include <cstdint>

namespace Cellerator::compiler::build {

enum class cuda_enablement_v1 : std::uint8_t { automatic, enabled, disabled };
enum class cuda_configuration_v1 : std::uint8_t {
    host_only,
    accelerator_enabled,
    missing_required_toolchain,
};

[[nodiscard]] constexpr cuda_configuration_v1 resolve_cuda_configuration_v1(
    cuda_enablement_v1 mode, bool toolchain_available) noexcept {
    if (mode == cuda_enablement_v1::disabled) {
        return cuda_configuration_v1::host_only;
    }
    if (toolchain_available) {
        return cuda_configuration_v1::accelerator_enabled;
    }
    return mode == cuda_enablement_v1::enabled
               ? cuda_configuration_v1::missing_required_toolchain
               : cuda_configuration_v1::host_only;
}

}  // namespace Cellerator::compiler::build
