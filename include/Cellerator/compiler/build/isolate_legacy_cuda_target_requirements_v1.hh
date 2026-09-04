#pragma once

namespace Cellerator::compiler::build {

struct legacy_cuda_isolation_v1 {
    bool discovery_is_target_scoped;
    bool architecture_is_target_scoped;
    bool provider_manifest_is_conditional;
    bool host_only_performs_cuda_detection;
};

inline constexpr legacy_cuda_isolation_v1 legacy_cuda_isolation_contract_v1{
    true, true, true, false,
};

}  // namespace Cellerator::compiler::build
