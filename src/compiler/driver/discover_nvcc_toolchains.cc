#include <Cellerator/compiler/driver/discover_nvcc_toolchains_v1.hh>
#include <algorithm>
namespace cellerator::compiler::driver {
nvcc_toolchain_v1 discover_nvcc_v1(const nvcc_discovery_input_v1& in, const nvcc_executable_probe_v1& exists) {
    std::vector<cuda_installation_v1> candidates;
    if (!in.explicit_root.empty()) candidates.push_back({in.explicit_root, "explicit", 0, 0, {}});
    candidates.insert(candidates.end(), in.installations.begin(), in.installations.end());
    for (const auto& item : candidates) {
        const auto nvcc = item.root + "/bin/nvcc";
        if (!exists(nvcc)) continue;
        if ((item.minimum_host_major && in.host_compiler_major < item.minimum_host_major) ||
            (item.maximum_host_major && in.host_compiler_major > item.maximum_host_major)) {
            return {{}, {}, {}, {}, {}, {}, "host compiler major " + std::to_string(in.host_compiler_major) + " is incompatible with CUDA " + item.version_identity};
        }
        if (!item.architectures.empty() && std::find(item.architectures.begin(), item.architectures.end(), in.requested_architecture) == item.architectures.end()) {
            return {{}, {}, {}, {}, {}, {}, "requested architecture is unsupported by CUDA " + item.version_identity};
        }
        return {nvcc, item.root, item.root + "/bin/ptxas", item.root + "/bin/nvlink", item.root + "/bin/fatbinary", item.version_identity, {}};
    }
    return {{}, {}, {}, {}, {}, {}, "NVCC toolchain unavailable"};
}
}  // namespace cellerator::compiler::driver
