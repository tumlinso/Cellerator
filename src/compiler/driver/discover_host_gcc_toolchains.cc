#include <Cellerator/compiler/driver/discover_host_gcc_toolchains_v1.hh>
namespace cellerator::compiler::driver {
gcc_toolchain_v1 discover_host_gcc_v1(const gcc_discovery_input_v1& input, const gcc_executable_probe_v1& exists) {
    for (const auto& root : input.roots) {
        const auto cxx = root + "/bin/g++"; const auto cc = root + "/bin/gcc";
        if (!exists(cxx) || !exists(cc)) continue;
        return {cxx, cc, root + "/bin/ld", root + "/include/c++", input.target_triple,
                input.version_identity, input.libstdcxx_abi_mode, {}};
    }
    gcc_toolchain_v1 out; out.diagnostic = "host GCC toolchain unavailable"; return out;
}
}  // namespace cellerator::compiler::driver
