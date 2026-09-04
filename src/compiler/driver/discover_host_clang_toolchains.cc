#include <Cellerator/compiler/driver/discover_host_clang_toolchains_v1.hh>
#include <array>
namespace cellerator::compiler::driver {
clang_toolchain_v1 discover_host_clang_v1(const clang_discovery_input_v1& in, const executable_probe_v1& exists) {
    std::vector<std::pair<std::string, discovery_source_v1>> roots;
    roots.emplace_back(in.explicit_root, discovery_source_v1::explicit_override);
    roots.emplace_back(in.environment_root, discovery_source_v1::environment);
    roots.emplace_back(in.configured_root, discovery_source_v1::configured_resource);
    for (const auto& root : in.path_roots) roots.emplace_back(root, discovery_source_v1::path);
    for (const auto& root : in.platform_roots) roots.emplace_back(root, discovery_source_v1::platform_default);
    for (const auto& [root, source] : roots) {
        if (root.empty()) continue;
        const auto compiler = root + "/bin/clang++";
        if (!exists(compiler)) continue;
        return {source, compiler, root + "/bin/ld.lld", root + "/lib/clang",
                "configured-target", "clang@" + root};
    }
    return {};
}
}  // namespace cellerator::compiler::driver
