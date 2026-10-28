#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

std::filesystem::path find_root(std::filesystem::path start) {
    for (int depth = 0; depth != 12; ++depth) {
        if (std::filesystem::exists(start / "AGENTS.md") &&
            std::filesystem::exists(start / "CMakeLists.txt")) {
            return start;
        }
        if (!start.has_parent_path() || start == start.parent_path()) break;
        start = start.parent_path();
    }
    return {};
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream stream(path);
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    return buffer.str();
}

}  // namespace

int main(int argc, char** argv) {
    const auto root = find_root(argc > 1 ? std::filesystem::path(argv[1])
                                         : std::filesystem::current_path());
    if (root.empty()) {
        std::cerr << "unable to locate Cellerator repository root\n";
        return 1;
    }
    const std::string text = read_file(root /
        "docs/compiler/source-layout/define_central_registry_and_generated_manifest_ownership.md");
    if (text.empty()) {
        std::cerr << "empty or missing registry ownership receipt\n";
        return 1;
    }

    const std::vector<std::string_view> required = {
        "CMakeLists.txt", "cmake/compiler/CelleratorCompilerTargets.cmake",
        "include/Cellerator/compiler.hh", "include/Cellerator/Cellerator.hh",
        "canonical grammar/token registry", "canonical CEIR dialect/operation manifest",
        "canonical backend and pass registries",
        "cmake/package/CelleratorConfig.cmake.in", "stdlib/manifest.json",
        ".gitmodules", "components/CellShard", "tests/CMakeLists.txt",
        "bench/CMakeLists.txt", "tools/CMakeLists.txt",
        "CE-CCP1-L-INTEGRATE-FOUNDATION",
        "src/compiler/frontend/parser/fragments/<stable-id>.json",
        "src/compiler/ir/<level>/fragments/<stable-id>.json",
        "src/compiler/backend/<provider>/fragments/<stable-id>.json",
        "src/compiler/pass/fragments/<stable-id>.json",
        "stdlib/cellerator/<area>/<stable-id>.cell",
        "stable identifier, schema version, owning task",
        "reject duplicate", "sort by stable identifier", "byte-identical",
        "requests integration through", "Project Control",
        "previous generated", "Existing runtime/provider registries",
        "no Part Two JIT"
    };
    bool valid = true;
    for (const auto item : required) {
        if (text.find(item) == std::string::npos) {
            std::cerr << "missing central ownership contract: " << item << '\n';
            valid = false;
        }
    }
    if (!valid) return 1;
    std::cout << "validated central registry ownership and isolated fragment routes\n";
    return 0;
}
