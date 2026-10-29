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
            std::filesystem::exists(start / "scope.md")) return start;
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
        "docs/compiler/source-layout/define_agents_and_ownership_inheritance.md");
    const std::vector<std::string_view> required = {
        "repository-root `AGENTS.md` remains", "Project Control authority",
        "never grants access", "include/Cellerator/compiler/AGENTS.md",
        "src/compiler/AGENTS.md", "src/compiler/ir/AGENTS.md",
        "src/compiler/backend/AGENTS.md", "src/compiler/tooling/AGENTS.md",
        "docs/compiler/migration/AGENTS.md", "public writable IR",
        "CPU/NVCC/NVPTX isolation", "no NVCC source parsing",
        "host-only protocol/queries", "no deletion before replacement proof",
        "docs/architecture.qmd", "docs/current_implementation.qmd",
        "docs/migration_roadmap.qmd", ".gitmodules", "CMakeLists.txt",
        "src/CMakeLists.txt", "tests/CMakeLists.txt", "bench/CMakeLists.txt",
        "tools/CMakeLists.txt", "cmake/compiler/CelleratorCompilerTargets.cmake",
        "cmake/package/CelleratorConfig.cmake.in",
        "cmake/package/CelleratorConfigVersion.cmake.in",
        "include/Cellerator/compiler.hh", "include/Cellerator/Cellerator.hh",
        "stdlib/manifest.json", "components/CellShard gitlink",
        "integration through Project", "src/geometry/AGENTS.md",
        "current runtime or JBC behavior", "no Part Two JIT",
        "does not create those files"
    };
    bool valid = !text.empty();
    for (const auto item : required) {
        if (text.find(item) == std::string::npos) {
            std::cerr << "missing ownership-inheritance contract: " << item << '\n';
            valid = false;
        }
    }
    if (!std::filesystem::exists(root / "AGENTS.md") ||
        !std::filesystem::exists(root / "src/geometry/AGENTS.md")) {
        std::cerr << "existing inherited guidance is missing\n";
        valid = false;
    }
    if (!valid) return 1;
    std::cout << "validated compiler AGENTS inheritance and central exclusions\n";
    return 0;
}
