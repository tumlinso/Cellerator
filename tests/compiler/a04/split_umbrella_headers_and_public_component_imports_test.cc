#include <Cellerator/abi.h>
#include <Cellerator/execution/identity.hh>

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
            std::filesystem::exists(start / "include/Cellerator/Cellerator.hh")) return start;
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
        "docs/compiler/source-layout/split_umbrella_headers_and_public_component_imports.md");
    const std::vector<std::string_view> required = {
        "#include <Cellerator/compiler.hh>", "#include <Cellerator/runtime.hh>",
        "#include <Cellerator/Cellerator.hh>", "host-safe compiler contracts",
        "includes `compiler.hh` and `runtime.hh`", "Small-umbrella rule",
        "never every", "CUDA compiler", "CellShard", "CelleraTorch",
        "#include <Cellerator/compiler/ir/common/ir_v1.hh>",
        "#include <Cellerator/compiler/profile/profile_artifact_v1.hh>",
        "#include <Cellerator/compiler/planning/planner_v1.hh>",
        "#include <Cellerator/compiler/backend/backend_v1.hh>",
        "#include <Cellerator/execution/program.hh>",
        "#include <Cellerator/geometry/semantic_geometry.hh>",
        "#include <Cellerator/runtime/session.cuh>",
        "compiler_umbrella_include_test.cc", "runtime_umbrella_include_test.cc",
        "cellerator_umbrella_include_test.cc", "trivial `main`",
        "current `include/Cellerator/Cellerator.hh` enumerates",
        "does not edit that header", "No current runtime or JBC behavior",
        "Part Two JIT"
    };
    bool valid = !text.empty();
    for (const auto item : required) {
        if (text.find(item) == std::string::npos) {
            std::cerr << "missing umbrella contract: " << item << '\n';
            valid = false;
        }
    }
    if (!valid) return 1;
    std::cout << "validated split umbrella and component import contracts\n";
    return 0;
}
