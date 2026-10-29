#include <array>
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
            std::filesystem::exists(start / "include/Cellerator")) return start;
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

bool require_all(const std::string& text,
                 const std::vector<std::string_view>& values,
                 std::string_view label) {
    bool valid = true;
    for (const auto value : values) {
        if (text.find(value) == std::string::npos) {
            std::cerr << "missing " << label << ": " << value << '\n';
            valid = false;
        }
    }
    return valid;
}

}  // namespace

int main(int argc, char** argv) {
    const auto root = find_root(argc > 1 ? std::filesystem::path(argv[1])
                                         : std::filesystem::current_path());
    if (root.empty()) {
        std::cerr << "unable to locate Cellerator repository root\n";
        return 1;
    }
    const std::string contract = read_file(root /
        "docs/compiler/source-layout/SOURCE_LAYOUT_V1.md");
    const std::string receipt = read_file(root /
        "docs/compiler/source-layout/freeze_the_source_layout_and_ownership_map.md");

    bool valid = true;
    valid = require_all(contract, {
        "CE-CCP1-I04-SOURCE-LAYOUT", "Version: `1`", "CE-CCP1-A04-010",
        "CE-CCP1-I03-COMPILER-OWNERSHIP", "3478e9787fbee8e66f1c12dba0d69641d01605ef2316420f7d74cac02421b1d0",
        "include/Cellerator/compiler/", "src/compiler/", "tools/cellerator/main.cc",
        "tools/celleratord/main.cc", "stdlib/cellerator/", "profiles/reference/",
        "${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/compiler/",
        "tests/compiler/", "bench/compiler/", "JBC evidence and discovery",
        "JBC certification", "JBC atom semantic states", "JBC composition and grammar",
        "JBC basis", "JBC superatom composition", "JBC partial algebra/legality",
        "JBC graph compiler", "JBC schedule compiler", "atom store, materialization",
        "legacy JBC compiler includes", "compiled ruleset export",
        "CellShard owns storage/application mechanics", "Lane write scopes",
        "Central locks", "Dependency direction", ".gitmodules", "CMakeLists.txt",
        "include/Cellerator/compiler.hh", "include/Cellerator/Cellerator.hh",
        "Common IR cannot depend", "CPU, NVCC, and NVPTX providers",
        "Project Control claims", "does not claim absent compiler paths",
        "no", "general JIT", "deep CellShard runtime integration"
    }, "source-layout contract") && valid;

    valid = require_all(receipt, {
        "Collision analysis", "Concurrent-write analysis", "Dependency checks",
        "include/Cellerator/geometry/compiler/", "src/geometry/compiler/",
        "tests/compiler/", "compatible reserved compiler test root",
        "exactly one active integration owner", "stable-ID fragments",
        "CE-CCP1-CP-A03", "CE-CCP1-I03-COMPILER-OWNERSHIP",
        "CE-CCP1-I04-SOURCE-LAYOUT", "does not broaden or reinterpret"
    }, "freeze receipt") && valid;

    constexpr std::array<std::string_view, 9> detail_files = {
        "freeze_the_public_compiler_header_tree.md",
        "freeze_the_compiler_implementation_tree.md",
        "freeze_compiler_executable_locations.md",
        "freeze_standard_library_and_resource_locations.md",
        "freeze_compiler_test_and_benchmark_locations.md",
        "define_central_registry_and_generated_manifest_ownership.md",
        "split_umbrella_headers_and_public_component_imports.md",
        "define_generated_source_and_build_tree_boundaries.md",
        "define_agents_and_ownership_inheritance.md"
    };
    for (const auto file : detail_files) {
        if (!std::filesystem::exists(root / "docs/compiler/source-layout" / file) ||
            contract.find(file) == std::string::npos) {
            std::cerr << "missing source-linked detail: " << file << '\n';
            valid = false;
        }
    }

    if (std::filesystem::exists(root / "include/Cellerator/compiler") ||
        std::filesystem::exists(root / "src/compiler") ||
        std::filesystem::exists(root / "tools/cellerator") ||
        std::filesystem::exists(root / "tools/celleratord")) {
        std::cerr << "planned compiler root unexpectedly exists during layout freeze\n";
        valid = false;
    }
    for (const auto existing : {"include/Cellerator/execution", "include/Cellerator/geometry",
                                "include/Cellerator/compute", "include/Cellerator/planner",
                                "include/Cellerator/runtime"}) {
        if (!std::filesystem::is_directory(root / existing)) {
            std::cerr << "current owner root missing: " << existing << '\n';
            valid = false;
        }
    }

    if (!valid) return 1;
    std::cout << "validated source layout, collisions, writes, and dependencies\n";
    return 0;
}
