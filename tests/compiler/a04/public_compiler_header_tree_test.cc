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
            std::filesystem::exists(start / "include/Cellerator")) {
            return start;
        }
        if (!start.has_parent_path() || start == start.parent_path()) {
            break;
        }
        start = start.parent_path();
    }
    return {};
}

bool require_contains(const std::string& text, std::string_view needle) {
    if (text.find(needle) != std::string::npos) {
        return true;
    }
    std::cerr << "missing source-layout contract: " << needle << '\n';
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    const auto start = argc > 1 ? std::filesystem::path(argv[1])
                                : std::filesystem::current_path();
    const auto root = find_root(start);
    if (root.empty()) {
        std::cerr << "unable to locate Cellerator repository root\n";
        return 1;
    }

    const auto receipt = root /
        "docs/compiler/source-layout/freeze_the_public_compiler_header_tree.md";
    std::ifstream stream(receipt);
    if (!stream) {
        std::cerr << "unable to read " << receipt << '\n';
        return 1;
    }
    std::ostringstream buffer;
    buffer << stream.rdbuf();
    const std::string text = buffer.str();

    const std::vector<std::string_view> required_paths = {
        "include/Cellerator/compiler/",
        "driver/", "frontend/", "source/", "parser/", "cxx/",
        "ast/", "sema/", "field/", "ir/", "common/", "text/",
        "semantic/", "planning/", "realization/", "profile/",
        "discovery/", "composition/", "program/", "reflection/",
        "pass/", "diagnostics/", "lto/", "backend/", "cpu/",
        "nvcc/", "nvptx/", "tooling/", "api/", "build/"
    };
    bool valid = true;
    for (const auto path : required_paths) {
        valid = require_contains(text, path) && valid;
    }

    const std::vector<std::string_view> boundary_contracts = {
        "Files below `include/Cellerator/compiler/` are installed SDK contracts",
        "Installed headers must not include `src/compiler/`",
        "Implementation-only declarations live below `src/compiler/`",
        "`ir/planning/` is the data model; `planning/` is the",
        "include/Cellerator/execution/",
        "include/Cellerator/geometry/",
        "include/Cellerator/compute/",
        "include/Cellerator/planner/",
        "include/Cellerator/runtime/",
        "preserves all existing runtime and JBC behavior",
        "does not create a Clang fork",
        "Part Two JIT/runtime"
    };
    for (const auto contract : boundary_contracts) {
        valid = require_contains(text, contract) && valid;
    }

    const std::vector<std::filesystem::path> existing_contract_roots = {
        "include/Cellerator/execution",
        "include/Cellerator/geometry",
        "include/Cellerator/compute",
        "include/Cellerator/planner",
        "include/Cellerator/runtime"
    };
    for (const auto& path : existing_contract_roots) {
        if (!std::filesystem::is_directory(root / path)) {
            std::cerr << "documented existing contract root is absent: " << path
                      << '\n';
            valid = false;
        }
    }

    if (!valid) {
        return 1;
    }
    std::cout << "validated frozen public compiler header tree and SDK boundary\n";
    return 0;
}
