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
            std::filesystem::exists(start / "src")) {
            return start;
        }
        if (!start.has_parent_path() || start == start.parent_path()) {
            break;
        }
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

bool require_contains(const std::string& text, std::string_view needle) {
    if (text.find(needle) != std::string::npos) {
        return true;
    }
    std::cerr << "missing implementation-layout contract: " << needle << '\n';
    return false;
}

bool is_installed_header(const std::filesystem::path& path) {
    const auto extension = path.extension().string();
    return extension == ".h" || extension == ".hh" || extension == ".hpp" ||
           extension == ".cuh";
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
        "docs/compiler/source-layout/freeze_the_compiler_implementation_tree.md";
    const std::string text = read_file(receipt);
    if (text.empty()) {
        std::cerr << "empty or missing implementation-layout receipt\n";
        return 1;
    }

    const std::vector<std::string_view> required = {
        "src/compiler/", "driver/", "frontend/", "source/", "parser/",
        "cxx/", "ast/", "sema/", "field/", "ir/", "common/", "text/",
        "semantic/", "planning/", "realization/", "profile/", "discovery/",
        "composition/", "program/", "reflection/", "pass/", "diagnostics/",
        "lto/", "backend/", "cpu/", "nvcc/", "nvptx/", "tooling/",
        "api/", "support/", "include/Cellerator/compiler/",
        "must never include a path", "provider selection crosses the public",
        "Nothing under `src/compiler/` is an install candidate",
        "src/execution/", "src/geometry/", "src/compute/", "src/planner/",
        "src/runtime/", "preserves current runtime and JBC behavior",
        "no Part Two JIT"
    };
    bool valid = true;
    for (const auto item : required) {
        valid = require_contains(text, item) && valid;
    }

    const auto include_root = root / "include/Cellerator";
    for (const auto& entry : std::filesystem::recursive_directory_iterator(include_root)) {
        if (!entry.is_regular_file() || !is_installed_header(entry.path())) {
            continue;
        }
        const auto header = read_file(entry.path());
        if (header.find("#include \"src/compiler/") != std::string::npos ||
            header.find("#include <src/compiler/") != std::string::npos) {
            std::cerr << "installed header leaks compiler implementation path: "
                      << entry.path() << '\n';
            valid = false;
        }
    }

    if (!valid) {
        return 1;
    }
    std::cout << "validated compiler implementation tree and SDK leak boundary\n";
    return 0;
}
