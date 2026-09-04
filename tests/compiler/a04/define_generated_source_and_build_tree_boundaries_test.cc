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
            std::filesystem::exists(start / ".gitignore")) return start;
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
        "docs/compiler/source-layout/define_generated_source_and_build_tree_boundaries.md");
    const std::vector<std::string_view> required = {
        "build/generated/Cellerator/compiler/", "frontend/parser/",
        "ir/dialects/", "backend/", "resources/", "build/",
        "${CMAKE_CURRENT_BINARY_DIR}/generated/Cellerator/compiler/",
        "No generator writes", "src/compiler/frontend/parser/grammar/",
        "src/compiler/frontend/parser/schemas/",
        "src/compiler/ir/<level>/fragments/",
        "src/compiler/backend/<provider>/fragments/", "stdlib/manifest.json",
        "cmake/compiler/templates/", "not committed", "explicit, sorted input list",
        "rejects duplicate", "avoids timestamps, absolute paths",
        "writes atomically", "separate empty binary roots", "SHA-256",
        "git status --porcelain=v1 --untracked-files=all",
        "git ls-files 'build/generated/**'", "build-interface include paths",
        "current build already uses", "no Part Two JIT"
    };
    bool valid = !text.empty();
    for (const auto item : required) {
        if (text.find(item) == std::string::npos) {
            std::cerr << "missing generated-source contract: " << item << '\n';
            valid = false;
        }
    }
    const std::string ignore = read_file(root / ".gitignore");
    if (ignore.find("build/") == std::string::npos) {
        std::cerr << "repository does not ignore the build output root\n";
        valid = false;
    }
    if (!valid) return 1;
    std::cout << "validated generated-source and build-tree boundary\n";
    return 0;
}
