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
            std::filesystem::exists(start / "docs")) {
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
    std::cerr << "missing resource-layout contract: " << needle << '\n';
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
        "docs/compiler/source-layout/freeze_standard_library_and_resource_locations.md";
    const std::string text = read_file(receipt);
    if (text.empty()) {
        std::cerr << "empty or missing resource-layout receipt\n";
        return 1;
    }

    const std::vector<std::string_view> required = {
        "stdlib/", "manifest.json", "cellerator/", "core.cell", "domain/",
        "relation/", "operation/", "profile/", "planning/", "reflection/",
        "ir/", "profiles/", "reference/", "schemas/", "`.cell` suffix",
        "${CMAKE_INSTALL_DATADIR}/cellerator/stdlib/",
        "${CMAKE_INSTALL_DATADIR}/cellerator/profiles/reference/",
        "${CMAKE_INSTALL_DATADIR}/cellerator/schemas/",
        "stable logical resource identity", "content hash",
        "Moving an installation does",
        "Missing, ambiguous, stale, or", "low-performance",
        "does not own or", "no fixed host path", "Part Two JIT"
    };
    bool valid = true;
    for (const auto item : required) {
        valid = require_contains(text, item) && valid;
    }
    for (const auto forbidden : {"/home/", "/usr/local/", "../share/cellerator"}) {
        if (text.find(forbidden) != std::string::npos) {
            std::cerr << "non-relocatable resource path in receipt: " << forbidden
                      << '\n';
            valid = false;
        }
    }

    if (!valid) {
        return 1;
    }
    std::cout << "validated relocatable standard-library and resource locations\n";
    return 0;
}
