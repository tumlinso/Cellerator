#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>

namespace {

std::filesystem::path find_root(std::filesystem::path start) {
    for (int depth = 0; depth != 12; ++depth) {
        if (std::filesystem::exists(start / "AGENTS.md") &&
            std::filesystem::exists(start / "bench")) {
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

std::size_t occurrences(const std::string& text, const std::string& needle) {
    std::size_t count = 0;
    for (std::size_t position = 0;
         (position = text.find(needle, position)) != std::string::npos;
         position += needle.size()) {
        ++count;
    }
    return count;
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
    const std::string text = read_file(root /
        "docs/compiler/source-layout/freeze_compiler_test_and_benchmark_locations.md");
    if (text.empty()) {
        std::cerr << "empty or missing test/benchmark layout receipt\n";
        return 1;
    }

    constexpr std::array<std::string_view, 9> families = {
        "frontend", "ir", "profile", "planning", "realization", "backends",
        "lto", "sdk", "language_server"
    };
    bool valid = true;
    for (const auto family : families) {
        const std::string test_path = "`tests/compiler/" + std::string(family) + "/`";
        const std::string bench_path = "`bench/compiler/" + std::string(family) + "/`";
        if (occurrences(text, test_path) != 1 || occurrences(text, bench_path) != 1) {
            std::cerr << "family does not have exactly one owner pair: " << family
                      << '\n';
            valid = false;
        }
    }

    constexpr std::array<std::string_view, 9> required = {
        "tests/compiler/a04/", "bootstrap validation receipts",
        "host-only compiler tests must not acquire CUDA dependencies",
        "repository benchmark mutex", "toolchain, build mode, input shape",
        "Existing runtime, geometry, JBC, CE-EXOP, CE-GEO, and CE-LIVE",
        "tests/CMakeLists.txt", "bench/CMakeLists.txt", "No Part Two JIT"
    };
    for (const auto item : required) {
        if (text.find(item) == std::string::npos) {
            std::cerr << "missing test/benchmark contract: " << item << '\n';
            valid = false;
        }
    }

    if (!valid) {
        return 1;
    }
    std::cout << "validated compiler test and benchmark owner mapping\n";
    return 0;
}
