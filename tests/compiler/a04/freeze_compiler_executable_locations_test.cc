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
            std::filesystem::exists(start / "tools")) {
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
    std::cerr << "missing executable-layout contract: " << needle << '\n';
    return false;
}

bool is_cpp_source(const std::filesystem::path& path) {
    const auto extension = path.extension().string();
    return extension == ".cc" || extension == ".cpp" || extension == ".cxx" ||
           extension == ".h" || extension == ".hh" || extension == ".hpp" ||
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
        "docs/compiler/source-layout/freeze_compiler_executable_locations.md";
    const std::string text = read_file(receipt);
    if (text.empty()) {
        std::cerr << "empty or missing executable-layout receipt\n";
        return 1;
    }

    const std::vector<std::string_view> required = {
        "tools/cellerator/main.cc", "tools/celleratord/main.cc",
        "thin applications over", "translate `argc` and `argv`",
        "must not own parsing", "include/Cellerator/compiler/",
        "src/compiler/", "only production C++", "is `main.cc`",
        "default to `cellerator` subcommands", "compiler libraries never link",
        "protocol and semantic code stays host-only", "tools/CMakeLists.txt",
        "preserves current runtime and JBC behavior", "no executable"
    };
    bool valid = true;
    for (const auto item : required) {
        valid = require_contains(text, item) && valid;
    }

    for (const auto tool : {"cellerator", "celleratord"}) {
        const auto directory = root / "tools" / tool;
        if (!std::filesystem::exists(directory)) {
            continue;
        }
        for (const auto& entry : std::filesystem::recursive_directory_iterator(directory)) {
            if (entry.is_regular_file() && is_cpp_source(entry.path()) &&
                entry.path().filename() != "main.cc") {
                std::cerr << "reusable compiler source leaked into executable tree: "
                          << entry.path() << '\n';
                valid = false;
            }
        }
    }

    if (!valid) {
        return 1;
    }
    std::cout << "validated thin compiler executable locations\n";
    return 0;
}
