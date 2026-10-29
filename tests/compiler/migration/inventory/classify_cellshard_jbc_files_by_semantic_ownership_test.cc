#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr const char* common_base =
    "7762a5925fe18b2ca45ab8a436f3461804ed2ad9";

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool contains_any(const std::string& value,
                  const std::initializer_list<const char*> needles) {
    for (const auto* needle : needles) {
        if (value.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string basename(const std::string& path) {
    const auto separator = path.find_last_of('/');
    return separator == std::string::npos ? path : path.substr(separator + 1);
}

std::vector<std::string> dispositions(const std::string& path) {
    std::vector<std::string> matches;
    const auto add = [&](bool condition, const char* disposition) {
        if (condition) {
            matches.emplace_back(disposition);
        }
    };

    add(starts_with(path, "include/CellShard/compiler/discovery/"),
        "compiler discovery");
    add(starts_with(path, "include/CellShard/compiler/certification/"),
        "exact certification");
    add(starts_with(path, "include/CellShard/compiler/atom/") ||
            starts_with(path, "src/compiler/atom/") ||
            starts_with(path, "include/CellShard/compiler/partial/"),
        "atom semantics");
    add(starts_with(path, "include/CellShard/compiler/grammar/") ||
            starts_with(path, "src/compiler/grammar/") ||
            starts_with(path, "include/CellShard/compiler/composition/"),
        "grammar/composition");
    add(starts_with(path, "include/CellShard/compiler/basis/"), "basis");
    add(starts_with(path, "include/CellShard/compiler/graph/") ||
            starts_with(path, "src/compiler/graph/") ||
            starts_with(path, "include/CellShard/compiler/schedule/") ||
            starts_with(path, "src/compiler/schedule/"),
        "global program/schedule");
    add(starts_with(path, "include/CellShard/artifact/atom_store/") ||
            starts_with(path, "src/artifact/atom_store/") ||
            path == "docs/SPEC_ATOM_STORE_V1.md",
        "concrete storage");

    const bool runtime = starts_with(path, "include/CellShard/runtime/v2/") ||
                         starts_with(path, "src/runtime/v2/");
    const bool materialization = runtime && contains_any(
        basename(path), {"async_file_atom_source", "atom_source.", "command_ir",
                         "exact_read_baseline", "read_plan", "runtime_recovery",
                         "worker_cuda_graph"});
    add(materialization, "concrete materialization");
    add(runtime && !materialization, "transport/residency");

    add(path == "CMakeLists.txt" || path == "include/CellShard/CellShard.hh" ||
            starts_with(path, "include/CellShard/interop/"),
        "bridge");
    add(starts_with(path, "tests/"), "test");
    add(starts_with(path, "bench/") ||
            starts_with(path, "docs/JBC/evidence/") ||
            starts_with(path, "include/CellShard/compiler/evidence/") ||
            starts_with(path, "src/compiler/evidence/") ||
            starts_with(path, "todos/") ||
            path == ".todo-orchestrator/state.snapshot.json" ||
            path == "todos.md" || path == "todo-status.md",
        "evidence");
    return matches;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char character : value) {
        quoted += character == '\'' ? "'\\''" : std::string(1, character);
    }
    return quoted + "'";
}

std::string run(const std::string& command) {
    std::array<char, 4096> buffer{};
    std::string output;
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr) {
        throw std::runtime_error("could not execute: " + command);
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }
    if (pclose(pipe) != 0) {
        throw std::runtime_error("command failed: " + command);
    }
    while (!output.empty() && (output.back() == '\n' || output.back() == '\r')) {
        output.pop_back();
    }
    return output;
}

std::vector<std::string> lines(const std::string& value) {
    std::vector<std::string> result;
    std::istringstream stream(value);
    for (std::string line; std::getline(stream, line);) {
        if (!line.empty()) {
            result.push_back(line);
        }
    }
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: classification_test <receipt> <CellShard repository>");
        const std::string git = "git -C " + shell_quote(argv[2]) + " ";
        const auto branches = lines(run(
            git + "for-each-ref --format='%(refname:short)' 'refs/heads/jbc/*'"));
        require(branches.size() == 24, "expected 24 local JBC branches");

        std::set<std::string> paths;
        for (const auto& branch : branches) {
            for (const auto& path : lines(run(
                     git + "diff --name-only " + common_base + ".." +
                     shell_quote(branch)))) {
                paths.insert(path);
            }
        }
        require(paths.size() == 979, "unexpected unique changed-path count");

        std::map<std::string, int> counts;
        for (const auto& path : paths) {
            const auto matches = dispositions(path);
            require(matches.size() == 1,
                    path + " has " + std::to_string(matches.size()) +
                        " primary dispositions");
            ++counts[matches.front()];
        }

        const std::map<std::string, int> expected{{
            {"compiler discovery", 82},
            {"exact certification", 16},
            {"atom semantics", 38},
            {"grammar/composition", 51},
            {"basis", 17},
            {"global program/schedule", 19},
            {"concrete storage", 43},
            {"concrete materialization", 13},
            {"transport/residency", 24},
            {"bridge", 3},
            {"test", 328},
            {"evidence", 345},
        }};
        require(counts == expected, "semantic-disposition counts changed");

        const std::string digest_command =
            "sh -c " + shell_quote(
                "for branch in $(git -C " + shell_quote(argv[2]) +
                " for-each-ref --format='%(refname:short)' 'refs/heads/jbc/*'); do "
                "git -C " + shell_quote(argv[2]) + " diff --name-only " + common_base +
                "..\"$branch\"; done | sort -u | sha256sum | cut -d' ' -f1");
        require(run(digest_command) ==
                    "af783b7c35be048289a8da5798e8b11c7895846f0d42d938dc6a235e73a5aee9",
                "changed-path source-set digest mismatch");

        std::ifstream receipt_stream(argv[1]);
        require(receipt_stream.good(), "could not open classification receipt");
        const std::string receipt((std::istreambuf_iterator<char>(receipt_stream)),
                                  std::istreambuf_iterator<char>());
        for (const auto& [name, count] : expected) {
            require(receipt.find("| " + name + " | " + std::to_string(count) + " |") !=
                        std::string::npos,
                    "receipt omits disposition: " + name);
        }

        std::cout << "classified 979 CellShard JBC paths into 12 exclusive dispositions\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
