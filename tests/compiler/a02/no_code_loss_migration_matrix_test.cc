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

const std::set<std::string>& split_headers() {
    static const std::set<std::string> paths{
        "include/CellShard/compiler/evidence/algorithm_provenance_v1.hh",
        "include/CellShard/compiler/graph/operation_provider.hh",
        "include/CellShard/compiler/atom/identity_classes_v1.hh",
        "include/CellShard/compiler/atom/evidence_plane_v1.hh",
        "include/CellShard/compiler/graph/operation_node.hh",
        "include/CellShard/compiler/graph/physical_realization.hh",
        "include/CellShard/compiler/composition/coverage_v1.hh",
        "include/CellShard/compiler/composition/relation_merge_v1.hh",
        "include/CellShard/compiler/composition/production_identity_v1.hh",
        "include/CellShard/compiler/composition/persistent_order_link_v1.hh",
        "include/CellShard/compiler/composition/grammar_symbol_v1.hh",
        "include/CellShard/compiler/composition/segment_alignment_v1.hh",
        "include/CellShard/compiler/composition/identity_spine_join_v1.hh",
        "include/CellShard/compiler/composition/physical_view_addition_v1.hh",
    };
    return paths;
}

std::string disposition(const std::string& path) {
    if (path == "include/CellShard/interop/cellerator/evidence_adapter_v1.hh") {
        return "wrap temporarily";
    }
    if (path == "CMakeLists.txt" || path == "include/CellShard/CellShard.hh"
        || starts_with(path, "tests/jbc/validation/")
        || split_headers().count(path) != 0) {
        return "split";
    }
    if (path == "include/CellShard/compiler/atom/persistent_identity_v1.hh"
        || path == "include/CellShard/compiler/atom/logical_coverage_v1.hh"
        || path == "include/CellShard/compiler/evidence/atom_evidence_record_v1.hh"
        || path == "include/CellShard/compiler/discovery/operation_trace/cellerator_identity_adapter_v1.hh") {
        return "retain as compatibility";
    }
    if (path == "include/CellShard/compiler/composition/derivation_dag_v1.hh"
        || path == "include/CellShard/compiler/grammar/derivation_dag_v1.hh"
        || path == "include/CellShard/compiler/graph/graph_recipe.hh") {
        return "retire after replacement proof";
    }
    if (starts_with(path, "include/CellShard/compiler/")
        || starts_with(path, "src/compiler/")) {
        return "move";
    }
    if (starts_with(path, "tests/jbc/atom_store/")
        || starts_with(path, "tests/jbc/runtime/")
        || starts_with(path, "bench/jbc/runtime/")) {
        return "preserve in place";
    }
    if (starts_with(path, "tests/") || starts_with(path, "bench/")) {
        return "adapt";
    }
    if (starts_with(path, "include/CellShard/artifact/atom_store/")
        || starts_with(path, "src/artifact/atom_store/")
        || starts_with(path, "include/CellShard/runtime/v2/")
        || starts_with(path, "src/runtime/v2/")
        || starts_with(path, "docs/JBC/evidence/")
        || starts_with(path, "todos/")
        || path == "docs/SPEC_ATOM_STORE_V1.md"
        || path == ".todo-orchestrator/state.snapshot.json"
        || path == "todos.md" || path == "todo-status.md") {
        return "preserve in place";
    }
    return {};
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "usage: test RECEIPT CELLSHARD_ROOT");
        const std::string git = "git -C " + shell_quote(argv[2]) + " ";
        const auto branches = lines(run(
            git + "for-each-ref --format='%(refname:short)' 'refs/heads/jbc/*'"));
        require(branches.size() == 24, "expected 24 JBC branches");
        std::set<std::string> paths;
        for (const auto& branch : branches) {
            for (const auto& path : lines(run(
                     git + "diff --name-only " + common_base + ".." +
                     shell_quote(branch)))) {
                paths.insert(path);
            }
        }
        require(paths.size() == 979, "raw worktree inventory is not 979 paths");

        std::map<std::string, int> counts;
        for (const auto& path : paths) {
            const std::string result = disposition(path);
            require(!result.empty(), "unclassified migration path: " + path);
            ++counts[result];
        }
        int reconciled = 0;
        for (const auto& item : counts) {
            std::cerr << item.first << '=' << item.second << '\n';
            reconciled += item.second;
        }
        require(reconciled == 979, "migration counts do not reconcile");

        std::ifstream receipt_stream(argv[1]);
        require(receipt_stream.good(), "could not open migration matrix receipt");
        const std::string receipt((std::istreambuf_iterator<char>(receipt_stream)),
                                  std::istreambuf_iterator<char>());
        for (const auto& item : counts) {
            require(receipt.find("| " + item.first + " | "
                                 + std::to_string(item.second) + " |")
                        != std::string::npos,
                    "receipt count mismatch for " + item.first);
        }
        require(receipt.find("979 |") != std::string::npos,
                "receipt lacks exact raw inventory reconciliation");
        require(receipt.find("No source or evidence path may be deleted")
                    != std::string::npos,
                "no-code-loss deletion rule missing");

        std::cout << "reconciled 979 JBC paths into seven migration dispositions\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
