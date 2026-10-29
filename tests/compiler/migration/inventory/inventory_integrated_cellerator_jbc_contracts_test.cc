#include <array>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct manifest_row {
    std::string path;
    std::string sha256;
    std::string task_id;
    std::string line;
};

bool require(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

std::vector<std::string> code_spans(const std::string& line) {
    std::vector<std::string> result;
    std::size_t cursor = 0;
    while (true) {
        const std::size_t begin = line.find('`', cursor);
        if (begin == std::string::npos) {
            break;
        }
        const std::size_t end = line.find('`', begin + 1);
        if (end == std::string::npos) {
            break;
        }
        result.push_back(line.substr(begin + 1, end - begin - 1));
        cursor = end + 1;
    }
    return result;
}

std::map<std::string, manifest_row> parse_manifest(const std::string& document,
                                                   const std::string& prefix) {
    std::map<std::string, manifest_row> rows;
    std::istringstream input(document);
    std::string line;
    while (std::getline(input, line)) {
        const auto spans = code_spans(line);
        if (spans.size() < 3 || spans[0].rfind(prefix, 0) != 0) {
            continue;
        }
        rows.emplace(spans[0], manifest_row{spans[0], spans[1], spans[2], line});
    }
    return rows;
}

std::set<std::string> enumerate_files(const std::vector<std::filesystem::path>& roots) {
    std::set<std::string> result;
    for (const auto& root : roots) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_regular_file()) {
                result.insert(entry.path().generic_string());
            }
        }
    }
    return result;
}

bool valid_sha256(const std::string& value) {
    if (value.size() != 64) {
        return false;
    }
    for (const unsigned char character : value) {
        if (!std::isxdigit(character) || std::isupper(character)) {
            return false;
        }
    }
    return true;
}

std::string sha256sum(const std::string& path) {
    const std::string command = "sha256sum '" + path + "'";
    std::array<char, 256> output{};
    FILE* pipe = popen(command.c_str(), "r");
    if (pipe == nullptr || fgets(output.data(), static_cast<int>(output.size()), pipe) == nullptr) {
        if (pipe != nullptr) {
            pclose(pipe);
        }
        return {};
    }
    const int status = pclose(pipe);
    if (status != 0) {
        return {};
    }
    return std::string(output.data()).substr(0, 64);
}

bool validate_rows(const std::map<std::string, manifest_row>& rows,
                   const std::string& authority) {
    bool valid = true;
    for (const auto& [path, row] : rows) {
        valid &= require(valid_sha256(row.sha256), "invalid SHA-256 for " + path);
        valid &= require(row.task_id.rfind("CE-JBC-", 0) == 0,
                         "invalid JBC task mapping for " + path);
        valid &= require(authority.find("\n" + row.task_id + ",") != std::string::npos,
                         "mapped JBC task is absent from planning authority: " + row.task_id);
        valid &= require(std::filesystem::is_regular_file(path), "manifest path is absent: " + path);
        valid &= require(sha256sum(path) == row.sha256, "SHA-256 drift for " + path);
    }
    return valid;
}

}  // namespace

int main(int argc, char** argv) {
    const std::filesystem::path receipt = argc > 1
        ? argv[1]
        : "planning/cellerator-compiler-preledger-v1/inventories/"
          "inventory_integrated_cellerator_jbc_contracts.md";
    const std::filesystem::path authority = argc > 2
        ? argv[2]
        : "planning/jbc-preledger-v1/proposed_todos.csv";
    if (!std::filesystem::is_regular_file(receipt) || !std::filesystem::is_regular_file(authority)) {
        std::cerr << "inventory receipt or JBC planning authority is absent\n";
        return 1;
    }

    const std::string document = read_file(receipt);
    const std::string authority_text = "\n" + read_file(authority);
    const auto sources = parse_manifest(document, "include/Cellerator/");
    const auto tests = parse_manifest(document, "tests/jbc/");
    const auto expected_sources = enumerate_files({
        "include/Cellerator/compute/decomposition",
        "include/Cellerator/execution/joint_compiler",
        "include/Cellerator/planner/external_cost",
        "include/Cellerator/profiling/joint_compiler",
    });
    const auto expected_tests = enumerate_files({"tests/jbc"});
    std::set<std::string> observed_sources;
    std::set<std::string> observed_tests;
    bool valid = true;
    for (const auto& [path, row] : sources) {
        observed_sources.insert(path);
        valid &= require(row.line.find("tests/jbc/") != std::string::npos,
                         "contract lacks current validation mapping: " + path);
    }
    for (const auto& item : tests) {
        observed_tests.insert(item.first);
    }

    valid &= require(document.find("31e491ed29de0fcde70259cbeab8c5c7ad353485") != std::string::npos,
                     "missing observed Cellerator commit");
    valid &= require(document.find("b9749ad3e5146a04f847533d8c6f1a54146aed20") != std::string::npos,
                     "missing embedded CellShard gitlink identity");
    valid &= require(sources.size() == 37, "expected exactly 37 integrated contract rows");
    valid &= require(tests.size() == 94, "expected exactly 94 current JBC test rows");
    valid &= require(observed_sources == expected_sources,
                     "integrated contract manifest is not the exact declared file set");
    valid &= require(observed_tests == expected_tests,
                     "JBC test manifest is not the exact tests/jbc file set");
    valid &= validate_rows(sources, authority_text);
    valid &= validate_rows(tests, authority_text);
    valid &= require(document.find("No intentional behavior difference is introduced") != std::string::npos,
                     "missing compatibility disposition");
    valid &= require(document.find("no Part Two JIT or deep CellShard-runtime claim") != std::string::npos,
                     "missing Part Two boundary");
    return valid ? 0 : 1;
}
