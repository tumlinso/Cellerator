#include <array>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr const char* common_base =
    "7762a5925fe18b2ca45ab8a436f3461804ed2ad9";
constexpr const char* source_commit =
    "b9749ad3e5146a04f847533d8c6f1a54146aed20";

struct record {
    std::string path;
    std::string sha256;
    std::uintmax_t bytes = 0;
    std::string disposition;
    std::string target;
    std::string task;
    std::string gate;
    std::string provenance;
};

bool starts_with(const std::string& value, const std::string& prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool contains(const std::string& value, const std::string& needle) {
    return value.find(needle) != std::string::npos;
}

std::string basename(const std::string& path) {
    const auto separator = path.find_last_of('/');
    return separator == std::string::npos ? path : path.substr(separator + 1);
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
        || starts_with(path, "bench/jbc/runtime/")
        || starts_with(path, "include/CellShard/artifact/atom_store/")
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
    if (starts_with(path, "tests/") || starts_with(path, "bench/")) {
        return "adapt";
    }
    return {};
}

std::string task_for(const std::string& path, const std::string& disposition_value) {
    if (disposition_value == "preserve in place") {
        if (contains(path, "atom_store") || contains(path, "runtime/v2")
            || starts_with(path, "tests/jbc/runtime/")
            || starts_with(path, "bench/jbc/runtime/")) {
            return "CE-CCP1-E03-013";
        }
        return "CE-CCP1-A02-012";
    }
    if (path == "CMakeLists.txt" || path == "include/CellShard/CellShard.hh") {
        return "CE-CCP1-A03-014";
    }
    if (starts_with(path, "tests/jbc/validation/")) {
        return "CE-CCP1-E03-015";
    }
    if (disposition_value == "wrap temporarily"
        || disposition_value == "retain as compatibility") {
        return "CE-CCP1-E02-015";
    }
    if (starts_with(path, "tests/jbc/")) {
        if (contains(path, "/atom/") || contains(path, "/certification/")
            || contains(path, "/discovery/") || contains(path, "/evidence/")) {
            return "CE-CCP1-E02-016";
        }
        return "CE-CCP1-E03-014";
    }
    if (starts_with(path, "bench/jbc/")) {
        return contains(path, "/bicluster/") || contains(path, "/trajectory/")
            ? "CE-CCP1-E02-016" : "CE-CCP1-E03-014";
    }
    if (contains(path, "/compiler/evidence/")) return "CE-CCP1-E02-002";
    if (contains(path, "/compiler/discovery/support_signature/")) return "CE-CCP1-E02-003";
    if (contains(path, "/compiler/discovery/co_support/")
        || contains(path, "/compiler/discovery/overlap/")) return "CE-CCP1-E02-004";
    if (contains(path, "/compiler/discovery/motif/")
        || contains(path, "/compiler/discovery/operation_trace/")) return "CE-CCP1-E02-005";
    if (contains(path, "/compiler/discovery/trajectory/")) return "CE-CCP1-E02-006";
    if (contains(path, "/compiler/discovery/multimodal/")
        || contains(path, "/compiler/discovery/sequence_compat/")) return "CE-CCP1-E02-007";
    if (contains(path, "/compiler/discovery/factor_topic/")
        || contains(path, "/compiler/discovery/bicluster/")) return "CE-CCP1-E02-008";
    if (contains(path, "/compiler/certification/")) return "CE-CCP1-E02-009";
    if (contains(path, "/compiler/atom/")) return "CE-CCP1-E02-010";
    if (contains(path, "/compiler/composition/superatom/")) return "CE-CCP1-E03-008";
    if (contains(path, "derivation_dag_v1.hh")) return "CE-CCP1-E03-002";
    if (contains(path, "/compiler/composition/")) return "CE-CCP1-E03-001";
    if (contains(path, "/compiler/grammar/") || contains(path, "/src/compiler/grammar/")) {
        return contains(path, "/induced/") ? "CE-CCP1-E03-004" : "CE-CCP1-E03-003";
    }
    if (contains(path, "/compiler/basis/")) return "CE-CCP1-E03-006";
    if (contains(path, "/compiler/partial/")) return "CE-CCP1-E03-009";
    if (contains(path, "/compiler/graph/")) return "CE-CCP1-E03-010";
    if (contains(path, "/compiler/schedule/")) return "CE-CCP1-E03-012";
    return "CE-CCP1-A02-012";
}

std::string gate_for(const std::string& task) {
    if (starts_with(task, "CE-CCP1-E02-")) return "CE-CCP1-E02-018-GATE";
    if (starts_with(task, "CE-CCP1-E03-")) return "CE-CCP1-E03-018-GATE";
    if (task == "CE-CCP1-A03-014") return "CE-CCP1-A03-014-GATE";
    return "CE-CCP1-A02-012-GATE";
}

std::string replace_prefix(const std::string& value, const std::string& prefix,
                           const std::string& replacement) {
    return starts_with(value, prefix)
        ? replacement + value.substr(prefix.size()) : std::string{};
}

std::string target_for(const std::string& path, const std::string& disposition_value,
                       const std::string& task) {
    if (disposition_value == "preserve in place") {
        return "CellShard:" + path;
    }
    if (disposition_value == "split") {
        return "Cellerator+CellShard:assigned-by-" + task;
    }
    std::string target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/evidence/",
                                  "include/Cellerator/compiler/discovery/evidence/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/discovery/",
                                  "include/Cellerator/compiler/discovery/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/certification/",
                                  "include/Cellerator/compiler/discovery/certification/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/atom/",
                                  "include/Cellerator/compiler/discovery/atom/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/composition/",
                                  "include/Cellerator/compiler/composition/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/grammar/",
                                  "include/Cellerator/compiler/composition/grammar/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/basis/",
                                  "include/Cellerator/compiler/composition/basis/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/partial/",
                                  "include/Cellerator/compiler/planning/partial/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/graph/",
                                  "include/Cellerator/compiler/program/")).empty()) return target;
    if (!(target = replace_prefix(path, "include/CellShard/compiler/schedule/",
                                  "include/Cellerator/compiler/program/schedule/")).empty()) return target;
    if (!(target = replace_prefix(path, "src/compiler/evidence/",
                                  "src/compiler/discovery/evidence/")).empty()) return target;
    if (!(target = replace_prefix(path, "src/compiler/grammar/",
                                  "src/compiler/composition/grammar/")).empty()) return target;
    if (!(target = replace_prefix(path, "src/compiler/graph/",
                                  "src/compiler/program/")).empty()) return target;
    if (!(target = replace_prefix(path, "src/compiler/schedule/",
                                  "src/compiler/program/schedule/")).empty()) return target;
    if (!(target = replace_prefix(path, "src/compiler/atom/",
                                  "src/compiler/discovery/atom/")).empty()) return target;
    if (starts_with(path, "tests/jbc/")) {
        return (starts_with(task, "CE-CCP1-E02-")
                    ? "tests/compiler/discovery/" : "tests/compiler/composition/")
            + path.substr(std::string("tests/jbc/").size());
    }
    if (starts_with(path, "bench/jbc/")) {
        return "bench/compiler/" + path.substr(std::string("bench/jbc/").size());
    }
    if (disposition_value == "wrap temporarily"
        || disposition_value == "retain as compatibility") {
        return "Cellerator:assigned-by-" + task + ";CellShard:" + path;
    }
    return "Cellerator:assigned-by-" + task + "/" + basename(path);
}

std::string provenance_for(const std::string& disposition_value) {
    std::string result = "JBC-PROV-";
    for (const char character : disposition_value) {
        result += character == ' ' ? '-' : static_cast<char>(std::toupper(
            static_cast<unsigned char>(character)));
    }
    return result;
}

std::vector<record> build_records(const fs::path& source_root) {
    const std::string git = "git -C " + shell_quote(source_root.string()) + " ";
    const auto branches = lines(run(
        git + "for-each-ref --format='%(refname:short)' 'refs/heads/jbc/*'"));
    if (branches.size() != 24) {
        throw std::runtime_error("expected 24 source JBC branches");
    }
    std::set<std::string> paths;
    for (const auto& branch : branches) {
        for (const auto& path : lines(run(
                 git + "diff --name-only " + common_base + ".." + shell_quote(branch)))) {
            paths.insert(path);
        }
    }
    if (paths.size() != 979) {
        throw std::runtime_error("expected 979 unique source paths");
    }
    std::vector<record> records;
    for (const auto& path : paths) {
        const fs::path source_path = source_root / path;
        if (!fs::is_regular_file(source_path)) {
            throw std::runtime_error("dangling source " + path);
        }
        record item;
        item.path = path;
        item.sha256 = run("sha256sum -- " + shell_quote(source_path.string())
                          + " | cut -d' ' -f1");
        item.bytes = fs::file_size(source_path);
        item.disposition = disposition(path);
        if (item.disposition.empty()) {
            throw std::runtime_error("unclassified source " + path);
        }
        item.task = task_for(path, item.disposition);
        item.gate = gate_for(item.task);
        item.target = target_for(path, item.disposition, item.task);
        item.provenance = provenance_for(item.disposition);
        records.push_back(std::move(item));
    }
    return records;
}

std::string json_escape(const std::string& value) {
    std::string result;
    for (const char character : value) {
        if (character == '\\' || character == '"') result += '\\';
        result += character;
    }
    return result;
}

void emit(const fs::path& source_root, const fs::path& csv_path,
          const fs::path& json_path) {
    const auto records = build_records(source_root);
    std::ofstream csv(csv_path);
    csv << "source_repository,source_branch,source_commit,source_path,sha256,bytes,"
           "disposition,proposed_target,migration_task,required_gate,provenance_rule,status\n";
    for (const auto& item : records) {
        csv << "git@github.com:tumlinso/CellShard.git,main," << source_commit << ','
            << item.path << ',' << item.sha256 << ',' << item.bytes << ','
            << item.disposition << ',' << item.target << ',' << item.task << ','
            << item.gate << ',' << item.provenance << ",source-frozen\n";
    }
    if (!csv) throw std::runtime_error("failed to write CSV");

    std::ofstream json(json_path);
    json << "{\n  \"schema_version\": 1,\n  \"source_repository\": "
            "\"git@github.com:tumlinso/CellShard.git\",\n  \"source_branch\": \"main\",\n"
            "  \"source_commit\": \"" << source_commit << "\",\n"
            "  \"common_base\": \"" << common_base << "\",\n"
            "  \"records\": [\n";
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& item = records[index];
        json << "    {\"source_path\":\"" << json_escape(item.path)
             << "\",\"sha256\":\"" << item.sha256 << "\",\"bytes\":" << item.bytes
             << ",\"disposition\":\"" << item.disposition
             << "\",\"proposed_target\":\"" << json_escape(item.target)
             << "\",\"migration_task\":\"" << item.task
             << "\",\"required_gate\":\"" << item.gate
             << "\",\"provenance_rule\":\"" << item.provenance
             << "\",\"status\":\"source-frozen\"}"
             << (index + 1 == records.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
    if (!json) throw std::runtime_error("failed to write JSON");
}

std::vector<std::string> csv_cells(const std::string& line) {
    std::vector<std::string> result;
    std::istringstream stream(line);
    for (std::string cell; std::getline(stream, cell, ',');) result.push_back(cell);
    return result;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::string(argv[1]) == "--generate") {
            emit(argv[2], argv[3], argv[4]);
            std::cout << "generated frozen 979-row CSV and JSON inventories\n";
            return EXIT_SUCCESS;
        }
        require(argc == 5, "usage: test RECEIPT CSV JSON CELLSHARD_ROOT");
        const fs::path receipt_path = argv[1];
        const fs::path csv_path = argv[2];
        const fs::path json_path = argv[3];
        const fs::path source_root = argv[4];
        require(run("git -C " + shell_quote(source_root.string()) + " rev-parse main")
                    == source_commit, "CellShard main source commit drifted");

        std::ifstream csv(csv_path);
        require(csv.good(), "missing CSV inventory");
        std::string line;
        require(static_cast<bool>(std::getline(csv, line)), "missing CSV header");
        require(line == "source_repository,source_branch,source_commit,source_path,sha256,bytes,"
                        "disposition,proposed_target,migration_task,required_gate,provenance_rule,status",
                "CSV schema mismatch");
        std::set<std::string> paths;
        std::map<std::string, int> counts;
        std::size_t row_count = 0;
        while (std::getline(csv, line)) {
            const auto fields = csv_cells(line);
            require(fields.size() == 12, "malformed CSV row " + std::to_string(row_count + 2));
            require(fields[0] == "git@github.com:tumlinso/CellShard.git", "source repository mismatch");
            require(fields[1] == "main" && fields[2] == source_commit, "source ref mismatch");
            require(paths.insert(fields[3]).second, "duplicate source path " + fields[3]);
            const fs::path source = source_root / fields[3];
            require(fs::is_regular_file(source), "dangling source " + fields[3]);
            require(run("sha256sum -- " + shell_quote(source.string()) + " | cut -d' ' -f1")
                        == fields[4], "content hash mismatch " + fields[3]);
            require(std::to_string(fs::file_size(source)) == fields[5],
                    "byte count mismatch " + fields[3]);
            require(disposition(fields[3]) == fields[6], "disposition mismatch " + fields[3]);
            require(!fields[7].empty() && !fields[8].empty() && !fields[9].empty(),
                    "missing destination/task/gate " + fields[3]);
            require(fields[10] == provenance_for(fields[6]), "provenance rule mismatch");
            require(fields[11] == "source-frozen", "invalid frozen status");
            ++counts[fields[6]];
            ++row_count;
        }
        require(row_count == 979, "CSV row count is not 979");
        const std::map<std::string, int> expected{{
            {"adapt", 242}, {"move", 220}, {"preserve in place", 457},
            {"retain as compatibility", 4}, {"retire after replacement proof", 3},
            {"split", 52}, {"wrap temporarily", 1},
        }};
        require(counts == expected, "CSV disposition counts do not reconcile");

        std::ifstream json_stream(json_path);
        require(json_stream.good(), "missing JSON inventory");
        const std::string json((std::istreambuf_iterator<char>(json_stream)),
                               std::istreambuf_iterator<char>());
        std::size_t json_rows = 0;
        for (std::size_t offset = 0;
             (offset = json.find("\"source_path\":", offset)) != std::string::npos;
             offset += 14) ++json_rows;
        require(json_rows == 979, "JSON row count is not 979");
        require(json.find(source_commit) != std::string::npos, "JSON source commit missing");

        const std::string receipt = [&] {
            std::ifstream input(receipt_path);
            require(input.good(), "missing human receipt");
            return std::string((std::istreambuf_iterator<char>(input)),
                               std::istreambuf_iterator<char>());
        }();
        const std::string csv_hash = run("sha256sum -- " + shell_quote(csv_path.string())
                                         + " | cut -d' ' -f1");
        const std::string json_hash = run("sha256sum -- " + shell_quote(json_path.string())
                                          + " | cut -d' ' -f1");
        require(receipt.find(csv_hash) != std::string::npos, "human receipt lacks CSV hash");
        require(receipt.find(json_hash) != std::string::npos, "human receipt lacks JSON hash");
        require(receipt.find("CE-CCP1-I02-JBC-MIGRATION-MANIFEST") != std::string::npos,
                "published interface identity missing");

        std::cout << "validated 979 unique, hashed, non-dangling migration rows\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
