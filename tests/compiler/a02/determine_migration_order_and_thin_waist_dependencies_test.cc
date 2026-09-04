#include <algorithm>
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

std::string read_text(const fs::path& path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot read " + path.string());
    }
    std::ostringstream text;
    text << stream.rdbuf();
    return text.str();
}

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> cells(const std::string& line) {
    std::vector<std::string> result;
    std::size_t begin = 1;
    while (begin < line.size()) {
        const std::size_t end = line.find('|', begin);
        if (end == std::string::npos) {
            break;
        }
        result.push_back(trim(line.substr(begin, end - begin)));
        begin = end + 1;
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
        require(argc == 4, "usage: test RECEIPT CELLERATOR_ROOT CELLSHARD_ROOT");
        const std::string receipt = read_text(argv[1]);
        const fs::path cellerator_root = argv[2];
        const fs::path cellshard_root = argv[3];
        const std::size_t nodes_begin = receipt.find("<!-- DAG-NODES-BEGIN -->");
        const std::size_t nodes_end = receipt.find("<!-- DAG-NODES-END -->");
        const std::size_t edges_begin = receipt.find("<!-- DAG-EDGES-BEGIN -->");
        const std::size_t edges_end = receipt.find("<!-- DAG-EDGES-END -->");
        require(nodes_begin < nodes_end && edges_begin < edges_end,
                "missing graph table markers");

        std::set<std::string> nodes;
        std::istringstream node_lines(receipt.substr(nodes_begin, nodes_end - nodes_begin));
        for (std::string line; std::getline(node_lines, line);) {
            const auto fields = cells(line);
            if (!fields.empty() && fields[0].rfind("MIG-", 0) == 0) {
                require(fields.size() == 4, "malformed node row: " + line);
                require(nodes.insert(fields[0]).second, "duplicate node " + fields[0]);
            }
        }
        require(nodes.size() == 26, "expected 26 migration nodes");

        std::map<std::string, std::vector<std::string>> outgoing;
        std::map<std::string, std::size_t> indegree;
        for (const auto& node : nodes) {
            indegree[node] = 0;
        }
        std::size_t edge_count = 0;
        std::istringstream edge_lines(receipt.substr(edges_begin, edges_end - edges_begin));
        for (std::string line; std::getline(edge_lines, line);) {
            const auto fields = cells(line);
            if (fields.size() >= 2 && fields[0].rfind("MIG-", 0) == 0) {
                require(fields.size() == 3, "malformed edge row: " + line);
                require(nodes.count(fields[0]) == 1, "unknown prerequisite " + fields[0]);
                require(nodes.count(fields[1]) == 1, "unknown consumer " + fields[1]);
                outgoing[fields[0]].push_back(fields[1]);
                ++indegree[fields[1]];
                ++edge_count;
            }
        }
        require(edge_count == 60, "expected 60 migration edges");

        std::set<std::string> ready;
        for (const auto& item : indegree) {
            if (item.second == 0) {
                ready.insert(item.first);
            }
        }
        std::size_t visited = 0;
        while (!ready.empty()) {
            const std::string node = *ready.begin();
            ready.erase(ready.begin());
            ++visited;
            for (const auto& consumer : outgoing[node]) {
                require(indegree[consumer] != 0, "duplicate or corrupt edge accounting");
                if (--indegree[consumer] == 0) {
                    ready.insert(consumer);
                }
            }
        }
        require(visited == nodes.size(), "migration dependency graph contains a cycle");

        for (const char* adapter : {
                 "MIG-A01", "MIG-A02", "MIG-A03", "MIG-A04",
                 "MIG-A05", "MIG-A06", "MIG-A07"}) {
            require(receipt.find(std::string("| ") + adapter + " |") != std::string::npos,
                    std::string("missing temporary or boundary adapter ") + adapter);
        }
        require(receipt.find("Temporary identity adapter") != std::string::npos,
                "missing temporary adapter retirement rule");
        require(receipt.find("Permanent schedule/materialization seam") != std::string::npos,
                "missing permanent adapter freeze rule");

        const std::string identity = read_text(cellerator_root /
            "include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh");
        const std::string coverage = read_text(cellerator_root /
            "include/Cellerator/execution/joint_compiler/logical_coverage_v1.hh");
        const std::string cost = read_text(cellerator_root /
            "include/Cellerator/planner/external_cost/vector_v1.hh");
        require(identity.find("producer_namespace") != std::string::npos,
                "identity thin-waist source mismatch");
        require(coverage.find("certified_exact_coverage_role_v1") != std::string::npos,
                "coverage thin-waist source mismatch");
        require(cost.find("pricing_epoch") != std::string::npos,
                "planner thin-waist source mismatch");

        for (const fs::path& source : {
                 fs::path("include/CellShard/compiler/atom/common_atom_v1.hh"),
                 fs::path("include/CellShard/compiler/evidence/evidence_atlas_v1.hh"),
                 fs::path("include/CellShard/compiler/certification/exact_atom_certificate_v1.hh"),
                 fs::path("include/CellShard/compiler/composition/derivation_dag_v1.hh"),
                 fs::path("include/CellShard/compiler/grammar/derivation_dag_v1.hh"),
                 fs::path("include/CellShard/compiler/basis/manifest.hpp"),
                 fs::path("include/CellShard/compiler/partial/partial_atom_v1.hh"),
                 fs::path("include/CellShard/compiler/graph/graph_recipe.hh"),
                 fs::path("include/CellShard/compiler/schedule/distributed_certificate.hh"),
                 fs::path("include/CellShard/compiler/schedule/portable_artifact.hh"),
                 fs::path("include/CellShard/artifact/atom_store/root_manifest_v1.hh"),
                 fs::path("include/CellShard/runtime/v2/residency_lease.hh")}) {
            require(fs::is_regular_file(cellshard_root / source),
                    "missing migration source " + source.string());
        }

        std::cout << "validated acyclic 26-node, 60-edge migration graph\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
