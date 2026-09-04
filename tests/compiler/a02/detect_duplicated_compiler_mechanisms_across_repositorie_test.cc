#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

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

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void contains(const std::string& text, const std::string& needle) {
    require(text.find(needle) != std::string::npos, "missing evidence: " + needle);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: test RECEIPT CELLERATOR_ROOT CELLSHARD_ROOT");
        const std::string receipt = read_text(argv[1]);
        const fs::path cellerator_root = argv[2];
        const fs::path cellshard_root = argv[3];

        for (int duplicate = 1; duplicate <= 16; ++duplicate) {
            std::ostringstream id;
            id << "JBC-D";
            if (duplicate < 10) {
                id << '0';
            }
            id << duplicate;
            const std::string row = "| " + id.str() + " |";
            const std::size_t first = receipt.find(row);
            require(first != std::string::npos, "missing duplicate decision " + id.str());
            require(receipt.find(row, first + row.size()) == std::string::npos,
                    "duplicate decision row " + id.str());
        }
        contains(receipt, "MERGE-CELLERATOR");
        contains(receipt, "ADAPT-CELLSHARD");
        contains(receipt, "RETAIN-DISTINCT");
        contains(receipt, "Blind-copy prohibition");
        contains(receipt, "must not create parallel Cellerator copies");

        const std::string ce_identity = read_text(cellerator_root /
            "include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh");
        const std::string cs_identity = read_text(cellshard_root /
            "include/CellShard/compiler/atom/persistent_identity_v1.hh");
        for (const char* field : {"producer_namespace", "local_identity"}) {
            contains(ce_identity, field);
            contains(cs_identity, field);
        }
        contains(ce_identity, "sizeof(persistent_identity_v1) == 16u");
        contains(cs_identity, "sizeof(atom_persistent_identity_v1) == 16");

        const std::string cs_coverage = read_text(cellshard_root /
            "include/CellShard/compiler/atom/logical_coverage_v1.hh");
        contains(cs_coverage, "Values intentionally match Cellerator");
        contains(cs_coverage, "const void *cellerator_coverage");
        contains(cs_coverage, "missing_exact_certification");

        const std::string ce_cost = read_text(cellerator_root /
            "include/Cellerator/planner/external_cost/vector_v1.hh");
        const std::string cs_realization = read_text(cellshard_root /
            "include/CellShard/compiler/graph/physical_realization.hh");
        contains(ce_cost, "fixed_ns");
        contains(ce_cost, "persistent_byte_ns");
        contains(ce_cost, "transfer_byte_ns");
        contains(cs_realization, "preparation_bytes");
        contains(cs_realization, "persistent_bytes");
        contains(cs_realization, "estimated_launches");

        const std::string composition_dag = read_text(cellshard_root /
            "include/CellShard/compiler/composition/derivation_dag_v1.hh");
        const std::string grammar_dag = read_text(cellshard_root /
            "include/CellShard/compiler/grammar/derivation_dag_v1.hh");
        contains(composition_dag, "cycle_detected");
        contains(grammar_dag, "derivation_dag_code_v1::cycle");
        contains(composition_dag, "max_derivation_nodes_v1 = 256");
        contains(composition_dag, "max_derivation_edges_v1 = 1024");

        const std::string ce_export = read_text(cellerator_root /
            "include/Cellerator/profiling/joint_compiler/execution_export_v2.hh");
        const std::string cs_schedule = read_text(cellshard_root /
            "include/CellShard/compiler/schedule/portable_artifact.hh");
        contains(ce_export, "correctness_receipt");
        contains(ce_export, "performance_freshness");
        contains(cs_schedule, "portable_command_kind");
        contains(cs_schedule, "transform_order");

        const std::string partial = read_text(cellshard_root /
            "include/CellShard/compiler/partial/partial_atom_v1.hh");
        contains(partial, "reconstruction_algebra_identity");
        contains(partial, "contribution_coverage_identity");
        contains(partial, "materialization_generation");

        std::cout << "validated 16 duplicate-mechanism decisions\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
