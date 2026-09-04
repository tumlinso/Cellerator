#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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

void require_contains(const std::string& text, const std::string& needle) {
    require(text.find(needle) != std::string::npos, "missing evidence: " + needle);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4, "usage: test RECEIPT CELLERATOR_ROOT CELLSHARD_ROOT");
        const fs::path receipt_path = argv[1];
        const fs::path cellerator_root = argv[2];
        const fs::path cellshard_root = argv[3];
        const std::string receipt = read_text(receipt_path);

        for (int interface_number = 1; interface_number <= 20; ++interface_number) {
            std::ostringstream identity;
            identity << "JBC-I";
            if (interface_number < 10) {
                identity << '0';
            }
            identity << interface_number;
            const std::string row_marker = "| " + identity.str() + " |";
            const std::size_t first = receipt.find(row_marker);
            require(first != std::string::npos, "missing interface row " + identity.str());
            require(receipt.find(row_marker, first + row_marker.size()) == std::string::npos,
                    "duplicate interface row " + identity.str());
        }

        for (const char* heading : {
                 "ID width", "Pointer ownership", "Generation model", "Exact coverage",
                 "Allocator assumption", "Target assumption", "CellShard dependency"}) {
            require_contains(receipt, heading);
        }
        require_contains(receipt, "Contradiction — rehome");
        require_contains(receipt, "JBC-I12, JBC-I13, JBC-I14, JBC-I15, JBC-I16");
        require_contains(receipt, "JBC-I18, and JBC-I19");
        require_contains(receipt, "No contradiction — retain:** JBC-I17");
        require_contains(receipt, "No contradiction — retain:** JBC-I20");

        const fs::path identity_path = cellerator_root /
            "include/Cellerator/execution/joint_compiler/persistent_identity_v1.hh";
        const std::string identity = read_text(identity_path);
        require_contains(identity, "std::uint64_t producer_namespace");
        require_contains(identity, "std::uint64_t local_identity");
        require_contains(identity, "sizeof(persistent_identity_v1) == 16u");

        const std::string binding = read_text(cellerator_root /
            "include/Cellerator/execution/joint_compiler/external_binding_v1.hh");
        require_contains(binding, "maximum_external_extents_v1 = 1024u");
        require_contains(binding, "opaque_runtime_token_v1 readiness");
        require_contains(binding, "opaque_runtime_token_v1 lease");
        require_contains(binding, "value_generation generation");

        const std::string evidence = read_text(cellshard_root /
            "include/CellShard/compiler/evidence/atom_evidence_record_v1.hh");
        require_contains(evidence, "proposal_only");
        require_contains(evidence, "exact-coverage");
        require_contains(evidence, "sizeof(atom_evidence_record_v1) == 80");

        const std::string partial = read_text(cellshard_root /
            "include/CellShard/compiler/partial/partial_atom_v1.hh");
        require_contains(partial, "structure_generation");
        require_contains(partial, "value_generation");
        require_contains(partial, "state_generation");
        require_contains(partial, "materialization_generation");
        require_contains(partial, "cost_model_generation");

        const fs::path include_root = cellerator_root / "include/Cellerator";
        for (const fs::path& relative : {
                 fs::path("execution/joint_compiler"),
                 fs::path("compute/decomposition"),
                 fs::path("planner/external_cost"),
                 fs::path("profiling/joint_compiler")}) {
            for (const auto& entry : fs::recursive_directory_iterator(include_root / relative)) {
                if (!entry.is_regular_file()) {
                    continue;
                }
                const std::string source = read_text(entry.path());
                require(source.find("#include <CellShard") == std::string::npos,
                        "direct CellShard include in " + entry.path().string());
                require(source.find("#include \"CellShard") == std::string::npos,
                        "direct CellShard include in " + entry.path().string());
            }
        }

        std::cout << "validated 20 JBC interface assumption mappings\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
