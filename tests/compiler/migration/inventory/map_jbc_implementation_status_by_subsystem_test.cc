#include <array>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

namespace {

struct subsystem_record {
    const char* name;
    const char* status;
    const char* source_anchor;
    const char* evidence_anchor;
};

constexpr std::array<subsystem_record, 35> subsystems{{
    {"CE semantic interfaces", "complete and tested", "include/Cellerator/execution/joint_compiler", "tests/jbc/interfaces"},
    {"CE atom-fragment preparation", "complete and tested", "src/execution/atom_fragment", "tests/jbc/fragment"},
    {"CE decomposition catalog", "complete and tested", "include/Cellerator/compute/decomposition", "tests/jbc/decomposition"},
    {"CE atom/value planes", "complete and tested", "include/Cellerator/execution/atom_plane", "tests/jbc/atom_plane"},
    {"CE multi-extent binding and candidate", "complete and tested", "include/Cellerator/execution/object_binding", "tests/jbc/multi_extent"},
    {"CE external complete-cost exchange", "complete and tested", "include/Cellerator/planner/external_cost", "tests/jbc/external_cost"},
    {"CE lowering resumption", "complete and tested", "include/Cellerator/execution/lowering_resumption/resumption_v1.hh", "tests/jbc/resumption"},
    {"CE aggregate/package surface", "complete and tested", "CMakeLists.txt", "tests/jbc/verification/standalone_abi_gate_v1_test.cc"},
    {"CE cross-operation validation scenarios", "test-only", "tests/jbc/cross_operation", "tests/jbc/cross_operation"},
    {"CE independent verifier helpers", "test-only", "tests/jbc/verification/atom_fragment_verifier_v1.hh", "tests/jbc/verification/numerical_verifier_v1_test.cc"},
    {"CS atom model", "complete and tested", "components/CellShard/include/CellShard/compiler/atom", "components/CellShard/tests/jbc/atom"},
    {"CS evidence atlas", "complete and tested", "components/CellShard/include/CellShard/compiler/evidence", "components/CellShard/tests/jbc/evidence"},
    {"CS exact certification", "complete and tested", "components/CellShard/include/CellShard/compiler/certification", "components/CellShard/tests/jbc/certification"},
    {"CS support-signature discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/support_signature", "components/CellShard/tests/jbc/discovery/support_signature"},
    {"CS co-support discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/co_support", "components/CellShard/tests/jbc/discovery/co_support"},
    {"CS bicluster discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/bicluster", "components/CellShard/tests/jbc/discovery/bicluster"},
    {"CS overlap discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/overlap", "components/CellShard/tests/jbc/discovery/overlap"},
    {"CS motif discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/motif", "components/CellShard/tests/jbc/discovery/motif"},
    {"CS factor/topic discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/factor_topic", "components/CellShard/tests/jbc/discovery/factor_topic"},
    {"CS operation-trace discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/operation_trace", "components/CellShard/tests/jbc/discovery/operation_trace"},
    {"CS trajectory discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/trajectory", "components/CellShard/tests/jbc/discovery/trajectory"},
    {"CS multimodal discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/multimodal", "components/CellShard/tests/jbc/discovery/multimodal"},
    {"CS sequence compatibility discovery", "complete and tested", "components/CellShard/include/CellShard/compiler/discovery/sequence_compat", "components/CellShard/tests/jbc/discovery/sequence_compat"},
    {"CS composition", "complete and tested", "components/CellShard/include/CellShard/compiler/composition", "components/CellShard/tests/jbc/composition"},
    {"CS explicit grammar", "complete and tested", "components/CellShard/include/CellShard/compiler/grammar", "components/CellShard/tests/jbc/grammar/explicit"},
    {"CS induced grammar experiment", "complete and tested", "components/CellShard/src/compiler/grammar/induced", "components/CellShard/tests/jbc/grammar/induced"},
    {"CS basis selection", "complete and tested", "components/CellShard/include/CellShard/compiler/basis", "components/CellShard/tests/jbc/basis"},
    {"CS superatoms", "complete and tested", "components/CellShard/include/CellShard/compiler/composition/superatom", "components/CellShard/tests/jbc/superatom"},
    {"CS persistent partials", "complete and tested", "components/CellShard/include/CellShard/compiler/partial", "components/CellShard/tests/jbc/partial"},
    {"CS global graph and schedule", "complete and tested", "components/CellShard/include/CellShard/compiler/graph", "components/CellShard/tests/jbc/global_ir"},
    {"CS atom store", "complete and tested", "components/CellShard/include/CellShard/artifact/atom_store", "components/CellShard/tests/jbc/atom_store"},
    {"CS runtime v2", "complete and tested", "components/CellShard/include/CellShard/runtime/v2", "components/CellShard/tests/jbc/runtime"},
    {"CS integrated validation/package matrix", "complete and tested", "components/CellShard/docs/JBC/evidence/integration_receipt.md", "components/CellShard/tests/jbc/validation"},
    {"CS biological novelty campaign result", "partial", "components/CellShard/docs/JBC/evidence/biological_novelty_readiness.md", "components/CellShard/docs/JBC/evidence/metric_schema.md"},
    {"Original JBC pre-ledger package", "design-only", "planning/jbc-preledger-v1", "planning/jbc-preledger-v1/10_PACKAGE_VALIDATION_REPORT.md"},
}};

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
    std::array<char, 256> buffer{};
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

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 4,
                "usage: subsystem_status_test <receipt> <Cellerator root> <CellShard root>");
        const std::filesystem::path root(argv[2]);
        const std::filesystem::path cellshard_root(argv[3]);
        std::ifstream receipt_stream(argv[1]);
        require(receipt_stream.good(), "could not open status receipt");
        const std::string receipt((std::istreambuf_iterator<char>(receipt_stream)),
                                  std::istreambuf_iterator<char>());

        std::map<std::string, int> status_counts;
        for (const auto& subsystem : subsystems) {
            const std::string row = std::string("| ") + subsystem.name + " | " +
                                    subsystem.status + " |";
            require(receipt.find(row) != std::string::npos,
                    std::string("missing or mismatched row: ") + subsystem.name);
            const auto resolve = [&](const char* anchor) {
                const std::string value(anchor);
                constexpr const char* prefix = "components/CellShard/";
                return starts_with(value, prefix)
                           ? cellshard_root / value.substr(std::char_traits<char>::length(prefix))
                           : root / value;
            };
            require(std::filesystem::exists(resolve(subsystem.source_anchor)),
                    std::string("missing source anchor: ") + subsystem.source_anchor);
            require(std::filesystem::exists(resolve(subsystem.evidence_anchor)),
                    std::string("missing evidence anchor: ") + subsystem.evidence_anchor);
            ++status_counts[subsystem.status];
        }

        require(status_counts["complete and tested"] == 31,
                "unexpected complete-and-tested count");
        require(status_counts["test-only"] == 2, "unexpected test-only count");
        require(status_counts["partial"] == 1, "unexpected partial count");
        require(status_counts["design-only"] == 1, "unexpected design-only count");
        for (const auto* empty_status : {"complete but unintegrated", "scaffold-only",
                                         "obsolete compatibility code"}) {
            require(receipt.find(std::string("no **") + empty_status + "**") !=
                        std::string::npos,
                    std::string("missing empty-status finding: ") + empty_status);
        }

        const std::string git = "git -C " + shell_quote(root.string()) + " ";
        require(run(git + "merge-base --is-ancestor 82ccaf5 HEAD && printf yes") == "yes",
                "Cellerator producer-history merge is not integrated");
        require(run(git + "merge-base --is-ancestor 8267f41 HEAD && printf yes") == "yes",
                "Cellerator aggregate build is not integrated");

        const auto cellshard = cellshard_root.string();
        const std::string cs_git = "git -C " + shell_quote(cellshard) + " ";
        require(run(cs_git + "merge-base --is-ancestor 1efc4df main && printf yes") == "yes",
                "CellShard producer-history merge is not integrated");
        require(run(cs_git + "rev-parse main") ==
                    "b9749ad3e5146a04f847533d8c6f1a54146aed20",
                "unexpected CellShard integration head");

        require(receipt.find("records 304") != std::string::npos &&
                    receipt.find("host tests passing") != std::string::npos,
                "receipt omits integrated host-test evidence");
        require(receipt.find("explicitly withholds a biological-performance promotion") !=
                        std::string::npos &&
                    receipt.find("reserved biological dataset campaign") !=
                        std::string::npos,
                "receipt overstates biological promotion evidence");

        std::cout << "validated 35 JBC subsystem statuses and integrated source anchors\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
