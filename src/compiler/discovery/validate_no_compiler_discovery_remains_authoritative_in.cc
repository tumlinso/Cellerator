#include <Cellerator/compiler/discovery/validate_no_compiler_discovery_remains_authoritative_in_v1.hh>

#include <array>

namespace Cellerator::compiler::discovery {
namespace {

constexpr std::array<cellshard_compiler_path_classification_v1, 10> k_classifications{{
    {"include/CellShard/compiler/atom;src/compiler/atom", "63119dc956c299d366192ecdd6cf4c478e9c219056cc35ef317ba0ad77742337", "Cellerator compiler atom contracts (E02)", 21},
    {"include/CellShard/compiler/evidence;src/compiler/evidence", "5244ce73b8e2b75ac5f28d35ea12031ebf87a8da7bce506de70c26bc83f1aaed", "Cellerator compiler evidence and discovery (E02)", 18},
    {"include/CellShard/compiler/certification;src/compiler/certification", "4c31f3762e5847c2a2c86e2470e5076188dfe0eb97e1683f3d14465a97f6f6a5", "Cellerator compiler certification (E02)", 16},
    {"include/CellShard/compiler/discovery;src/compiler/discovery", "1f0f164b1db62db9b4e87d1522ca5a4ef091631006ad7dcbe09eea58950ce6e7", "Cellerator compiler discovery and Baseplane-facing adapters (E02)", 82},
    {"include/CellShard/compiler/composition;src/compiler/composition", "4c6eaf5a6184172294707386dbbdf2447fb800504d41bd4076699e609f294e6b", "Cellerator compiler composition and grammar (E03)", 31},
    {"include/CellShard/compiler/grammar;src/compiler/grammar", "e698032bc6fe77dfe7da244d400a9253bebc13b77acdd06799660b5c9efc9158", "Cellerator compiler grammar (E03)", 20},
    {"include/CellShard/compiler/basis;src/compiler/basis", "43fda877f66ba61427601ed7c63ec1d13be9c4c72b13d14f9a031d5e4ff69e2a", "Cellerator compiler basis planning (E03/A03)", 17},
    {"include/CellShard/compiler/graph;src/compiler/graph", "a0d5f7f0c6f6dc85f850ed15efcc57a7554921265b0850636890670dbdb94db2", "Cellerator semantic and planning IR (A03/E03)", 15},
    {"include/CellShard/compiler/partial;src/compiler/partial", "99901c757c96a34c8b15693fd6c4eb58820bb3beba44ba6f97f9060465ad2857", "Cellerator compiler partial-result contracts (E03)", 17},
    {"include/CellShard/compiler/schedule;src/compiler/schedule", "97bacb15a923f60762d95b94655b8ee2806ecb1ac16df78acbb814f597ae0d61", "Cellerator planning and realization IR (A03)", 4},
}};

constexpr cellshard_compiler_authority_audit_v1 k_audit{
    "tumlinso/CellShard",
    "b9749ad3e5146a04f847533d8c6f1a54146aed20",
    1,
    0,
    241,
    0,
    0,
    0,
};

bool lowercase_hex_digest(std::string_view digest) noexcept {
    if (digest.size() != 64) {
        return false;
    }
    for (const char value : digest) {
        if (!((value >= '0' && value <= '9') || (value >= 'a' && value <= 'f'))) {
            return false;
        }
    }
    return true;
}

}  // namespace

const cellshard_compiler_path_classification_v1*
cellshard_compiler_path_classifications_v1(std::size_t* count) noexcept {
    if (count != nullptr) {
        *count = k_classifications.size();
    }
    return k_classifications.data();
}

const cellshard_compiler_authority_audit_v1&
cellshard_compiler_authority_audit_receipt_v1() noexcept {
    return k_audit;
}

bool valid_cellshard_compiler_authority_audit_v1() noexcept {
    std::size_t classified_count = 0;
    for (const auto& classification : k_classifications) {
        if (classification.source_prefix.empty() ||
            !lowercase_hex_digest(classification.source_tree_sha256) ||
            classification.cellerator_owner.empty() || classification.source_file_count == 0) {
            return false;
        }
        classified_count += classification.source_file_count;
    }
    return k_audit.repository == "tumlinso/CellShard" && k_audit.audited_commit.size() == 40 &&
           k_audit.audited_branch_count == 1 && k_audit.jbc_branch_count == 0 &&
           classified_count == k_audit.classified_compiler_path_count &&
           k_audit.unclassified_compiler_path_count == 0 &&
           k_audit.production_authority_consumer_count == 0 &&
           k_audit.retained_authoritative_api_count == 0;
}

}  // namespace Cellerator::compiler::discovery
