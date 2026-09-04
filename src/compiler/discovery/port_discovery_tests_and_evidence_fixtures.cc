#include <Cellerator/compiler/discovery/port_discovery_tests_and_evidence_fixtures_v1.hh>

#include <array>

namespace Cellerator::compiler::discovery {
namespace {

constexpr std::array<migrated_fixture_family_v1, 12> k_fixture_families{{
    {"components/CellShard/tests/jbc/atom", "c8f61927c42add8afb70f946a22eed0e1dac44e4ea8c01fdd25775090e516c89", "CE-CCP1-E02-001;010;011;012;015", "property and malformed atom fixtures are consolidated behind Cellerator-owned contracts", 20},
    {"components/CellShard/tests/jbc/evidence", "1d6c6a25af26286706d401aef586a1b3358ad338542d6501643c83ff43da4ac7", "CE-CCP1-E02-002", "expected scoring results are retained in the deterministic evidence test", 16},
    {"components/CellShard/tests/jbc/certification", "1178b5b918098686d56bcda0a9c9b30854199d3ee38f3107bcbd893eebd2b03f", "CE-CCP1-E02-009;013", "certification fixtures are split between exact rescan and scalable index gates", 16},
    {"components/CellShard/tests/jbc/discovery/support_signature", "7a8f20995747566e8e03909a339ab1eb58124386f1311f320c6328c940694acd", "CE-CCP1-E02-003;008", "support property and malformed cases share the typed provider gate", 10},
    {"components/CellShard/tests/jbc/discovery/co_support", "bd807e1b9bcbff600380c6fe5ff49e74b12275b76f039ee576050eb8efa87039", "CE-CCP1-E02-004", "benchmark evidence remains cold and the exact result remains the acceptance oracle", 11},
    {"components/CellShard/tests/jbc/discovery/overlap", "f7a23eb25e7455862badd6b464f285148d34b56dd7d5fd4224fab7cb68619330", "CE-CCP1-E02-002;004", "overlap expectations are covered by scoring and co-support gates", 6},
    {"components/CellShard/tests/jbc/discovery/motif", "6b1b5b337262a89ea33f50b000034f5d7f5505b61f436ca8000e61cda1bac652", "CE-CCP1-E02-005", "negative-result fixtures remain legitimate empty motif outcomes", 8},
    {"components/CellShard/tests/jbc/discovery/operation_trace", "8aa122052e22e89dc35c3e053440bd9faea893770e87dafc72b9aa6f48416bfc", "CE-CCP1-E02-005", "trace evidence is normalized into the shared motif and trace provider test", 8},
    {"components/CellShard/tests/jbc/discovery/trajectory", "425650ecbcbc27ed49745fe4a79b8cd6a67c77289219b8cfac9f00e6f2951bb6", "CE-CCP1-E02-006", "lineage fixtures preserve deterministic order and malformed rejection", 12},
    {"components/CellShard/tests/jbc/discovery/multimodal", "0a833cce86f42722f6b14460858f019bc7eef41b158cff61e42ea6c88b098402", "CE-CCP1-E02-007", "multimodal fixtures now use explicit modality identity", 10},
    {"components/CellShard/tests/jbc/discovery/factor_topic", "4deafe9f985333a6b33315e6003b46b7967ac1f0f369d09eda1b25809300cc53", "CE-CCP1-E02-008", "factor fixtures share the provider family validation gate", 6},
    {"components/CellShard/tests/jbc/discovery/bicluster", "3c51c5b5cd8d50e25b76839d49f05291290785f8b289d9504d30c76be328976e", "CE-CCP1-E02-008", "bicluster fixtures share the provider family validation gate", 8},
}};

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

const migrated_fixture_family_v1* migrated_fixture_inventory_v1(std::size_t* count) noexcept {
    if (count != nullptr) {
        *count = k_fixture_families.size();
    }
    return k_fixture_families.data();
}

bool valid_migrated_fixture_family_v1(const migrated_fixture_family_v1& family) noexcept {
    return family.source_path.rfind("components/CellShard/tests/jbc/", 0) == 0 &&
           lowercase_hex_digest(family.source_tree_sha256) && !family.focused_gate.empty() &&
           !family.intentional_change.empty() && family.source_file_count != 0;
}

std::size_t migrated_fixture_source_file_count_v1() noexcept {
    std::size_t count = 0;
    for (const auto& family : k_fixture_families) {
        count += family.source_file_count;
    }
    return count;
}

}  // namespace Cellerator::compiler::discovery
