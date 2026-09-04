#include <Cellerator/compiler/discovery/preserve_migration_provenance_in_source_and_artifacts_v1.hh>

#include <cassert>
#include <filesystem>
#include <string>

using namespace Cellerator::compiler::discovery;

int main() {
    std::size_t count = 0;
    const auto* records = jbc_discovery_migration_manifest_v1(&count);
    assert(records != nullptr);
    assert(count == 13);
    for (std::size_t index = 0; index < count; ++index) {
        assert(valid_jbc_migration_provenance_v1(records[index]));
    }

    const std::filesystem::path root{CELLERATOR_SOURCE_ROOT};
    const std::filesystem::path directories[]{
        root / "include/Cellerator/compiler/discovery",
        root / "src/compiler/discovery",
        root / "tests/compiler/discovery",
    };
    std::size_t moved_files = 0;
    for (const auto& directory : directories) {
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            const auto name = entry.path().filename().string();
            if (name.rfind("import_", 0) != 0) {
                continue;
            }
            assert(find_jbc_migration_provenance_v1(name) != nullptr);
            ++moved_files;
        }
    }
    assert(moved_files == 39);

    const auto* trajectory = find_jbc_migration_provenance_v1(
        "src/compiler/discovery/import_trajectory_and_lineage_pattern_discovery.cc");
    assert(trajectory != nullptr);
    assert(trajectory->cellerator_todo == "CE-CCP1-E02-006");
    assert(find_jbc_migration_provenance_v1("unrelated_runtime.cc") == nullptr);
}
