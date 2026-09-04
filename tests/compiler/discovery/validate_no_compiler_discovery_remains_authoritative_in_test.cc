#include <Cellerator/compiler/discovery/validate_no_compiler_discovery_remains_authoritative_in_v1.hh>

#include <cassert>
#include <string_view>
#include <unordered_set>

using namespace Cellerator::compiler::discovery;

int main() {
    assert(valid_cellshard_compiler_authority_audit_v1());

    std::size_t count = 0;
    const auto* classifications = cellshard_compiler_path_classifications_v1(&count);
    assert(classifications != nullptr);
    assert(count == 10);

    std::unordered_set<std::string_view> prefixes;
    std::size_t source_files = 0;
    for (std::size_t index = 0; index < count; ++index) {
        assert(prefixes.emplace(classifications[index].source_prefix).second);
        source_files += classifications[index].source_file_count;
    }
    assert(source_files == 241);

    const auto& audit = cellshard_compiler_authority_audit_receipt_v1();
    assert(audit.classified_compiler_path_count == source_files);
    assert(audit.unclassified_compiler_path_count == 0);
    assert(audit.production_authority_consumer_count == 0);
    assert(audit.retained_authoritative_api_count == 0);
    assert(audit.jbc_branch_count == 0);
}
