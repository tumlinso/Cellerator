#include <Cellerator/compiler/discovery/port_discovery_tests_and_evidence_fixtures_v1.hh>

#include <cassert>
#include <string_view>
#include <unordered_set>

using namespace Cellerator::compiler::discovery;

int main() {
    std::size_t count = 0;
    const auto* families = migrated_fixture_inventory_v1(&count);
    assert(families != nullptr);
    assert(count == 12);

    std::unordered_set<std::string_view> paths;
    bool saw_property_and_malformed = false;
    bool saw_benchmark = false;
    bool saw_negative_result = false;
    for (std::size_t index = 0; index < count; ++index) {
        const auto& family = families[index];
        assert(valid_migrated_fixture_family_v1(family));
        assert(paths.emplace(family.source_path).second);
        saw_property_and_malformed |=
            family.intentional_change.find("property and malformed") != std::string_view::npos;
        saw_benchmark |= family.intentional_change.find("benchmark") != std::string_view::npos;
        saw_negative_result |=
            family.intentional_change.find("negative-result") != std::string_view::npos;
    }

    assert(migrated_fixture_source_file_count_v1() == 131);
    assert(saw_property_and_malformed);
    assert(saw_benchmark);
    assert(saw_negative_result);
}
