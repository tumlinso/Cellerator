#include <Cellerator/compiler/discovery/import_support_signature_discovery_v1.hh>

#include <cassert>

using namespace Cellerator::compiler::discovery;

int main() {
    const std::uint64_t offsets[]{0, 3, 6, 8, 9};
    const std::uint64_t sources[]{1, 3, 7, 1, 3, 7, 2, 8, 9};
    const support_relation_view_v1 relation{{3, 4}, offsets, sources, 4, 9};
    const support_signature_config_v1 config{16, 2, 99, {8, 12}};

    support_signature_discovery_v1 first;
    support_signature_discovery_v1 second;
    assert(discover_support_signatures_v1(relation, config, &first) ==
           support_signature_status_v1::success);
    assert(discover_support_signatures_v1(relation, config, &second) ==
           support_signature_status_v1::success);
    assert(first.minima == second.minima);
    assert(first.proposals.size() == 2);
    assert(first.proposals[0].first_destination == 0);
    assert(first.proposals[0].second_destination == 1);
    assert(first.proposals[0].matching_minima == 16);
    assert(first.proposals[0].biological_stratum == config.biological_stratum);
    assert(first.hashed_edges == 9 * 16);
    assert(first.compared_pairs == 6);

    auto invalid = relation;
    const std::uint64_t bad_offsets[]{1, 3, 6, 8, 9};
    invalid.destination_offsets = bad_offsets;
    assert(discover_support_signatures_v1(invalid, config, &first) ==
           support_signature_status_v1::invalid_offsets);
}
