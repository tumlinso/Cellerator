#include <Cellerator/compiler/discovery/import_co_support_and_overlap_discovery_v1.hh>

#include <cassert>
#include <random>

using namespace Cellerator::compiler::discovery;

int main() {
    const std::uint64_t offsets[]{0, 3, 5, 8};
    const std::uint64_t sources[]{0, 1, 2, 0, 1, 0, 1, 3};
    const support_relation_view_v1 relation{{7, 8}, offsets, sources, 3, 8};
    co_support_discovery_v1 result;
    assert(discover_co_support_and_overlap_v1(relation, 4, 3, &result) ==
           co_support_status_v1::success);
    assert((result.source_prevalence == std::vector<std::uint64_t>{3, 3, 1, 1}));
    assert((result.destination_convergence == std::vector<std::uint64_t>{3, 2, 3}));
    assert(result.proposals[0].first_source == 0);
    assert(result.proposals[0].second_source == 1);
    assert(result.proposals[0].observed_together == 3);
    assert(result.proposals[0].null_numerator == 9);
    assert(result.proposals[0].null_denominator == 3);
    assert(result.enumerated_pairs == 7);

    std::mt19937_64 random(19);
    for (unsigned trial = 0; trial < 32; ++trial) {
        std::vector<std::uint64_t> random_offsets{0};
        std::vector<std::uint64_t> random_sources;
        std::vector<std::uint64_t> oracle(6, 0);
        for (unsigned destination = 0; destination < 10; ++destination) {
            for (unsigned source = 0; source < 6; ++source)
                if ((random() & 3u) == 0) {
                    random_sources.push_back(source);
                    ++oracle[source];
                }
            random_offsets.push_back(random_sources.size());
        }
        const support_relation_view_v1 random_relation{
            {9, trial + 1}, random_offsets.data(), random_sources.data(), 10,
            random_sources.size()};
        assert(discover_co_support_and_overlap_v1(
                   random_relation, 6, 8, &result) == co_support_status_v1::success);
        assert(result.source_prevalence == oracle);
    }
}
