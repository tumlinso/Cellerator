#include <Cellerator/compiler/discovery/import_scalable_certification_indexes_v1.hh>

#include <cassert>
#include <chrono>
#include <vector>

using namespace Cellerator::compiler::discovery;

namespace {

persistent_atom_identity_v1 id(std::uint64_t value) {
    return {1, value};
}

struct measurement_v1 {
    std::uint64_t elapsed_nanoseconds = 0;
    certification_index_metrics_v1 metrics{};
};

measurement_v1 measure(std::size_t count) {
    std::vector<canonical_relation_edge_v1> canonical;
    std::vector<indexed_proposal_membership_v1> memberships;
    canonical.reserve(count);
    memberships.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        canonical.push_back({id(index + 1), id(count + index + 1),
                             id(2 * count + index + 1)});
        memberships.push_back({id(index + 1), id(4 * count - index)});
    }
    scalable_certification_result_v1 result;
    const auto begin = std::chrono::steady_clock::now();
    assert(build_scalable_certification_index_v1(
               id(900000), 7, canonical, memberships, 2 * count,
               memberships.size() * sizeof(indexed_proposal_membership_v1),
               &result) == scalable_certification_status_v1::success);
    const auto end = std::chrono::steady_clock::now();
    assert(result.certificate.exact_cover);
    return {static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
                    .count()),
            result.metrics};
}

}  // namespace

int main() {
    const auto small = measure(1024);
    const auto medium = measure(4096);
    const auto large = measure(16384);
    assert(medium.metrics.peak_workspace_bytes ==
           4 * small.metrics.peak_workspace_bytes);
    assert(large.metrics.peak_workspace_bytes ==
           4 * medium.metrics.peak_workspace_bytes);
    assert(large.metrics.exact_lookups == 16384);
    assert(large.elapsed_nanoseconds < small.elapsed_nanoseconds * 128 + 1000000);

    const std::vector<canonical_relation_edge_v1> canonical{
        {id(1), id(10), id(20)}, {id(2), id(11), id(21)}};
    scalable_certification_result_v1 result;
    assert(build_scalable_certification_index_v1(
               id(100), 1, canonical,
               {{id(1), id(200)}, {id(1), id(201)}}, 8, 1024, &result) ==
           scalable_certification_status_v1::success);
    assert(result.certificate.omitted_edge_identities ==
           std::vector<persistent_atom_identity_v1>{id(2)});
    assert(result.certificate.duplicate_receipts.size() == 1);
    assert(build_scalable_certification_index_v1(
               id(100), 1, canonical, {{id(99), id(200)}}, 8, 1024, &result) ==
           scalable_certification_status_v1::unknown_edge);
}
