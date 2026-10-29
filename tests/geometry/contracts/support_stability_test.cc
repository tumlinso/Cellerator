#include <Cellerator/geometry/support_atlas.hh>

#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace geo = cellerator::geometry;

namespace cellerator::geometry {

bool query_support_multiresolution_requirements_v1(
    const support_atlas_view_v1 &base,
    const support_atlas_view_v1 *resamples,
    std::uint32_t resample_count,
    const biological_stratum_v1 *strata,
    std::uint64_t stratum_count,
    std::uint32_t resolution_count,
    std::uint64_t work_identity,
    support_atlas_requirements_v1 *out,
    std::string *error = nullptr);

bool build_support_multiresolution_v1(
    const support_atlas_view_v1 &base,
    const support_atlas_view_v1 *resamples,
    std::uint32_t resample_count,
    const biological_stratum_v1 *strata,
    std::uint64_t stratum_count,
    std::uint32_t resolution_count,
    std::uint64_t work_identity,
    const support_atlas_buffers_v1 &buffers,
    support_atlas_view_v1 *out,
    std::string *error = nullptr);

} // namespace cellerator::geometry

namespace {

geo::source_affinity_record_v1 edge(std::uint32_t source,
                                    std::uint32_t neighbor,
                                    std::uint32_t rank) {
    geo::source_affinity_record_v1 result{};
    result.source_id = source;
    result.neighbor_source_id = neighbor;
    result.rank = rank;
    result.score_numerator = 1;
    result.score_denominator = rank + 1u;
    return result;
}

geo::support_atlas_view_v1 atlas_view(
    const std::vector<geo::source_affinity_record_v1> &affinity,
    const std::vector<geo::destination_degree_v1> &degrees,
    std::uint64_t evidence_identity) {
    geo::support_atlas_view_v1 atlas{};
    atlas.flags = geo::support_atlas_flag_sampled
        | geo::support_atlas_flag_weighted
        | geo::support_atlas_flag_normalized
        | geo::support_atlas_flag_top_l;
    atlas.evidence_identity = evidence_identity;
    atlas.relation_identity = 0x11u;
    atlas.structure_identity = 0x22u;
    atlas.structure_epoch = 3u;
    atlas.source_axis_identity = 0x33u;
    atlas.destination_axis_identity = 0x44u;
    atlas.source_count = 5u;
    atlas.destination_count = degrees.size();
    atlas.affinity = affinity.data();
    atlas.affinity_count = affinity.size();
    atlas.destination_degrees = degrees.data();
    atlas.destination_degree_count = degrees.size();
    return atlas;
}

struct output_storage {
    std::vector<geo::community_assignment_v1> communities;
    std::vector<geo::work_signature_v1> work_signatures;
    std::vector<geo::biological_stratum_v1> strata;
    std::vector<geo::resampling_stability_v1> stability;
    geo::support_atlas_buffers_v1 buffers{};

    explicit output_storage(const geo::support_atlas_requirements_v1 &requirements)
        : communities(requirements.community_capacity),
          work_signatures(requirements.work_signature_capacity),
          strata(requirements.stratum_capacity),
          stability(requirements.stability_capacity) {
        buffers.communities = communities.data();
        buffers.community_capacity = communities.size();
        buffers.work_signatures = work_signatures.data();
        buffers.work_signature_capacity = work_signatures.size();
        buffers.strata = strata.data();
        buffers.stratum_capacity = strata.size();
        buffers.stability = stability.data();
        buffers.stability_capacity = stability.size();
    }
};

void test_multiresolution_stability_and_metadata() {
    const std::vector<geo::source_affinity_record_v1> base_affinity{
        edge(0u, 1u, 0u), edge(0u, 2u, 1u),
        edge(1u, 0u, 0u), edge(1u, 2u, 1u),
        edge(2u, 3u, 0u), edge(2u, 0u, 1u),
        edge(3u, 2u, 0u)};
    const std::vector<geo::source_affinity_record_v1> changed_affinity{
        edge(0u, 1u, 0u), edge(0u, 2u, 1u),
        edge(1u, 0u, 0u), edge(1u, 2u, 1u),
        edge(2u, 0u, 0u), edge(2u, 3u, 1u),
        edge(3u, 2u, 0u)};
    std::vector<geo::destination_degree_v1> degrees(4u);
    for (std::uint32_t destination = 0u; destination < degrees.size(); ++destination) {
        degrees[destination].destination_id = destination;
        degrees[destination].degree = destination + 1u;
    }
    const geo::support_atlas_view_v1 base = atlas_view(base_affinity, degrees, 0x101u);
    const geo::support_atlas_view_v1 resamples[] = {
        atlas_view(base_affinity, degrees, 0x102u),
        atlas_view(changed_affinity, degrees, 0x103u)};
    const geo::biological_stratum_v1 strata[] = {
        {9u, 20u, 3u, 1u},
        {9u, 10u, 0u, 0u}};

    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_support_multiresolution_requirements_v1(
        base, resamples, 2u, strata, 2u, 2u, 0x777u,
        &requirements, &error));
    assert(requirements.community_capacity == 10u);
    assert(requirements.stability_capacity == 10u);
    assert(requirements.stratum_capacity == 2u);
    assert(requirements.work_signature_capacity == 1u);
    assert(requirements.workspace_bytes == 0u);

    output_storage storage(requirements);
    geo::support_atlas_view_v1 output{};
    assert(geo::build_support_multiresolution_v1(
        base, resamples, 2u, strata, 2u, 2u, 0x777u,
        storage.buffers, &output, &error));
    assert(output.flags == (base.flags
        | geo::support_atlas_flag_multiresolution
        | geo::support_atlas_flag_stratified
        | geo::support_atlas_flag_resampled));
    assert(output.community_count == 10u);
    const std::uint32_t coarse[] = {0u, 0u, 0u, 0u, 4u};
    const std::uint32_t fine[] = {0u, 0u, 2u, 2u, 4u};
    for (std::uint32_t source = 0u; source < 5u; ++source) {
        assert(output.communities[source].resolution == 0u);
        assert(output.communities[source].community_id == coarse[source]);
        assert(output.communities[5u + source].resolution == 1u);
        assert(output.communities[5u + source].community_id == fine[source]);
    }
    assert(output.stability_count == 10u);
    for (std::uint32_t source = 0u; source < 5u; ++source) {
        assert(output.stability[source].stable_assignment_count == 2u);
        assert(output.stability[source].resample_count == 2u);
    }
    assert(output.stability[5u].stable_assignment_count == 2u);
    assert(output.stability[6u].stable_assignment_count == 2u);
    assert(output.stability[7u].stable_assignment_count == 1u);
    assert(output.stability[8u].stable_assignment_count == 1u);
    assert(output.stability[9u].stable_assignment_count == 2u);

    assert(output.stratum_count == 2u);
    assert(output.strata[0].stratum_identity == 10u);
    assert(output.strata[1].stratum_identity == 20u);
    assert(output.work_signature_count == 1u);
    assert(output.work_signatures[0].work_identity == 0x777u);
    assert(output.work_signatures[0].destination_count == 4u);
    assert(output.work_signatures[0].edge_count == 10u);

    output_storage repeated(requirements);
    geo::support_atlas_view_v1 repeated_output{};
    assert(geo::build_support_multiresolution_v1(
        base, resamples, 2u, strata, 2u, 2u, 0x777u,
        repeated.buffers, &repeated_output, &error));
    assert(repeated_output.evidence_identity == output.evidence_identity);
    assert(std::memcmp(repeated.communities.data(), storage.communities.data(),
                       output.community_count * sizeof(geo::community_assignment_v1)) == 0);
    assert(std::memcmp(repeated.stability.data(), storage.stability.data(),
                       output.stability_count * sizeof(geo::resampling_stability_v1)) == 0);
}

void test_validation_and_capacity() {
    std::vector<geo::source_affinity_record_v1> affinity{
        edge(0u, 1u, 0u), edge(1u, 0u, 0u)};
    std::vector<geo::destination_degree_v1> degrees(1u);
    degrees[0].degree = 2u;
    geo::support_atlas_view_v1 base = atlas_view(affinity, degrees, 1u);
    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_support_multiresolution_requirements_v1(
        base, nullptr, 0u, nullptr, 0u, 1u, 0u,
        &requirements, &error));
    output_storage storage(requirements);
    --storage.buffers.community_capacity;
    geo::support_atlas_view_v1 output{};
    assert(!geo::build_support_multiresolution_v1(
        base, nullptr, 0u, nullptr, 0u, 1u, 0u,
        storage.buffers, &output, &error));

    affinity[0].rank = 1u;
    assert(!geo::query_support_multiresolution_requirements_v1(
        base, nullptr, 0u, nullptr, 0u, 1u, 0u,
        &requirements, &error));
}

} // namespace

int main() {
    test_multiresolution_stability_and_metadata();
    test_validation_and_capacity();
    return 0;
}
