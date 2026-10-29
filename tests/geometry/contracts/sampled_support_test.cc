#include <Cellerator/geometry/support_atlas.hh>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace geo = cellerator::geometry;

namespace cellerator::geometry {

// CE-GEO-51 owns implementation but not the CE-GEO-50 public contract header.
// These exact declarations are published to the integration lane for addition
// to that header before the support-atlas interface is frozen.
bool query_sampled_support_requirements_v1(
    const support_relation_view_v1 &relation,
    const support_sampling_policy_v1 &policy,
    support_atlas_requirements_v1 *out,
    std::string *error = nullptr);

bool build_sampled_support_v1(
    const support_relation_view_v1 &relation,
    const support_sampling_policy_v1 &policy,
    const support_atlas_buffers_v1 &buffers,
    support_atlas_view_v1 *out,
    std::string *error = nullptr);

} // namespace cellerator::geometry

namespace {

geo::support_relation_view_v1 make_relation(
    std::uint32_t sources,
    const std::vector<std::uint64_t> &offsets,
    const std::vector<std::uint32_t> &source_ids) {
    geo::support_relation_view_v1 relation{};
    relation.relation_identity = 0x101u;
    relation.structure_identity = 0x202u;
    relation.structure_epoch = 7u;
    relation.source_axis_identity = 0x303u;
    relation.destination_axis_identity = 0x404u;
    relation.source_count = sources;
    relation.destination_count = static_cast<std::uint32_t>(offsets.size() - 1u);
    relation.edge_count = source_ids.size();
    relation.destination_offsets = offsets.data();
    relation.source_ids = source_ids.data();
    return relation;
}

geo::support_sampling_policy_v1 policy(std::uint64_t seed,
                                       std::uint64_t destinations,
                                       std::uint32_t pairs) {
    geo::support_sampling_policy_v1 result{};
    result.seed = seed;
    result.maximum_sampled_destinations = destinations;
    result.maximum_pairs_per_destination = pairs;
    return result;
}

bool same_record(const geo::co_support_record_v1 &left,
                 const geo::co_support_record_v1 &right) {
    return std::memcmp(&left, &right, sizeof(left)) == 0;
}

void test_bounded_high_degree_sampling() {
    constexpr std::uint32_t degree = 1024u;
    std::vector<std::uint64_t> offsets{0u, degree};
    std::vector<std::uint32_t> sources(degree);
    for (std::uint32_t source = 0u; source < degree; ++source) sources[source] = source;
    const geo::support_relation_view_v1 relation = make_relation(degree, offsets, sources);
    const geo::support_sampling_policy_v1 sample_policy = policy(0xabcdefu, 1u, 7u);

    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_sampled_support_requirements_v1(
        relation, sample_policy, &requirements, &error));
    assert(requirements.co_support_capacity == 7u);
    assert(requirements.workspace_bytes == 0u);

    std::vector<geo::co_support_record_v1> records(requirements.co_support_capacity);
    geo::support_atlas_view_v1 atlas{};
    geo::support_atlas_buffers_v1 buffers{};
    buffers.co_support = records.data();
    buffers.co_support_capacity = records.size();
    assert(geo::build_sampled_support_v1(relation, sample_policy, buffers, &atlas, &error));
    assert(atlas.co_support_count == 7u);
    assert(atlas.provenance.sampled_destination_count == 1u);
    assert(atlas.provenance.sampled_pair_observation_count == 7u);
    for (std::size_t i = 0u; i < records.size(); ++i) {
        assert(records[i].source_a < records[i].source_b);
        assert(records[i].sampled_support == 1u);
        assert(records[i].weighted_support == (degree * (degree - 1u) / 2u) / 7.0);
        for (std::size_t j = 0u; j < i; ++j) {
            assert(records[i].source_a != records[j].source_a
                || records[i].source_b != records[j].source_b);
        }
    }
}

void test_seeded_byte_determinism_and_bounds() {
    const std::vector<std::uint64_t> offsets{0u, 3u, 7u, 12u, 14u, 20u};
    const std::vector<std::uint32_t> sources{
        0u, 2u, 5u,
        1u, 3u, 5u, 7u,
        0u, 1u, 2u, 6u, 7u,
        3u, 4u,
        0u, 2u, 3u, 4u, 6u, 7u};
    const geo::support_relation_view_v1 relation = make_relation(8u, offsets, sources);
    const geo::support_sampling_policy_v1 sample_policy = policy(91u, 3u, 4u);

    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_sampled_support_requirements_v1(
        relation, sample_policy, &requirements, &error));
    assert(requirements.co_support_capacity <= 12u);

    std::vector<geo::co_support_record_v1> first(requirements.co_support_capacity);
    std::vector<geo::co_support_record_v1> second(requirements.co_support_capacity);
    geo::support_atlas_buffers_v1 first_buffers{};
    first_buffers.co_support = first.data();
    first_buffers.co_support_capacity = first.size();
    geo::support_atlas_buffers_v1 second_buffers{};
    second_buffers.co_support = second.data();
    second_buffers.co_support_capacity = second.size();
    geo::support_atlas_view_v1 first_view{};
    geo::support_atlas_view_v1 second_view{};
    assert(geo::build_sampled_support_v1(
        relation, sample_policy, first_buffers, &first_view, &error));
    assert(geo::build_sampled_support_v1(
        relation, sample_policy, second_buffers, &second_view, &error));
    assert(first_view.evidence_identity == second_view.evidence_identity);
    assert(first_view.provenance.input_identity == second_view.provenance.input_identity);
    assert(first_view.provenance.seed == 91u);
    assert(first_view.provenance.sampled_destination_count == 3u);
    assert(first_view.co_support_count == second_view.co_support_count);
    for (std::size_t i = 0u; i < first.size(); ++i) {
        assert(same_record(first[i], second[i]));
    }

    const geo::support_sampling_policy_v1 other_seed = policy(92u, 3u, 4u);
    geo::support_atlas_requirements_v1 other_requirements{};
    assert(geo::query_sampled_support_requirements_v1(
        relation, other_seed, &other_requirements, &error));
    std::vector<geo::co_support_record_v1> other(other_requirements.co_support_capacity);
    geo::support_atlas_buffers_v1 other_buffers{};
    other_buffers.co_support = other.data();
    other_buffers.co_support_capacity = other.size();
    geo::support_atlas_view_v1 other_view{};
    assert(geo::build_sampled_support_v1(
        relation, other_seed, other_buffers, &other_view, &error));
    assert(other_view.evidence_identity != first_view.evidence_identity);
}

void test_exact_small_rows_and_validation() {
    const std::vector<std::uint64_t> offsets{0u, 3u, 5u};
    const std::vector<std::uint32_t> sources{0u, 1u, 3u, 2u, 4u};
    geo::support_relation_view_v1 relation = make_relation(5u, offsets, sources);
    geo::support_sampling_policy_v1 sample_policy = policy(3u, 2u, 99u);
    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_sampled_support_requirements_v1(
        relation, sample_policy, &requirements, &error));
    assert(requirements.co_support_capacity == 4u);

    std::vector<geo::co_support_record_v1> records(requirements.co_support_capacity);
    geo::support_atlas_buffers_v1 buffers{};
    buffers.co_support = records.data();
    buffers.co_support_capacity = records.size();
    geo::support_atlas_view_v1 atlas{};
    assert(geo::build_sampled_support_v1(relation, sample_policy, buffers, &atlas, &error));
    assert(atlas.flags == (geo::support_atlas_flag_sampled | geo::support_atlas_flag_weighted));
    assert(atlas.co_support_count == 4u);
    assert(atlas.relation_identity == relation.relation_identity);
    assert(atlas.structure_identity == relation.structure_identity);
    assert(atlas.provenance.exact_rescan_edge_count == 0u);

    std::vector<std::uint64_t> observed;
    for (const geo::co_support_record_v1 &record : records) {
        observed.push_back((static_cast<std::uint64_t>(record.source_a) << 32u)
            | record.source_b);
        assert(record.weighted_support == 1.0);
    }
    std::sort(observed.begin(), observed.end());
    const std::vector<std::uint64_t> expected{
        (0ull << 32u) | 1u,
        (0ull << 32u) | 3u,
        (1ull << 32u) | 3u,
        (2ull << 32u) | 4u};
    assert(observed == expected);

    geo::support_atlas_buffers_v1 too_small{};
    too_small.co_support = records.data();
    too_small.co_support_capacity = records.size() - 1u;
    assert(!geo::build_sampled_support_v1(
        relation, sample_policy, too_small, &atlas, &error));

    const std::vector<std::uint32_t> duplicate_sources{0u, 1u, 1u, 2u, 4u};
    relation.source_ids = duplicate_sources.data();
    assert(!geo::query_sampled_support_requirements_v1(
        relation, sample_policy, &requirements, &error));

    relation.source_ids = sources.data();
    sample_policy.maximum_pairs_per_destination = 0u;
    assert(!geo::query_sampled_support_requirements_v1(
        relation, sample_policy, &requirements, &error));
}

} // namespace

int main() {
    test_bounded_high_degree_sampling();
    test_seeded_byte_determinism_and_bounds();
    test_exact_small_rows_and_validation();
    return 0;
}
