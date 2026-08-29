#include <Cellerator/geometry/support_atlas.hh>

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace geo = cellerator::geometry;

namespace cellerator::geometry {

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

bool query_support_affinity_requirements_v1(
    const support_relation_view_v1 &relation,
    const support_atlas_view_v1 &sampled_support,
    const support_sampling_policy_v1 &policy,
    support_atlas_requirements_v1 *out,
    std::string *error = nullptr);

bool build_support_affinity_v1(
    const support_relation_view_v1 &relation,
    const support_atlas_view_v1 &sampled_support,
    const support_sampling_policy_v1 &policy,
    const support_atlas_buffers_v1 &buffers,
    support_atlas_view_v1 *out,
    std::string *error = nullptr);

} // namespace cellerator::geometry

namespace {

struct fixture {
    std::vector<std::uint64_t> offsets{0u, 3u, 5u, 7u, 9u, 12u};
    std::vector<std::uint32_t> sources{
        0u, 1u, 2u,
        0u, 1u,
        0u, 2u,
        1u, 2u,
        0u, 1u, 2u};
    std::vector<double> weights{
        1.0, -2.0, 3.0,
        4.0, 5.0,
        6.0, -7.0,
        8.0, 9.0,
        -10.0, 11.0, 12.0};
    std::vector<geo::co_support_record_v1> observations;
    geo::support_relation_view_v1 relation{};
    geo::support_atlas_view_v1 sampled{};
    geo::support_sampling_policy_v1 policy{};

    fixture() {
        relation.relation_identity = 0x11u;
        relation.structure_identity = 0x22u;
        relation.structure_epoch = 3u;
        relation.source_axis_identity = 0x33u;
        relation.destination_axis_identity = 0x44u;
        relation.source_count = 3u;
        relation.destination_count = 5u;
        relation.edge_count = sources.size();
        relation.destination_offsets = offsets.data();
        relation.source_ids = sources.data();
        relation.edge_weights = weights.data();

        // Deliberately noncanonical observation order with repeated pairs.
        const std::uint32_t pairs[][2] = {
            {1u, 2u}, {0u, 1u}, {0u, 2u},
            {0u, 1u}, {1u, 2u}, {0u, 2u},
            {0u, 1u}, {1u, 2u}, {0u, 2u}};
        for (const auto &pair : pairs) {
            geo::co_support_record_v1 record{};
            record.source_a = pair[0];
            record.source_b = pair[1];
            record.sampled_support = 1u;
            record.weighted_support = pair[0] == 0u && pair[1] == 1u ? 1.0
                : pair[0] == 0u ? 0.5 : 0.75;
            observations.push_back(record);
        }
        sampled.flags = geo::support_atlas_flag_sampled | geo::support_atlas_flag_weighted;
        sampled.evidence_identity = 0x55u;
        sampled.relation_identity = relation.relation_identity;
        sampled.structure_identity = relation.structure_identity;
        sampled.structure_epoch = relation.structure_epoch;
        sampled.source_axis_identity = relation.source_axis_identity;
        sampled.destination_axis_identity = relation.destination_axis_identity;
        sampled.source_count = relation.source_count;
        sampled.destination_count = relation.destination_count;
        sampled.provenance.seed = 9u;
        sampled.provenance.sampled_destination_count = relation.destination_count;
        sampled.provenance.sampled_pair_observation_count = observations.size();
        sampled.co_support = observations.data();
        sampled.co_support_count = observations.size();

        policy.seed = 9u;
        policy.maximum_sampled_destinations = relation.destination_count;
        policy.maximum_pairs_per_destination = 16u;
        policy.top_l_per_source = 1u;
    }
};

struct output_storage {
    std::vector<geo::source_prevalence_v1> prevalence;
    std::vector<geo::destination_degree_v1> degrees;
    std::vector<geo::co_support_record_v1> co_support;
    std::vector<geo::source_affinity_record_v1> affinity;
    geo::support_atlas_buffers_v1 buffers{};

    explicit output_storage(const geo::support_atlas_requirements_v1 &requirements)
        : prevalence(requirements.prevalence_capacity),
          degrees(requirements.destination_degree_capacity),
          co_support(requirements.co_support_capacity),
          affinity(requirements.affinity_capacity) {
        buffers.prevalence = prevalence.data();
        buffers.prevalence_capacity = prevalence.size();
        buffers.destination_degrees = degrees.data();
        buffers.destination_degree_capacity = degrees.size();
        buffers.co_support = co_support.data();
        buffers.co_support_capacity = co_support.size();
        buffers.affinity = affinity.data();
        buffers.affinity_capacity = affinity.size();
    }
};

void test_sampled_support_pipeline_compatibility() {
    fixture data;
    data.policy.top_l_per_source = 2u;
    geo::support_atlas_requirements_v1 sampled_requirements{};
    std::string error;
    assert(geo::query_sampled_support_requirements_v1(
        data.relation, data.policy, &sampled_requirements, &error));
    assert(sampled_requirements.co_support_capacity == 9u);
    std::vector<geo::co_support_record_v1> sampled_records(
        sampled_requirements.co_support_capacity);
    geo::support_atlas_buffers_v1 sampled_buffers{};
    sampled_buffers.co_support = sampled_records.data();
    sampled_buffers.co_support_capacity = sampled_records.size();
    geo::support_atlas_view_v1 sampled{};
    assert(geo::build_sampled_support_v1(
        data.relation, data.policy, sampled_buffers, &sampled, &error));

    geo::support_atlas_requirements_v1 affinity_requirements{};
    assert(geo::query_support_affinity_requirements_v1(
        data.relation, sampled, data.policy, &affinity_requirements, &error));
    output_storage storage(affinity_requirements);
    geo::support_atlas_view_v1 atlas{};
    assert(geo::build_support_affinity_v1(
        data.relation, sampled, data.policy, storage.buffers, &atlas, &error));
    assert(atlas.co_support_count == 3u);
    assert(atlas.affinity_count == 6u);
    for (std::uint64_t pair = 0u; pair < atlas.co_support_count; ++pair) {
        assert(atlas.co_support[pair].sampled_support == 3u);
        assert(atlas.co_support[pair].weighted_support == 3.0);
        assert(atlas.co_support[pair].association_numerator == 3);
        assert(atlas.co_support[pair].association_denominator == 4u);
    }
}

void test_affinity_aggregation_and_top_l() {
    fixture data;
    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_support_affinity_requirements_v1(
        data.relation, data.sampled, data.policy, &requirements, &error));
    assert(requirements.prevalence_capacity == 3u);
    assert(requirements.destination_degree_capacity == 5u);
    assert(requirements.co_support_capacity == 32u);
    assert(requirements.affinity_capacity == 3u);
    assert(requirements.workspace_bytes == 0u);

    output_storage storage(requirements);
    geo::support_atlas_view_v1 atlas{};
    assert(geo::build_support_affinity_v1(
        data.relation, data.sampled, data.policy, storage.buffers, &atlas, &error));
    assert(atlas.flags == (geo::support_atlas_flag_sampled
        | geo::support_atlas_flag_weighted
        | geo::support_atlas_flag_normalized
        | geo::support_atlas_flag_top_l));
    assert(atlas.prevalence_count == 3u);
    for (std::uint32_t source = 0u; source < 3u; ++source) {
        assert(atlas.prevalence[source].source_id == source);
        assert(atlas.prevalence[source].destination_support == 4u);
    }
    assert(atlas.prevalence[0].weighted_destination_support == 21.0);
    assert(atlas.prevalence[1].weighted_destination_support == 26.0);
    assert(atlas.prevalence[2].weighted_destination_support == 31.0);
    const std::uint32_t expected_degrees[] = {3u, 2u, 2u, 2u, 3u};
    for (std::uint32_t destination = 0u; destination < 5u; ++destination) {
        assert(atlas.destination_degrees[destination].degree == expected_degrees[destination]);
    }
    assert(atlas.destination_degrees[0].total_edge_weight == 6.0);
    assert(atlas.destination_degrees[4].total_edge_weight == 33.0);

    assert(atlas.co_support_count == 3u);
    for (std::uint64_t pair = 0u; pair < atlas.co_support_count; ++pair) {
        const geo::co_support_record_v1 &record = atlas.co_support[pair];
        assert(record.sampled_support == 3u);
        if (record.source_a == 0u && record.source_b == 1u) {
            assert(record.weighted_support == 3.0);
            assert(record.association_numerator == 3);
            assert(record.association_denominator == 4u);
        } else if (record.source_a == 0u && record.source_b == 2u) {
            assert(record.weighted_support == 1.5);
            assert(record.association_numerator == 3);
            assert(record.association_denominator == 8u);
        } else {
            assert(record.source_a == 1u && record.source_b == 2u);
            assert(record.weighted_support == 2.25);
            assert(record.association_numerator == 9);
            assert(record.association_denominator == 16u);
        }
    }
    assert(atlas.affinity_count == 3u);
    assert(atlas.affinity[0].source_id == 0u);
    assert(atlas.affinity[0].neighbor_source_id == 1u);
    assert(atlas.affinity[1].source_id == 1u);
    assert(atlas.affinity[1].neighbor_source_id == 0u);
    assert(atlas.affinity[2].source_id == 2u);
    assert(atlas.affinity[2].neighbor_source_id == 1u);
    assert(atlas.affinity[0].rank == 0u);
    assert(atlas.affinity[0].score_numerator == 3);
    assert(atlas.affinity[0].score_denominator == 4u);
    assert(atlas.affinity[1].rank == 0u);
    assert(atlas.affinity[1].score_numerator == 3);
    assert(atlas.affinity[1].score_denominator == 4u);
    assert(atlas.affinity[2].rank == 0u);
    assert(atlas.affinity[2].score_numerator == 9);
    assert(atlas.affinity[2].score_denominator == 16u);

    output_storage repeated(requirements);
    geo::support_atlas_view_v1 repeated_atlas{};
    assert(geo::build_support_affinity_v1(
        data.relation, data.sampled, data.policy,
        repeated.buffers, &repeated_atlas, &error));
    assert(repeated_atlas.evidence_identity == atlas.evidence_identity);
    assert(std::memcmp(repeated.prevalence.data(), storage.prevalence.data(),
                       atlas.prevalence_count * sizeof(geo::source_prevalence_v1)) == 0);
    assert(std::memcmp(repeated.degrees.data(), storage.degrees.data(),
                       atlas.destination_degree_count * sizeof(geo::destination_degree_v1)) == 0);
    assert(std::memcmp(repeated.co_support.data(), storage.co_support.data(),
                       atlas.co_support_count * sizeof(geo::co_support_record_v1)) == 0);
    assert(std::memcmp(repeated.affinity.data(), storage.affinity.data(),
                       atlas.affinity_count * sizeof(geo::source_affinity_record_v1)) == 0);
}

void test_capacity_and_identity_rejection() {
    fixture data;
    geo::support_atlas_requirements_v1 requirements{};
    std::string error;
    assert(geo::query_support_affinity_requirements_v1(
        data.relation, data.sampled, data.policy, &requirements, &error));
    output_storage storage(requirements);
    geo::support_atlas_view_v1 atlas{};
    --storage.buffers.affinity_capacity;
    assert(!geo::build_support_affinity_v1(
        data.relation, data.sampled, data.policy, storage.buffers, &atlas, &error));

    ++storage.buffers.affinity_capacity;
    data.sampled.structure_epoch += 1u;
    assert(!geo::query_support_affinity_requirements_v1(
        data.relation, data.sampled, data.policy, &requirements, &error));

    data.sampled.structure_epoch = data.relation.structure_epoch;
    data.policy.top_l_per_source = 0u;
    assert(!geo::query_support_affinity_requirements_v1(
        data.relation, data.sampled, data.policy, &requirements, &error));
}

} // namespace

int main() {
    test_sampled_support_pipeline_compatibility();
    test_affinity_aggregation_and_top_l();
    test_capacity_and_identity_rejection();
    return 0;
}
