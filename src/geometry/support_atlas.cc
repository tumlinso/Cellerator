#include <Cellerator/compute/sampling.hh>
#include <Cellerator/geometry/support_atlas.hh>

#include <algorithm>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

namespace cellerator::geometry {

namespace {

constexpr std::uint64_t destination_sampling_domain = 0x737570706f727431ull;
constexpr std::uint64_t pair_sampling_domain = 0x70616972735f7631ull;
constexpr std::uint64_t evidence_identity_domain = 0x61746c61735f7631ull;

void set_error(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
}

bool checked_add(std::uint64_t left, std::uint64_t right,
                 std::uint64_t *out) noexcept {
    if (out == nullptr || right > std::numeric_limits<std::uint64_t>::max() - left) {
        return false;
    }
    *out = left + right;
    return true;
}

std::uint64_t mix_identity(std::uint64_t identity,
                           std::uint64_t value) noexcept {
    using ::cellerator::compute::sampling::splitmix64_hash;
    return splitmix64_hash(identity ^ splitmix64_hash(value));
}

std::uint64_t relation_input_identity(
    const support_relation_view_v1 &relation) noexcept {
    std::uint64_t identity = evidence_identity_domain;
    identity = mix_identity(identity, relation.relation_identity);
    identity = mix_identity(identity, relation.structure_identity);
    identity = mix_identity(identity, relation.structure_epoch);
    identity = mix_identity(identity, relation.source_axis_identity);
    identity = mix_identity(identity, relation.destination_axis_identity);
    identity = mix_identity(identity, relation.source_count);
    identity = mix_identity(identity, relation.destination_count);
    identity = mix_identity(identity, relation.edge_count);
    for (std::uint64_t destination = 0u;
         destination <= relation.destination_count; ++destination) {
        identity = mix_identity(identity, relation.destination_offsets[destination]);
    }
    for (std::uint64_t edge = 0u; edge < relation.edge_count; ++edge) {
        identity = mix_identity(identity, relation.source_ids[edge]);
        if (relation.edge_weights != nullptr) {
            std::uint64_t bits = 0u;
            std::memcpy(&bits, relation.edge_weights + edge, sizeof(bits));
            identity = mix_identity(identity, bits);
        }
    }
    return identity;
}

bool validate_relation(const support_relation_view_v1 &relation,
                       std::string *error) {
    if (relation.destination_offsets == nullptr) {
        set_error(error, "support relation requires destination offsets");
        return false;
    }
    if (relation.edge_count != 0u && relation.source_ids == nullptr) {
        set_error(error, "non-empty support relation requires source IDs");
        return false;
    }
    if (relation.destination_offsets[0] != 0u
        || relation.destination_offsets[relation.destination_count] != relation.edge_count) {
        set_error(error, "support relation offsets do not span edge_count");
        return false;
    }
    for (std::uint64_t destination = 0u;
         destination < relation.destination_count; ++destination) {
        const std::uint64_t begin = relation.destination_offsets[destination];
        const std::uint64_t end = relation.destination_offsets[destination + 1u];
        if (end < begin || end > relation.edge_count) {
            set_error(error, "support relation offsets are not monotonic");
            return false;
        }
        std::uint32_t previous = 0u;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const std::uint32_t source = relation.source_ids[edge];
            if (source >= relation.source_count) {
                set_error(error, "support relation source ID is out of range");
                return false;
            }
            if (edge != begin && source <= previous) {
                set_error(error, "destination supports must contain sorted unique source IDs");
                return false;
            }
            previous = source;
        }
    }
    return true;
}

bool validate_policy(const support_sampling_policy_v1 &policy,
                     std::uint32_t destination_count,
                     std::string *error) {
    if (policy.schema_version != support_atlas_schema_version_v1
        || policy.sampling_algorithm_version != support_sampling_algorithm_version_v1) {
        set_error(error, "unsupported support sampling policy version");
        return false;
    }
    if (destination_count != 0u && policy.maximum_sampled_destinations == 0u) {
        set_error(error, "support sampling requires a positive destination bound");
        return false;
    }
    if (destination_count != 0u && policy.maximum_pairs_per_destination == 0u) {
        set_error(error, "support sampling requires a positive pair bound");
        return false;
    }
    return true;
}

std::uint64_t pair_count(std::uint64_t degree) noexcept {
    return degree < 2u ? 0u : degree * (degree - 1u) / 2u;
}

std::uint64_t coprime_step(std::uint64_t extent,
                           std::uint64_t hash) noexcept {
    if (extent <= 1u) return 0u;
    std::uint64_t step = hash % extent;
    if (step == 0u) step = 1u;
    while (std::gcd(step, extent) != 1u) {
        ++step;
        if (step == extent) step = 1u;
    }
    return step;
}

struct deterministic_permutation {
    std::uint64_t extent = 0u;
    std::uint64_t start = 0u;
    std::uint64_t step = 0u;

    std::uint64_t operator()(std::uint64_t index) const noexcept {
        if (extent <= 1u) return 0u;
        const __uint128_t offset = static_cast<__uint128_t>(index) * step;
        return (start + static_cast<std::uint64_t>(offset % extent)) % extent;
    }
};

deterministic_permutation make_permutation(std::uint64_t extent,
                                           std::uint64_t seed,
                                           std::uint64_t domain) noexcept {
    using ::cellerator::compute::sampling::hash_global_row_index;
    deterministic_permutation result;
    result.extent = extent;
    if (extent == 0u) return result;
    result.start = hash_global_row_index(domain, seed) % extent;
    result.step = coprime_step(extent,
        hash_global_row_index(domain ^ 0x9e3779b97f4a7c15ull, seed));
    return result;
}

void pair_from_rank(std::uint64_t degree, std::uint64_t rank,
                    std::uint64_t *first, std::uint64_t *second) noexcept {
    std::uint64_t low = 0u;
    std::uint64_t high = degree - 1u;
    while (low + 1u < high) {
        const std::uint64_t middle = low + (high - low) / 2u;
        const __uint128_t before = static_cast<__uint128_t>(middle)
            * (2u * static_cast<__uint128_t>(degree) - middle - 1u) / 2u;
        if (before <= rank) low = middle;
        else high = middle;
    }
    const __uint128_t before = static_cast<__uint128_t>(low)
        * (2u * static_cast<__uint128_t>(degree) - low - 1u) / 2u;
    *first = low;
    *second = low + 1u + static_cast<std::uint64_t>(rank - before);
}

std::uint64_t sampled_destination_count(
    const support_relation_view_v1 &relation,
    const support_sampling_policy_v1 &policy) noexcept {
    return std::min<std::uint64_t>(relation.destination_count,
                                   policy.maximum_sampled_destinations);
}

bool query_observation_count(const support_relation_view_v1 &relation,
                             const support_sampling_policy_v1 &policy,
                             std::uint64_t *out,
                             std::string *error) {
    const std::uint64_t destination_samples = sampled_destination_count(relation, policy);
    const deterministic_permutation destinations = make_permutation(
        relation.destination_count, policy.seed, destination_sampling_domain);
    std::uint64_t observations = 0u;
    for (std::uint64_t sample = 0u; sample < destination_samples; ++sample) {
        const std::uint64_t destination = destinations(sample);
        const std::uint64_t degree = relation.destination_offsets[destination + 1u]
            - relation.destination_offsets[destination];
        const std::uint64_t bounded = std::min<std::uint64_t>(
            pair_count(degree), policy.maximum_pairs_per_destination);
        if (!checked_add(observations, bounded, &observations)) {
            set_error(error, "sampled support observation count overflows");
            return false;
        }
    }
    *out = observations;
    return true;
}

} // namespace

bool query_sampled_support_requirements_v1(
    const support_relation_view_v1 &relation,
    const support_sampling_policy_v1 &policy,
    support_atlas_requirements_v1 *out,
    std::string *error) {
    if (out == nullptr) {
        set_error(error, "sampled support requirements output is null");
        return false;
    }
    *out = support_atlas_requirements_v1{};
    if (!validate_relation(relation, error)
        || !validate_policy(policy, relation.destination_count, error)) {
        return false;
    }
    if (!query_observation_count(relation, policy, &out->co_support_capacity, error)) {
        *out = support_atlas_requirements_v1{};
        return false;
    }
    return true;
}

bool build_sampled_support_v1(
    const support_relation_view_v1 &relation,
    const support_sampling_policy_v1 &policy,
    const support_atlas_buffers_v1 &buffers,
    support_atlas_view_v1 *out,
    std::string *error) {
    support_atlas_requirements_v1 requirements;
    if (out == nullptr) {
        set_error(error, "sampled support output view is null");
        return false;
    }
    *out = support_atlas_view_v1{};
    if (!query_sampled_support_requirements_v1(relation, policy,
                                               &requirements, error)) {
        return false;
    }
    if (buffers.co_support_capacity < requirements.co_support_capacity
        || (requirements.co_support_capacity != 0u && buffers.co_support == nullptr)) {
        set_error(error, "sampled support co-support capacity is insufficient");
        return false;
    }

    const std::uint64_t destination_samples = sampled_destination_count(relation, policy);
    const deterministic_permutation destinations = make_permutation(
        relation.destination_count, policy.seed, destination_sampling_domain);
    std::uint64_t observation = 0u;
    std::uint64_t evidence_identity = mix_identity(
        evidence_identity_domain, relation_input_identity(relation));
    evidence_identity = mix_identity(evidence_identity, policy.seed);
    evidence_identity = mix_identity(evidence_identity, destination_samples);
    evidence_identity = mix_identity(evidence_identity,
                                     policy.maximum_pairs_per_destination);

    for (std::uint64_t sample = 0u; sample < destination_samples; ++sample) {
        const std::uint64_t destination = destinations(sample);
        const std::uint64_t begin = relation.destination_offsets[destination];
        const std::uint64_t degree = relation.destination_offsets[destination + 1u] - begin;
        const std::uint64_t available_pairs = pair_count(degree);
        const std::uint64_t sampled_pairs = std::min<std::uint64_t>(
            available_pairs, policy.maximum_pairs_per_destination);
        if (sampled_pairs == 0u) continue;

        const deterministic_permutation pairs = make_permutation(
            available_pairs,
            policy.seed ^ ::cellerator::compute::sampling::hash_global_row_index(
                destination, pair_sampling_domain),
            pair_sampling_domain);
        const double inverse_destination_probability = static_cast<double>(
            relation.destination_count) / static_cast<double>(destination_samples);
        const double inverse_pair_probability = static_cast<double>(available_pairs)
            / static_cast<double>(sampled_pairs);

        for (std::uint64_t pair_sample = 0u;
             pair_sample < sampled_pairs; ++pair_sample) {
            std::uint64_t first = 0u;
            std::uint64_t second = 0u;
            pair_from_rank(degree, pairs(pair_sample), &first, &second);
            co_support_record_v1 record{};
            record.source_a = relation.source_ids[begin + first];
            record.source_b = relation.source_ids[begin + second];
            record.sampled_support = 1u;
            record.weighted_support = inverse_destination_probability
                * inverse_pair_probability;
            buffers.co_support[observation++] = record;
            evidence_identity = mix_identity(evidence_identity,
                (static_cast<std::uint64_t>(record.source_a) << 32u)
                    | record.source_b);
        }
    }

    out->flags = support_atlas_flag_sampled | support_atlas_flag_weighted;
    out->evidence_identity = mix_identity(evidence_identity, observation);
    out->relation_identity = relation.relation_identity;
    out->structure_identity = relation.structure_identity;
    out->structure_epoch = relation.structure_epoch;
    out->source_axis_identity = relation.source_axis_identity;
    out->destination_axis_identity = relation.destination_axis_identity;
    out->source_count = relation.source_count;
    out->destination_count = relation.destination_count;
    out->provenance.seed = policy.seed;
    out->provenance.input_identity = relation_input_identity(relation);
    out->provenance.sampled_destination_count = destination_samples;
    out->provenance.sampled_pair_observation_count = observation;
    out->co_support = observation == 0u ? nullptr : buffers.co_support;
    out->co_support_count = observation;
    return true;
}

} // namespace cellerator::geometry
