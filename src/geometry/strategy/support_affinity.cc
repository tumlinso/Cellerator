#include <Cellerator/geometry/support_atlas.hh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <string>

namespace cellerator::geometry {

namespace {

constexpr std::uint64_t association_scale = 1000000u;
constexpr std::uint64_t affinity_identity_domain = 0x616666696e697431ull;

void set_error(std::string *error, const char *message) {
    if (error != nullptr) *error = message;
}

bool checked_multiply(std::uint64_t left, std::uint64_t right,
                      std::uint64_t *out) noexcept {
    if (out == nullptr
        || (left != 0u && right > std::numeric_limits<std::uint64_t>::max() / left)) {
        return false;
    }
    *out = left * right;
    return true;
}

std::uint64_t mix_identity(std::uint64_t identity,
                           std::uint64_t value) noexcept {
    identity ^= value + 0x9e3779b97f4a7c15ull + (identity << 6u) + (identity >> 2u);
    identity ^= identity >> 30u;
    identity *= 0xbf58476d1ce4e5b9ull;
    identity ^= identity >> 27u;
    identity *= 0x94d049bb133111ebull;
    return identity ^ (identity >> 31u);
}

bool validate_relation(const support_relation_view_v1 &relation,
                       std::string *error) {
    if (relation.destination_offsets == nullptr) {
        set_error(error, "support affinity requires destination offsets");
        return false;
    }
    if (relation.edge_count != 0u && relation.source_ids == nullptr) {
        set_error(error, "non-empty support affinity relation requires source IDs");
        return false;
    }
    if (relation.destination_offsets[0] != 0u
        || relation.destination_offsets[relation.destination_count] != relation.edge_count) {
        set_error(error, "support affinity offsets do not span edge_count");
        return false;
    }
    for (std::uint64_t destination = 0u;
         destination < relation.destination_count; ++destination) {
        const std::uint64_t begin = relation.destination_offsets[destination];
        const std::uint64_t end = relation.destination_offsets[destination + 1u];
        if (end < begin || end > relation.edge_count) {
            set_error(error, "support affinity offsets are not monotonic");
            return false;
        }
        std::uint32_t previous = 0u;
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const std::uint32_t source = relation.source_ids[edge];
            if (source >= relation.source_count) {
                set_error(error, "support affinity source ID is out of range");
                return false;
            }
            if (edge != begin && source <= previous) {
                set_error(error, "support affinity requires sorted unique destination supports");
                return false;
            }
            if (relation.edge_weights != nullptr
                && !std::isfinite(relation.edge_weights[edge])) {
                set_error(error, "support affinity edge weights must be finite");
                return false;
            }
            previous = source;
        }
    }
    return true;
}

bool validate_sampled_support(const support_relation_view_v1 &relation,
                              const support_atlas_view_v1 &sampled,
                              std::string *error) {
    if (sampled.schema_version != support_atlas_schema_version_v1
        || sampled.provenance.schema_version != support_atlas_schema_version_v1
        || sampled.provenance.sampling_algorithm_version
            != support_sampling_algorithm_version_v1) {
        set_error(error, "sampled support schema is unsupported");
        return false;
    }
    if ((sampled.flags & support_atlas_flag_sampled) == 0u
        || (sampled.flags & support_atlas_flag_weighted) == 0u) {
        set_error(error, "support affinity requires sampled weighted observations");
        return false;
    }
    if (sampled.relation_identity != relation.relation_identity
        || sampled.structure_identity != relation.structure_identity
        || sampled.structure_epoch != relation.structure_epoch
        || sampled.source_axis_identity != relation.source_axis_identity
        || sampled.destination_axis_identity != relation.destination_axis_identity
        || sampled.source_count != relation.source_count
        || sampled.destination_count != relation.destination_count) {
        set_error(error, "sampled support identity does not match the relation");
        return false;
    }
    if (sampled.co_support_count != 0u && sampled.co_support == nullptr) {
        set_error(error, "sampled support records are absent");
        return false;
    }
    if (sampled.provenance.sampled_pair_observation_count != sampled.co_support_count) {
        set_error(error, "sampled support observation provenance is inconsistent");
        return false;
    }
    for (std::uint64_t i = 0u; i < sampled.co_support_count; ++i) {
        const co_support_record_v1 &record = sampled.co_support[i];
        if (record.source_a >= record.source_b
            || record.source_b >= relation.source_count
            || record.sampled_support == 0u
            || !std::isfinite(record.weighted_support)
            || record.weighted_support < 0.0) {
            set_error(error, "sampled support observation is invalid");
            return false;
        }
    }
    return true;
}

bool checked_accumulate(co_support_record_v1 *target,
                        const co_support_record_v1 &value,
                        std::string *error) {
    if (value.sampled_support
        > std::numeric_limits<std::uint64_t>::max() - target->sampled_support) {
        set_error(error, "raw co-support accumulation overflows");
        return false;
    }
    const double weighted = target->weighted_support + value.weighted_support;
    if (!std::isfinite(weighted)) {
        set_error(error, "weighted co-support accumulation overflows");
        return false;
    }
    target->sampled_support += value.sampled_support;
    target->weighted_support = weighted;
    return true;
}

bool query_pair_table_capacity(std::uint64_t observations,
                               std::uint64_t *out,
                               std::string *error) {
    if (observations == 0u) {
        *out = 0u;
        return true;
    }
    if (observations > (std::uint64_t{1} << 62u)) {
        set_error(error, "co-support hash-table capacity overflows");
        return false;
    }
    const std::uint64_t required = std::max<std::uint64_t>(8u, observations * 2u);
    std::uint64_t capacity = 8u;
    while (capacity < required) capacity <<= 1u;
    *out = capacity;
    return true;
}

std::uint64_t pair_key(const co_support_record_v1 &record) noexcept {
    return (static_cast<std::uint64_t>(record.source_a) << 32u)
        | record.source_b;
}

bool aggregate_observations(const support_atlas_view_v1 &sampled_support,
                            co_support_record_v1 *table,
                            std::uint64_t table_capacity,
                            std::uint64_t *unique_count,
                            std::string *error) {
    constexpr std::uint32_t empty_source = std::numeric_limits<std::uint32_t>::max();
    if (sampled_support.co_support_count == 0u) {
        *unique_count = 0u;
        return true;
    }
    for (std::uint64_t slot = 0u; slot < table_capacity; ++slot) {
        table[slot] = co_support_record_v1{};
        table[slot].source_a = empty_source;
    }
    const std::uint64_t mask = table_capacity - 1u;
    for (std::uint64_t observation = 0u;
         observation < sampled_support.co_support_count; ++observation) {
        const co_support_record_v1 &record = sampled_support.co_support[observation];
        std::uint64_t slot = mix_identity(affinity_identity_domain,
                                          pair_key(record)) & mask;
        while (table[slot].source_a != empty_source
               && (table[slot].source_a != record.source_a
                   || table[slot].source_b != record.source_b)) {
            slot = (slot + 1u) & mask;
        }
        if (table[slot].source_a == empty_source) {
            table[slot] = record;
        } else if (!checked_accumulate(table + slot, record, error)) {
            return false;
        }
    }
    std::uint64_t count = 0u;
    for (std::uint64_t slot = 0u; slot < table_capacity; ++slot) {
        if (table[slot].source_a == empty_source) continue;
        if (count != slot) table[count] = table[slot];
        ++count;
    }
    *unique_count = count;
    return true;
}

bool encode_overlap_association(co_support_record_v1 *record,
                                std::uint64_t support_a,
                                std::uint64_t support_b,
                                std::string *error) {
    const std::uint64_t shared_bound = std::min(support_a, support_b);
    if (shared_bound == 0u) {
        set_error(error, "co-support pair refers to a source with zero prevalence");
        return false;
    }
    const long double scaled = static_cast<long double>(record->weighted_support)
        * static_cast<long double>(association_scale);
    if (!std::isfinite(scaled)
        || scaled > static_cast<long double>(std::numeric_limits<std::int64_t>::max())) {
        set_error(error, "normalized co-support is not representable");
        return false;
    }
    std::uint64_t numerator = static_cast<std::uint64_t>(std::llround(scaled));
    std::uint64_t denominator = 0u;
    if (!checked_multiply(shared_bound, association_scale, &denominator)
        || denominator == 0u
        || numerator > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        set_error(error, "normalized co-support denominator is not representable");
        return false;
    }
    const std::uint64_t divisor = std::gcd(numerator, denominator);
    record->association_numerator = static_cast<std::int64_t>(numerator / divisor);
    record->association_denominator = denominator / divisor;
    return true;
}

bool affinity_better(const source_affinity_record_v1 &left,
                     const source_affinity_record_v1 &right) noexcept {
    const __int128_t lhs = static_cast<__int128_t>(left.score_numerator)
        * right.score_denominator;
    const __int128_t rhs = static_cast<__int128_t>(right.score_numerator)
        * left.score_denominator;
    if (lhs != rhs) return lhs > rhs;
    return left.neighbor_source_id < right.neighbor_source_id;
}

// Heap root is the least desirable retained neighbor.
bool heap_better(const source_affinity_record_v1 &left,
                 const source_affinity_record_v1 &right) noexcept {
    return affinity_better(left, right);
}

void retain_affinity(source_affinity_record_v1 *storage,
                     std::uint32_t source,
                     std::uint32_t neighbor,
                     std::int64_t numerator,
                     std::uint64_t denominator,
                     std::uint32_t top_l) {
    source_affinity_record_v1 *const heap = storage
        + static_cast<std::uint64_t>(source) * top_l;
    std::uint32_t count = heap[0].reserved;
    source_affinity_record_v1 candidate{};
    candidate.source_id = source;
    candidate.neighbor_source_id = neighbor;
    candidate.score_numerator = numerator;
    candidate.score_denominator = denominator;
    if (count < top_l) {
        heap[count] = candidate;
        ++count;
        std::push_heap(heap, heap + count, heap_better);
    } else if (affinity_better(candidate, heap[0])) {
        std::pop_heap(heap, heap + count, heap_better);
        heap[count - 1u] = candidate;
        std::push_heap(heap, heap + count, heap_better);
    }
    heap[0].reserved = count;
}

} // namespace

bool query_support_affinity_requirements_v1(
    const support_relation_view_v1 &relation,
    const support_atlas_view_v1 &sampled_support,
    const support_sampling_policy_v1 &policy,
    support_atlas_requirements_v1 *out,
    std::string *error) {
    if (out == nullptr) {
        set_error(error, "support affinity requirements output is null");
        return false;
    }
    *out = support_atlas_requirements_v1{};
    if (!validate_relation(relation, error)
        || !validate_sampled_support(relation, sampled_support, error)) {
        return false;
    }
    if (policy.schema_version != support_atlas_schema_version_v1
        || policy.sampling_algorithm_version != support_sampling_algorithm_version_v1) {
        set_error(error, "support affinity policy version is unsupported");
        return false;
    }
    const std::uint32_t top_l = relation.source_count < 2u ? 0u
        : std::min<std::uint32_t>(policy.top_l_per_source, relation.source_count - 1u);
    if (relation.source_count > 1u && top_l == 0u) {
        set_error(error, "support affinity requires a positive top-L bound");
        return false;
    }
    out->prevalence_capacity = relation.source_count;
    out->destination_degree_capacity = relation.destination_count;
    if (!query_pair_table_capacity(sampled_support.co_support_count,
                                   &out->co_support_capacity, error)) {
        *out = support_atlas_requirements_v1{};
        return false;
    }
    if (!checked_multiply(relation.source_count, top_l,
                          &out->affinity_capacity)) {
        set_error(error, "support affinity capacity overflows");
        *out = support_atlas_requirements_v1{};
        return false;
    }
    return true;
}

bool build_support_affinity_v1(
    const support_relation_view_v1 &relation,
    const support_atlas_view_v1 &sampled_support,
    const support_sampling_policy_v1 &policy,
    const support_atlas_buffers_v1 &buffers,
    support_atlas_view_v1 *out,
    std::string *error) {
    support_atlas_requirements_v1 requirements;
    if (out == nullptr) {
        set_error(error, "support affinity output view is null");
        return false;
    }
    *out = support_atlas_view_v1{};
    if (!query_support_affinity_requirements_v1(
            relation, sampled_support, policy, &requirements, error)) {
        return false;
    }
    if ((requirements.prevalence_capacity != 0u && buffers.prevalence == nullptr)
        || buffers.prevalence_capacity < requirements.prevalence_capacity
        || (requirements.destination_degree_capacity != 0u
            && buffers.destination_degrees == nullptr)
        || buffers.destination_degree_capacity < requirements.destination_degree_capacity
        || (requirements.co_support_capacity != 0u && buffers.co_support == nullptr)
        || buffers.co_support_capacity < requirements.co_support_capacity
        || (requirements.affinity_capacity != 0u && buffers.affinity == nullptr)
        || buffers.affinity_capacity < requirements.affinity_capacity) {
        set_error(error, "support affinity output capacity is insufficient");
        return false;
    }

    for (std::uint32_t source = 0u; source < relation.source_count; ++source) {
        buffers.prevalence[source] = source_prevalence_v1{};
        buffers.prevalence[source].source_id = source;
    }
    for (std::uint32_t destination = 0u;
         destination < relation.destination_count; ++destination) {
        const std::uint64_t begin = relation.destination_offsets[destination];
        const std::uint64_t end = relation.destination_offsets[destination + 1u];
        destination_degree_v1 degree{};
        degree.destination_id = destination;
        degree.degree = static_cast<std::uint32_t>(end - begin);
        for (std::uint64_t edge = begin; edge < end; ++edge) {
            const std::uint32_t source = relation.source_ids[edge];
            const double magnitude = relation.edge_weights == nullptr
                ? 1.0 : std::abs(relation.edge_weights[edge]);
            ++buffers.prevalence[source].destination_support;
            buffers.prevalence[source].weighted_destination_support += magnitude;
            degree.total_edge_weight += magnitude;
            if (!std::isfinite(buffers.prevalence[source].weighted_destination_support)
                || !std::isfinite(degree.total_edge_weight)) {
                set_error(error, "support weight accumulation overflows");
                return false;
            }
        }
        buffers.destination_degrees[destination] = degree;
    }

    std::uint64_t unique_count = 0u;
    if (!aggregate_observations(sampled_support, buffers.co_support,
                                requirements.co_support_capacity,
                                &unique_count, error)) return false;
    for (std::uint64_t pair = 0u; pair < unique_count; ++pair) {
        co_support_record_v1 &record = buffers.co_support[pair];
        if (!encode_overlap_association(
                &record,
                buffers.prevalence[record.source_a].destination_support,
                buffers.prevalence[record.source_b].destination_support,
                error)) {
            return false;
        }
    }

    const std::uint32_t top_l = relation.source_count < 2u ? 0u
        : std::min<std::uint32_t>(policy.top_l_per_source, relation.source_count - 1u);
    std::uint64_t affinity_count = 0u;
    if (top_l != 0u) {
        std::fill(buffers.affinity, buffers.affinity + requirements.affinity_capacity,
                  source_affinity_record_v1{});
        for (std::uint64_t pair = 0u; pair < unique_count; ++pair) {
            const co_support_record_v1 &record = buffers.co_support[pair];
            retain_affinity(buffers.affinity, record.source_a, record.source_b,
                            record.association_numerator,
                            record.association_denominator, top_l);
            retain_affinity(buffers.affinity, record.source_b, record.source_a,
                            record.association_numerator,
                            record.association_denominator, top_l);
        }
        for (std::uint32_t source = 0u; source < relation.source_count; ++source) {
            source_affinity_record_v1 *const heap = buffers.affinity
                + static_cast<std::uint64_t>(source) * top_l;
            const std::uint32_t count = heap[0].reserved;
            if (count != 0u) {
                heap[0].reserved = 0u;
                std::sort(heap, heap + count, affinity_better);
            }
            for (std::uint32_t rank = 0u; rank < count; ++rank) {
                source_affinity_record_v1 record = heap[rank];
                record.rank = rank;
                record.reserved = 0u;
                buffers.affinity[affinity_count++] = record;
            }
        }
    }

    *out = sampled_support;
    out->flags |= support_atlas_flag_normalized | support_atlas_flag_top_l;
    out->evidence_identity = mix_identity(
        affinity_identity_domain, sampled_support.evidence_identity);
    out->evidence_identity = mix_identity(out->evidence_identity, unique_count);
    out->evidence_identity = mix_identity(out->evidence_identity, affinity_count);
    out->prevalence = relation.source_count == 0u ? nullptr : buffers.prevalence;
    out->prevalence_count = relation.source_count;
    out->destination_degrees = relation.destination_count == 0u
        ? nullptr : buffers.destination_degrees;
    out->destination_degree_count = relation.destination_count;
    out->co_support = unique_count == 0u ? nullptr : buffers.co_support;
    out->co_support_count = unique_count;
    out->affinity = affinity_count == 0u ? nullptr : buffers.affinity;
    out->affinity_count = affinity_count;
    return true;
}

} // namespace cellerator::geometry
