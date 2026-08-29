#include <Cellerator/geometry/support_atlas.hh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace cellerator::geometry {

namespace {

constexpr std::uint64_t multiresolution_identity_domain = 0x6d756c7469726573ull;

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

bool validate_affinity(const support_atlas_view_v1 &atlas,
                       const support_atlas_view_v1 *identity_source,
                       std::string *error) {
    if (atlas.schema_version != support_atlas_schema_version_v1
        || (atlas.flags & support_atlas_flag_top_l) == 0u) {
        set_error(error, "multiresolution support requires a top-L atlas");
        return false;
    }
    if (atlas.affinity_count != 0u && atlas.affinity == nullptr) {
        set_error(error, "multiresolution affinity records are absent");
        return false;
    }
    if (identity_source != nullptr
        && (atlas.relation_identity != identity_source->relation_identity
            || atlas.structure_identity != identity_source->structure_identity
            || atlas.structure_epoch != identity_source->structure_epoch
            || atlas.source_axis_identity != identity_source->source_axis_identity
            || atlas.destination_axis_identity != identity_source->destination_axis_identity
            || atlas.source_count != identity_source->source_count
            || atlas.destination_count != identity_source->destination_count)) {
        set_error(error, "resampled support identity does not match the base atlas");
        return false;
    }
    std::uint32_t previous_source = 0u;
    std::uint32_t expected_rank = 0u;
    bool have_previous = false;
    for (std::uint64_t i = 0u; i < atlas.affinity_count; ++i) {
        const source_affinity_record_v1 &record = atlas.affinity[i];
        if (record.source_id >= atlas.source_count
            || record.neighbor_source_id >= atlas.source_count
            || record.source_id == record.neighbor_source_id
            || record.score_denominator == 0u) {
            set_error(error, "multiresolution affinity record is invalid");
            return false;
        }
        if (!have_previous || record.source_id != previous_source) {
            if (have_previous && record.source_id <= previous_source) {
                set_error(error, "multiresolution affinity sources are not canonical");
                return false;
            }
            previous_source = record.source_id;
            expected_rank = 0u;
            have_previous = true;
        }
        if (record.rank != expected_rank++) {
            set_error(error, "multiresolution affinity ranks are not contiguous");
            return false;
        }
    }
    return true;
}

bool validate_strata(const support_atlas_view_v1 &base,
                     const biological_stratum_v1 *strata,
                     std::uint64_t stratum_count,
                     std::string *error) {
    if (stratum_count != 0u && strata == nullptr) {
        set_error(error, "biological strata records are absent");
        return false;
    }
    for (std::uint64_t i = 0u; i < stratum_count; ++i) {
        if (strata[i].stratum_axis_identity == 0u
            || strata[i].stratum_identity == 0u
            || strata[i].destination_id >= base.destination_count) {
            set_error(error, "biological stratum record is invalid");
            return false;
        }
    }
    return true;
}

std::uint32_t maximum_neighbor_count(const support_atlas_view_v1 &atlas) noexcept {
    std::uint32_t maximum = 0u;
    for (std::uint64_t i = 0u; i < atlas.affinity_count; ++i) {
        maximum = std::max(maximum, atlas.affinity[i].rank + 1u);
    }
    return maximum;
}

std::uint32_t rank_limit(std::uint32_t maximum,
                         std::uint32_t resolution) noexcept {
    if (maximum == 0u) return 0u;
    if (resolution >= 31u) return 1u;
    const std::uint32_t divisor = std::uint32_t{1} << resolution;
    return std::max<std::uint32_t>(1u, (maximum + divisor - 1u) / divisor);
}

std::uint32_t find_community(community_assignment_v1 *records,
                             std::uint32_t source) noexcept {
    std::uint32_t root = source;
    while (records[root].community_id != root) root = records[root].community_id;
    while (records[source].community_id != source) {
        const std::uint32_t parent = records[source].community_id;
        records[source].community_id = root;
        source = parent;
    }
    return root;
}

void union_community(community_assignment_v1 *records,
                     std::uint32_t left,
                     std::uint32_t right) noexcept {
    left = find_community(records, left);
    right = find_community(records, right);
    if (left == right) return;
    if (left < right) records[right].community_id = left;
    else records[left].community_id = right;
}

std::uint32_t find_stability(resampling_stability_v1 *records,
                             std::uint32_t source) noexcept {
    std::uint32_t root = source;
    while (records[root].source_id != root) root = records[root].source_id;
    while (records[source].source_id != source) {
        const std::uint32_t parent = records[source].source_id;
        records[source].source_id = root;
        source = parent;
    }
    return root;
}

void union_stability(resampling_stability_v1 *records,
                     std::uint32_t left,
                     std::uint32_t right) noexcept {
    left = find_stability(records, left);
    right = find_stability(records, right);
    if (left == right) return;
    if (left < right) records[right].source_id = left;
    else records[left].source_id = right;
}

bool contributes_at_resolution(const source_affinity_record_v1 &record,
                               std::uint32_t limit) noexcept {
    return record.rank < limit && record.score_numerator > 0;
}

void build_base_communities(const support_atlas_view_v1 &atlas,
                            std::uint32_t resolution_count,
                            community_assignment_v1 *communities) {
    const std::uint32_t maximum = maximum_neighbor_count(atlas);
    for (std::uint32_t resolution = 0u;
         resolution < resolution_count; ++resolution) {
        community_assignment_v1 *const records = communities
            + static_cast<std::uint64_t>(resolution) * atlas.source_count;
        for (std::uint32_t source = 0u; source < atlas.source_count; ++source) {
            records[source] = community_assignment_v1{};
            records[source].resolution = resolution;
            records[source].source_id = source;
            records[source].community_id = source;
        }
        const std::uint32_t limit = rank_limit(maximum, resolution);
        for (std::uint64_t edge = 0u; edge < atlas.affinity_count; ++edge) {
            const source_affinity_record_v1 &record = atlas.affinity[edge];
            if (contributes_at_resolution(record, limit)) {
                union_community(records, record.source_id, record.neighbor_source_id);
            }
        }
        for (std::uint32_t source = 0u; source < atlas.source_count; ++source) {
            records[source].community_id = find_community(records, source);
        }
    }
}

void accumulate_stability(const support_atlas_view_v1 &base,
                          const support_atlas_view_v1 *resamples,
                          std::uint32_t resample_count,
                          std::uint32_t resolution_count,
                          const community_assignment_v1 *communities,
                          resampling_stability_v1 *stability) {
    const std::uint32_t maximum = maximum_neighbor_count(base);
    for (std::uint32_t resolution = 0u;
         resolution < resolution_count; ++resolution) {
        resampling_stability_v1 *const records = stability
            + static_cast<std::uint64_t>(resolution) * base.source_count;
        const community_assignment_v1 *const baseline = communities
            + static_cast<std::uint64_t>(resolution) * base.source_count;
        for (std::uint32_t source = 0u; source < base.source_count; ++source) {
            records[source] = resampling_stability_v1{};
            records[source].resolution = resolution;
            records[source].source_id = source;
            records[source].resample_count = resample_count;
        }
        const std::uint32_t limit = rank_limit(maximum, resolution);
        for (std::uint32_t resample = 0u; resample < resample_count; ++resample) {
            for (std::uint32_t source = 0u; source < base.source_count; ++source) {
                records[source].source_id = source;
            }
            for (std::uint64_t edge = 0u;
                 edge < resamples[resample].affinity_count; ++edge) {
                const source_affinity_record_v1 &record = resamples[resample].affinity[edge];
                if (contributes_at_resolution(record, limit)) {
                    union_stability(records, record.source_id, record.neighbor_source_id);
                }
            }
            for (std::uint32_t source = 0u; source < base.source_count; ++source) {
                if (find_stability(records, source) == baseline[source].community_id) {
                    ++records[source].stable_assignment_count;
                }
            }
        }
        for (std::uint32_t source = 0u; source < base.source_count; ++source) {
            records[source].source_id = source;
        }
    }
}

bool stratum_less(const biological_stratum_v1 &left,
                  const biological_stratum_v1 &right) noexcept {
    if (left.stratum_axis_identity != right.stratum_axis_identity) {
        return left.stratum_axis_identity < right.stratum_axis_identity;
    }
    if (left.stratum_identity != right.stratum_identity) {
        return left.stratum_identity < right.stratum_identity;
    }
    if (left.destination_id != right.destination_id) {
        return left.destination_id < right.destination_id;
    }
    return left.stratum_id < right.stratum_id;
}

} // namespace

bool query_support_multiresolution_requirements_v1(
    const support_atlas_view_v1 &base,
    const support_atlas_view_v1 *resamples,
    std::uint32_t resample_count,
    const biological_stratum_v1 *strata,
    std::uint64_t stratum_count,
    std::uint32_t resolution_count,
    std::uint64_t work_identity,
    support_atlas_requirements_v1 *out,
    std::string *error) {
    if (out == nullptr) {
        set_error(error, "multiresolution requirements output is null");
        return false;
    }
    *out = support_atlas_requirements_v1{};
    if (resolution_count == 0u) {
        set_error(error, "multiresolution support requires at least one resolution");
        return false;
    }
    if (!validate_affinity(base, nullptr, error)
        || !validate_strata(base, strata, stratum_count, error)) {
        return false;
    }
    if (resample_count != 0u && resamples == nullptr) {
        set_error(error, "resampled support atlas views are absent");
        return false;
    }
    for (std::uint32_t i = 0u; i < resample_count; ++i) {
        if (!validate_affinity(resamples[i], &base, error)) return false;
    }
    if (!checked_multiply(base.source_count, resolution_count,
                          &out->community_capacity)) {
        set_error(error, "multiresolution community capacity overflows");
        return false;
    }
    out->stratum_capacity = stratum_count;
    out->work_signature_capacity = work_identity == 0u ? 0u : 1u;
    if (resample_count != 0u
        && !checked_multiply(base.source_count, resolution_count,
                             &out->stability_capacity)) {
        set_error(error, "multiresolution stability capacity overflows");
        *out = support_atlas_requirements_v1{};
        return false;
    }
    if (work_identity != 0u
        && (base.destination_degree_count != base.destination_count
            || (base.destination_count != 0u && base.destination_degrees == nullptr))) {
        set_error(error, "work signatures require complete destination degrees");
        *out = support_atlas_requirements_v1{};
        return false;
    }
    return true;
}

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
    std::string *error) {
    support_atlas_requirements_v1 requirements;
    if (out == nullptr) {
        set_error(error, "multiresolution output view is null");
        return false;
    }
    *out = support_atlas_view_v1{};
    if (!query_support_multiresolution_requirements_v1(
            base, resamples, resample_count, strata, stratum_count,
            resolution_count, work_identity, &requirements, error)) {
        return false;
    }
    if ((requirements.community_capacity != 0u && buffers.communities == nullptr)
        || buffers.community_capacity < requirements.community_capacity
        || (requirements.stratum_capacity != 0u && buffers.strata == nullptr)
        || buffers.stratum_capacity < requirements.stratum_capacity
        || (requirements.stability_capacity != 0u && buffers.stability == nullptr)
        || buffers.stability_capacity < requirements.stability_capacity
        || (requirements.work_signature_capacity != 0u
            && buffers.work_signatures == nullptr)
        || buffers.work_signature_capacity < requirements.work_signature_capacity) {
        set_error(error, "multiresolution output capacity is insufficient");
        return false;
    }

    build_base_communities(base, resolution_count, buffers.communities);
    if (resample_count != 0u) {
        accumulate_stability(base, resamples, resample_count,
                             resolution_count, buffers.communities,
                             buffers.stability);
    }
    if (stratum_count != 0u) {
        std::copy(strata, strata + stratum_count, buffers.strata);
        std::sort(buffers.strata, buffers.strata + stratum_count, stratum_less);
    }

    std::uint64_t identity = mix_identity(
        multiresolution_identity_domain, base.evidence_identity);
    for (std::uint64_t i = 0u; i < requirements.community_capacity; ++i) {
        identity = mix_identity(identity,
            (static_cast<std::uint64_t>(buffers.communities[i].resolution) << 32u)
                | buffers.communities[i].source_id);
        identity = mix_identity(identity, buffers.communities[i].community_id);
    }
    for (std::uint64_t i = 0u; i < stratum_count; ++i) {
        identity = mix_identity(identity, buffers.strata[i].stratum_axis_identity);
        identity = mix_identity(identity, buffers.strata[i].stratum_identity);
        identity = mix_identity(identity,
            (static_cast<std::uint64_t>(buffers.strata[i].destination_id) << 32u)
                | buffers.strata[i].stratum_id);
    }
    for (std::uint32_t resample = 0u; resample < resample_count; ++resample) {
        identity = mix_identity(identity, resamples[resample].evidence_identity);
    }
    for (std::uint64_t i = 0u; i < requirements.stability_capacity; ++i) {
        identity = mix_identity(identity,
            (static_cast<std::uint64_t>(buffers.stability[i].resolution) << 32u)
                | buffers.stability[i].source_id);
        identity = mix_identity(identity,
            (static_cast<std::uint64_t>(buffers.stability[i].stable_assignment_count) << 32u)
                | buffers.stability[i].resample_count);
    }

    if (work_identity != 0u) {
        work_signature_v1 signature{};
        signature.work_identity = work_identity;
        signature.support_hash = identity;
        signature.destination_count = base.destination_count;
        for (std::uint64_t destination = 0u;
             destination < base.destination_degree_count; ++destination) {
            signature.edge_count += base.destination_degrees[destination].degree;
        }
        buffers.work_signatures[0] = signature;
        identity = mix_identity(identity, work_identity);
        identity = mix_identity(identity, signature.edge_count);
    }

    *out = base;
    out->flags |= support_atlas_flag_multiresolution;
    if (stratum_count != 0u) out->flags |= support_atlas_flag_stratified;
    if (resample_count != 0u) out->flags |= support_atlas_flag_resampled;
    out->evidence_identity = identity;
    out->communities = requirements.community_capacity == 0u
        ? nullptr : buffers.communities;
    out->community_count = requirements.community_capacity;
    out->work_signatures = work_identity == 0u ? nullptr : buffers.work_signatures;
    out->work_signature_count = work_identity == 0u ? 0u : 1u;
    out->strata = stratum_count == 0u ? nullptr : buffers.strata;
    out->stratum_count = stratum_count;
    out->stability = requirements.stability_capacity == 0u
        ? nullptr : buffers.stability;
    out->stability_count = requirements.stability_capacity;
    return true;
}

} // namespace cellerator::geometry
