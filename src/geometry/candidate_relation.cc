#include "Cellerator/geometry/candidate_relation.hh"

#include <algorithm>
#include <limits>
#include <vector>

namespace cellpack {
namespace {

bool valid_score_kind(candidate_score_kind kind) {
    return kind == candidate_score_kind::none
        || kind == candidate_score_kind::exact_merge_gain
        || kind == candidate_score_kind::support_intersection
        || kind == candidate_score_kind::jaccard
        || kind == candidate_score_kind::minhash_similarity
        || kind == candidate_score_kind::structural_rank;
}

int evidence_rank(const candidate_relation &relation) {
    if ((relation.evidence_flags & candidate_evidence_exact) != 0u) return 2;
    if ((relation.evidence_flags & candidate_evidence_approximate) != 0u) return 1;
    return 0;
}

int compare_rational(const candidate_relation &lhs, const candidate_relation &rhs) {
    const __int128 left = static_cast<__int128>(lhs.score_numerator)
        * static_cast<__int128>(rhs.score_denominator);
    const __int128 right = static_cast<__int128>(rhs.score_numerator)
        * static_cast<__int128>(lhs.score_denominator);
    return left < right ? -1 : (left > right ? 1 : 0);
}

bool relation_less(const candidate_relation &lhs, const candidate_relation &rhs) {
    if (lhs.feature_a != rhs.feature_a) return lhs.feature_a < rhs.feature_a;
    if (lhs.feature_b != rhs.feature_b) return lhs.feature_b < rhs.feature_b;
    const int lhs_rank = evidence_rank(lhs), rhs_rank = evidence_rank(rhs);
    if (lhs_rank != rhs_rank) return lhs_rank > rhs_rank;
    if (lhs.score_kind != rhs.score_kind) {
        return static_cast<u32>(lhs.score_kind) < static_cast<u32>(rhs.score_kind);
    }
    const int score_compare = compare_rational(lhs, rhs);
    if (score_compare != 0) return score_compare > 0;
    if (lhs.evidence_flags != rhs.evidence_flags) return lhs.evidence_flags > rhs.evidence_flags;
    if (lhs.support_intersection != rhs.support_intersection) {
        return lhs.support_intersection > rhs.support_intersection;
    }
    if (lhs.support_a != rhs.support_a) return lhs.support_a > rhs.support_a;
    return lhs.support_b > rhs.support_b;
}

bool exact_duplicate_conflicts(const candidate_relation &lhs, const candidate_relation &rhs) {
    if ((lhs.evidence_flags & candidate_evidence_exact) == 0u
        || (rhs.evidence_flags & candidate_evidence_exact) == 0u) return false;
    if (compare_rational(lhs, rhs) != 0) return true;
    const bool lhs_counts = (lhs.evidence_flags & candidate_evidence_support_counts) != 0u;
    const bool rhs_counts = (rhs.evidence_flags & candidate_evidence_support_counts) != 0u;
    if (!lhs_counts || !rhs_counts) return false;
    if (lhs.support_a != rhs.support_a || lhs.support_b != rhs.support_b) return true;
    const bool lhs_intersection = (lhs.evidence_flags & candidate_evidence_intersection) != 0u;
    const bool rhs_intersection = (rhs.evidence_flags & candidate_evidence_intersection) != 0u;
    return lhs_intersection && rhs_intersection
        && lhs.support_intersection != rhs.support_intersection;
}

validation_result validate_one(const candidate_relation &relation, u32 feature_count, u32 index) {
    if (relation.feature_a >= feature_count || relation.feature_b >= feature_count) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate feature endpoint is out of range");
    }
    if (relation.score_denominator == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate score denominator must be positive");
    }
    if (!valid_score_kind(relation.score_kind)) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate score kind is unsupported");
    }
    const u32 precision = relation.evidence_flags
        & (candidate_evidence_exact | candidate_evidence_approximate);
    constexpr u32 allowed_flags = candidate_evidence_exact | candidate_evidence_approximate
        | candidate_evidence_support_counts | candidate_evidence_intersection;
    if ((relation.evidence_flags & ~allowed_flags) != 0u) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate evidence flags are unsupported");
    }
    if (precision == (candidate_evidence_exact | candidate_evidence_approximate)) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate evidence cannot be exact and approximate");
    }
    if ((relation.evidence_flags & candidate_evidence_intersection) != 0u
        && (relation.evidence_flags & candidate_evidence_support_counts) == 0u) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate intersection requires support counts");
    }
    if ((relation.evidence_flags & candidate_evidence_support_counts) != 0u
        && ((relation.evidence_flags & candidate_evidence_intersection) != 0u)
        && relation.support_intersection > std::min(relation.support_a, relation.support_b)) {
        return validation_error(validation_code::invalid_plan_geometry, index, "candidate support intersection exceeds endpoint support");
    }
    return validation_ok();
}

} // namespace

validation_result validate_candidate_relation_view(candidate_relation_view candidates, u32 feature_count) {
    if (candidates.relation_count != 0u && candidates.relations == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "candidate relation array is null");
    }
    if (candidates.relation_count > static_cast<u64>(std::numeric_limits<std::size_t>::max())) {
        return validation_error(validation_code::integer_overflow, invalid_id, "candidate relation count exceeds host address space");
    }
    for (u64 i = 0u; i < candidates.relation_count; ++i) {
        const validation_result status = validate_one(
            candidates.relations[i], feature_count,
            i > invalid_id ? invalid_id : static_cast<u32>(i));
        if (!status) return status;
    }
    return validation_ok();
}

validation_result normalize_candidate_relations(
    candidate_relation_view candidates,
    u32 feature_count,
    normalized_candidate_relations *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id, "normalized candidate output is null");
    }
    const validation_result input_status = validate_candidate_relation_view(candidates, feature_count);
    if (!input_status) return input_status;

    candidate_normalization_statistics statistics;
    statistics.input_relations = candidates.relation_count;
    std::vector<candidate_relation> normalized;
    normalized.reserve(static_cast<std::size_t>(candidates.relation_count));
    for (u64 i = 0u; i < candidates.relation_count; ++i) {
        candidate_relation relation = candidates.relations[i];
        if (relation.feature_a == relation.feature_b) {
            ++statistics.self_edges_discarded;
            continue;
        }
        if (relation.feature_b < relation.feature_a) {
            std::swap(relation.feature_a, relation.feature_b);
            std::swap(relation.support_a, relation.support_b);
        }
        normalized.push_back(relation);
    }
    std::sort(normalized.begin(), normalized.end(), relation_less);

    std::vector<candidate_relation> unique;
    unique.reserve(normalized.size());
    std::size_t begin = 0u;
    while (begin < normalized.size()) {
        std::size_t end = begin + 1u;
        while (end < normalized.size()
            && normalized[end].feature_a == normalized[begin].feature_a
            && normalized[end].feature_b == normalized[begin].feature_b
            && normalized[end].score_kind == normalized[begin].score_kind) {
            ++end;
        }
        for (std::size_t i = begin; i < end; ++i) {
            if ((normalized[i].evidence_flags & candidate_evidence_exact) == 0u) continue;
            for (std::size_t j = i + 1u; j < end; ++j) {
                if (exact_duplicate_conflicts(normalized[i], normalized[j])) {
                    return validation_error(validation_code::invalid_plan_geometry,
                        normalized[i].feature_a,
                        "conflicting exact duplicate candidate scores");
                }
            }
        }
        unique.push_back(normalized[begin]);
        statistics.duplicates_collapsed += static_cast<u64>(end - begin - 1u);
        begin = end;
    }

    normalized_candidate_relations result;
    result.count_ = static_cast<u64>(unique.size());
    result.statistics_ = statistics;
    result.statistics_.output_relations = result.count_;
    if (!unique.empty()) {
        result.relations_.reset(new candidate_relation[unique.size()]);
        std::copy(unique.begin(), unique.end(), result.relations_.get());
    }
    *out = std::move(result);
    return validation_ok();
}

} // namespace cellpack
