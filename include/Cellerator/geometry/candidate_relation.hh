#pragma once

#include "Cellerator/geometry/validate.hh"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cellpack {

enum class candidate_score_kind : u32 {
    none = 0u,
    exact_merge_gain = 1u,
    support_intersection = 2u,
    jaccard = 3u,
    minhash_similarity = 4u,
    structural_rank = 5u
};

enum candidate_evidence_flags : u32 {
    candidate_evidence_none = 0u,
    candidate_evidence_exact = 1u << 0u,
    candidate_evidence_approximate = 1u << 1u,
    candidate_evidence_support_counts = 1u << 2u,
    candidate_evidence_intersection = 1u << 3u
};

// Candidate scores are rational so normalization and tie-breaking are exact.
// Scores from different kinds are never compared numerically.
struct candidate_relation {
    u32 feature_a = invalid_id;
    u32 feature_b = invalid_id;
    std::int64_t score_numerator = 0;
    u64 score_denominator = 1u;
    candidate_score_kind score_kind = candidate_score_kind::none;
    u32 evidence_flags = candidate_evidence_none;
    u64 support_a = 0u;
    u64 support_b = 0u;
    u64 support_intersection = 0u;
};

struct candidate_relation_view {
    const candidate_relation *relations = nullptr;
    u64 relation_count = 0u;
};

struct candidate_normalization_statistics {
    u64 input_relations = 0u;
    u64 output_relations = 0u;
    u64 self_edges_discarded = 0u;
    u64 duplicates_collapsed = 0u;
};

class normalized_candidate_relations {
public:
    normalized_candidate_relations() = default;
    normalized_candidate_relations(const normalized_candidate_relations &) = delete;
    normalized_candidate_relations &operator=(const normalized_candidate_relations &) = delete;
    normalized_candidate_relations(normalized_candidate_relations &&) noexcept = default;
    normalized_candidate_relations &operator=(normalized_candidate_relations &&) noexcept = default;

    candidate_relation_view view() const noexcept {
        return {relations_.get(), count_};
    }
    const candidate_normalization_statistics &statistics() const noexcept { return statistics_; }

private:
    std::unique_ptr<candidate_relation[]> relations_;
    u64 count_ = 0u;
    candidate_normalization_statistics statistics_{};

    friend validation_result normalize_candidate_relations(
        candidate_relation_view, u32, normalized_candidate_relations *);
};

validation_result validate_candidate_relation_view(candidate_relation_view candidates, u32 feature_count);

validation_result normalize_candidate_relations(
    candidate_relation_view candidates,
    u32 feature_count,
    normalized_candidate_relations *out);

} // namespace cellpack
