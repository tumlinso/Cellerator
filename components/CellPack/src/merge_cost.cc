#include "CellPack/merge_cost.hh"

#include "merge_cost_internal.cuh"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>

namespace cellpack {
namespace {

namespace gc = ::cellerator::compute::gene_candidates;
namespace gs = ::cellerator::compute::gene_support;
namespace sampling = ::cellerator::compute::sampling;

bool power_of_two(u32 value) {
    return value != 0u && (value & (value - 1u)) == 0u;
}

bool same_sampling_provenance(
    const sampling::sample_provenance &lhs,
    const sampling::sample_provenance &rhs) {
    return lhs.seed == rhs.seed
        && lhs.hash_algorithm == rhs.hash_algorithm
        && lhs.hash_version == rhs.hash_version
        && lhs.total_rows == rhs.total_rows
        && lhs.selected_rows == rhs.selected_rows
        && lhs.mode == rhs.mode
        && lhs.split_name == rhs.split_name
        && lhs.cell_identity == rhs.cell_identity
        && lhs.quantile.begin.numerator == rhs.quantile.begin.numerator
        && lhs.quantile.begin.denominator == rhs.quantile.begin.denominator
        && lhs.quantile.end.numerator == rhs.quantile.end.numerator
        && lhs.quantile.end.denominator == rhs.quantile.end.denominator
        && lhs.requested_row_count == rhs.requested_row_count
        && lhs.requested_density_strata == rhs.requested_density_strata
        && lhs.density_strata == rhs.density_strata
        && lhs.density_bin_upper_bounds_inclusive == rhs.density_bin_upper_bounds_inclusive
        && lhs.stratum_total_rows == rhs.stratum_total_rows
        && lhs.stratum_sampled_rows == rhs.stratum_sampled_rows
        && lhs.weighting_rule == rhs.weighting_rule;
}

validation_result validate_support(const gs::gene_support_bitset_view &support) {
    if (support.provenance == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "merge-cost sampling provenance is null");
    }
    if (support.provenance->hash_algorithm != sampling::splitmix64_algorithm_name
        || support.provenance->hash_version != sampling::splitmix64_algorithm_version
        || support.provenance->selected_rows != support.layout.sampled_cell_count) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                "merge-cost sampling provenance is inconsistent");
    }
    if (support.layout.gene_count > static_cast<u64>(UINT32_MAX)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "merge-cost gene count exceeds canonical id width");
    }
    if (support.layout.sampled_cell_count > UINT64_MAX - 31u) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "merge-cost sampled row count overflows support layout");
    }
    const u64 expected_words_per_gene = (support.layout.sampled_cell_count + 31u) / 32u;
    u64 expected_word_count = 0u, expected_bytes = 0u, expected_count_bytes = 0u;
    if (!detail::checked_mul_u64(expected_words_per_gene, support.layout.gene_count,
                                &expected_word_count)
        || !detail::checked_mul_u64(expected_word_count, sizeof(gs::support_word_t),
                                    &expected_bytes)
        || !detail::checked_mul_u64(support.layout.gene_count,
                                    sizeof(::cellerator::types::count_value_t),
                                    &expected_count_bytes)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "merge-cost support layout overflows addressable bytes");
    }
    if (expected_words_per_gene != support.layout.words_per_gene
        || expected_word_count != support.layout.support_word_count
        || expected_bytes != support.layout.support_bytes
        || expected_count_bytes != support.layout.detection_count_bytes) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                "merge-cost support layout is inconsistent");
    }
    if (expected_word_count != 0u && support.gene_support == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "merge-cost support words are null");
    }
    if (support.layout.gene_count != 0u && support.detected_cell_counts == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "merge-cost detection counts are null");
    }
    if (support.layout.sampled_cell_count != 0u
        && support.sampled_position_to_global_row == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "merge-cost sampled-row mapping is null");
    }
    for (u64 row = 0u; row < support.layout.sampled_cell_count; ++row) {
        const u64 global_row = support.sampled_position_to_global_row[row];
        if (global_row >= support.provenance->total_rows
            || (row != 0u && global_row <= support.sampled_position_to_global_row[row - 1u])) {
            return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                    "merge-cost sampled-row mapping is not canonical");
        }
    }
    for (u64 gene = 0u; gene < support.layout.gene_count; ++gene) {
        if (support.detected_cell_counts[gene] > support.layout.sampled_cell_count) {
            return validation_error(validation_code::invalid_plan_geometry,
                                    static_cast<u32>(gene),
                                    "merge-cost detection count exceeds sampled rows");
        }
    }
    const u32 tail_bits = static_cast<u32>(support.layout.sampled_cell_count % 32u);
    if (tail_bits != 0u && support.layout.words_per_gene != 0u) {
        const u32 valid_mask = static_cast<u32>((1ull << tail_bits) - 1ull);
        const std::size_t last = support.layout.words_per_gene - 1u;
        for (u64 gene = 0u; gene < support.layout.gene_count; ++gene) {
            const u32 word = support.gene_support[
                static_cast<std::size_t>(gene) * support.layout.words_per_gene + last];
            if ((word & ~valid_mask) != 0u) {
                return validation_error(validation_code::invalid_plan_geometry,
                                        static_cast<u32>(gene),
                                        "merge-cost support has nonzero tail padding");
            }
        }
    }
    return validation_ok();
}

validation_result validate_candidates(
    const gs::gene_support_bitset_view &support,
    const gc::gene_candidate_pair_view &candidates) {
    if (candidates.provenance == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "merge-cost candidate provenance is null");
    }
    if (candidates.count != 0u && candidates.pairs == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "merge-cost candidate pairs are null");
    }
    if (candidates.count > static_cast<u64>(std::numeric_limits<std::size_t>::max())) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "merge-cost candidate count exceeds host address space");
    }
    const gc::candidate_discovery_provenance &provenance = *candidates.provenance;
    if (provenance.algorithm != gc::candidate_algorithm_name
        || provenance.hash_version != gc::candidate_hash_version
        || provenance.sampled_cell_count != support.layout.sampled_cell_count
        || provenance.gene_count != support.layout.gene_count
        || provenance.unique_candidate_count != candidates.count
        || !same_sampling_provenance(provenance.sampling, *support.provenance)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                "merge-cost candidate/support provenance mismatch");
    }
    for (u64 index = 0u; index < candidates.count; ++index) {
        const gc::gene_candidate_pair &pair = candidates.pairs[index];
        const u32 error_index = index > invalid_id ? invalid_id : static_cast<u32>(index);
        if (pair.gene_a >= pair.gene_b || pair.gene_b >= support.layout.gene_count) {
            return validation_error(validation_code::invalid_plan_geometry, error_index,
                                    "merge-cost candidate pair is not canonical and in range");
        }
        if (index != 0u) {
            const gc::gene_candidate_pair &previous = candidates.pairs[index - 1u];
            if (previous.gene_a > pair.gene_a
                || (previous.gene_a == pair.gene_a && previous.gene_b >= pair.gene_b)) {
                return validation_error(validation_code::invalid_plan_geometry, error_index,
                                        "merge-cost candidate pairs are not sorted and unique");
            }
        }
    }
    return validation_ok();
}

validation_result score_one(
    const gs::gene_support_bitset_view &support,
    u32 gene_a,
    u32 gene_b,
    const exact_merge_cost_policy &policy,
    exact_gene_merge_cost *out) {
    u64 count_a = 0u, count_b = 0u, intersection = 0u, support_union = 0u;
    const std::size_t words = support.layout.words_per_gene;
    const gs::support_word_t *a = words == 0u ? nullptr
        : support.gene_support + static_cast<std::size_t>(gene_a) * words;
    const gs::support_word_t *b = words == 0u ? nullptr
        : support.gene_support + static_cast<std::size_t>(gene_b) * words;
    for (std::size_t word = 0u; word < words; ++word) {
        const u32 lhs = a[word], rhs = b[word];
        count_a += static_cast<u64>(__builtin_popcount(lhs));
        count_b += static_cast<u64>(__builtin_popcount(rhs));
        intersection += static_cast<u64>(__builtin_popcount(lhs & rhs));
        support_union += static_cast<u64>(__builtin_popcount(lhs | rhs));
    }
    if (count_a != support.detected_cell_counts[gene_a]
        || count_b != support.detected_cell_counts[gene_b]) {
        return validation_error(validation_code::invalid_plan_geometry, gene_a,
                                "merge-cost bitset/count evidence disagrees");
    }
    if (!detail::calculate_merge_cost(count_a, count_b, intersection, policy, out)) {
        return validation_error(validation_code::integer_overflow, gene_a,
                                "merge-cost byte accounting overflows");
    }
    if (out->support_union != support_union) {
        return validation_error(validation_code::invalid_plan_geometry, gene_a,
                                "merge-cost support union is inconsistent");
    }
    return validation_ok();
}

} // namespace

owned_exact_gene_merge_scores::owned_exact_gene_merge_scores(
    std::unique_ptr<candidate_relation[]> relations,
    std::unique_ptr<exact_gene_merge_cost[]> costs,
    u64 count,
    exact_merge_scoring_provenance provenance) noexcept
    : relations_(std::move(relations)), costs_(std::move(costs)), count_(count),
      provenance_(std::move(provenance)) {}

exact_gene_merge_score_view owned_exact_gene_merge_scores::view() const noexcept {
    return {relations_.get(), costs_.get(), count_, &provenance_};
}

const exact_merge_scoring_provenance &
owned_exact_gene_merge_scores::scoring_provenance() const noexcept {
    return provenance_;
}

validation_result validate_exact_merge_cost_policy(const exact_merge_cost_policy &policy) {
    if (policy.version != exact_merge_cost_policy_version) {
        return validation_error(validation_code::unsupported_version, policy.version,
                                "unsupported exact merge-cost policy version");
    }
    if (policy.maximum_block_width < 2u || policy.mask_word_bits == 0u
        || !power_of_two(policy.metadata_alignment)
        || !power_of_two(policy.payload_alignment) || !power_of_two(policy.record_alignment)) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                "invalid exact merge-cost width, mask, or alignment policy");
    }
    if (policy.value_storage != merge_value_storage::compact_nonzeros
        && policy.value_storage != merge_value_storage::dense_active_rows) {
        return validation_error(validation_code::invalid_plan_geometry, invalid_id,
                                "unsupported exact merge-cost value storage policy");
    }
    return validation_ok();
}

validation_result estimate_exact_block_cost(
    u32 block_width,
    u64 active_rows,
    u64 nnz,
    const exact_merge_cost_policy &policy,
    exact_block_cost *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "exact block-cost output is null");
    }
    const validation_result policy_status = validate_exact_merge_cost_policy(policy);
    if (!policy_status) return policy_status;
    if (block_width == 0u || block_width > policy.maximum_block_width) {
        return validation_error(validation_code::invalid_plan_geometry, block_width,
                                "exact block width is outside policy bounds");
    }
    u64 maximum_nnz = 0u;
    if (!detail::checked_mul_u64(active_rows, block_width, &maximum_nnz)) {
        return validation_error(validation_code::integer_overflow, block_width,
                                "exact block logical slots overflow");
    }
    if (nnz > maximum_nnz) {
        return validation_error(validation_code::invalid_plan_geometry, block_width,
                                "exact block nnz exceeds logical slots");
    }
    if (!detail::calculate_block_cost(block_width, active_rows, nnz, policy, out)) {
        return validation_error(validation_code::integer_overflow, block_width,
                                "exact block byte accounting overflows");
    }
    return validation_ok();
}

namespace detail {

validation_result validate_scoring_inputs(
    const gs::gene_support_bitset_view &support,
    const gc::gene_candidate_pair_view &candidates,
    const exact_merge_cost_policy &policy) {
    const validation_result policy_status = validate_exact_merge_cost_policy(policy);
    if (!policy_status) return policy_status;
    const validation_result support_status = validate_support(support);
    if (!support_status) return support_status;
    const validation_result candidate_status = validate_candidates(support, candidates);
    if (!candidate_status) return candidate_status;
    u64 maximum_nnz = 0u;
    exact_gene_merge_cost maximum_cost;
    if (!checked_mul_u64(support.layout.sampled_cell_count, 2u, &maximum_nnz)
        || !calculate_merge_cost(support.layout.sampled_cell_count,
                                support.layout.sampled_cell_count,
                                support.layout.sampled_cell_count,
                                policy, &maximum_cost)) {
        return validation_error(validation_code::integer_overflow, invalid_id,
                                "merge-cost policy overflows supported sample bounds");
    }
    (void) maximum_nnz;
    return validation_ok();
}

candidate_relation make_exact_relation(
    u32 gene_a,
    u32 gene_b,
    const exact_gene_merge_cost &cost) noexcept {
    candidate_relation relation;
    relation.feature_a = gene_a;
    relation.feature_b = gene_b;
    relation.score_numerator = cost.merge_gain_bytes;
    relation.score_denominator = 1u;
    relation.score_kind = candidate_score_kind::exact_merge_gain;
    relation.evidence_flags = candidate_evidence_exact
        | candidate_evidence_support_counts | candidate_evidence_intersection;
    relation.support_a = cost.support_a;
    relation.support_b = cost.support_b;
    relation.support_intersection = cost.support_intersection;
    return relation;
}

} // namespace detail

validation_result estimate_merge_gain(
    const gs::gene_support_bitset_view &support,
    u32 gene_a,
    u32 gene_b,
    const exact_merge_cost_policy &policy,
    exact_gene_merge_cost *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "exact merge-gain output is null");
    }
    const validation_result policy_status = validate_exact_merge_cost_policy(policy);
    if (!policy_status) return policy_status;
    const validation_result support_status = validate_support(support);
    if (!support_status) return support_status;
    if (gene_a >= gene_b || gene_b >= support.layout.gene_count) {
        return validation_error(validation_code::invalid_plan_geometry, gene_a,
                                "exact merge endpoints are not canonical and in range");
    }
    return score_one(support, gene_a, gene_b, policy, out);
}

validation_result score_gene_merges_cpu(
    const gs::gene_support_bitset_view &support,
    const gc::gene_candidate_pair_view &candidates,
    const exact_merge_cost_policy &policy,
    owned_exact_gene_merge_scores *out) {
    if (out == nullptr) {
        return validation_error(validation_code::null_pointer, invalid_id,
                                "CPU exact merge-score output is null");
    }
    const validation_result input_status = detail::validate_scoring_inputs(
        support, candidates, policy);
    if (!input_status) return input_status;

    std::unique_ptr<candidate_relation[]> relations;
    std::unique_ptr<exact_gene_merge_cost[]> costs;
    if (candidates.count != 0u) {
        relations.reset(new (std::nothrow) candidate_relation[static_cast<std::size_t>(candidates.count)]);
        costs.reset(new (std::nothrow) exact_gene_merge_cost[static_cast<std::size_t>(candidates.count)]);
        if (relations == nullptr || costs == nullptr) {
            return validation_error(validation_code::insufficient_capacity, invalid_id,
                                    "failed to allocate CPU exact merge-score output");
        }
    }
    for (u64 index = 0u; index < candidates.count; ++index) {
        const gc::gene_candidate_pair &pair = candidates.pairs[index];
        const validation_result score_status = score_one(
            support, pair.gene_a, pair.gene_b, policy, &costs[index]);
        if (!score_status) return score_status;
        relations[index] = detail::make_exact_relation(pair.gene_a, pair.gene_b, costs[index]);
    }
    try {
        exact_merge_scoring_provenance provenance;
        provenance.algorithm_version = policy.version;
        provenance.policy = policy;
        provenance.candidates = *candidates.provenance;
        *out = owned_exact_gene_merge_scores(
            std::move(relations), std::move(costs), candidates.count, std::move(provenance));
    } catch (const std::bad_alloc &) {
        return validation_error(validation_code::insufficient_capacity, invalid_id,
                                "failed to copy exact merge-score provenance");
    }
    return validation_ok();
}

} // namespace cellpack
