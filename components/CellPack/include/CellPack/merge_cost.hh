#pragma once

#include "CellPack/candidate_relation.hh"

#include <Cellerator/compute/gene_candidate_discovery.hh>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cellpack {

inline constexpr u32 exact_merge_cost_policy_version = 1u;

enum class merge_value_storage : u32 {
    compact_nonzeros = 0u,
    dense_active_rows = 1u
};

// Provisional, replaceable physical-cost policy. A zero byte term explicitly
// defers that component. This is not the CP-BP-06 record ABI.
struct exact_merge_cost_policy {
    u32 version = exact_merge_cost_policy_version;
    u32 maximum_block_width = 32u;
    u32 block_metadata_bytes = 16u;
    u32 feature_identifier_bytes = 4u;
    u32 block_offset_bytes = 8u;
    u32 active_row_offset_bytes = 4u;
    u32 mask_word_bits = 32u;
    u32 mask_word_bytes = 4u;
    u32 value_bytes = 2u;
    u32 metadata_alignment = 8u;
    u32 payload_alignment = 8u;
    u32 record_alignment = 16u;
    merge_value_storage value_storage = merge_value_storage::compact_nonzeros;
};

struct exact_block_cost {
    u64 block_width = 0u;
    u64 active_rows = 0u;
    u64 nnz = 0u;
    u64 value_slots = 0u;
    u64 metadata_bytes = 0u;
    u64 identifier_bytes = 0u;
    u64 block_offset_bytes = 0u;
    u64 active_row_offset_bytes = 0u;
    u64 mask_bytes = 0u;
    u64 value_bytes = 0u;
    u64 alignment_padding_bytes = 0u;
    u64 total_bytes = 0u;
};

struct exact_gene_merge_cost {
    u64 support_a = 0u;
    u64 support_b = 0u;
    u64 support_intersection = 0u;
    u64 support_union = 0u;
    exact_block_cost cost_a{};
    exact_block_cost cost_b{};
    exact_block_cost merged_cost{};
    u64 separated_cost_bytes = 0u;
    std::int64_t merge_gain_bytes = 0;
};

struct exact_merge_scoring_provenance {
    u32 algorithm_version = exact_merge_cost_policy_version;
    exact_merge_cost_policy policy{};
    ::cellerator::compute::gene_candidates::candidate_discovery_provenance candidates{};
};

struct exact_gene_merge_score_view {
    const candidate_relation *relations = nullptr;
    const exact_gene_merge_cost *costs = nullptr;
    u64 count = 0u;
    const exact_merge_scoring_provenance *provenance = nullptr;

    candidate_relation_view relation_view() const noexcept {
        return {relations, count};
    }
};

class owned_exact_gene_merge_scores {
public:
    owned_exact_gene_merge_scores() = default;
    owned_exact_gene_merge_scores(
        std::unique_ptr<candidate_relation[]> relations,
        std::unique_ptr<exact_gene_merge_cost[]> costs,
        u64 count,
        exact_merge_scoring_provenance provenance) noexcept;
    owned_exact_gene_merge_scores(const owned_exact_gene_merge_scores &) = delete;
    owned_exact_gene_merge_scores &operator=(const owned_exact_gene_merge_scores &) = delete;
    owned_exact_gene_merge_scores(owned_exact_gene_merge_scores &&) noexcept = default;
    owned_exact_gene_merge_scores &operator=(owned_exact_gene_merge_scores &&) noexcept = default;

    exact_gene_merge_score_view view() const noexcept;
    const exact_merge_scoring_provenance &scoring_provenance() const noexcept;

private:
    std::unique_ptr<candidate_relation[]> relations_;
    std::unique_ptr<exact_gene_merge_cost[]> costs_;
    u64 count_ = 0u;
    exact_merge_scoring_provenance provenance_{};
};

validation_result validate_exact_merge_cost_policy(const exact_merge_cost_policy &policy);

validation_result estimate_exact_block_cost(
    u32 block_width,
    u64 active_rows,
    u64 nnz,
    const exact_merge_cost_policy &policy,
    exact_block_cost *out);

validation_result estimate_merge_gain(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    u32 gene_a,
    u32 gene_b,
    const exact_merge_cost_policy &policy,
    exact_gene_merge_cost *out);

validation_result score_gene_merges_cpu(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    const ::cellerator::compute::gene_candidates::gene_candidate_pair_view &candidates,
    const exact_merge_cost_policy &policy,
    owned_exact_gene_merge_scores *out);

// Correctness-first V100 path. It transiently stages the immutable host support
// and candidate pairs, performs exact wordwise scoring, and returns host-owned
// relations/costs. Persistent device ownership is deliberately deferred.
validation_result score_gene_merges_cuda(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    const ::cellerator::compute::gene_candidates::gene_candidate_pair_view &candidates,
    const exact_merge_cost_policy &policy,
    int device,
    owned_exact_gene_merge_scores *out);

} // namespace cellpack
