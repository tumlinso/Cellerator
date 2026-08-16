#pragma once

/*
 * Shared CPU/device integer byte-accounting helpers for CP-BP-03. Prefer these
 * helpers over copying the formula into either path. On 2026-08-16,
 * ./build-cp-bp03/cellPackMergeCostBench compared the CPU reference with the
 * sm_70 CUDA scorer at 65,536 cells, 30,000 genes, and 105,000 candidates:
 * 308.250 ms CPU versus 77.924/78.895 ms GPU minimum/median, including staging.
 * Every support, cost, and gain integer matched exactly (zero tolerance).
 */

#include "CellPack/merge_cost.hh"

#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define CELLPACK_MERGE_HD __host__ __device__
#else
#define CELLPACK_MERGE_HD
#endif

namespace cellpack::detail {

CELLPACK_MERGE_HD inline bool checked_add_u64(u64 lhs, u64 rhs, u64 *out) {
    if (lhs > UINT64_MAX - rhs) return false;
    *out = lhs + rhs;
    return true;
}

CELLPACK_MERGE_HD inline bool checked_mul_u64(u64 lhs, u64 rhs, u64 *out) {
    if (lhs != 0u && rhs > UINT64_MAX / lhs) return false;
    *out = lhs * rhs;
    return true;
}

CELLPACK_MERGE_HD inline bool align_up_u64(u64 value, u32 alignment, u64 *out) {
    const u64 mask = static_cast<u64>(alignment) - 1u;
    if (value > UINT64_MAX - mask) return false;
    *out = (value + mask) & ~mask;
    return true;
}

CELLPACK_MERGE_HD inline bool calculate_block_cost(
    u32 block_width,
    u64 active_rows,
    u64 nnz,
    const exact_merge_cost_policy &policy,
    exact_block_cost *out) {
    exact_block_cost estimate;
    estimate.block_width = block_width;
    estimate.active_rows = active_rows;
    estimate.nnz = nnz;
    estimate.metadata_bytes = policy.block_metadata_bytes;
    estimate.block_offset_bytes = policy.block_offset_bytes;
    if (!checked_mul_u64(block_width, policy.feature_identifier_bytes,
                         &estimate.identifier_bytes)) return false;

    u64 row_offset_count = 0u;
    if (!checked_add_u64(active_rows, 1u, &row_offset_count)
        || !checked_mul_u64(row_offset_count, policy.active_row_offset_bytes,
                            &estimate.active_row_offset_bytes)) return false;
    const u64 mask_words = (static_cast<u64>(block_width) + policy.mask_word_bits - 1u)
        / policy.mask_word_bits;
    u64 mask_bytes_per_row = 0u;
    if (!checked_mul_u64(mask_words, policy.mask_word_bytes, &mask_bytes_per_row)
        || !checked_mul_u64(active_rows, mask_bytes_per_row, &estimate.mask_bytes)) return false;

    if (policy.value_storage == merge_value_storage::compact_nonzeros) {
        estimate.value_slots = nnz;
    } else {
        if (!checked_mul_u64(active_rows, block_width, &estimate.value_slots)) return false;
    }
    if (!checked_mul_u64(estimate.value_slots, policy.value_bytes,
                         &estimate.value_bytes)) return false;

    u64 metadata_raw = 0u, metadata_aligned = 0u;
    if (!checked_add_u64(estimate.metadata_bytes, estimate.identifier_bytes, &metadata_raw)
        || !checked_add_u64(metadata_raw, estimate.block_offset_bytes, &metadata_raw)
        || !align_up_u64(metadata_raw, policy.metadata_alignment, &metadata_aligned)) return false;

    u64 payload_raw = 0u, payload_aligned = 0u;
    if (!checked_add_u64(estimate.active_row_offset_bytes, estimate.mask_bytes, &payload_raw)
        || !align_up_u64(payload_raw, policy.payload_alignment, &payload_aligned)) return false;

    u64 unaligned_total = 0u, total = 0u;
    if (!checked_add_u64(metadata_aligned, payload_aligned, &unaligned_total)
        || !checked_add_u64(unaligned_total, estimate.value_bytes, &unaligned_total)
        || !align_up_u64(unaligned_total, policy.record_alignment, &total)) return false;

    const u64 raw_components = estimate.metadata_bytes + estimate.identifier_bytes
        + estimate.block_offset_bytes + estimate.active_row_offset_bytes
        + estimate.mask_bytes + estimate.value_bytes;
    estimate.alignment_padding_bytes = total - raw_components;
    estimate.total_bytes = total;
    *out = estimate;
    return true;
}

CELLPACK_MERGE_HD inline bool signed_gain(u64 separated, u64 merged, std::int64_t *out) {
    if (separated >= merged) {
        const u64 magnitude = separated - merged;
        if (magnitude > static_cast<u64>(INT64_MAX)) return false;
        *out = static_cast<std::int64_t>(magnitude);
        return true;
    }
    const u64 magnitude = merged - separated;
    const u64 minimum_magnitude = static_cast<u64>(INT64_MAX) + 1u;
    if (magnitude > minimum_magnitude) return false;
    *out = magnitude == minimum_magnitude
        ? INT64_MIN
        : -static_cast<std::int64_t>(magnitude);
    return true;
}

CELLPACK_MERGE_HD inline bool calculate_merge_cost(
    u64 support_a,
    u64 support_b,
    u64 support_intersection,
    const exact_merge_cost_policy &policy,
    exact_gene_merge_cost *out) {
    if (support_intersection > support_a || support_intersection > support_b) return false;
    exact_gene_merge_cost estimate;
    estimate.support_a = support_a;
    estimate.support_b = support_b;
    estimate.support_intersection = support_intersection;
    if (!checked_add_u64(support_a, support_b, &estimate.support_union)
        || estimate.support_union < support_intersection) return false;
    estimate.support_union -= support_intersection;
    u64 merged_nnz = 0u;
    if (!checked_add_u64(support_a, support_b, &merged_nnz)
        || !calculate_block_cost(1u, support_a, support_a, policy, &estimate.cost_a)
        || !calculate_block_cost(1u, support_b, support_b, policy, &estimate.cost_b)
        || !calculate_block_cost(2u, estimate.support_union, merged_nnz,
                                 policy, &estimate.merged_cost)
        || !checked_add_u64(estimate.cost_a.total_bytes, estimate.cost_b.total_bytes,
                            &estimate.separated_cost_bytes)
        || !signed_gain(estimate.separated_cost_bytes, estimate.merged_cost.total_bytes,
                        &estimate.merge_gain_bytes)) return false;
    *out = estimate;
    return true;
}

validation_result validate_scoring_inputs(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    const ::cellerator::compute::gene_candidates::gene_candidate_pair_view &candidates,
    const exact_merge_cost_policy &policy);

candidate_relation make_exact_relation(
    u32 gene_a,
    u32 gene_b,
    const exact_gene_merge_cost &cost) noexcept;

} // namespace cellpack::detail

#undef CELLPACK_MERGE_HD
