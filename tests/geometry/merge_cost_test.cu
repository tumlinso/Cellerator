/*
 * CP-BP-03 exact merge-cost contract tests. CPU is the reference; CUDA must
 * match every integer support, byte component, gain, and candidate relation.
 * Native target: Tesla V100 sm_70.
 */

#include "Cellerator/geometry/merge_cost.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cp = ::cellpack;
namespace gc = ::cellerator::compute::gene_candidates;
namespace gs = ::cellerator::compute::gene_support;
namespace sampling = ::cellerator::compute::sampling;

int require(bool condition, const char *message) {
    if (!condition) std::fprintf(stderr, "%s\n", message);
    return condition ? 1 : 0;
}

struct scoring_fixture {
    gs::gene_support_layout layout{};
    std::vector<gs::support_word_t> words;
    std::vector<::cellerator::types::count_value_t> counts;
    std::vector<std::uint64_t> global_rows;
    sampling::sample_provenance sampling_provenance;
    std::vector<gc::gene_candidate_pair> pairs;
    gc::candidate_discovery_provenance candidate_provenance;

    gs::gene_support_bitset_view support_view() const {
        return {layout, words.data(), counts.data(), global_rows.data(), &sampling_provenance};
    }

    gc::gene_candidate_pair_view candidate_view() const {
        return {pairs.data(), pairs.size(), &candidate_provenance};
    }
};

bool build_fixture(
    std::uint64_t cells,
    const std::vector<std::vector<std::uint64_t>> &gene_cells,
    std::vector<gc::gene_candidate_pair> pairs,
    scoring_fixture *out) {
    scoring_fixture fixture;
    fixture.layout.sampled_cell_count = cells;
    fixture.layout.gene_count = gene_cells.size();
    fixture.layout.words_per_gene = static_cast<std::size_t>((cells + 31u) / 32u);
    fixture.layout.support_word_count = fixture.layout.words_per_gene * gene_cells.size();
    fixture.layout.support_bytes = fixture.layout.support_word_count * sizeof(gs::support_word_t);
    fixture.layout.detection_count_bytes = gene_cells.size()
        * sizeof(::cellerator::types::count_value_t);
    fixture.words.assign(fixture.layout.support_word_count, 0u);
    fixture.counts.assign(gene_cells.size(), 0u);
    fixture.global_rows.resize(static_cast<std::size_t>(cells));
    for (std::uint64_t row = 0u; row < cells; ++row) fixture.global_rows[row] = row;
    for (std::size_t gene = 0u; gene < gene_cells.size(); ++gene) {
        std::vector<std::uint64_t> unique = gene_cells[gene];
        std::sort(unique.begin(), unique.end());
        unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
        for (std::uint64_t row : unique) {
            if (row >= cells) return false;
            fixture.words[gene * fixture.layout.words_per_gene + row / 32u]
                |= static_cast<std::uint32_t>(1u << (row % 32u));
        }
        fixture.counts[gene] = static_cast<::cellerator::types::count_value_t>(unique.size());
    }
    fixture.sampling_provenance.seed = 0x1234abcdu;
    fixture.sampling_provenance.total_rows = cells;
    fixture.sampling_provenance.selected_rows = cells;
    fixture.sampling_provenance.mode = sampling::selection_mode::exact_lowest_hash;
    fixture.sampling_provenance.split_name = "cp-bp-03-fixture";
    fixture.sampling_provenance.requested_row_count = cells;
    fixture.pairs = std::move(pairs);
    fixture.candidate_provenance.sampling = fixture.sampling_provenance;
    fixture.candidate_provenance.sampled_cell_count = cells;
    fixture.candidate_provenance.gene_count = gene_cells.size();
    fixture.candidate_provenance.unique_candidate_count = fixture.pairs.size();
    *out = std::move(fixture);
    return true;
}

bool equal_block_cost(const cp::exact_block_cost &lhs, const cp::exact_block_cost &rhs) {
    return lhs.block_width == rhs.block_width
        && lhs.active_rows == rhs.active_rows && lhs.nnz == rhs.nnz
        && lhs.value_slots == rhs.value_slots && lhs.metadata_bytes == rhs.metadata_bytes
        && lhs.identifier_bytes == rhs.identifier_bytes
        && lhs.block_offset_bytes == rhs.block_offset_bytes
        && lhs.active_row_offset_bytes == rhs.active_row_offset_bytes
        && lhs.mask_bytes == rhs.mask_bytes && lhs.value_bytes == rhs.value_bytes
        && lhs.alignment_padding_bytes == rhs.alignment_padding_bytes
        && lhs.total_bytes == rhs.total_bytes;
}

bool equal_cost(const cp::exact_gene_merge_cost &lhs, const cp::exact_gene_merge_cost &rhs) {
    return lhs.support_a == rhs.support_a && lhs.support_b == rhs.support_b
        && lhs.support_intersection == rhs.support_intersection
        && lhs.support_union == rhs.support_union
        && equal_block_cost(lhs.cost_a, rhs.cost_a)
        && equal_block_cost(lhs.cost_b, rhs.cost_b)
        && equal_block_cost(lhs.merged_cost, rhs.merged_cost)
        && lhs.separated_cost_bytes == rhs.separated_cost_bytes
        && lhs.merge_gain_bytes == rhs.merge_gain_bytes;
}

bool equal_relation(const cp::candidate_relation &lhs, const cp::candidate_relation &rhs) {
    return lhs.feature_a == rhs.feature_a && lhs.feature_b == rhs.feature_b
        && lhs.score_numerator == rhs.score_numerator
        && lhs.score_denominator == rhs.score_denominator
        && lhs.score_kind == rhs.score_kind && lhs.evidence_flags == rhs.evidence_flags
        && lhs.support_a == rhs.support_a && lhs.support_b == rhs.support_b
        && lhs.support_intersection == rhs.support_intersection;
}

int test_policy_and_block_cost() {
    cp::exact_merge_cost_policy policy;
    cp::exact_block_cost cost;
    cp::validation_result status = cp::validate_exact_merge_cost_policy(policy);
    if (!require(static_cast<bool>(status), "default exact merge-cost policy rejected")) return 1;
    status = cp::estimate_exact_block_cost(32u, 4u, 64u, policy, &cost);
    if (!require(static_cast<bool>(status), "maximum-width block cost failed")) return 2;
    const std::uint64_t raw = cost.metadata_bytes + cost.identifier_bytes
        + cost.block_offset_bytes + cost.active_row_offset_bytes + cost.mask_bytes
        + cost.value_bytes;
    if (!require(cost.block_width == 32u && cost.active_rows == 4u && cost.nnz == 64u
                 && cost.value_slots == 64u && cost.total_bytes == raw + cost.alignment_padding_bytes,
                 "maximum-width block accounting mismatch")) return 3;
    status = cp::estimate_exact_block_cost(33u, 4u, 64u, policy, &cost);
    if (!require(status.code == cp::validation_code::invalid_plan_geometry,
                 "over-width block was accepted")) return 4;

    cp::exact_merge_cost_policy deferred = policy;
    deferred.block_metadata_bytes = 0u;
    deferred.feature_identifier_bytes = 0u;
    deferred.block_offset_bytes = 0u;
    deferred.active_row_offset_bytes = 0u;
    deferred.mask_word_bytes = 0u;
    deferred.value_bytes = 0u;
    status = cp::estimate_exact_block_cost(2u, 9u, 10u, deferred, &cost);
    if (!require(static_cast<bool>(status) && cost.total_bytes == 0u,
                 "explicitly deferred byte terms are not zero cost")) return 5;

    cp::exact_merge_cost_policy overflow = policy;
    overflow.value_storage = cp::merge_value_storage::dense_active_rows;
    status = cp::estimate_exact_block_cost(
        32u, std::numeric_limits<std::uint64_t>::max(), 0u, overflow, &cost);
    if (!require(status.code == cp::validation_code::integer_overflow,
                 "overflowing block cost was accepted")) return 6;
    return 0;
}

int test_reference_cases(bool gpu_available) {
    scoring_fixture fixture;
    const std::vector<std::vector<std::uint64_t>> genes = {
        {}, {}, {0u, 1u, 32u}, {0u, 1u, 32u},
        {1u, 2u, 32u}, {3u, 4u, 5u}, {32u}, {0u, 6u}
    };
    const std::vector<gc::gene_candidate_pair> pairs = {
        {0u, 1u}, {2u, 3u}, {2u, 4u}, {4u, 5u}, {6u, 7u}
    };
    if (!require(build_fixture(33u, genes, pairs, &fixture), "failed to build merge fixture")) return 10;
    cp::exact_merge_cost_policy policy;
    cp::owned_exact_gene_merge_scores cpu, repeated, gpu;
    cp::validation_result status = cp::score_gene_merges_cpu(
        fixture.support_view(), fixture.candidate_view(), policy, &cpu);
    if (!require(static_cast<bool>(status), status.message)) return 11;
    status = cp::score_gene_merges_cpu(
        fixture.support_view(), fixture.candidate_view(), policy, &repeated);
    if (!require(static_cast<bool>(status), status.message)) return 12;
    const cp::exact_gene_merge_score_view cpu_view = cpu.view(), repeated_view = repeated.view();
    if (!require(cpu_view.count == pairs.size() && cpu_view.provenance != nullptr,
                 "CPU exact merge-score view mismatch")) return 13;
    if (!require(cpu_view.costs[0].support_union == 0u
                 && cpu_view.costs[0].support_intersection == 0u,
                 "empty support merge mismatch")) return 14;
    if (!require(cpu_view.costs[1].support_a == 3u && cpu_view.costs[1].support_b == 3u
                 && cpu_view.costs[1].support_intersection == 3u
                 && cpu_view.costs[1].support_union == 3u,
                 "identical support merge mismatch")) return 15;
    if (!require(cpu_view.costs[2].support_intersection == 2u
                 && cpu_view.costs[2].support_union == 4u,
                 "overlapping support merge mismatch")) return 16;
    if (!require(cpu_view.costs[3].support_intersection == 0u
                 && cpu_view.costs[3].support_union == 6u,
                 "disjoint support merge mismatch")) return 17;
    if (!require(cpu_view.costs[4].support_a == 1u
                 && cpu_view.costs[4].support_intersection == 0u,
                 "tail-word support merge mismatch")) return 18;
    for (std::uint64_t index = 0u; index < cpu_view.count; ++index) {
        const cp::exact_gene_merge_cost &cost = cpu_view.costs[index];
        const std::int64_t expected = static_cast<std::int64_t>(cost.separated_cost_bytes)
            - static_cast<std::int64_t>(cost.merged_cost.total_bytes);
        if (!require(cost.merge_gain_bytes == expected,
                     "merge_gain != cost(A)+cost(B)-cost(A union B)")) return 19;
        if (!require(equal_cost(cost, repeated_view.costs[index])
                     && equal_relation(cpu_view.relations[index], repeated_view.relations[index]),
                     "CPU exact merge scoring is not deterministic")) return 20;
    }
    status = cp::validate_candidate_relation_view(
        {cpu_view.relations, cpu_view.count}, static_cast<cp::u32>(genes.size()));
    if (!require(static_cast<bool>(status), "exact scored candidate relations are invalid")) return 21;
    cp::normalized_candidate_relations normalized;
    status = cp::normalize_candidate_relations(
        {cpu_view.relations, cpu_view.count}, static_cast<cp::u32>(genes.size()), &normalized);
    if (!require(static_cast<bool>(status) && normalized.view().relation_count == cpu_view.count
                 && normalized.view().relations[0].score_kind
                     == cp::candidate_score_kind::exact_merge_gain,
                 "optimizer candidate normalization rejected exact merge evidence")) return 22;

    cp::exact_gene_merge_cost one;
    status = cp::estimate_merge_gain(fixture.support_view(), 2u, 4u, policy, &one);
    if (!require(static_cast<bool>(status) && equal_cost(one, cpu_view.costs[2]),
                 "single-pair merge estimate differs from batch reference")) return 23;

    if (gpu_available) {
        status = cp::score_gene_merges_cuda(
            fixture.support_view(), fixture.candidate_view(), policy, 0, &gpu);
        if (!require(static_cast<bool>(status), status.message)) return 24;
        const cp::exact_gene_merge_score_view gpu_view = gpu.view();
        if (!require(gpu_view.count == cpu_view.count, "CPU/CUDA merge-score count mismatch")) return 25;
        for (std::uint64_t index = 0u; index < cpu_view.count; ++index) {
            if (!require(equal_cost(cpu_view.costs[index], gpu_view.costs[index])
                         && equal_relation(cpu_view.relations[index], gpu_view.relations[index]),
                         "CPU/CUDA exact merge-score mismatch")) return 26;
        }
    }

    scoring_fixture zero_rows;
    if (!require(build_fixture(0u, {{}, {}}, {{0u, 1u}}, &zero_rows),
                 "failed to build zero-row merge fixture")) return 27;
    cp::owned_exact_gene_merge_scores zero_cpu, zero_gpu;
    status = cp::score_gene_merges_cpu(
        zero_rows.support_view(), zero_rows.candidate_view(), policy, &zero_cpu);
    if (!require(static_cast<bool>(status) && zero_cpu.view().costs[0].support_union == 0u,
                 "zero-row CPU merge scoring failed")) return 28;
    if (gpu_available) {
        status = cp::score_gene_merges_cuda(
            zero_rows.support_view(), zero_rows.candidate_view(), policy, 0, &zero_gpu);
        if (!require(static_cast<bool>(status)
                     && equal_cost(zero_cpu.view().costs[0], zero_gpu.view().costs[0]),
                     "zero-row CUDA merge scoring failed")) return 29;
    }
    return 0;
}

int test_unprofitable_and_invalid_inputs() {
    std::vector<std::uint64_t> left(32u), right(32u);
    for (std::uint64_t row = 0u; row < 32u; ++row) {
        left[row] = row;
        right[row] = row + 32u;
    }
    scoring_fixture fixture;
    if (!require(build_fixture(64u, {left, right}, {{0u, 1u}}, &fixture),
                 "failed to build unprofitable merge fixture")) return 30;
    cp::exact_merge_cost_policy dense;
    dense.value_storage = cp::merge_value_storage::dense_active_rows;
    cp::owned_exact_gene_merge_scores scores;
    cp::validation_result status = cp::score_gene_merges_cpu(
        fixture.support_view(), fixture.candidate_view(), dense, &scores);
    if (!require(static_cast<bool>(status) && scores.view().costs[0].merge_gain_bytes < 0,
                 "dense disjoint merge was not unprofitable")) return 31;

    scoring_fixture bad_count = fixture;
    ++bad_count.counts[0];
    status = cp::score_gene_merges_cpu(
        bad_count.support_view(), bad_count.candidate_view(), dense, &scores);
    if (!require(status.code == cp::validation_code::invalid_plan_geometry,
                 "bitset/count mismatch was accepted")) return 32;

    scoring_fixture bad_provenance = fixture;
    bad_provenance.candidate_provenance.sampling.split_name = "wrong-split";
    status = cp::score_gene_merges_cpu(
        bad_provenance.support_view(), bad_provenance.candidate_view(), dense, &scores);
    if (!require(status.code == cp::validation_code::invalid_plan_geometry,
                 "candidate/support provenance mismatch was accepted")) return 33;

    scoring_fixture bad_pair = fixture;
    bad_pair.pairs[0] = {1u, 0u};
    status = cp::score_gene_merges_cpu(
        bad_pair.support_view(), bad_pair.candidate_view(), dense, &scores);
    if (!require(status.code == cp::validation_code::invalid_plan_geometry,
                 "noncanonical candidate pair was accepted")) return 34;
    return 0;
}

int main() {
    int device_count = 0;
    const bool gpu_available = cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
    int status = test_policy_and_block_cost();
    if (status != 0) return status;
    status = test_reference_cases(gpu_available);
    if (status != 0) return status;
    status = test_unprofitable_and_invalid_inputs();
    if (status != 0) return status;
    std::fprintf(stdout, "MERGE_COST_TEST cpu=pass cuda=%s policy_version=%u\n",
                 gpu_available ? "pass" : "skipped", cp::exact_merge_cost_policy_version);
    return 0;
}
