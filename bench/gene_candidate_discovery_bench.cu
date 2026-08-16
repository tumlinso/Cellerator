/*
 * Representative CP-BP-02 end-to-end V100 smoke. The benchmark includes
 * transient H2D staging, MinHash, CUB LSH grouping/deduplication, and final D2H
 * pair transfer. It makes no production-quality recall or threshold claim.
 * Command: ./build-sampling/geneCandidateDiscoveryBench.
 */

#include <Cellerator/compute/gene_candidate_discovery.hh>

#include "benchmark_mutex.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace cg = ::cellerator::compute::gene_support;
namespace cc = ::cellerator::compute::gene_candidates;
namespace cs = ::cellerator::compute::sampling;
namespace ct = ::cellerator::types;

bool build_clustered_support(std::uint64_t cells,
                             std::uint64_t genes,
                             std::uint32_t cluster_size,
                             cg::owned_gene_support_bitsets *out,
                             std::string *error) {
    cg::gene_support_layout layout;
    if (!cg::calculate_gene_support_layout(cells, genes, &layout, error)) return false;
    std::unique_ptr<cg::support_word_t[]> words(
        new (std::nothrow) cg::support_word_t[layout.support_word_count]());
    std::unique_ptr<ct::count_value_t[]> counts(
        new (std::nothrow) ct::count_value_t[(std::size_t) genes]());
    std::unique_ptr<std::uint64_t[]> mapping(
        new (std::nothrow) std::uint64_t[(std::size_t) cells]);
    if (words == nullptr || counts == nullptr || mapping == nullptr) {
        if (error != nullptr) *error = "failed to allocate benchmark support";
        return false;
    }
    for (std::uint64_t cell = 0u; cell < cells; ++cell) mapping[cell] = cell;
    for (std::uint64_t gene = 0u; gene < genes; ++gene) {
        const std::uint64_t cluster = gene / cluster_size;
        for (std::uint64_t entry = 0u; entry < 32u; ++entry) {
            const std::uint64_t cell = cc::candidate_splitmix64_v1(
                cluster * 0xd2b74407b1ce6e93ull + entry) % cells;
            words[(std::size_t) gene * layout.words_per_gene + (std::size_t) cell / 32u]
                |= (ct::u32) 1u << (cell % 32u);
        }
        ct::count_value_t count = 0u;
        for (std::size_t word = 0u; word < layout.words_per_gene; ++word) {
            count += (ct::count_value_t) __builtin_popcount(
                words[(std::size_t) gene * layout.words_per_gene + word]);
        }
        counts[gene] = count;
    }
    cs::sample_provenance provenance;
    provenance.seed = 20260814u;
    provenance.total_rows = cells;
    provenance.selected_rows = cells;
    provenance.mode = cs::selection_mode::exact_lowest_hash;
    provenance.split_name = "candidate-v100-smoke";
    provenance.requested_row_count = cells;
    *out = cg::owned_gene_support_bitsets(
        layout, std::move(words), std::move(counts), std::move(mapping), std::move(provenance));
    return true;
}

int main() {
    constexpr std::uint64_t cells = 65536u, genes = 30000u;
    constexpr std::uint32_t cluster_size = 8u;
    constexpr int warmups = 1, repeats = 3;
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("geneCandidateDiscoveryBench", 0);
    cg::owned_gene_support_bitsets support;
    cc::candidate_discovery_config config;
    cc::candidate_discovery_bounds bounds;
    std::string error;
    config.seed = 0xdecafbad12345678ull;
    if (!build_clustered_support(cells, genes, cluster_size, &support, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    if (!cc::calculate_candidate_discovery_bounds(support.view().layout, config, &bounds, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    cc::owned_gene_candidates candidates;
    for (int warmup = 0; warmup < warmups; ++warmup) {
        if (!cc::discover_gene_candidates_cuda(support.view(), config, 0, &candidates, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 3;
        }
    }
    std::vector<double> milliseconds;
    milliseconds.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        if (!cc::discover_gene_candidates_cuda(support.view(), config, 0, &candidates, &error)) {
            std::fprintf(stderr, "%s\n", error.c_str());
            return 4;
        }
        const auto end = std::chrono::steady_clock::now();
        milliseconds.push_back(std::chrono::duration<double, std::milli>(end - begin).count());
    }
    std::sort(milliseconds.begin(), milliseconds.end());
    const auto view = candidates.view();
    std::uint64_t retained_cluster_pairs = 0u;
    for (std::uint64_t i = 0u; i < view.count; ++i) {
        if (view.pairs[i].gene_a / cluster_size == view.pairs[i].gene_b / cluster_size) {
            ++retained_cluster_pairs;
        }
    }
    const std::uint64_t full_clusters = genes / cluster_size, remainder = genes % cluster_size;
    const std::uint64_t expected_cluster_pairs = full_clusters * cluster_size * (cluster_size - 1u) / 2u
        + remainder * (remainder - 1u) / 2u;
    const std::uint64_t exhaustive_pairs = genes * (genes - 1u) / 2u;
    const double recall = (double) retained_cluster_pairs / expected_cluster_pairs;
    const double reduction = 1.0 - (double) view.count / exhaustive_pairs;
    std::fprintf(stdout,
                 "CANDIDATE_BENCH device=0 gpu=Tesla_V100 sm=70 cells=%llu genes=%llu "
                 "sketches=%u bands=%u rows_per_band=%u bucket_cap=%u "
                 "candidates=%llu raw_pairs=%llu exhaustive_pairs=%llu "
                 "cluster_pairs=%llu retained_cluster_pairs=%llu recall=%.6f reduction=%.6f "
                 "support_bytes=%zu fixed_device_bytes_excluding_cub=%zu "
                 "cub_temporary_bytes=%zu accounted_peak_device_bytes=%zu "
                 "warmups=%d repeats=%d min_ms=%.3f median_ms=%.3f\n",
                 (unsigned long long) cells, (unsigned long long) genes,
                 config.sketch_count, config.lsh_bands, config.rows_per_band,
                 config.maximum_bucket_size, (unsigned long long) view.count,
                 (unsigned long long) view.provenance->raw_pair_occurrences,
                 (unsigned long long) exhaustive_pairs,
                 (unsigned long long) expected_cluster_pairs,
                 (unsigned long long) retained_cluster_pairs, recall, reduction,
                 support.view().layout.support_bytes, bounds.fixed_device_bytes_excluding_cub,
                 view.provenance->device_cub_temporary_bytes,
                 view.provenance->device_peak_bytes,
                 warmups, repeats, milliseconds.front(), milliseconds[milliseconds.size() / 2u]);
    return recall == 1.0 ? 0 : 5;
}
