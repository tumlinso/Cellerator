/*
 * CP-BP-03 end-to-end host-staged exact scorer benchmark. It compares the CPU
 * oracle with the regular-CUDA V100 path and holds the repository GPU mutex.
 * Command: ./build-cp-bp03/cellPackMergeCostBench.
 */

#include "CellPack/merge_cost.hh"

#include "benchmark_mutex.hh"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <new>
#include <utility>
#include <vector>

namespace cp = ::cellpack;
namespace gc = ::cellerator::compute::gene_candidates;
namespace gs = ::cellerator::compute::gene_support;
namespace sampling = ::cellerator::compute::sampling;

struct benchmark_fixture {
    gs::gene_support_layout layout{};
    std::unique_ptr<gs::support_word_t[]> words;
    std::unique_ptr<::cellerator::types::count_value_t[]> counts;
    std::unique_ptr<std::uint64_t[]> global_rows;
    sampling::sample_provenance sampling_provenance;
    std::unique_ptr<gc::gene_candidate_pair[]> pairs;
    std::uint64_t pair_count = 0u;
    gc::candidate_discovery_provenance candidate_provenance;

    gs::gene_support_bitset_view support_view() const {
        return {layout, words.get(), counts.get(), global_rows.get(), &sampling_provenance};
    }

    gc::gene_candidate_pair_view candidate_view() const {
        return {pairs.get(), pair_count, &candidate_provenance};
    }
};

bool build_fixture(std::uint64_t cells,
                   std::uint64_t genes,
                   std::uint32_t cluster_width,
                   benchmark_fixture *out) {
    if (genes == 0u || genes % cluster_width != 0u) return false;
    benchmark_fixture fixture;
    fixture.layout.sampled_cell_count = cells;
    fixture.layout.gene_count = genes;
    fixture.layout.words_per_gene = static_cast<std::size_t>((cells + 31u) / 32u);
    fixture.layout.support_word_count = fixture.layout.words_per_gene * static_cast<std::size_t>(genes);
    fixture.layout.support_bytes = fixture.layout.support_word_count * sizeof(gs::support_word_t);
    fixture.layout.detection_count_bytes = static_cast<std::size_t>(genes)
        * sizeof(::cellerator::types::count_value_t);
    fixture.words.reset(new (std::nothrow) gs::support_word_t[fixture.layout.support_word_count]());
    fixture.counts.reset(new (std::nothrow) ::cellerator::types::count_value_t[genes]());
    fixture.global_rows.reset(new (std::nothrow) std::uint64_t[cells]);
    const std::uint64_t clusters = genes / cluster_width;
    fixture.pair_count = clusters * cluster_width * (cluster_width - 1u) / 2u;
    fixture.pairs.reset(new (std::nothrow) gc::gene_candidate_pair[fixture.pair_count]);
    if (fixture.words == nullptr || fixture.counts == nullptr
        || fixture.global_rows == nullptr || fixture.pairs == nullptr) return false;
    for (std::uint64_t row = 0u; row < cells; ++row) fixture.global_rows[row] = row;
    for (std::uint64_t gene = 0u; gene < genes; ++gene) {
        const std::uint64_t cluster = gene / cluster_width;
        for (std::uint64_t entry = 0u; entry < 32u; ++entry) {
            const std::uint64_t row = (cluster * 17u + entry * 7919u) % cells;
            fixture.words[static_cast<std::size_t>(gene) * fixture.layout.words_per_gene + row / 32u]
                |= static_cast<std::uint32_t>(1u << (row % 32u));
        }
        fixture.counts[gene] = 32u;
    }
    std::uint64_t cursor = 0u;
    for (std::uint64_t cluster = 0u; cluster < clusters; ++cluster) {
        const std::uint32_t begin = static_cast<std::uint32_t>(cluster * cluster_width);
        for (std::uint32_t a = 0u; a < cluster_width; ++a) {
            for (std::uint32_t b = a + 1u; b < cluster_width; ++b) {
                fixture.pairs[cursor++] = {begin + a, begin + b};
            }
        }
    }
    fixture.sampling_provenance.seed = 20260816u;
    fixture.sampling_provenance.total_rows = cells;
    fixture.sampling_provenance.selected_rows = cells;
    fixture.sampling_provenance.mode = sampling::selection_mode::exact_lowest_hash;
    fixture.sampling_provenance.split_name = "cp-bp-03-v100-smoke";
    fixture.sampling_provenance.requested_row_count = cells;
    fixture.candidate_provenance.sampling = fixture.sampling_provenance;
    fixture.candidate_provenance.sampled_cell_count = cells;
    fixture.candidate_provenance.gene_count = genes;
    fixture.candidate_provenance.nonempty_gene_count = genes;
    fixture.candidate_provenance.unique_candidate_count = fixture.pair_count;
    *out = std::move(fixture);
    return true;
}

bool equal_result(const cp::exact_gene_merge_score_view &cpu,
                  const cp::exact_gene_merge_score_view &gpu) {
    if (cpu.count != gpu.count) return false;
    for (std::uint64_t index = 0u; index < cpu.count; ++index) {
        if (cpu.relations[index].feature_a != gpu.relations[index].feature_a
            || cpu.relations[index].feature_b != gpu.relations[index].feature_b
            || cpu.relations[index].score_numerator != gpu.relations[index].score_numerator
            || cpu.costs[index].support_a != gpu.costs[index].support_a
            || cpu.costs[index].support_b != gpu.costs[index].support_b
            || cpu.costs[index].support_intersection != gpu.costs[index].support_intersection
            || cpu.costs[index].merged_cost.total_bytes != gpu.costs[index].merged_cost.total_bytes
            || cpu.costs[index].separated_cost_bytes != gpu.costs[index].separated_cost_bytes) {
            return false;
        }
    }
    return true;
}

int main() {
    constexpr std::uint64_t cells = 65536u, genes = 30000u;
    constexpr std::uint32_t cluster_width = 8u;
    constexpr int warmups = 1, repeats = 3;
    cellerator::bench::benchmark_mutex_guard benchmark_mutex("cellPackMergeCostBench", 0);
    benchmark_fixture fixture;
    if (!build_fixture(cells, genes, cluster_width, &fixture)) {
        std::fprintf(stderr, "failed to build CP-BP-03 benchmark fixture\n");
        return 1;
    }
    cp::exact_merge_cost_policy policy;
    cp::owned_exact_gene_merge_scores cpu, gpu;
    const auto cpu_begin = std::chrono::steady_clock::now();
    cp::validation_result status = cp::score_gene_merges_cpu(
        fixture.support_view(), fixture.candidate_view(), policy, &cpu);
    const auto cpu_end = std::chrono::steady_clock::now();
    if (!status) {
        std::fprintf(stderr, "%s\n", status.message);
        return 2;
    }
    for (int warmup = 0; warmup < warmups; ++warmup) {
        status = cp::score_gene_merges_cuda(
            fixture.support_view(), fixture.candidate_view(), policy, 0, &gpu);
        if (!status) {
            std::fprintf(stderr, "%s\n", status.message);
            return 3;
        }
    }
    std::vector<double> gpu_milliseconds;
    gpu_milliseconds.reserve(repeats);
    for (int repeat = 0; repeat < repeats; ++repeat) {
        const auto begin = std::chrono::steady_clock::now();
        status = cp::score_gene_merges_cuda(
            fixture.support_view(), fixture.candidate_view(), policy, 0, &gpu);
        const auto end = std::chrono::steady_clock::now();
        if (!status) {
            std::fprintf(stderr, "%s\n", status.message);
            return 4;
        }
        gpu_milliseconds.push_back(
            std::chrono::duration<double, std::milli>(end - begin).count());
    }
    if (!equal_result(cpu.view(), gpu.view())) {
        std::fprintf(stderr, "CPU/CUDA CP-BP-03 benchmark outputs differ\n");
        return 5;
    }
    std::sort(gpu_milliseconds.begin(), gpu_milliseconds.end());
    const double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_begin).count();
    const std::size_t staged_device_bytes = fixture.layout.support_bytes
        + static_cast<std::size_t>(fixture.pair_count) * sizeof(gc::gene_candidate_pair)
        + static_cast<std::size_t>(fixture.pair_count) * sizeof(cp::exact_gene_merge_cost)
        + sizeof(cp::u32);
    std::fprintf(stdout,
                 "MERGE_COST_BENCH device=0 gpu=Tesla_V100 sm=70 cells=%llu genes=%llu "
                 "candidates=%llu words_per_gene=%zu support_bytes=%zu staged_device_bytes=%zu "
                 "policy_version=%u value_storage=compact_nonzeros warmups=%d repeats=%d "
                 "cpu_ms=%.3f gpu_min_ms=%.3f gpu_median_ms=%.3f exact_match=1\n",
                 static_cast<unsigned long long>(cells),
                 static_cast<unsigned long long>(genes),
                 static_cast<unsigned long long>(fixture.pair_count),
                 fixture.layout.words_per_gene, fixture.layout.support_bytes,
                 staged_device_bytes, policy.version, warmups, repeats, cpu_ms,
                 gpu_milliseconds.front(), gpu_milliseconds[gpu_milliseconds.size() / 2u]);
    return 0;
}
