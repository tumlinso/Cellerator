/*
 * CPU/CUDA correctness validation for the CP-BP-02 integer hash, MinHash,
 * bounded LSH grouping, and canonical deduplication contract. Target: V100
 * sm_70. Command: ./build-sampling/geneCandidateDiscoveryRuntimeTest. Exact
 * CPU/GPU agreement is required; the representative benchmark is 60.904 ms
 * median for 65,536 cells x 30,000 genes on 2026-08-14.
 */

#include <Cellerator/compute/gene_candidate_discovery.hh>

#include "src/compute/packing/gene_candidate_hash.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cg = ::cellerator::compute::gene_support;
namespace cc = ::cellerator::compute::gene_candidates;
namespace cs = ::cellerator::compute::sampling;
namespace ct = ::cellerator::types;

int require(bool ok, const char *label) {
    if (!ok) std::fprintf(stderr, "%s\n", label);
    return ok ? 1 : 0;
}

bool make_support(std::uint64_t cells,
                  const std::vector<std::vector<std::uint64_t>> &gene_cells,
                  const char *split_name,
                  cg::owned_gene_support_bitsets *out,
                  std::string *error) {
    cg::gene_support_layout layout;
    if (!cg::calculate_gene_support_layout(cells, gene_cells.size(), &layout, error)) return false;
    std::unique_ptr<cg::support_word_t[]> words;
    std::unique_ptr<ct::count_value_t[]> counts;
    std::unique_ptr<std::uint64_t[]> mapping;
    if (layout.support_word_count != 0u) {
        words.reset(new (std::nothrow) cg::support_word_t[layout.support_word_count]());
    }
    if (!gene_cells.empty()) counts.reset(new (std::nothrow) ct::count_value_t[gene_cells.size()]());
    if (cells != 0u) mapping.reset(new (std::nothrow) std::uint64_t[(std::size_t) cells]);
    if ((layout.support_word_count != 0u && words == nullptr)
        || (!gene_cells.empty() && counts == nullptr) || (cells != 0u && mapping == nullptr)) {
        if (error != nullptr) *error = "failed to allocate candidate test support";
        return false;
    }
    for (std::uint64_t cell = 0u; cell < cells; ++cell) mapping[cell] = 1000u + 3u * cell;
    for (std::size_t gene = 0u; gene < gene_cells.size(); ++gene) {
        std::vector<std::uint64_t> unique_cells = gene_cells[gene];
        std::sort(unique_cells.begin(), unique_cells.end());
        unique_cells.erase(std::unique(unique_cells.begin(), unique_cells.end()), unique_cells.end());
        for (std::uint64_t cell : unique_cells) {
            if (cell >= cells) {
                if (error != nullptr) *error = "candidate test support cell is out of range";
                return false;
            }
            words[gene * layout.words_per_gene + (std::size_t) cell / 32u]
                |= (ct::u32) 1u << (cell % 32u);
        }
        counts[gene] = (ct::count_value_t) unique_cells.size();
    }
    cs::sample_provenance provenance;
    provenance.seed = 0xabcddcba12344321ull;
    provenance.total_rows = cells == 0u ? 0u : 1000u + 3u * (cells - 1u) + 1u;
    provenance.selected_rows = cells;
    provenance.mode = cs::selection_mode::exact_lowest_hash;
    provenance.split_name = split_name;
    provenance.requested_row_count = cells;
    *out = cg::owned_gene_support_bitsets(
        layout, std::move(words), std::move(counts), std::move(mapping), std::move(provenance));
    return true;
}

bool equal_results(const cc::gene_candidate_pair_view &lhs,
                   const cc::gene_candidate_pair_view &rhs) {
    if (lhs.count != rhs.count || lhs.provenance == nullptr || rhs.provenance == nullptr) return false;
    for (std::uint64_t i = 0u; i < lhs.count; ++i) {
        if (lhs.pairs[i].gene_a != rhs.pairs[i].gene_a
            || lhs.pairs[i].gene_b != rhs.pairs[i].gene_b) return false;
    }
    return lhs.provenance->algorithm == rhs.provenance->algorithm
        && lhs.provenance->hash_version == rhs.provenance->hash_version
        && lhs.provenance->config.seed == rhs.provenance->config.seed
        && lhs.provenance->bucket_count == rhs.provenance->bucket_count
        && lhs.provenance->oversized_bucket_count == rhs.provenance->oversized_bucket_count
        && lhs.provenance->discarded_bucket_members == rhs.provenance->discarded_bucket_members
        && lhs.provenance->raw_pair_occurrences == rhs.provenance->raw_pair_occurrences
        && lhs.provenance->sampling.split_name == rhs.provenance->sampling.split_name;
}

bool contains_pair(const cc::gene_candidate_pair_view &view, ct::idx_t a, ct::idx_t b) {
    if (a > b) std::swap(a, b);
    for (std::uint64_t i = 0u; i < view.count; ++i) {
        if (view.pairs[i].gene_a == a && view.pairs[i].gene_b == b) return true;
    }
    return false;
}

bool canonical_sorted_unique(const cc::gene_candidate_pair_view &view) {
    for (std::uint64_t i = 0u; i < view.count; ++i) {
        if (view.pairs[i].gene_a >= view.pairs[i].gene_b) return false;
        if (i != 0u) {
            const auto &previous = view.pairs[i - 1u], &current = view.pairs[i];
            if (previous.gene_a > current.gene_a
                || (previous.gene_a == current.gene_a && previous.gene_b >= current.gene_b)) {
                return false;
            }
        }
    }
    return true;
}

__global__ void hash_golden_kernel(std::uint64_t *out) {
    const std::uint64_t seed = 0x123456789abcdef0ull;
    out[0] = cc::detail::splitmix64_v1(0u);
    out[1] = cc::detail::splitmix64_v1(1u);
    out[2] = cc::detail::splitmix64_v1(UINT64_MAX);
    for (std::uint32_t sketch = 0u; sketch < 4u; ++sketch) {
        out[3u + sketch] = cc::detail::minhash_value_v1(7u, seed, sketch);
    }
    out[7] = cc::detail::lsh_band_key_v1(out + 3u, 4u, seed, 0u);
}

int test_hash_goldens(bool gpu_available) {
    const std::uint64_t expected[] = {
        0xe220a8397b1dcdafull, 0x910a2dec89025cc1ull, 0xe4d971771b652c20ull,
        0x989946415e8475a2ull, 0xb2e094d64a7680ebull, 0x14752360a22aae64ull,
        0x8db45e50588b0128ull, 0x69cb3ae25a3bc59aull
    };
    if (!require(cc::candidate_splitmix64_v1(0u) == expected[0]
                 && cc::candidate_splitmix64_v1(1u) == expected[1]
                 && cc::candidate_splitmix64_v1(UINT64_MAX) == expected[2],
                 "candidate SplitMix64 golden mismatch")) return 1;
    std::uint64_t sketches[4];
    for (std::uint32_t sketch = 0u; sketch < 4u; ++sketch) {
        sketches[sketch] = cc::candidate_minhash_value_v1(
            7u, 0x123456789abcdef0ull, sketch);
        if (!require(sketches[sketch] == expected[3u + sketch],
                     "candidate MinHash golden mismatch")) return 2;
    }
    if (!require(cc::candidate_lsh_band_key_v1(
                     sketches, 4u, 0x123456789abcdef0ull, 0u) == expected[7],
                 "candidate LSH band-key golden mismatch")) return 3;
    if (gpu_available) {
        std::uint64_t *device_values = nullptr;
        std::uint64_t actual[8]{};
        if (!require(cudaMalloc((void **) &device_values, sizeof(actual)) == cudaSuccess,
                     "failed to allocate CUDA hash goldens")) return 4;
        hash_golden_kernel<<<1, 1>>>(device_values);
        cudaError_t status = cudaGetLastError();
        if (status == cudaSuccess) status = cudaMemcpy(
            actual, device_values, sizeof(actual), cudaMemcpyDeviceToHost);
        cudaFree(device_values);
        if (!require(status == cudaSuccess, cudaGetErrorString(status))) return 5;
        for (std::size_t i = 0u; i < 8u; ++i) {
            if (!require(actual[i] == expected[i], "CUDA candidate hash golden mismatch")) return 6;
        }
    }
    return 0;
}

int test_support_semantics_and_determinism(bool gpu_available) {
    const std::vector<std::vector<std::uint64_t>> genes = {
        {0u, 1u, 2u, 3u}, {0u, 1u, 2u, 3u}, {10u, 11u},
        {0u, 1u, 2u, 4u}, {}, {20u}, {20u}, {32u, 33u, 64u}
    };
    cg::owned_gene_support_bitsets support;
    cc::owned_gene_candidates first, second, gpu;
    cc::candidate_discovery_config config;
    config.seed = 77u;
    std::string error;
    if (!require(make_support(65u, genes, "candidate-stable", &support, &error), error.c_str())) return 10;
    if (!require(cc::discover_gene_candidates_cpu(support.view(), config, &first, &error), error.c_str())) return 11;
    if (!require(cc::discover_gene_candidates_cpu(support.view(), config, &second, &error), error.c_str())) return 12;
    const cc::gene_candidate_pair_view view = first.view();
    if (!require(equal_results(view, second.view()), "repeated CPU candidate discovery differs")) return 13;
    if (!require(canonical_sorted_unique(view), "candidate pairs are not canonical sorted unique")) return 14;
    if (!require(contains_pair(view, 0u, 1u), "identical common genes were not retained")) return 15;
    if (!require(contains_pair(view, 5u, 6u), "identical rare genes were not retained")) return 16;
    if (!require(!contains_pair(view, 0u, 2u), "fixed disjoint support unexpectedly collided")) return 17;
    for (std::uint64_t i = 0u; i < view.count; ++i) {
        if (!require(view.pairs[i].gene_a != 4u && view.pairs[i].gene_b != 4u,
                     "empty gene entered candidate output")) return 18;
    }
    if (!require(view.provenance->raw_pair_occurrences > view.count,
                 "duplicate LSH collisions were not exercised")) return 19;
    if (!require(view.provenance->sampling.seed == support.view().provenance->seed
                 && view.provenance->sampling.split_name == "candidate-stable"
                 && view.provenance->sampled_cell_count == 65u,
                 "Step 1 provenance was not retained")) return 20;
    if (gpu_available) {
        if (!require(cc::discover_gene_candidates_cuda(support.view(), config, 0, &gpu, &error), error.c_str())) return 21;
        if (!require(equal_results(view, gpu.view()), "CPU/GPU candidate results differ")) return 22;
    }
    return 0;
}

int test_tail_word_counts(bool gpu_available) {
    for (std::uint64_t cells : {31u, 32u, 33u, 65u}) {
        const std::vector<std::vector<std::uint64_t>> genes = {
            {0u, cells - 1u}, {0u, cells - 1u}, {}, {cells / 2u}
        };
        cg::owned_gene_support_bitsets support;
        cc::owned_gene_candidates cpu, gpu;
        cc::candidate_discovery_config config;
        config.seed = cells;
        std::string error;
        if (!require(make_support(cells, genes, "candidate-tail", &support, &error), error.c_str())) return 30;
        if (!require(cc::discover_gene_candidates_cpu(support.view(), config, &cpu, &error), error.c_str())) return 31;
        if (!require(contains_pair(cpu.view(), 0u, 1u), "tail-word identical pair was not retained")) return 32;
        if (gpu_available) {
            if (!require(cc::discover_gene_candidates_cuda(support.view(), config, 0, &gpu, &error), error.c_str())) return 33;
            if (!require(equal_results(cpu.view(), gpu.view()), "tail-word CPU/GPU candidates differ")) return 34;
        }
    }
    return 0;
}

int test_bucket_cap(bool gpu_available) {
    std::vector<std::vector<std::uint64_t>> genes(12u, {0u, 1u, 2u, 32u});
    cg::owned_gene_support_bitsets support;
    cc::owned_gene_candidates first, second, gpu;
    cc::candidate_discovery_config config;
    config.seed = 991u;
    config.sketch_count = 8u;
    config.lsh_bands = 8u;
    config.rows_per_band = 1u;
    config.maximum_bucket_size = 4u;
    config.maximum_raw_pair_occurrences = 1000u;
    std::string error;
    if (!require(make_support(33u, genes, "candidate-cap", &support, &error), error.c_str())) return 40;
    if (!require(cc::discover_gene_candidates_cpu(support.view(), config, &first, &error)
                 && cc::discover_gene_candidates_cpu(support.view(), config, &second, &error),
                 error.c_str())) return 41;
    const auto view = first.view();
    if (!require(equal_results(view, second.view()), "bucket-cap output is not deterministic")) return 42;
    if (!require(view.provenance->oversized_bucket_count == 8u
                 && view.provenance->discarded_bucket_members == 64u
                 && view.provenance->raw_pair_occurrences == 48u
                 && view.count <= 48u,
                 "bucket-cap accounting is incorrect")) return 43;
    if (!require(canonical_sorted_unique(view), "bucket-cap candidates are not canonical")) return 44;
    if (gpu_available) {
        if (!require(cc::discover_gene_candidates_cuda(support.view(), config, 0, &gpu, &error), error.c_str())) return 45;
        if (!require(equal_results(view, gpu.view()), "bucket-cap CPU/GPU candidates differ")) return 46;
    }
    return 0;
}

int test_zero_cells_and_genes(bool gpu_available) {
    std::string error;
    for (const auto &fixture : std::vector<std::pair<
             std::uint64_t, std::vector<std::vector<std::uint64_t>>>>{
             {0u, {{}, {}, {}}}, {5u, {}}}) {
        cg::owned_gene_support_bitsets support;
        cc::owned_gene_candidates cpu, gpu;
        cc::candidate_discovery_config config;
        if (!require(make_support(fixture.first, fixture.second, "candidate-zero",
                                  &support, &error), error.c_str())) return 47;
        if (!require(cc::discover_gene_candidates_cpu(support.view(), config, &cpu, &error),
                     error.c_str())) return 48;
        if (!require(cpu.view().count == 0u && cpu.view().pairs == nullptr,
                     "zero-shape candidate output is not empty")) return 49;
        if (gpu_available) {
            if (!require(cc::discover_gene_candidates_cuda(
                             support.view(), config, 0, &gpu, &error), error.c_str())) return 50;
            if (!require(equal_results(cpu.view(), gpu.view()),
                         "zero-shape CPU/GPU candidates differ")) return 51;
        }
    }
    return 0;
}

int test_invalid_config_and_bounds() {
    cg::gene_support_layout layout;
    cc::candidate_discovery_bounds bounds;
    cc::candidate_discovery_config config;
    std::string error;
    if (!require(cg::calculate_gene_support_layout(65536u, 30000u, &layout, &error), error.c_str())) return 50;
    if (!require(cc::calculate_candidate_discovery_bounds(layout, config, &bounds, &error), error.c_str())) return 51;
    if (!require(bounds.maximum_lsh_records == 480000u
                 && bounds.maximum_raw_pair_occurrences == 15113856u
                 && bounds.sketch_bytes == 15360000u,
                 "representative candidate sizing formula is incorrect")) return 52;
    config.sketch_count = 63u;
    if (!require(!cc::calculate_candidate_discovery_bounds(layout, config, &bounds, &error),
                 "mismatched sketch/band configuration was accepted")) return 53;
    config = {};
    config.maximum_bucket_size = 1u;
    if (!require(!cc::calculate_candidate_discovery_bounds(layout, config, &bounds, &error),
                 "invalid bucket cap was accepted")) return 54;
    config = {};
    config.maximum_raw_pair_occurrences = 100u;
    if (!require(!cc::calculate_candidate_discovery_bounds(layout, config, &bounds, &error),
                 "insufficient raw-pair budget was accepted")) return 55;
    config = {};
    layout.gene_count = UINT64_MAX;
    if (!require(!cc::calculate_candidate_discovery_bounds(layout, config, &bounds, &error),
                 "overflowing candidate dimensions were accepted")) return 56;
    config = {};
    config.sketch_count = 257u;
    config.lsh_bands = 257u;
    config.rows_per_band = 1u;
    layout.gene_count = 10u;
    if (!require(!cc::calculate_candidate_discovery_bounds(layout, config, &bounds, &error),
                 "oversized sketch count was accepted")) return 57;
    return 0;
}

double jaccard(const cg::gene_support_bitset_view &support, std::size_t a, std::size_t b) {
    std::uint64_t intersection = 0u, union_count = 0u;
    for (std::size_t word = 0u; word < support.layout.words_per_gene; ++word) {
        const ct::u32 lhs = support.gene_support[a * support.layout.words_per_gene + word];
        const ct::u32 rhs = support.gene_support[b * support.layout.words_per_gene + word];
        intersection += (std::uint64_t) __builtin_popcount(lhs & rhs);
        union_count += (std::uint64_t) __builtin_popcount(lhs | rhs);
    }
    return union_count == 0u ? 0.0 : (double) intersection / (double) union_count;
}

int test_synthetic_recall(bool gpu_available) {
    constexpr std::size_t gene_count = 64u;
    std::vector<std::vector<std::uint64_t>> genes(gene_count);
    for (std::size_t cluster = 0u; cluster < 8u; ++cluster) {
        const std::vector<std::uint64_t> support = {
            cluster, cluster + 8u, cluster + 16u, cluster + 32u
        };
        for (std::size_t member = 0u; member < 4u; ++member) {
            genes[cluster * 4u + member] = support;
        }
    }
    for (std::size_t gene = 32u; gene < gene_count; ++gene) genes[gene].push_back(gene);
    cg::owned_gene_support_bitsets support;
    cc::owned_gene_candidates cpu, gpu;
    cc::candidate_discovery_config config;
    config.seed = 424242u;
    std::string error;
    if (!require(make_support(65u, genes, "candidate-recall", &support, &error), error.c_str())) return 60;
    if (!require(cc::discover_gene_candidates_cpu(support.view(), config, &cpu, &error), error.c_str())) return 61;
    std::uint64_t high_overlap = 0u, retained = 0u;
    for (std::size_t a = 0u; a < gene_count; ++a) {
        for (std::size_t b = a + 1u; b < gene_count; ++b) {
            if (jaccard(support.view(), a, b) >= 0.75) {
                ++high_overlap;
                if (contains_pair(cpu.view(), (ct::idx_t) a, (ct::idx_t) b)) ++retained;
            }
        }
    }
    const std::uint64_t exhaustive = gene_count * (gene_count - 1u) / 2u;
    const double recall = high_overlap == 0u ? 1.0 : (double) retained / high_overlap;
    const double reduction = 1.0 - (double) cpu.view().count / exhaustive;
    std::fprintf(stderr,
                 "CANDIDATE_SYNTHETIC genes=%zu exhaustive=%llu candidates=%llu "
                 "high_overlap=%llu retained=%llu recall=%.6f reduction=%.6f\n",
                 gene_count, (unsigned long long) exhaustive,
                 (unsigned long long) cpu.view().count,
                 (unsigned long long) high_overlap, (unsigned long long) retained,
                 recall, reduction);
    if (!require(high_overlap == 48u && retained == high_overlap && recall == 1.0,
                 "deliberately high-overlap candidate recall is incomplete")) return 62;
    if (!require(reduction > 0.90, "synthetic candidate reduction is too small")) return 63;
    if (gpu_available) {
        if (!require(cc::discover_gene_candidates_cuda(support.view(), config, 0, &gpu, &error), error.c_str())) return 64;
        if (!require(equal_results(cpu.view(), gpu.view()), "synthetic CPU/GPU candidates differ")) return 65;
    }
    return 0;
}

int main() {
    static_assert(std::is_same<decltype(cc::gene_candidate_pair{}.gene_a), ct::idx_t>::value,
                  "candidate endpoints must use Cellerator canonical gene indices");
    static_assert(sizeof(cc::gene_candidate_pair) == 2u * sizeof(ct::idx_t),
                  "candidate pair ABI must remain two canonical indices");
    static_assert(std::is_trivially_copyable<cc::gene_candidate_pair>::value,
                  "candidate pairs must remain directly transferable");
    int device_count = 0;
    const cudaError_t device_status = cudaGetDeviceCount(&device_count);
    const bool gpu_available = device_status == cudaSuccess && device_count > 0;
    if (!gpu_available) {
        cudaGetLastError();
        std::fprintf(stderr, "SKIP: candidate CUDA agreement checks (no CUDA device)\n");
    }
    int status = 0;
    if ((status = test_hash_goldens(gpu_available)) != 0) return status;
    if ((status = test_support_semantics_and_determinism(gpu_available)) != 0) return status;
    if ((status = test_tail_word_counts(gpu_available)) != 0) return status;
    if ((status = test_bucket_cap(gpu_available)) != 0) return status;
    if ((status = test_zero_cells_and_genes(gpu_available)) != 0) return status;
    if ((status = test_invalid_config_and_bounds()) != 0) return status;
    if ((status = test_synthetic_recall(gpu_available)) != 0) return status;
    return 0;
}
