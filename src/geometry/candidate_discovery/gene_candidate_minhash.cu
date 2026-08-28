/*
 * Validation/benchmark (2026-08-14, Cellerator 1ebb734): the sm_70
 * one-block-per-gene kernel matches discover_gene_candidates_cpu exactly on
 * 31/32/33/65-cell fixtures. On Tesla V100, the 65,536-cell x 30,000-gene,
 * 64-sketch end-to-end command ./build-sampling/geneCandidateDiscoveryBench
 * measured 59.832 ms minimum and 60.904 ms median over three timed runs after
 * one warmup. No NVIDIA library owns support-bitset MinHash; CUB owns sorting.
 */

#include <Cellerator/geometry/candidate_discovery/gene_candidate_hash.cuh>
#include "gene_candidate_internal.hh"

#include <cuda_runtime.h>

#include <limits>

namespace cellerator::compute::gene_candidates::detail {

namespace cg = ::cellerator::compute::gene_support;
namespace ct = ::cellerator::types;

__global__ void gene_minhash_kernel(const cg::support_word_t *support,
                                    const std::uint64_t *sampled_position_to_global_row,
                                    const ct::idx_t *nonempty_genes,
                                    std::uint64_t nonempty_gene_count,
                                    std::size_t words_per_gene,
                                    std::uint32_t sketch_count,
                                    std::uint64_t seed,
                                    std::uint64_t *sketches) {
    const std::uint64_t gene_position = blockIdx.x;
    const std::uint32_t sketch = threadIdx.x;
    if (gene_position >= nonempty_gene_count || sketch >= sketch_count) return;
    const ct::idx_t gene = nonempty_genes[gene_position];
    std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
    for (std::size_t word = 0u; word < words_per_gene; ++word) {
        ct::u32 bits = support[(std::size_t) gene * words_per_gene + word];
        while (bits != 0u) {
            const std::uint32_t bit = (std::uint32_t) (__ffs((int) bits) - 1);
            const std::uint64_t cell = word * 32u + bit;
            minimum = min(minimum, minhash_value_v1(
                sampled_position_to_global_row[cell], seed, sketch));
            bits &= bits - 1u;
        }
    }
    sketches[gene_position * sketch_count + sketch] = minimum;
}

cudaError_t launch_gene_minhash(const cg::support_word_t *support,
                                const std::uint64_t *sampled_position_to_global_row,
                                const ct::idx_t *nonempty_genes,
                                std::uint64_t nonempty_gene_count,
                                std::size_t words_per_gene,
                                std::uint32_t sketch_count,
                                std::uint64_t seed,
                                std::uint64_t *sketches,
                                cudaStream_t stream) {
    if (nonempty_gene_count == 0u) return cudaSuccess;
    const unsigned int threads = ((sketch_count + 31u) / 32u) * 32u;
    gene_minhash_kernel<<<(unsigned int) nonempty_gene_count, threads, 0u, stream>>>(
        support, sampled_position_to_global_row, nonempty_genes, nonempty_gene_count,
        words_per_gene, sketch_count, seed, sketches);
    return cudaGetLastError();
}

} // namespace cellerator::compute::gene_candidates::detail
