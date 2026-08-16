#pragma once

#include <Cellerator/compute/gene_candidate_discovery.hh>

#include <memory>
#include <string>

#include <cuda_runtime_api.h>

namespace cellerator::compute::gene_candidates::detail {

void set_error(std::string *error, const std::string &message);
bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t *out);
bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t *out);
bool validate_config(const candidate_discovery_config &config, std::string *error);
bool validate_support_view(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    std::string *error);
bool collect_nonempty_genes(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    std::unique_ptr< ::cellerator::types::idx_t[]> *genes,
    std::uint64_t *count,
    std::string *error);
candidate_discovery_provenance make_provenance(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    const candidate_discovery_config &config,
    std::uint64_t nonempty_gene_count);

cudaError_t launch_gene_minhash(
    const ::cellerator::compute::gene_support::support_word_t *support,
    const std::uint64_t *sampled_position_to_global_row,
    const ::cellerator::types::idx_t *nonempty_genes,
    std::uint64_t nonempty_gene_count,
    std::size_t words_per_gene,
    std::uint32_t sketch_count,
    std::uint64_t seed,
    std::uint64_t *sketches,
    cudaStream_t stream);

} // namespace cellerator::compute::gene_candidates::detail
