#pragma once

#include <Cellerator/geometry/gene_support_bitset.hh>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace cellerator::compute::gene_candidates {

inline constexpr const char *candidate_algorithm_name = "cellerator_gene_minhash_lsh_v1";
inline constexpr std::uint32_t candidate_hash_version = 1u;
inline constexpr const char *candidate_bucket_cap_rule = "splitmix64_circular_window_v1";

struct gene_candidate_pair {
    ::cellerator::types::idx_t gene_a = 0u;
    ::cellerator::types::idx_t gene_b = 0u;
};

struct candidate_discovery_config {
    std::uint64_t seed = 0u;
    std::uint32_t sketch_count = 64u;
    std::uint32_t lsh_bands = 16u;
    std::uint32_t rows_per_band = 4u;
    std::uint32_t maximum_bucket_size = 64u;
    std::uint64_t maximum_raw_pair_occurrences = 16000000u;
};

struct candidate_discovery_bounds {
    std::uint64_t maximum_lsh_records = 0u;
    std::uint64_t maximum_raw_pair_occurrences = 0u;
    std::size_t support_staging_bytes = 0u;
    std::size_t mapping_staging_bytes = 0u;
    std::size_t nonempty_gene_staging_bytes = 0u;
    std::size_t sketch_bytes = 0u;
    std::size_t lsh_ping_pong_bytes = 0u;
    std::size_t grouping_bytes = 0u;
    std::size_t raw_pair_ping_pong_bytes = 0u;
    std::size_t fixed_device_bytes_excluding_cub = 0u;
};

struct candidate_discovery_provenance {
    std::string algorithm = candidate_algorithm_name;
    std::uint32_t hash_version = candidate_hash_version;
    candidate_discovery_config config;
    ::cellerator::compute::sampling::sample_provenance sampling;
    std::uint64_t sampled_cell_count = 0u;
    std::uint64_t gene_count = 0u;
    std::uint64_t nonempty_gene_count = 0u;
    std::uint64_t lsh_record_count = 0u;
    std::uint64_t bucket_count = 0u;
    std::uint64_t oversized_bucket_count = 0u;
    std::uint64_t discarded_bucket_members = 0u;
    std::uint64_t raw_pair_occurrences = 0u;
    std::uint64_t unique_candidate_count = 0u;
    std::size_t device_cub_temporary_bytes = 0u;
    std::size_t device_peak_bytes = 0u;
    std::string bucket_cap_rule = candidate_bucket_cap_rule;
};

// Host-borrowed view. Pointers remain valid until the owning result is moved,
// move-assigned, or destroyed.
struct gene_candidate_pair_view {
    const gene_candidate_pair *pairs = nullptr;
    std::uint64_t count = 0u;
    const candidate_discovery_provenance *provenance = nullptr;
};

class owned_gene_candidates {
public:
    owned_gene_candidates() = default;
    owned_gene_candidates(std::unique_ptr<gene_candidate_pair[]> pairs,
                          std::uint64_t count,
                          candidate_discovery_provenance provenance) noexcept;
    ~owned_gene_candidates() = default;
    owned_gene_candidates(const owned_gene_candidates &) = delete;
    owned_gene_candidates &operator=(const owned_gene_candidates &) = delete;
    owned_gene_candidates(owned_gene_candidates &&) noexcept = default;
    owned_gene_candidates &operator=(owned_gene_candidates &&) noexcept = default;

    gene_candidate_pair_view view() const noexcept;
    const candidate_discovery_provenance &discovery_provenance() const noexcept;

private:
    std::unique_ptr<gene_candidate_pair[]> pairs_;
    std::uint64_t count_ = 0u;
    candidate_discovery_provenance provenance_;
};

// Fixed unsigned-arithmetic contract shared by CPU and CUDA implementations.
std::uint64_t candidate_splitmix64_v1(std::uint64_t value) noexcept;
std::uint64_t candidate_minhash_value_v1(std::uint64_t global_row_index,
                                         std::uint64_t seed,
                                         std::uint32_t sketch_index) noexcept;
std::uint64_t candidate_lsh_band_key_v1(const std::uint64_t *sketch_values,
                                        std::uint32_t rows_per_band,
                                        std::uint64_t seed,
                                        std::uint32_t band) noexcept;

bool calculate_candidate_discovery_bounds(
    const ::cellerator::compute::gene_support::gene_support_layout &support_layout,
    const candidate_discovery_config &config,
    candidate_discovery_bounds *out,
    std::string *error = nullptr);

bool discover_gene_candidates_cpu(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    const candidate_discovery_config &config,
    owned_gene_candidates *out,
    std::string *error = nullptr);

// Correctness-first convenience path: transiently stages the host-owned Step 1
// support on the requested device and returns a host-owned canonical pair list.
bool discover_gene_candidates_cuda(
    const ::cellerator::compute::gene_support::gene_support_bitset_view &support,
    const candidate_discovery_config &config,
    int device,
    owned_gene_candidates *out,
    std::string *error = nullptr);

} // namespace cellerator::compute::gene_candidates
