#pragma once

#include <Cellerator/matrix/compressed.cuh>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cellerator::compute::sampling {

inline constexpr const char *splitmix64_algorithm_name = "cellerator_splitmix64_v1";
inline constexpr std::uint32_t splitmix64_algorithm_version = 1u;
inline constexpr std::uint64_t maximum_exact_sample_rows = 65536u;
inline constexpr const char *inverse_stratum_weighting_rule =
    "weight_h=total_rows_in_stratum/sampled_rows_in_stratum_v1";

enum class selection_mode : std::uint32_t {
    hash_quantile_range = 1u,
    exact_lowest_hash = 2u,
    density_quantile_exact = 3u
};

enum class cell_identity_kind : std::uint32_t {
    // Canonical dataset/global matrix row identity, not a later physical pack
    // or execution-layout position.
    global_row_index = 1u,
    stable_item_id = 2u,
    // Compatibility spelling for the frozen v1 numeric provenance value.
    stable_cellshard_cell_id = stable_item_id
};

struct quantile_boundary {
    std::uint64_t numerator = 0u;
    std::uint64_t denominator = 1u;
};

struct hash_quantile_range {
    quantile_boundary begin = {0u, 1u};
    quantile_boundary end = {1u, 1u};
};

struct sample_spec {
    selection_mode mode = selection_mode::hash_quantile_range;
    std::uint64_t seed = 0u;
    std::string split_name;
    hash_quantile_range quantile;
    // Exact mode accepts requests through maximum_exact_sample_rows. If the
    // population is smaller, every row is selected and the original request
    // remains recorded in provenance.
    std::uint64_t requested_row_count = 0u;
};

struct cell_identity_view {
    cell_identity_kind kind = cell_identity_kind::global_row_index;
    const char * const *stable_cell_ids = nullptr;
    std::uint64_t count = 0u;
};

struct row_nnz_view {
    // Values must be accurate structural nonzeros for complete global rows.
    // Padded Blocked-ELL/Sliced-ELL widths are not valid substitutes.
    const std::uint64_t *values = nullptr;
    std::uint64_t count = 0u;
};

struct density_sample_spec {
    // Exact host quantiles hold and sort all row lengths. Equal row lengths
    // are never split between strata.
    std::uint64_t seed = 0u;
    std::string split_name;
    std::uint32_t requested_strata = 1u;
    std::uint64_t requested_row_count = 0u;
};

struct sample_provenance {
    std::uint64_t seed = 0u;
    std::string hash_algorithm = splitmix64_algorithm_name;
    std::uint32_t hash_version = splitmix64_algorithm_version;
    std::uint64_t total_rows = 0u;
    std::uint64_t selected_rows = 0u;
    selection_mode mode = selection_mode::hash_quantile_range;
    std::string split_name;
    cell_identity_kind cell_identity = cell_identity_kind::global_row_index;
    hash_quantile_range quantile;
    std::uint64_t requested_row_count = 0u;
    std::uint32_t requested_density_strata = 0u;
    std::uint32_t density_strata = 0u;
    std::vector<std::uint64_t> density_bin_upper_bounds_inclusive;
    std::vector<std::uint64_t> stratum_total_rows;
    std::vector<std::uint64_t> stratum_sampled_rows;
    std::string weighting_rule;
};

struct sample_plan {
    // Rows are always ascending so downstream materializers read canonical
    // complete rows. Vector position is the deterministic materialized sample
    // position, and global_row_indices[position] preserves its source mapping.
    // Hashes, strata, and weights are aligned to the same positions.
    std::vector<std::uint64_t> global_row_indices;
    std::vector<std::uint64_t> identity_hashes;
    std::vector<std::uint32_t> row_strata;
    std::vector<double> sampling_weights;
    sample_provenance provenance;
};

// Fixed v1 contract. Numeric identities use domain-separated SplitMix64;
// stable ID bytes use FNV-1a plus length/domain separation before SplitMix64.
// Unsigned overflow is intentional.
std::uint64_t splitmix64_hash(std::uint64_t value) noexcept;
std::uint64_t hash_global_row_index(std::uint64_t global_row_index, std::uint64_t seed) noexcept;
std::uint64_t hash_stable_cell_id(const char *cell_id, std::size_t length, std::uint64_t seed) noexcept;

bool build_sample_plan(std::uint64_t total_rows,
                       const sample_spec &spec,
                       const cell_identity_view &identities,
                       sample_plan *out,
                       std::string *error = nullptr);

bool reproduce_sample_plan(const sample_provenance &provenance,
                           const cell_identity_view &identities,
                           sample_plan *out,
                           std::string *error = nullptr);

bool build_density_sample_plan(std::uint64_t total_rows,
                               const row_nnz_view &row_nnz,
                               const density_sample_spec &spec,
                               const cell_identity_view &identities,
                               sample_plan *out,
                               std::string *error = nullptr);

bool build_csr_density_sample_plan(const ::cellerator::matrix::compressed *source,
                                   const density_sample_spec &spec,
                                   const cell_identity_view &identities,
                                   sample_plan *out,
                                   std::string *error = nullptr);

bool reproduce_density_sample_plan(const sample_provenance &provenance,
                                   const row_nnz_view &row_nnz,
                                   const cell_identity_view &identities,
                                   sample_plan *out,
                                   std::string *error = nullptr);

} // namespace cellerator::compute::sampling
