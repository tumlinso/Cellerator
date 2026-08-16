#include <Cellerator/compute/gene_candidate_discovery.hh>

#include "gene_candidate_hash.cuh"
#include "gene_candidate_internal.hh"

#include <algorithm>
#include <climits>
#include <limits>
#include <new>
#include <tuple>
#include <utility>
#include <vector>

namespace cellerator::compute::gene_candidates {

namespace detail {

namespace cg = ::cellerator::compute::gene_support;
namespace cs = ::cellerator::compute::sampling;
namespace ct = ::cellerator::types;

void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

bool checked_add(std::size_t lhs, std::size_t rhs, std::size_t *out) {
    if (out == nullptr || rhs > std::numeric_limits<std::size_t>::max() - lhs) return false;
    *out = lhs + rhs;
    return true;
}

bool checked_multiply(std::size_t lhs, std::size_t rhs, std::size_t *out) {
    if (out == nullptr || (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs)) {
        return false;
    }
    *out = lhs * rhs;
    return true;
}

bool validate_config(const candidate_discovery_config &config, std::string *error) {
    std::uint64_t product = 0u;
    if (config.sketch_count == 0u || config.sketch_count > 256u) {
        set_error(error, "candidate sketch_count must be in [1, 256]");
        return false;
    }
    if (config.lsh_bands == 0u || config.rows_per_band == 0u) {
        set_error(error, "candidate LSH bands and rows_per_band must be nonzero");
        return false;
    }
    product = (std::uint64_t) config.lsh_bands * config.rows_per_band;
    if (product != config.sketch_count) {
        set_error(error, "candidate sketch_count must equal lsh_bands * rows_per_band");
        return false;
    }
    if (config.maximum_bucket_size < 2u) {
        set_error(error, "candidate maximum_bucket_size must be at least two");
        return false;
    }
    if (config.maximum_raw_pair_occurrences == 0u
        || config.maximum_raw_pair_occurrences > (std::uint64_t) INT_MAX) {
        set_error(error, "candidate raw-pair budget must be in [1, INT_MAX]");
        return false;
    }
    return true;
}

bool validate_support_view(const cg::gene_support_bitset_view &support,
                           std::string *error) {
    if (support.provenance == nullptr) {
        set_error(error, "candidate input sampling provenance is null");
        return false;
    }
    if (support.provenance->hash_algorithm != cs::splitmix64_algorithm_name
        || support.provenance->hash_version != cs::splitmix64_algorithm_version
        || support.provenance->selected_rows != support.layout.sampled_cell_count) {
        set_error(error, "candidate input sampling provenance is inconsistent");
        return false;
    }
    cg::gene_support_layout expected;
    if (!cg::calculate_gene_support_layout(
            support.layout.sampled_cell_count, support.layout.gene_count, &expected, error)) {
        return false;
    }
    if (expected.words_per_gene != support.layout.words_per_gene
        || expected.support_word_count != support.layout.support_word_count
        || expected.support_bytes != support.layout.support_bytes
        || expected.detection_count_bytes != support.layout.detection_count_bytes) {
        set_error(error, "candidate input gene-support layout is inconsistent");
        return false;
    }
    if (support.layout.support_word_count != 0u && support.gene_support == nullptr) {
        set_error(error, "candidate input support words are null");
        return false;
    }
    if (support.layout.gene_count != 0u && support.detected_cell_counts == nullptr) {
        set_error(error, "candidate input detection counts are null");
        return false;
    }
    if (support.layout.sampled_cell_count != 0u
        && support.sampled_position_to_global_row == nullptr) {
        set_error(error, "candidate input sampled-position/global-row mapping is null");
        return false;
    }
    for (std::uint64_t cell = 0u; cell < support.layout.sampled_cell_count; ++cell) {
        const std::uint64_t global_row = support.sampled_position_to_global_row[cell];
        if (global_row >= support.provenance->total_rows
            || (cell != 0u && global_row <= support.sampled_position_to_global_row[cell - 1u])) {
            set_error(error, "candidate input global-row mapping is not strictly ascending and in range");
            return false;
        }
    }
    for (std::uint64_t gene = 0u; gene < support.layout.gene_count; ++gene) {
        if ((std::uint64_t) support.detected_cell_counts[gene] > support.layout.sampled_cell_count) {
            set_error(error, "candidate input detected-cell count exceeds sampled cells");
            return false;
        }
    }
    const std::uint32_t tail = (std::uint32_t) (support.layout.sampled_cell_count % 32u);
    if (tail != 0u && support.layout.words_per_gene != 0u) {
        const ct::u32 valid_mask = (ct::u32) ((1ull << tail) - 1ull);
        const std::size_t last_word = support.layout.words_per_gene - 1u;
        for (std::uint64_t gene = 0u; gene < support.layout.gene_count; ++gene) {
            if ((support.gene_support[(std::size_t) gene * support.layout.words_per_gene + last_word]
                 & ~valid_mask) != 0u) {
                set_error(error, "candidate input support has bits beyond sampled_cell_count");
                return false;
            }
        }
    }
    return true;
}

bool collect_nonempty_genes(const cg::gene_support_bitset_view &support,
                            std::unique_ptr<ct::idx_t[]> *genes,
                            std::uint64_t *count,
                            std::string *error) {
    if (genes == nullptr || count == nullptr) {
        set_error(error, "candidate nonempty-gene output is null");
        return false;
    }
    std::uint64_t nonempty = 0u;
    for (std::uint64_t gene = 0u; gene < support.layout.gene_count; ++gene) {
        if (support.detected_cell_counts[gene] != 0u) ++nonempty;
    }
    std::unique_ptr<ct::idx_t[]> staged;
    if (nonempty != 0u) staged.reset(new (std::nothrow) ct::idx_t[(std::size_t) nonempty]);
    if (nonempty != 0u && staged == nullptr) {
        set_error(error, "failed to allocate nonempty gene identities");
        return false;
    }
    std::size_t cursor = 0u;
    for (std::uint64_t gene = 0u; gene < support.layout.gene_count; ++gene) {
        if (support.detected_cell_counts[gene] != 0u) staged[cursor++] = (ct::idx_t) gene;
    }
    *genes = std::move(staged);
    *count = nonempty;
    return true;
}

candidate_discovery_provenance make_provenance(
    const cg::gene_support_bitset_view &support,
    const candidate_discovery_config &config,
    std::uint64_t nonempty_gene_count) {
    candidate_discovery_provenance provenance;
    provenance.config = config;
    provenance.sampling = *support.provenance;
    provenance.sampled_cell_count = support.layout.sampled_cell_count;
    provenance.gene_count = support.layout.gene_count;
    provenance.nonempty_gene_count = nonempty_gene_count;
    provenance.lsh_record_count = nonempty_gene_count * config.lsh_bands;
    return provenance;
}

} // namespace detail

namespace {

namespace cg = ::cellerator::compute::gene_support;
namespace ct = ::cellerator::types;

struct lsh_record {
    std::uint64_t key = 0u;
    std::uint32_t band = 0u;
    ct::idx_t gene = 0u;
};

std::uint64_t encode_pair(ct::idx_t gene_a, ct::idx_t gene_b) {
    const ct::idx_t a = std::min(gene_a, gene_b), b = std::max(gene_a, gene_b);
    return ((std::uint64_t) a << 32u) | (std::uint64_t) b;
}

bool allocate_pairs_from_encoded(const std::vector<std::uint64_t> &encoded,
                                 candidate_discovery_provenance provenance,
                                 owned_gene_candidates *out,
                                 std::string *error) {
    std::unique_ptr<gene_candidate_pair[]> pairs;
    if (!encoded.empty()) {
        pairs.reset(new (std::nothrow) gene_candidate_pair[encoded.size()]);
        if (pairs == nullptr) {
            detail::set_error(error, "failed to allocate candidate-pair output");
            return false;
        }
    }
    for (std::size_t i = 0u; i < encoded.size(); ++i) {
        pairs[i].gene_a = (ct::idx_t) (encoded[i] >> 32u);
        pairs[i].gene_b = (ct::idx_t) encoded[i];
    }
    provenance.unique_candidate_count = encoded.size();
    *out = owned_gene_candidates(
        std::move(pairs), (std::uint64_t) encoded.size(), std::move(provenance));
    return true;
}

} // namespace

owned_gene_candidates::owned_gene_candidates(
    std::unique_ptr<gene_candidate_pair[]> pairs,
    std::uint64_t count,
    candidate_discovery_provenance provenance) noexcept
    : pairs_(std::move(pairs)), count_(count), provenance_(std::move(provenance)) {}

gene_candidate_pair_view owned_gene_candidates::view() const noexcept {
    return {pairs_.get(), count_, &provenance_};
}

const candidate_discovery_provenance &owned_gene_candidates::discovery_provenance() const noexcept {
    return provenance_;
}

std::uint64_t candidate_splitmix64_v1(std::uint64_t value) noexcept {
    return detail::splitmix64_v1(value);
}

std::uint64_t candidate_minhash_value_v1(std::uint64_t global_row_index,
                                         std::uint64_t seed,
                                         std::uint32_t sketch_index) noexcept {
    return detail::minhash_value_v1(global_row_index, seed, sketch_index);
}

std::uint64_t candidate_lsh_band_key_v1(const std::uint64_t *sketch_values,
                                        std::uint32_t rows_per_band,
                                        std::uint64_t seed,
                                        std::uint32_t band) noexcept {
    if (sketch_values == nullptr && rows_per_band != 0u) return 0u;
    return detail::lsh_band_key_v1(sketch_values, rows_per_band, seed, band);
}

bool calculate_candidate_discovery_bounds(const cg::gene_support_layout &support_layout,
                                          const candidate_discovery_config &config,
                                          candidate_discovery_bounds *out,
                                          std::string *error) {
    if (out == nullptr) {
        detail::set_error(error, "candidate-discovery bounds output is null");
        return false;
    }
    if (!detail::validate_config(config, error)) return false;
    if (support_layout.gene_count > (std::uint64_t) std::numeric_limits<ct::idx_t>::max()) {
        detail::set_error(error, "candidate gene count exceeds canonical index range");
        return false;
    }

    const std::uint64_t genes = support_layout.gene_count;
    const std::uint64_t cap = std::min<std::uint64_t>(config.maximum_bucket_size, genes);
    const std::uint64_t full_buckets = cap == 0u ? 0u : genes / cap;
    const std::uint64_t remainder = cap == 0u ? 0u : genes % cap;
    const auto choose_two = [](std::uint64_t count) -> std::uint64_t {
        return count < 2u ? 0u : count * (count - 1u) / 2u;
    };
    if (genes != 0u && config.lsh_bands > (std::uint64_t) INT_MAX / genes) {
        detail::set_error(error, "candidate LSH record count exceeds CUB item range");
        return false;
    }
    const std::uint64_t records = genes * config.lsh_bands;
    const std::uint64_t per_band_pairs = full_buckets * choose_two(cap) + choose_two(remainder);
    if (per_band_pairs != 0u
        && config.lsh_bands > std::numeric_limits<std::uint64_t>::max() / per_band_pairs) {
        detail::set_error(error, "candidate raw-pair bound overflows uint64_t");
        return false;
    }
    const std::uint64_t raw_pairs = per_band_pairs * config.lsh_bands;
    if (raw_pairs > config.maximum_raw_pair_occurrences || raw_pairs > (std::uint64_t) INT_MAX) {
        detail::set_error(error, "candidate worst-case raw-pair bound exceeds configured budget");
        return false;
    }

    candidate_discovery_bounds bounds;
    bounds.maximum_lsh_records = records;
    bounds.maximum_raw_pair_occurrences = raw_pairs;
    bounds.support_staging_bytes = support_layout.support_bytes;
    if (!detail::checked_multiply((std::size_t) support_layout.sampled_cell_count,
                                  sizeof(std::uint64_t), &bounds.mapping_staging_bytes)
        || !detail::checked_multiply((std::size_t) genes, sizeof(ct::idx_t),
                                     &bounds.nonempty_gene_staging_bytes)
        || !detail::checked_multiply((std::size_t) genes, config.sketch_count * sizeof(std::uint64_t),
                                     &bounds.sketch_bytes)
        || !detail::checked_multiply((std::size_t) records, 4u * sizeof(std::uint64_t),
                                     &bounds.lsh_ping_pong_bytes)
        || !detail::checked_multiply((std::size_t) raw_pairs, 2u * sizeof(std::uint64_t),
                                     &bounds.raw_pair_ping_pong_bytes)) {
        detail::set_error(error, "candidate-discovery allocation size overflows size_t");
        return false;
    }
    std::size_t heads = 0u, ids = 0u, bucket_offsets = 0u, pair_counts = 0u, pair_offsets = 0u;
    if (!detail::checked_multiply((std::size_t) records, sizeof(ct::u32), &heads)
        || !detail::checked_multiply((std::size_t) records, sizeof(ct::u32), &ids)
        || !detail::checked_multiply((std::size_t) records + 1u, sizeof(std::uint64_t), &bucket_offsets)
        || !detail::checked_multiply((std::size_t) records, sizeof(std::uint64_t), &pair_counts)
        || !detail::checked_multiply((std::size_t) records + 1u, sizeof(std::uint64_t), &pair_offsets)
        || !detail::checked_add(heads, ids, &bounds.grouping_bytes)
        || !detail::checked_add(bounds.grouping_bytes, bucket_offsets, &bounds.grouping_bytes)
        || !detail::checked_add(bounds.grouping_bytes, pair_counts, &bounds.grouping_bytes)
        || !detail::checked_add(bounds.grouping_bytes, pair_offsets, &bounds.grouping_bytes)) {
        detail::set_error(error, "candidate grouping allocation size overflows size_t");
        return false;
    }
    std::size_t total = 0u;
    const std::size_t pieces[] = {
        bounds.support_staging_bytes, bounds.mapping_staging_bytes,
        bounds.nonempty_gene_staging_bytes,
        bounds.sketch_bytes, bounds.lsh_ping_pong_bytes, bounds.grouping_bytes,
        bounds.raw_pair_ping_pong_bytes
    };
    for (std::size_t piece : pieces) {
        if (!detail::checked_add(total, piece, &total)) {
            detail::set_error(error, "candidate fixed device-memory sum overflows size_t");
            return false;
        }
    }
    bounds.fixed_device_bytes_excluding_cub = total;
    *out = bounds;
    return true;
}

bool discover_gene_candidates_cpu(const cg::gene_support_bitset_view &support,
                                  const candidate_discovery_config &config,
                                  owned_gene_candidates *out,
                                  std::string *error) {
    if (out == nullptr) {
        detail::set_error(error, "owned candidate output is null");
        return false;
    }
    candidate_discovery_bounds bounds;
    if (!detail::validate_config(config, error)
        || !detail::validate_support_view(support, error)
        || !calculate_candidate_discovery_bounds(support.layout, config, &bounds, error)) {
        return false;
    }
    std::unique_ptr<ct::idx_t[]> nonempty_genes;
    std::uint64_t nonempty_count = 0u;
    if (!detail::collect_nonempty_genes(support, &nonempty_genes, &nonempty_count, error)) return false;
    candidate_discovery_provenance provenance =
        detail::make_provenance(support, config, nonempty_count);
    if (nonempty_count < 2u) {
        std::vector<std::uint64_t> empty;
        return allocate_pairs_from_encoded(empty, std::move(provenance), out, error);
    }

    try {
        std::vector<std::uint64_t> sketches((std::size_t) nonempty_count * config.sketch_count,
                                            std::numeric_limits<std::uint64_t>::max());
        for (std::size_t gene_position = 0u; gene_position < nonempty_count; ++gene_position) {
            const ct::idx_t gene = nonempty_genes[gene_position];
            for (std::uint32_t sketch = 0u; sketch < config.sketch_count; ++sketch) {
                std::uint64_t minimum = std::numeric_limits<std::uint64_t>::max();
                for (std::size_t word = 0u; word < support.layout.words_per_gene; ++word) {
                    ct::u32 bits = support.gene_support[(std::size_t) gene * support.layout.words_per_gene + word];
                    while (bits != 0u) {
                        const std::uint32_t bit = (std::uint32_t) __builtin_ctz(bits);
                        const std::uint64_t cell = word * 32u + bit;
                        const std::uint64_t value = detail::minhash_value_v1(
                            support.sampled_position_to_global_row[cell], config.seed, sketch);
                        minimum = std::min(minimum, value);
                        bits &= bits - 1u;
                    }
                }
                sketches[gene_position * config.sketch_count + sketch] = minimum;
            }
        }

        std::vector<lsh_record> records;
        records.reserve((std::size_t) nonempty_count * config.lsh_bands);
        for (std::uint32_t band = 0u; band < config.lsh_bands; ++band) {
            for (std::size_t gene_position = 0u; gene_position < nonempty_count; ++gene_position) {
                const std::uint64_t *band_values = sketches.data()
                    + gene_position * config.sketch_count + (std::size_t) band * config.rows_per_band;
                records.push_back({detail::lsh_band_key_v1(
                                       band_values, config.rows_per_band, config.seed, band),
                                   band, nonempty_genes[gene_position]});
            }
        }
        std::sort(records.begin(), records.end(), [](const lsh_record &lhs, const lsh_record &rhs) {
            return std::tie(lhs.key, lhs.band, lhs.gene) < std::tie(rhs.key, rhs.band, rhs.gene);
        });

        std::vector<std::uint64_t> raw_pairs;
        raw_pairs.reserve((std::size_t) std::min<std::uint64_t>(
            bounds.maximum_raw_pair_occurrences, config.maximum_raw_pair_occurrences));
        for (std::size_t begin = 0u; begin < records.size();) {
            std::size_t end = begin + 1u;
            while (end < records.size() && records[end].key == records[begin].key
                   && records[end].band == records[begin].band) ++end;
            ++provenance.bucket_count;
            const std::size_t bucket_size = end - begin;
            const std::size_t selected = std::min<std::size_t>(bucket_size, config.maximum_bucket_size);
            std::size_t start = 0u;
            if (bucket_size > config.maximum_bucket_size) {
                ++provenance.oversized_bucket_count;
                provenance.discarded_bucket_members += bucket_size - config.maximum_bucket_size;
                start = (std::size_t) detail::oversized_bucket_window_start_v1(
                    records[begin].key, config.seed, records[begin].band, bucket_size);
            }
            for (std::size_t i = 0u; i < selected; ++i) {
                const ct::idx_t gene_i = records[begin + (start + i) % bucket_size].gene;
                for (std::size_t j = i + 1u; j < selected; ++j) {
                    const ct::idx_t gene_j = records[begin + (start + j) % bucket_size].gene;
                    if (gene_i != gene_j) raw_pairs.push_back(encode_pair(gene_i, gene_j));
                }
            }
            begin = end;
        }
        if (raw_pairs.size() > config.maximum_raw_pair_occurrences) {
            detail::set_error(error, "candidate raw pairs exceeded configured budget");
            return false;
        }
        provenance.raw_pair_occurrences = raw_pairs.size();
        std::sort(raw_pairs.begin(), raw_pairs.end());
        raw_pairs.erase(std::unique(raw_pairs.begin(), raw_pairs.end()), raw_pairs.end());
        return allocate_pairs_from_encoded(raw_pairs, std::move(provenance), out, error);
    } catch (const std::bad_alloc &) {
        detail::set_error(error, "failed to allocate CPU candidate-discovery scratch");
        return false;
    }
}

} // namespace cellerator::compute::gene_candidates
